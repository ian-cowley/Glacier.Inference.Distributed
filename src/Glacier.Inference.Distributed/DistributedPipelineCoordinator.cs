namespace Glacier.Inference.Distributed;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO.Pipelines;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Cluster.Shuffle;

/// <summary>
/// Pipeline partition coordinator managing multi-stage activation handoff across heterogeneous inference stages
/// (e.g. Stage 0 Attention/Root -> Stage 1 Warm Expert Pool -> Stage 2 Residual Accumulator).
/// Supports both zero-allocation in-memory execution and zero-copy wire frame transport over System.IO.Pipelines.
/// </summary>
public sealed class DistributedPipelineCoordinator : IDisposable
{
    private readonly AttentionRootStageWorker _stage0;
    private readonly IInferenceStageWorker _stage1;
    private readonly AccumulatorStageWorker _stage2;
    private readonly int _hiddenDim;
    private int _disposed;

    /// <summary>
    /// Gets the hidden dimension (D) of activations processed by this coordinator.
    /// </summary>
    public int HiddenDim => _hiddenDim;

    /// <summary>
    /// Gets Stage 0 (Attention / Ingress).
    /// </summary>
    public AttentionRootStageWorker Stage0 => _stage0;

    /// <summary>
    /// Gets Stage 1 (Expert Pool).
    /// </summary>
    public IInferenceStageWorker Stage1 => _stage1;

    /// <summary>
    /// Gets Stage 2 (Accumulator).
    /// </summary>
    public AccumulatorStageWorker Stage2 => _stage2;

    public DistributedPipelineCoordinator(
        AttentionRootStageWorker stage0,
        IInferenceStageWorker stage1,
        AccumulatorStageWorker stage2,
        int hiddenDim)
    {
        _stage0 = stage0 ?? throw new ArgumentNullException(nameof(stage0));
        _stage1 = stage1 ?? throw new ArgumentNullException(nameof(stage1));
        _stage2 = stage2 ?? throw new ArgumentNullException(nameof(stage2));
        _hiddenDim = hiddenDim;
    }

    /// <summary>
    /// Executes a full 3-stage inference step (Stage 0 -> Stage 1 -> Stage 2) for an activated set of experts.
    /// Hot-path zero GC allocation execution.
    /// </summary>
    /// <param name="sequenceId">Logical request sequence identifier.</param>
    /// <param name="tokenIndex">Token index in sequence.</param>
    /// <param name="layerIndex">Transformer layer index.</param>
    /// <param name="inputEmbedding">Input activation or token embedding vector [D].</param>
    /// <param name="selectedExperts">Array of activated expert IDs.</param>
    /// <param name="expertWeights">Softmax gating weights corresponding to selected experts.</param>
    /// <returns>A ref-counted partition buffer containing the final accumulated activation tensor.</returns>
    public unsafe RefCountedPartitionBuffer ExecuteStep(
        ulong sequenceId,
        int tokenIndex,
        int layerIndex,
        ReadOnlySpan<byte> inputEmbedding,
        ReadOnlySpan<int> selectedExperts,
        ReadOnlySpan<float> expertWeights)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);

        if (selectedExperts.Length != expertWeights.Length)
        {
            throw new ArgumentException("selectedExperts and expertWeights must have identical lengths.");
        }

        int activationBytes = _hiddenDim * sizeof(float);
        if (inputEmbedding.Length < activationBytes)
        {
            throw new ArgumentException($"inputEmbedding must be at least {activationBytes} bytes.", nameof(inputEmbedding));
        }

        // --- STAGE 0: Root Attention & Normalization ---
        var header0 = new ActivationHeader(
            sequenceId: sequenceId,
            tokenIndex: tokenIndex,
            layerIndex: layerIndex,
            hiddenDim: _hiddenDim,
            expertId: selectedExperts.Length > 0 ? selectedExperts[0] : 0,
            flags: ActivationFlags.None);

        using var stage0Buffer = _stage0.ProcessActivation(header0, inputEmbedding);

        // --- STAGE 2: Allocate accumulation buffer ---
        var accumulator = RefCountedPartitionBuffer.Allocate(activationBytes);
        float* pAcc = (float*)accumulator.UnmanagedPointer;
        fixed (byte* inPtr = inputEmbedding)
        {
            float* pResidual = (float*)inPtr;
            // Initialize with residual stream
            accumulator.Span.Clear();
            _stage2.AddResidual(pAcc, pResidual);
        }

        // --- STAGE 1: Process each selected expert through Warm Expert Pool ---
        for (int k = 0; k < selectedExperts.Length; k++)
        {
            int expertId = selectedExperts[k];
            float weight = expertWeights[k];

            var header1 = new ActivationHeader(
                sequenceId: sequenceId,
                tokenIndex: tokenIndex,
                layerIndex: layerIndex,
                hiddenDim: _hiddenDim,
                expertId: expertId,
                flags: k == selectedExperts.Length - 1 ? ActivationFlags.FinalLayer : ActivationFlags.None);

            using var expertOutput = _stage1.ProcessActivation(header1, stage0Buffer.Span);
            float* pExpert = (float*)expertOutput.UnmanagedPointer;

            // --- STAGE 2: Accumulate weighted expert output ---
            _stage2.AccumulateWeightedExpert(pAcc, pExpert, weight);
        }

        return accumulator;
    }

    /// <summary>
    /// Zero-allocation hot path: executes a full 3-stage inference step directly into caller-provided unmanaged buffers.
    /// Guarantees exactly 0 bytes allocated on the managed GC heap.
    /// </summary>
    public unsafe void ExecuteStepZeroAlloc(
        ulong sequenceId,
        int tokenIndex,
        int layerIndex,
        float* inputEmbedding,
        ReadOnlySpan<int> selectedExperts,
        ReadOnlySpan<float> expertWeights,
        float* destination,
        float* scratchStage0,
        float* scratchExpert)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        ArgumentNullException.ThrowIfNull(inputEmbedding);
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(scratchStage0);
        ArgumentNullException.ThrowIfNull(scratchExpert);

        if (selectedExperts.Length != expertWeights.Length)
        {
            throw new ArgumentException("selectedExperts and expertWeights must have identical lengths.");
        }

        // --- STAGE 0: Root Attention & Normalization ---
        var header0 = new ActivationHeader(
            sequenceId: sequenceId,
            tokenIndex: tokenIndex,
            layerIndex: layerIndex,
            hiddenDim: _hiddenDim,
            expertId: selectedExperts.Length > 0 ? selectedExperts[0] : 0,
            flags: ActivationFlags.None);

        _stage0.ProcessActivation(header0, inputEmbedding, scratchStage0);

        // --- STAGE 2: Initialize destination with residual stream ---
        NativeMemory.Clear(destination, (nuint)(_hiddenDim * sizeof(float)));
        _stage2.AddResidual(destination, inputEmbedding);

        // --- STAGE 1: Process each selected expert ---
        for (int k = 0; k < selectedExperts.Length; k++)
        {
            int expertId = selectedExperts[k];
            float weight = expertWeights[k];

            var header1 = new ActivationHeader(
                sequenceId: sequenceId,
                tokenIndex: tokenIndex,
                layerIndex: layerIndex,
                hiddenDim: _hiddenDim,
                expertId: expertId,
                flags: k == selectedExperts.Length - 1 ? ActivationFlags.FinalLayer : ActivationFlags.None);

            if (_stage1 is CpuWarmExpertPoolWorker cpuWorker)
            {
                cpuWorker.ProcessActivation(header1, scratchStage0, scratchExpert);
            }
            else
            {
                using var expertOutput = _stage1.ProcessActivation(header1, new ReadOnlySpan<byte>(scratchStage0, _hiddenDim * sizeof(float)));
                NativeMemory.Copy(expertOutput.UnmanagedPointer, scratchExpert, (nuint)(_hiddenDim * sizeof(float)));
            }

            // --- STAGE 2: Accumulate weighted expert output ---
            _stage2.AccumulateWeightedExpert(destination, scratchExpert, weight);
        }
    }

    /// <summary>
    /// Asynchronously executes a 3-stage inference step.
    /// </summary>
    public ValueTask<RefCountedPartitionBuffer> ExecuteStepAsync(
        ulong sequenceId,
        int tokenIndex,
        int layerIndex,
        ReadOnlyMemory<byte> inputEmbedding,
        int[] selectedExperts,
        float[] expertWeights,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var buffer = ExecuteStep(sequenceId, tokenIndex, layerIndex, inputEmbedding.Span, selectedExperts, expertWeights);
        return ValueTask.FromResult(buffer);
    }

    /// <summary>
    /// Writes an activation frame to an output pipe over <see cref="ClusterFrameHeader"/> wire framing.
    /// Frame layout: 32-byte ClusterFrameHeader + 32-byte ActivationHeader + Activation payload bytes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static async ValueTask WriteActivationFrameAsync(
        PipeWriter writer,
        ActivationHeader header,
        ReadOnlyMemory<byte> activation,
        ClusterMessageType messageType = ClusterMessageType.StageDispatch,
        ulong correlationId = 0,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(writer);

        int payloadLen = ActivationHeader.HeaderSize + activation.Length;
        var clusterHeader = new ClusterFrameHeader(
            messageType: messageType,
            correlationId: correlationId != 0 ? correlationId : header.SequenceId,
            payloadLength: (ulong)payloadLen,
            flags: ClusterFrameFlags.None);

        // Pre-allocate header chunk
        int headerBytes = ClusterFrameHeader.HeaderSize + ActivationHeader.HeaderSize;
        var span = writer.GetSpan(headerBytes);
        clusterHeader.WriteTo(span.Slice(0, ClusterFrameHeader.HeaderSize));
        header.WriteTo(span.Slice(ClusterFrameHeader.HeaderSize, ActivationHeader.HeaderSize));
        writer.Advance(headerBytes);

        if (!activation.IsEmpty)
        {
            var result = await writer.WriteAsync(activation, cancellationToken).ConfigureAwait(false);
            if (result.IsCanceled) throw new OperationCanceledException(cancellationToken);
        }
        else
        {
            await writer.FlushAsync(cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Reads an activation frame from an input pipe over <see cref="ClusterFrameHeader"/> wire framing.
    /// Validates complete payload arrival before slicing.
    /// </summary>
    public static async ValueTask<(ActivationHeader Header, RefCountedPartitionBuffer Payload)> ReadActivationFrameAsync(
        PipeReader reader,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(reader);

        while (true)
        {
            var readResult = await reader.ReadAsync(cancellationToken).ConfigureAwait(false);
            var buffer = readResult.Buffer;

            if (ClusterFrameReader.TryReadFrame(ref buffer, out var clusterHeader, out var payload))
            {
                if (payload.Length < ActivationHeader.HeaderSize)
                {
                    reader.AdvanceTo(buffer.Start, buffer.End);
                    throw new InvalidOperationException($"Payload length ({payload.Length}) is smaller than ActivationHeader size (32).");
                }

                // Parse ActivationHeader from first 32 bytes of payload
                Span<byte> headerBytes = stackalloc byte[ActivationHeader.HeaderSize];
                payload.Slice(0, ActivationHeader.HeaderSize).CopyTo(headerBytes);
                var actHeader = ActivationHeader.ReadFrom(headerBytes);

                // Copy remaining payload into RefCountedPartitionBuffer
                var actPayload = payload.Slice(ActivationHeader.HeaderSize);
                var bufferOwner = RefCountedPartitionBuffer.CreateCopy(actPayload);

                reader.AdvanceTo(buffer.Start);
                return (actHeader, bufferOwner);
            }

            reader.AdvanceTo(buffer.Start, buffer.End);

            if (readResult.IsCompleted)
            {
                throw new InvalidOperationException("Pipe completed before full activation frame arrived.");
            }
        }
    }

    /// <summary>
    /// Serves an asynchronous stage worker service loop over an <see cref="IDuplexPipe"/>.
    /// Continuously reads incoming activation frames, processes them via <paramref name="worker"/>,
    /// and streams output frames back to the client.
    /// </summary>
    public static async Task RunWorkerServiceLoopAsync(
        IDuplexPipe duplexPipe,
        IInferenceStageWorker worker,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(duplexPipe);
        ArgumentNullException.ThrowIfNull(worker);

        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                var (header, inputBuffer) = await ReadActivationFrameAsync(duplexPipe.Input, cancellationToken).ConfigureAwait(false);
                using (inputBuffer)
                {
                    using var outputBuffer = await worker.ProcessActivationAsync(
                        header,
                        inputBuffer.Memory,
                        cancellationToken).ConfigureAwait(false);

                    var responseHeader = new ActivationHeader(
                        sequenceId: header.SequenceId,
                        tokenIndex: header.TokenIndex,
                        layerIndex: header.LayerIndex,
                        hiddenDim: header.HiddenDim,
                        expertId: header.ExpertId,
                        flags: header.Flags | (uint)ActivationFlags.AcknowledgeRequired);

                    await WriteActivationFrameAsync(
                        duplexPipe.Output,
                        responseHeader,
                        outputBuffer.Memory,
                        messageType: ClusterMessageType.PartitionFetchResponse,
                        correlationId: header.SequenceId,
                        cancellationToken: cancellationToken).ConfigureAwait(false);
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Graceful cancellation
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _stage0.Dispose();
        _stage1.Dispose();
        _stage2.Dispose();
    }
}
