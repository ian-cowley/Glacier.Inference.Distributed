namespace Glacier.Inference.Distributed;

using System;
using System.Diagnostics.CodeAnalysis;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Shuffle;

/// <summary>
/// Stage 0 synthetic RMSNorm ingress stage worker for 3-stage MoE shuffle micro-pipelines.
/// Note: Executes vectorized RMSNorm ingress normalization only (full self-attention and learned router
/// gating are handled by <c>IClusterStageModel</c> / <c>IPipelineStage</c> in full model pipelines).
/// </summary>
[Experimental("GLACIER_EXP001")]
public sealed unsafe class AttentionRootStageWorker : IInferenceStageWorker
{
    private readonly int _dim;
    private readonly int _topK;
    private readonly int _numExperts;
    private int _disposed;

    public int StageRank => 0;
    public string Name { get; }
    public int TopK => _topK;
    public int NumExperts => _numExperts;

    public AttentionRootStageWorker(
        int dim,
        int topK = 2,
        int numExperts = 8,
        string name = "Stage 0: RMSNorm Ingress")
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dim);
        _dim = dim;
        _topK = Math.Clamp(topK, 1, Math.Max(1, numExperts));
        _numExperts = numExperts;
        Name = name;
    }

    public ValueTask<RefCountedPartitionBuffer> ProcessActivationAsync(
        ActivationHeader header,
        ReadOnlyMemory<byte> inputActivation,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(ProcessActivation(header, inputActivation.Span));
    }

    public RefCountedPartitionBuffer ProcessActivation(
        ActivationHeader header,
        ReadOnlySpan<byte> inputActivation)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);

        int bytes = _dim * sizeof(float);
        if (inputActivation.Length < bytes)
        {
            throw new ArgumentException($"Input activation must be at least {bytes} bytes.", nameof(inputActivation));
        }

        var output = RefCountedPartitionBuffer.Allocate(bytes);

        fixed (byte* inPtr = inputActivation)
        {
            ProcessActivation(header, (float*)inPtr, (float*)output.UnmanagedPointer);
        }

        return output;
    }

    /// <summary>
    /// Processes vectorized RMSNorm activation directly into caller-provided unmanaged float buffers.
    /// Exactly zero GC heap allocations.
    /// </summary>
    public void ProcessActivation(
        ActivationHeader header,
        float* inputActivation,
        float* outputActivation)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        ArgumentNullException.ThrowIfNull(inputActivation);
        ArgumentNullException.ThrowIfNull(outputActivation);

        var inSpan = new ReadOnlySpan<float>(inputActivation, _dim);
        var outSpan = new Span<float>(outputActivation, _dim);

        float sumSq = TensorPrimitives.SumOfSquares(inSpan);
        float rms = 1.0f / MathF.Sqrt((sumSq / _dim) + 1e-5f);
        TensorPrimitives.Multiply(inSpan, rms, outSpan);
    }

    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
    }
}

/// <summary>
/// Stage 2 worker: accumulates weighted expert outputs and merges with residual connection
/// via hardware-accelerated <see cref="TensorPrimitives"/> SIMD kernels.
/// </summary>
public sealed unsafe class AccumulatorStageWorker : IInferenceStageWorker
{
    private readonly int _dim;
    private int _disposed;

    public int StageRank => 2;
    public string Name { get; }

    public AccumulatorStageWorker(
        int dim,
        string name = "Stage 2: Residual Accumulator")
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dim);
        _dim = dim;
        Name = name;
    }

    public ValueTask<RefCountedPartitionBuffer> ProcessActivationAsync(
        ActivationHeader header,
        ReadOnlyMemory<byte> inputActivation,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(ProcessActivation(header, inputActivation.Span));
    }

    public RefCountedPartitionBuffer ProcessActivation(
        ActivationHeader header,
        ReadOnlySpan<byte> inputActivation)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);

        int bytes = _dim * sizeof(float);
        if (inputActivation.Length < bytes)
        {
            throw new ArgumentException($"Input activation must be at least {bytes} bytes.", nameof(inputActivation));
        }

        var output = RefCountedPartitionBuffer.Allocate(bytes);
        inputActivation.Slice(0, bytes).CopyTo(output.Span);
        return output;
    }

    /// <summary>
    /// Accumulates expert output into an existing accumulation buffer with SIMD FMA weighting:
    /// <c>accumulator = accumulator + weight * expertOutput</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AccumulateWeightedExpert(
        float* accumulator,
        float* expertOutput,
        float weight)
    {
        var accSpan = new Span<float>(accumulator, _dim);
        var expSpan = new ReadOnlySpan<float>(expertOutput, _dim);
        TensorPrimitives.MultiplyAdd(expSpan, weight, accSpan, accSpan);
    }

    /// <summary>
    /// Adds residual vector into accumulator: <c>accumulator = accumulator + residual</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddResidual(
        float* accumulator,
        float* residual)
    {
        var accSpan = new Span<float>(accumulator, _dim);
        var resSpan = new ReadOnlySpan<float>(residual, _dim);
        TensorPrimitives.Add(accSpan, resSpan, accSpan);
    }

    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
    }
}
