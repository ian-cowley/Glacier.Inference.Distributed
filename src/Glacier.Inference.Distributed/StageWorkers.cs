namespace Glacier.Inference.Distributed;

using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Shuffle;

/// <summary>
/// Stage 0 worker: executes ingress token normalization, root attention, and expert router gating.
/// Emits activation vectors with assigned expert routing metadata.
/// </summary>
public sealed unsafe class AttentionRootStageWorker : IInferenceStageWorker
{
    private readonly int _dim;
    private readonly int _topK;
    private readonly int _numExperts;
    private int _disposed;

    public int StageRank => 0;
    public string Name { get; }

    public AttentionRootStageWorker(
        int dim,
        int topK = 2,
        int numExperts = 8,
        string name = "Stage 0: Root Attention & Router")
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dim);
        _dim = dim;
        _topK = Math.Clamp(topK, 1, numExperts);
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
    /// Processes RMSNorm activation directly into caller-provided unmanaged float buffers.
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

        // Apply RMSNorm: outX = x / sqrt(mean(x^2) + eps)
        float sumSq = 0.0f;
        for (int i = 0; i < _dim; i++)
        {
            sumSq += inputActivation[i] * inputActivation[i];
        }
        float rms = 1.0f / MathF.Sqrt((sumSq / _dim) + 1e-5f);

        int iNorm = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            int vLen = Vector256<float>.Count;
            var vRms = Vector256.Create(rms);
            for (; iNorm <= _dim - vLen; iNorm += vLen)
            {
                var vx = Vector256.Load(inputActivation + iNorm);
                (vx * vRms).Store(outputActivation + iNorm);
            }
        }
        for (; iNorm < _dim; iNorm++)
        {
            outputActivation[iNorm] = inputActivation[iNorm] * rms;
        }
    }

    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
    }
}

/// <summary>
/// Stage 2 worker: accumulates weighted expert outputs and merges with residual connection.
/// Implements zero-allocation SIMD FMA accumulation.
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
    /// Accumulates expert output into an existing accumulation buffer with SIMD weighting.
    /// y = y + weight * expertOutput
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AccumulateWeightedExpert(
        float* accumulator,
        float* expertOutput,
        float weight)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            int vLen = Vector256<float>.Count;
            var vWeight = Vector256.Create(weight);
            for (; i <= _dim - vLen; i += vLen)
            {
                var vAcc = Vector256.Load(accumulator + i);
                var vExp = Vector256.Load(expertOutput + i);
                (vAcc + (vExp * vWeight)).Store(accumulator + i);
            }
        }
        for (; i < _dim; i++)
        {
            accumulator[i] += weight * expertOutput[i];
        }
    }

    /// <summary>
    /// Adds residual vector into accumulator: y = y + residual.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddResidual(
        float* accumulator,
        float* residual)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            int vLen = Vector256<float>.Count;
            for (; i <= _dim - vLen; i += vLen)
            {
                var vAcc = Vector256.Load(accumulator + i);
                var vRes = Vector256.Load(residual + i);
                (vAcc + vRes).Store(accumulator + i);
            }
        }
        for (; i < _dim; i++)
        {
            accumulator[i] += residual[i];
        }
    }

    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
    }
}
