namespace Glacier.Inference.Distributed;

using System;

/// <summary>
/// Pre-allocated pinned memory arena for large batched prompt prefill passes.
/// Eliminates >32 MB Large Object Heap (LOH) allocations and eliminates Gen 2 GC collection freezes
/// during batch prompt prefill and parallel token ingestion.
/// Arrays allocated via <see cref="GC.AllocateArray{T}(int, bool)"/> with <c>pinned: true</c> reside
/// directly in the Pinned Object Heap (POH) without requiring redundant <c>GCHandle</c> table entries.
/// </summary>
public sealed class PinnedBatchBufferArena : IDisposable
{
    private const int DefaultMaxBatchTokens = 2048;
    private const int DefaultHiddenDim = 5120;
    private const int DefaultMaxVocab = 152064;

    /// <summary>
    /// Reserved capacity for full remote sampling trailer (32B header + up to 256 recent token IDs = 1,056 bytes).
    /// Ensures full-chunk FP32 prefill frames with <c>computeLogits == true</c> never overflow wire buffers.
    /// </summary>
    public const int MaxSamplingTrailerBytes = 32 + (256 * sizeof(int));

    private readonly int _maxTotalFloats;
    private readonly int _maxBytePayload;

    private readonly byte[] _batchWireSendBuffer;
    private readonly byte[] _batchWireRecvBuffer;
    private readonly float[] _floatInputBuffer;
    private readonly float[] _floatOutputBuffer;
    private readonly float[] _logitsBuffer;
    private bool _disposed;

    /// <summary>
    /// Constructs a pinned batch arena with capacity for large prefill chunks.
    /// </summary>
    public PinnedBatchBufferArena(
        int maxBatchTokens = DefaultMaxBatchTokens,
        int hiddenDim = DefaultHiddenDim,
        int vocabSize = DefaultMaxVocab)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxBatchTokens);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hiddenDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabSize);

        _maxTotalFloats = maxBatchTokens * hiddenDim;
        // 64 bytes combined header + FP32 payload + full sampling trailer headroom
        _maxBytePayload = 64 + (_maxTotalFloats * sizeof(float)) + MaxSamplingTrailerBytes;

        _batchWireSendBuffer = GC.AllocateArray<byte>(_maxBytePayload, pinned: true);
        _batchWireRecvBuffer = GC.AllocateArray<byte>(_maxBytePayload, pinned: true);
        _floatInputBuffer = GC.AllocateArray<float>(_maxTotalFloats, pinned: true);
        _floatOutputBuffer = GC.AllocateArray<float>(_maxTotalFloats, pinned: true);
        _logitsBuffer = GC.AllocateArray<float>(vocabSize, pinned: true);
    }

    public int MaxTotalFloats => _maxTotalFloats;
    public int MaxBytePayload => _maxBytePayload;

    public byte[] BatchWireSendArray => _batchWireSendBuffer;
    public byte[] BatchWireRecvArray => _batchWireRecvBuffer;
    public float[] FloatInputArray => _floatInputBuffer;
    public float[] FloatOutputArray => _floatOutputBuffer;
    public float[] LogitsArray => _logitsBuffer;

    public Span<byte> GetBatchSendSpan(int requiredBytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (requiredBytes < 0 || requiredBytes > _maxBytePayload)
            throw new ArgumentOutOfRangeException(nameof(requiredBytes), $"Required bytes ({requiredBytes}) exceeds arena capacity ({_maxBytePayload}).");
        return _batchWireSendBuffer.AsSpan(0, requiredBytes);
    }

    public Span<byte> GetBatchRecvSpan(int requiredBytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (requiredBytes < 0 || requiredBytes > _maxBytePayload)
            throw new ArgumentOutOfRangeException(nameof(requiredBytes), $"Required bytes ({requiredBytes}) exceeds arena capacity ({_maxBytePayload}).");
        return _batchWireRecvBuffer.AsSpan(0, requiredBytes);
    }

    public Span<float> GetFloatInputSpan(int requiredCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (requiredCount < 0 || requiredCount > _maxTotalFloats)
            throw new ArgumentOutOfRangeException(nameof(requiredCount), $"Required float count ({requiredCount}) exceeds arena capacity ({_maxTotalFloats}).");
        return _floatInputBuffer.AsSpan(0, requiredCount);
    }

    public Span<float> GetFloatOutputSpan(int requiredCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (requiredCount < 0 || requiredCount > _maxTotalFloats)
            throw new ArgumentOutOfRangeException(nameof(requiredCount), $"Required float count ({requiredCount}) exceeds arena capacity ({_maxTotalFloats}).");
        return _floatOutputBuffer.AsSpan(0, requiredCount);
    }

    public Span<float> GetLogitsSpan(int requiredCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (requiredCount < 0 || requiredCount > _logitsBuffer.Length)
            throw new ArgumentOutOfRangeException(nameof(requiredCount), $"Required logits count ({requiredCount}) exceeds arena capacity ({_logitsBuffer.Length}).");
        return _logitsBuffer.AsSpan(0, requiredCount);
    }

    /// <summary>
    /// Resets all buffers in the arena to zero.
    /// </summary>
    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Array.Clear(_batchWireSendBuffer);
        Array.Clear(_batchWireRecvBuffer);
        Array.Clear(_floatInputBuffer);
        Array.Clear(_floatOutputBuffer);
        Array.Clear(_logitsBuffer);
    }

    public void Dispose()
    {
        _disposed = true;
    }
}
