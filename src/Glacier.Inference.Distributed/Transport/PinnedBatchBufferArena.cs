namespace Glacier.Inference.Distributed;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// Pre-allocated pinned memory arena for large batched prompt prefill passes.
/// Eliminates >32 MB Large Object Heap (LOH) allocations and eliminates Gen 2 GC collection freezes
/// during batch prompt prefill and parallel token ingestion.
/// </summary>
public sealed class PinnedBatchBufferArena : IDisposable
{
    private const int DefaultMaxBatchTokens = 2048;
    private const int DefaultHiddenDim = 5120;
    private const int DefaultMaxVocab = 152064;

    private readonly int _maxTotalFloats;
    private readonly int _maxBytePayload;

    private readonly byte[] _batchWireSendBuffer;
    private readonly byte[] _batchWireRecvBuffer;
    private readonly float[] _floatInputBuffer;
    private readonly float[] _floatOutputBuffer;
    private readonly float[] _logitsBuffer;

    private GCHandle _hSend;
    private GCHandle _hRecv;
    private GCHandle _hIn;
    private GCHandle _hOut;
    private GCHandle _hLogits;
    private bool _disposed;

    /// <summary>
    /// Constructs a pinned batch arena with capacity for large prefill chunks.
    /// </summary>
    public PinnedBatchBufferArena(
        int maxBatchTokens = DefaultMaxBatchTokens,
        int hiddenDim = DefaultHiddenDim,
        int vocabSize = DefaultMaxVocab)
    {
        _maxTotalFloats = maxBatchTokens * hiddenDim;
        // 64 bytes combined header + FP32 payload
        _maxBytePayload = 64 + (_maxTotalFloats * sizeof(float));

        _batchWireSendBuffer = GC.AllocateArray<byte>(_maxBytePayload, pinned: true);
        _batchWireRecvBuffer = GC.AllocateArray<byte>(_maxBytePayload, pinned: true);
        _floatInputBuffer = GC.AllocateArray<float>(_maxTotalFloats, pinned: true);
        _floatOutputBuffer = GC.AllocateArray<float>(_maxTotalFloats, pinned: true);
        _logitsBuffer = GC.AllocateArray<float>(vocabSize, pinned: true);

        _hSend = GCHandle.Alloc(_batchWireSendBuffer, GCHandleType.Pinned);
        _hRecv = GCHandle.Alloc(_batchWireRecvBuffer, GCHandleType.Pinned);
        _hIn = GCHandle.Alloc(_floatInputBuffer, GCHandleType.Pinned);
        _hOut = GCHandle.Alloc(_floatOutputBuffer, GCHandleType.Pinned);
        _hLogits = GCHandle.Alloc(_logitsBuffer, GCHandleType.Pinned);
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
        if (requiredBytes > _maxBytePayload)
            throw new ArgumentOutOfRangeException(nameof(requiredBytes), $"Required bytes ({requiredBytes}) exceeds arena capacity ({_maxBytePayload}).");
        return _batchWireSendBuffer.AsSpan(0, requiredBytes);
    }

    public Span<byte> GetBatchRecvSpan(int requiredBytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (requiredBytes > _maxBytePayload)
            throw new ArgumentOutOfRangeException(nameof(requiredBytes), $"Required bytes ({requiredBytes}) exceeds arena capacity ({_maxBytePayload}).");
        return _batchWireRecvBuffer.AsSpan(0, requiredBytes);
    }

    public Span<float> GetFloatInputSpan(int requiredCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (requiredCount > _maxTotalFloats)
            throw new ArgumentOutOfRangeException(nameof(requiredCount), $"Required float count ({requiredCount}) exceeds arena capacity ({_maxTotalFloats}).");
        return _floatInputBuffer.AsSpan(0, requiredCount);
    }

    public Span<float> GetFloatOutputSpan(int requiredCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (requiredCount > _maxTotalFloats)
            throw new ArgumentOutOfRangeException(nameof(requiredCount), $"Required float count ({requiredCount}) exceeds arena capacity ({_maxTotalFloats}).");
        return _floatOutputBuffer.AsSpan(0, requiredCount);
    }

    public Span<float> GetLogitsSpan(int requiredCount)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (requiredCount > _logitsBuffer.Length)
            throw new ArgumentOutOfRangeException(nameof(requiredCount), $"Required logits count ({requiredCount}) exceeds arena capacity ({_logitsBuffer.Length}).");
        return _logitsBuffer.AsSpan(0, requiredCount);
    }

    /// <summary>
    /// Resets all buffers in the arena to zero without releasing pinned handles.
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
        if (_disposed) return;
        _disposed = true;

        if (_hSend.IsAllocated) _hSend.Free();
        if (_hRecv.IsAllocated) _hRecv.Free();
        if (_hIn.IsAllocated) _hIn.Free();
        if (_hOut.IsAllocated) _hOut.Free();
        if (_hLogits.IsAllocated) _hLogits.Free();
    }
}
