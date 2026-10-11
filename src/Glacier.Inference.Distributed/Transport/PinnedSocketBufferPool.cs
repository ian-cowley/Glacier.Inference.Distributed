namespace Glacier.Inference.Distributed;

using System;
using System.Runtime.InteropServices;

/// <summary>
/// Pre-allocated unmanaged pinned buffer pool for zero-allocation socket transport.
/// Eliminates buffer resizing, pinned memory copying, and GC LOH fragmentation.
/// </summary>
public sealed class PinnedSocketBufferPool : IDisposable
{
    private readonly int _bufferSize;
    private readonly int _maxBuffers;
    private readonly byte[][] _pool;
    private readonly GCHandle[] _handles;
    private readonly bool[] _inUse;
    private readonly object _lock = new();
    private bool _disposed;

    /// <summary>
    /// Initializes a new pool of pinned byte arrays.
    /// </summary>
    /// <param name="bufferSize">Size in bytes for each contiguous buffer.</param>
    /// <param name="maxBuffers">Maximum number of pre-allocated buffers.</param>
    public PinnedSocketBufferPool(int bufferSize = 64 * 1024, int maxBuffers = 16)
    {
        if (bufferSize <= 0) throw new ArgumentOutOfRangeException(nameof(bufferSize), "Buffer size must be positive.");
        if (maxBuffers <= 0) throw new ArgumentOutOfRangeException(nameof(maxBuffers), "Max buffers must be positive.");

        _bufferSize = bufferSize;
        _maxBuffers = maxBuffers;
        _pool = new byte[maxBuffers][];
        _handles = new GCHandle[maxBuffers];
        _inUse = new bool[maxBuffers];

        for (int i = 0; i < maxBuffers; i++)
        {
            _pool[i] = GC.AllocateArray<byte>(bufferSize, pinned: true);
            _handles[i] = GCHandle.Alloc(_pool[i], GCHandleType.Pinned);
            _inUse[i] = false;
        }
    }

    /// <summary>Gets the fixed byte size of each pooled buffer.</summary>
    public int BufferSize => _bufferSize;

    /// <summary>Gets the total capacity of the pool.</summary>
    public int MaxBuffers => _maxBuffers;

    /// <summary>Gets the count of currently available buffers in the pool.</summary>
    public int AvailableCount
    {
        get
        {
            lock (_lock)
            {
                if (_disposed) return 0;
                int count = 0;
                for (int i = 0; i < _maxBuffers; i++)
                {
                    if (!_inUse[i]) count++;
                }
                return count;
            }
        }
    }

    /// <summary>
    /// Rents a pinned memory slice from the pool.
    /// </summary>
    public Memory<byte> Rent()
    {
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            for (int i = 0; i < _maxBuffers; i++)
            {
                if (!_inUse[i])
                {
                    _inUse[i] = true;
                    return _pool[i].AsMemory();
                }
            }
            throw new InvalidOperationException("Pinned buffer pool exhausted!");
        }
    }

    /// <summary>
    /// Rents the underlying pinned byte array directly.
    /// </summary>
    public byte[] RentArray()
    {
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            for (int i = 0; i < _maxBuffers; i++)
            {
                if (!_inUse[i])
                {
                    _inUse[i] = true;
                    return _pool[i];
                }
            }
            throw new InvalidOperationException("Pinned buffer pool exhausted!");
        }
    }

    /// <summary>
    /// Returns a previously rented memory slice back to the pool.
    /// Safely ignores foreign buffers.
    /// </summary>
    public void Return(Memory<byte> buffer)
    {
        lock (_lock)
        {
            if (_disposed) return;
            for (int i = 0; i < _maxBuffers; i++)
            {
                if (MemoryMarshal.TryGetArray(buffer, out ArraySegment<byte> seg) && ReferenceEquals(seg.Array, _pool[i]))
                {
                    _inUse[i] = false;
                    return;
                }
            }
        }
    }

    /// <summary>
    /// Returns a previously rented byte array directly back to the pool.
    /// </summary>
    public void ReturnArray(byte[] buffer)
    {
        lock (_lock)
        {
            if (_disposed) return;
            for (int i = 0; i < _maxBuffers; i++)
            {
                if (ReferenceEquals(buffer, _pool[i]))
                {
                    _inUse[i] = false;
                    return;
                }
            }
        }
    }

    /// <summary>
    /// Disposes the pool, releasing all pinned GCHandles.
    /// </summary>
    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            for (int i = 0; i < _maxBuffers; i++)
            {
                if (_handles[i].IsAllocated)
                {
                    _handles[i].Free();
                }
            }
        }
    }
}
