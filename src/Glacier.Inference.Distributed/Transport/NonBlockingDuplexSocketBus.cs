namespace Glacier.Inference.Distributed.Transport;

using System;
using System.Diagnostics;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;

/// <summary>
/// Unmanaged 24-byte struct representing a single-token result received from remote pipeline stage.
/// Strictly zero-allocation on the channel bus hot path.
/// </summary>
public readonly record struct TokenResultFrame(
    ulong SequenceId,
    int TokenId,
    float TopLogit,
    double ComputeMs);

/// <summary>
/// High-throughput non-blocking duplex transport bus decoupling GPU execution from TCP socket I/O.
/// Features a background receive loop reading cluster result frames into a bounded Channel,
/// serialized full-duplex socket writes, lock-free atomic watchdog tracking, and clean cancellation.
/// </summary>
public sealed class NonBlockingDuplexSocketBus : IDisposable, IAsyncDisposable
{
    public const double DefaultWatchdogSeconds = 60.0;
    public const int DefaultInlinePayloadCapacityBytes = 256 * 1024;

    private readonly Socket _socket;
    private readonly Channel<TokenResultFrame> _channel;
    private readonly byte[] _recvBuffer;
    private readonly Memory<byte> _recvMemory;
    private readonly byte[] _pongBuffer;
    private readonly SemaphoreSlim _sendLock = new(1, 1);
    private readonly CancellationTokenSource _cts = new();
    private long _lastActivityTimestamp;
    private Task? _receiveTask;
    private int _disposed;
    private int _resourcesCleaned;
    private readonly TimeSpan _watchdogTimeout;

    public ChannelReader<TokenResultFrame> Reader => _channel.Reader;
    public Socket Socket => _socket;
    public TimeSpan WatchdogTimeout => _watchdogTimeout;

    public NonBlockingDuplexSocketBus(
        Socket socket,
        int channelCapacity = 4,
        TimeSpan? watchdogTimeout = null,
        int maxPayloadBytes = DefaultInlinePayloadCapacityBytes)
    {
        _socket = socket ?? throw new ArgumentNullException(nameof(socket));
        ArgumentOutOfRangeException.ThrowIfLessThan(channelCapacity, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxPayloadBytes, 1024);

        _watchdogTimeout = watchdogTimeout ?? TimeSpan.FromSeconds(DefaultWatchdogSeconds);
        _recvBuffer = GC.AllocateArray<byte>(FusedSocketTransport.CombinedHeaderSize + maxPayloadBytes, pinned: true);
        _recvMemory = _recvBuffer.AsMemory();
        _pongBuffer = GC.AllocateArray<byte>(ClusterFrameHeader.HeaderSize, pinned: true);
        _lastActivityTimestamp = Stopwatch.GetTimestamp();

        _channel = Channel.CreateBounded<TokenResultFrame>(new BoundedChannelOptions(channelCapacity)
        {
            SingleWriter = true,
            SingleReader = true,
            FullMode = BoundedChannelFullMode.Wait,
            AllowSynchronousContinuations = true
        });
    }

    /// <summary>
    /// Computes an adaptive watchdog timeout scaled to prompt token length to accommodate longer prefill delays.
    /// </summary>
    public static TimeSpan CalculateAdaptiveTimeout(int promptTokens)
    {
        return TimeSpan.FromSeconds(Math.Max(DefaultWatchdogSeconds, 0.05 * promptTokens + 10.0));
    }

    /// <summary>
    /// Atomically updates the last peer activity timestamp to extend the watchdog window without cross-thread CTS races.
    /// </summary>
    public void ResetWatchdog()
    {
        Interlocked.Exchange(ref _lastActivityTimestamp, Stopwatch.GetTimestamp());
    }

    /// <summary>
    /// Serializes outbound socket writes through an internal semaphore so foreground activation sends
    /// never race concurrently with background HeartbeatPong responses.
    /// </summary>
    public async ValueTask SendSerializedAsync(ReadOnlyMemory<byte> buffer, CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        bool acquired = false;
        try
        {
            await _sendLock.WaitAsync(ct).ConfigureAwait(false);
            acquired = true;
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

            int sent = 0;
            while (sent < buffer.Length)
            {
                int r = await _socket.SendAsync(buffer.Slice(sent), SocketFlags.None, ct).ConfigureAwait(false);
                if (r <= 0) throw new SocketException((int)SocketError.ConnectionReset);
                sent += r;
            }
        }
        catch (ObjectDisposedException) when (Volatile.Read(ref _disposed) != 0)
        {
            throw new ObjectDisposedException(nameof(NonBlockingDuplexSocketBus));
        }
        finally
        {
            if (acquired)
            {
                try { _sendLock.Release(); }
                catch (ObjectDisposedException) { }
            }
        }
    }

    /// <summary>
    /// Starts the background receive loop task.
    /// </summary>
    public void Start()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_receiveTask != null) return;
        ResetWatchdog();

        CancellationToken loopToken;
        try
        {
            loopToken = _cts.Token;
        }
        catch (ObjectDisposedException)
        {
            throw new ObjectDisposedException(nameof(NonBlockingDuplexSocketBus));
        }

        _receiveTask = Task.Run(() => ReceiveLoopAsync(loopToken));
    }

    private async Task ReceiveLoopAsync(CancellationToken ct)
    {
        try
        {
            while (!ct.IsCancellationRequested && Volatile.Read(ref _disposed) == 0)
            {
                var (clusterHdr, resultHdr, _) = await FusedSocketTransport.ReceiveResultAsync(
                    _socket, _recvMemory, ct).ConfigureAwait(false);

                if (clusterHdr.Magic != ClusterFrameHeader.MagicConstant)
                {
                    _channel.Writer.TryComplete();
                    break;
                }

                ResetWatchdog();

                if (clusterHdr.MessageType == ClusterMessageType.HeartbeatPing ||
                    clusterHdr.MessageType == ClusterMessageType.HeartbeatPong)
                {
                    if (clusterHdr.MessageType == ClusterMessageType.HeartbeatPing && Volatile.Read(ref _disposed) == 0)
                    {
                        bool acquired = false;
                        try
                        {
                            await _sendLock.WaitAsync(ct).ConfigureAwait(false);
                            acquired = true;
                            if (Volatile.Read(ref _disposed) == 0)
                            {
                                var pongHdr = new ClusterFrameHeader(ClusterMessageType.HeartbeatPong, clusterHdr.CorrelationId, 0);
                                MemoryMarshal.Write(_pongBuffer, in pongHdr);
                                int sent = 0;
                                while (sent < ClusterFrameHeader.HeaderSize)
                                {
                                    int r = await _socket.SendAsync(
                                        _pongBuffer.AsMemory(sent, ClusterFrameHeader.HeaderSize - sent),
                                        SocketFlags.None,
                                        ct).ConfigureAwait(false);
                                    if (r <= 0) throw new SocketException((int)SocketError.ConnectionReset);
                                    sent += r;
                                }
                            }
                        }
                        finally
                        {
                            if (acquired)
                            {
                                try { _sendLock.Release(); }
                                catch (ObjectDisposedException) { }
                            }
                        }
                    }

                    continue;
                }

                if (clusterHdr.MessageType != FusedSocketTransport.MsgResult ||
                    clusterHdr.PayloadLength < (ulong)FusedActivationResultHeader.HeaderSize)
                {
                    continue;
                }

                var frame = new TokenResultFrame(
                    clusterHdr.CorrelationId,
                    resultHdr.NextTokenId,
                    resultHdr.TopLogit,
                    resultHdr.ComputeMs);

                await _channel.Writer.WriteAsync(frame, ct).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException)
        {
            _channel.Writer.TryComplete();
        }
        catch (ObjectDisposedException) when (Volatile.Read(ref _disposed) != 0)
        {
            _channel.Writer.TryComplete();
        }
        catch (SocketException) when (Volatile.Read(ref _disposed) != 0)
        {
            _channel.Writer.TryComplete();
        }
        catch (Exception ex)
        {
            _channel.Writer.TryComplete(ex);
        }
        finally
        {
            if (Volatile.Read(ref _disposed) != 0)
            {
                CleanupKernelPrimitivesOnce();
            }
        }
    }

    /// <summary>
    /// Asynchronously receives the next token result from the channel, bounded by the watchdog timeout.
    /// Hot path is strictly zero-allocation when a token frame is already buffered.
    /// </summary>
    public ValueTask<TokenResultFrame> ReceiveTokenAsync(CancellationToken cancellationToken = default)
    {
        if (_channel.Reader.TryRead(out var frame))
        {
            return new ValueTask<TokenResultFrame>(frame);
        }

        return ReceiveTokenSlowAsync(cancellationToken);
    }

    private async ValueTask<TokenResultFrame> ReceiveTokenSlowAsync(CancellationToken cancellationToken)
    {
        ResetWatchdog();

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (_channel.Reader.TryRead(out var readyFrame))
            {
                return readyFrame;
            }

            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

            long lastTicks = Interlocked.Read(ref _lastActivityTimestamp);
            TimeSpan elapsedSinceActivity = Stopwatch.GetElapsedTime(lastTicks);
            TimeSpan remaining = _watchdogTimeout - elapsedSinceActivity;

            if (remaining <= TimeSpan.Zero)
            {
                throw new TimeoutException($"Stage token response timed out after {_watchdogTimeout.TotalSeconds:F1}s watchdog limit.");
            }

            CancellationTokenSource waitCts;
            try
            {
                waitCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, _cts.Token);
            }
            catch (ObjectDisposedException)
            {
                throw new ObjectDisposedException(nameof(NonBlockingDuplexSocketBus));
            }

            using (waitCts)
            {
                waitCts.CancelAfter(remaining);

                try
                {
                    if (await _channel.Reader.WaitToReadAsync(waitCts.Token).ConfigureAwait(false))
                    {
                        if (_channel.Reader.TryRead(out var frame))
                        {
                            return frame;
                        }
                    }
                    else
                    {
                        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
                        return await _channel.Reader.ReadAsync(cancellationToken).ConfigureAwait(false);
                    }
                }
                catch (OperationCanceledException) when (Volatile.Read(ref _disposed) != 0 && !cancellationToken.IsCancellationRequested)
                {
                    throw new ObjectDisposedException(nameof(NonBlockingDuplexSocketBus));
                }
                catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested && !_cts.IsCancellationRequested)
                {
                    // Watchdog slice expired; loop re-evaluates _lastActivityTimestamp in case heartbeats arrived.
                }
            }
        }
    }

    private void SignalDisposeCore()
    {
        try { _cts.Cancel(); }
        catch (ObjectDisposedException) { }

        _channel.Writer.TryComplete();

        // Unblock any pending socket ReadAsync/SendAsync cleanly without disposing _sendLock/_cts while ReceiveLoopAsync unwinds
        try
        {
            if (_socket.Connected)
            {
                _socket.Shutdown(SocketShutdown.Both);
            }
        }
        catch { }

        try { _socket.Close(); }
        catch { }
    }

    private void CleanupKernelPrimitivesOnce()
    {
        if (Interlocked.Exchange(ref _resourcesCleaned, 1) != 0) return;
        try { _cts.Dispose(); } catch { }
        try { _sendLock.Dispose(); } catch { }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        SignalDisposeCore();

        var receiveTask = _receiveTask;
        if (receiveTask == null || receiveTask.IsCompleted)
        {
            CleanupKernelPrimitivesOnce();
        }
        else
        {
            _ = receiveTask.ContinueWith(
                static (_, state) => ((NonBlockingDuplexSocketBus)state!).CleanupKernelPrimitivesOnce(),
                this,
                CancellationToken.None,
                TaskContinuationOptions.ExecuteSynchronously,
                TaskScheduler.Default);
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            SignalDisposeCore();
        }

        var receiveTask = _receiveTask;
        if (receiveTask != null)
        {
            try { await receiveTask.ConfigureAwait(false); }
            catch { }
        }

        CleanupKernelPrimitivesOnce();
    }
}
