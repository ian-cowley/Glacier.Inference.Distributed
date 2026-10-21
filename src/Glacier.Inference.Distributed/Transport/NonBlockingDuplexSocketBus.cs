namespace Glacier.Inference.Distributed.Transport;

using System;
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
/// zero-allocation unmanaged struct dispatch, an adaptive 60.0s per-token watchdog timeout, and clean cancellation.
/// </summary>
public sealed class NonBlockingDuplexSocketBus : IDisposable
{
    public const double DefaultWatchdogSeconds = 60.0;

    private readonly Socket _socket;
    private readonly Channel<TokenResultFrame> _channel;
    private readonly byte[] _recvBuffer;
    private readonly Memory<byte> _recvMemory;
    private readonly CancellationTokenSource _cts = new();
    private CancellationTokenSource? _reusableWatchdogCts;
    private volatile CancellationTokenSource? _activeWatchdogCts;
    private Task? _receiveTask;
    private int _disposed;
    private readonly TimeSpan _watchdogTimeout;

    public ChannelReader<TokenResultFrame> Reader => _channel.Reader;
    public Socket Socket => _socket;
    public TimeSpan WatchdogTimeout => _watchdogTimeout;

    public NonBlockingDuplexSocketBus(Socket socket, int channelCapacity = 4, TimeSpan? watchdogTimeout = null)
    {
        _socket = socket ?? throw new ArgumentNullException(nameof(socket));
        _watchdogTimeout = watchdogTimeout ?? TimeSpan.FromSeconds(DefaultWatchdogSeconds);
        _recvBuffer = new byte[FusedSocketTransport.CombinedHeaderSize + 1024];
        _recvMemory = _recvBuffer.AsMemory();
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
    /// Resets the active watchdog cancellation timer upon receiving a heartbeat or activity from the peer.
    /// </summary>
    public void ResetWatchdog()
    {
        try
        {
            _activeWatchdogCts?.CancelAfter(_watchdogTimeout);
        }
        catch (ObjectDisposedException) { }
    }

    /// <summary>
    /// Starts the background receive loop task.
    /// </summary>
    public void Start()
    {
        if (_receiveTask != null) return;
        _receiveTask = Task.Run(() => ReceiveLoopAsync(_cts.Token));
    }

    private async Task ReceiveLoopAsync(CancellationToken ct)
    {
        try
        {
            while (!ct.IsCancellationRequested)
            {
                var (clusterHdr, resultHdr, _) = await FusedSocketTransport.ReceiveResultAsync(
                    _socket, _recvMemory, ct).ConfigureAwait(false);

                if (clusterHdr.Magic != ClusterFrameHeader.MagicConstant)
                {
                    _channel.Writer.TryComplete();
                    break;
                }

                // Interleaved heartbeats keep the channel alive and reset the watchdog
                if (clusterHdr.MessageType == ClusterMessageType.HeartbeatPing ||
                    clusterHdr.MessageType == ClusterMessageType.HeartbeatPong)
                {
                    if (clusterHdr.MessageType == ClusterMessageType.HeartbeatPing)
                    {
                        var pongHdr = new ClusterFrameHeader(ClusterMessageType.HeartbeatPong, clusterHdr.CorrelationId, 0);
                        byte[] pBytes = new byte[ClusterFrameHeader.HeaderSize];
                        MemoryMarshal.Write(pBytes, in pongHdr);
                        await _socket.SendAsync(pBytes, SocketFlags.None, ct).ConfigureAwait(false);
                    }

                    ResetWatchdog();
                    continue;
                }

                // If not a token result frame, skip dispatch
                if (clusterHdr.MessageType != (ClusterMessageType)0x71 || clusterHdr.PayloadLength < (ulong)FusedActivationResultHeader.HeaderSize)
                {
                    ResetWatchdog();
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
        catch (Exception ex)
        {
            _channel.Writer.TryComplete(ex);
        }
    }

    /// <summary>
    /// Asynchronously receives the next token result from the channel, bounded by the watchdog timeout.
    /// Hot path is strictly zero-allocation: checks TryRead first, and reuses cached CancellationTokenSource when waiting.
    /// </summary>
    public ValueTask<TokenResultFrame> ReceiveTokenAsync(CancellationToken cancellationToken = default)
    {
        // 1. Hot path: if token result is already available in the channel, return immediately with zero allocation
        if (_channel.Reader.TryRead(out var frame))
        {
            return new ValueTask<TokenResultFrame>(frame);
        }

        // 2. Slow path: asynchronously wait with watchdog timeout
        return ReceiveTokenSlowAsync(cancellationToken);
    }

    private async ValueTask<TokenResultFrame> ReceiveTokenSlowAsync(CancellationToken cancellationToken)
    {
        CancellationTokenSource cts;
        bool isLinked = false;

        if (cancellationToken == default)
        {
            // Reuse cached CTS for 0-allocation watchdog when caller passes default token
            if (_reusableWatchdogCts == null || !_reusableWatchdogCts.TryReset())
            {
                _reusableWatchdogCts?.Dispose();
                _reusableWatchdogCts = new CancellationTokenSource();
            }
            cts = _reusableWatchdogCts;
            cts.CancelAfter(_watchdogTimeout);
        }
        else
        {
            cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            cts.CancelAfter(_watchdogTimeout);
            isLinked = true;
        }

        _activeWatchdogCts = cts;

        try
        {
            var frame = await _channel.Reader.ReadAsync(cts.Token).ConfigureAwait(false);
            if (!isLinked)
            {
                cts.CancelAfter(Timeout.InfiniteTimeSpan);
            }
            return frame;
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException($"Stage token response timed out after {_watchdogTimeout.TotalSeconds:F1}s watchdog limit.");
        }
        finally
        {
            if (ReferenceEquals(_activeWatchdogCts, cts))
            {
                _activeWatchdogCts = null;
            }
            if (isLinked)
            {
                cts.Dispose();
            }
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _cts.Cancel();
        _channel.Writer.TryComplete();
        _cts.Dispose();
        _reusableWatchdogCts?.Dispose();
    }
}
