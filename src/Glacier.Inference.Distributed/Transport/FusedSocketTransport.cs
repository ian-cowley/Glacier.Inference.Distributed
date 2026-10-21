namespace Glacier.Inference.Distributed;

using System;
using System.Buffers;
using System.IO;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;

/// <summary>
/// Fused zero-allocation socket transport implementing atomic socket send/receive operations.
/// Fuses ClusterFrameHeader (32B) + FusedActivationHeader (32B) + Activation payload into single
/// unified socket writes/reads, eliminating syscall fragmentation, buffer resizing, and ThreadPool IOCP context switches.
/// </summary>
public static class FusedSocketTransport
{
    public const int CombinedHeaderSize = ClusterFrameHeader.HeaderSize + FusedActivationHeader.HeaderSize; // Exactly 64 bytes
    public const ClusterMessageType MsgResult = (ClusterMessageType)0x71;

    /// <summary>
    /// Configures high-performance socket options for ultra-low latency cluster transit.
    /// Sets TCP Keep-Alive, aggressive probe intervals (5s idle, 1s interval, 5 retries),
    /// and 120s safety timeouts to prevent socket drops and silent connection resets.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ConfigureSocket(Socket socket, int bufferSizeBytes = 4 * 1024 * 1024)
    {
        ArgumentNullException.ThrowIfNull(socket);
        socket.NoDelay = true;
        socket.ReceiveBufferSize = bufferSizeBytes;
        socket.SendBufferSize = bufferSizeBytes;

        socket.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.KeepAlive, true);
        if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux() || OperatingSystem.IsMacOS())
        {
            socket.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveTime, 5);
            socket.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveInterval, 1);
            socket.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveRetryCount, 5);
        }
        socket.ReceiveTimeout = 120_000;
        socket.SendTimeout = 120_000;
    }

    /// <summary>
    /// Synchronously sends a fused activation frame in a single atomic socket write with zero heap allocations.
    /// </summary>
    public static void SendFusedActivation(
        Socket socket,
        ulong correlationId,
        in FusedActivationHeader actHeader,
        ReadOnlySpan<byte> activationPayload,
        Span<byte> stagingBuffer)
    {
        ArgumentNullException.ThrowIfNull(socket);

        int payloadLen = FusedActivationHeader.HeaderSize + activationPayload.Length;
        int totalFrameBytes = ClusterFrameHeader.HeaderSize + payloadLen;

        if (stagingBuffer.Length < totalFrameBytes)
            throw new ArgumentException($"Staging buffer too small ({stagingBuffer.Length} < {totalFrameBytes})", nameof(stagingBuffer));

        var clusterHeader = new ClusterFrameHeader(
            messageType: (ClusterMessageType)0x70, // MsgActivation
            correlationId: correlationId,
            payloadLength: (ulong)payloadLen,
            flags: ClusterFrameFlags.None);

        clusterHeader.WriteTo(stagingBuffer.Slice(0, ClusterFrameHeader.HeaderSize));
        actHeader.WriteTo(stagingBuffer.Slice(ClusterFrameHeader.HeaderSize, FusedActivationHeader.HeaderSize));

        if (!activationPayload.IsEmpty)
        {
            activationPayload.CopyTo(stagingBuffer.Slice(CombinedHeaderSize, activationPayload.Length));
        }

        // Single atomic socket write
        int sent = 0;
        var toSend = stagingBuffer.Slice(0, totalFrameBytes);
        while (sent < totalFrameBytes)
        {
            int r = socket.Send(toSend.Slice(sent), SocketFlags.None);
            if (r <= 0) throw new SocketException((int)SocketError.ConnectionReset);
            sent += r;
        }
    }

    /// <summary>
    /// Asynchronously sends a fused activation frame returning a non-allocating ValueTask.
    /// </summary>
    public static async ValueTask SendFusedActivationAsync(
        Socket socket,
        ulong correlationId,
        FusedActivationHeader actHeader,
        ReadOnlyMemory<byte> activationPayload,
        Memory<byte> stagingBuffer,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(socket);

        int payloadLen = FusedActivationHeader.HeaderSize + activationPayload.Length;
        int totalFrameBytes = ClusterFrameHeader.HeaderSize + payloadLen;

        if (stagingBuffer.Length < totalFrameBytes)
            throw new ArgumentException($"Staging buffer too small ({stagingBuffer.Length} < {totalFrameBytes})", nameof(stagingBuffer));

        var clusterHeader = new ClusterFrameHeader(
            messageType: (ClusterMessageType)0x70, // MsgActivation
            correlationId: correlationId,
            payloadLength: (ulong)payloadLen,
            flags: ClusterFrameFlags.None);

        clusterHeader.WriteTo(stagingBuffer.Span.Slice(0, ClusterFrameHeader.HeaderSize));
        actHeader.WriteTo(stagingBuffer.Span.Slice(ClusterFrameHeader.HeaderSize, FusedActivationHeader.HeaderSize));

        if (!activationPayload.IsEmpty)
        {
            activationPayload.Span.CopyTo(stagingBuffer.Span.Slice(CombinedHeaderSize, activationPayload.Length));
        }

        int sent = 0;
        var toSend = stagingBuffer.Slice(0, totalFrameBytes);
        while (sent < totalFrameBytes)
        {
            int r = await socket.SendAsync(toSend.Slice(sent), SocketFlags.None, cancellationToken).ConfigureAwait(false);
            if (r <= 0) throw new SocketException((int)SocketError.ConnectionReset);
            sent += r;
        }
    }

    /// <summary>
    /// Synchronously receives a fused activation frame into a pre-allocated buffer with zero heap allocations.
    /// </summary>
    public static bool ReceiveFusedActivation(
        Socket socket,
        Span<byte> targetBuffer,
        out ClusterFrameHeader clusterHeader,
        out FusedActivationHeader actHeader,
        out ReadOnlySpan<byte> payloadSpan)
    {
        ArgumentNullException.ThrowIfNull(socket);

        // 1. Read at least CombinedHeaderSize (64 bytes)
        int received = ReadExact(socket, targetBuffer.Slice(0, CombinedHeaderSize));
        if (received < CombinedHeaderSize)
        {
            clusterHeader = default;
            actHeader = default;
            payloadSpan = ReadOnlySpan<byte>.Empty;
            return false;
        }

        clusterHeader = MemoryMarshal.Read<ClusterFrameHeader>(targetBuffer.Slice(0, ClusterFrameHeader.HeaderSize));
        if (clusterHeader.Magic != ClusterFrameHeader.MagicConstant)
        {
            actHeader = default;
            payloadSpan = ReadOnlySpan<byte>.Empty;
            return false;
        }

        actHeader = FusedActivationHeader.ReadFrom(targetBuffer.Slice(ClusterFrameHeader.HeaderSize, FusedActivationHeader.HeaderSize));
        int activationPayloadLen = (int)clusterHeader.PayloadLength - FusedActivationHeader.HeaderSize;

        if (activationPayloadLen > 0)
        {
            int remainingRead = ReadExact(socket, targetBuffer.Slice(CombinedHeaderSize, activationPayloadLen));
            if (remainingRead < activationPayloadLen)
            {
                payloadSpan = ReadOnlySpan<byte>.Empty;
                return false;
            }
        }

        payloadSpan = targetBuffer.Slice(CombinedHeaderSize, activationPayloadLen);
        return true;
    }

    /// <summary>
    /// Asynchronously receives a fused activation frame returning a non-allocating ValueTask.
    /// </summary>
    public static async ValueTask<(ClusterFrameHeader ClusterHdr, FusedActivationHeader ActHdr, int PayloadLength)> ReceiveFusedActivationAsync(
        Socket socket,
        Memory<byte> targetBuffer,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(socket);

        int received = await ReadExactAsync(socket, targetBuffer.Slice(0, CombinedHeaderSize), cancellationToken).ConfigureAwait(false);
        if (received < CombinedHeaderSize)
        {
            return (default, default, 0);
        }

        var clusterHeader = MemoryMarshal.Read<ClusterFrameHeader>(targetBuffer.Span.Slice(0, ClusterFrameHeader.HeaderSize));
        var actHeader = FusedActivationHeader.ReadFrom(targetBuffer.Span.Slice(ClusterFrameHeader.HeaderSize, FusedActivationHeader.HeaderSize));
        int activationPayloadLen = (int)clusterHeader.PayloadLength - FusedActivationHeader.HeaderSize;

        if (activationPayloadLen > 0)
        {
            int remaining = await ReadExactAsync(socket, targetBuffer.Slice(CombinedHeaderSize, activationPayloadLen), cancellationToken).ConfigureAwait(false);
            if (remaining < activationPayloadLen)
            {
                return (default, default, 0);
            }
        }

        return (clusterHeader, actHeader, activationPayloadLen);
    }

    /// <summary>
    /// Synchronously sends an activation result frame in a single atomic socket write.
    /// </summary>
    public static void SendResult(
        Socket socket,
        ulong correlationId,
        in FusedActivationResultHeader resultHeader,
        ReadOnlySpan<byte> activationPayload,
        Span<byte> stagingBuffer)
    {
        ArgumentNullException.ThrowIfNull(socket);

        int payloadLen = FusedActivationResultHeader.HeaderSize + activationPayload.Length;
        int totalFrameBytes = ClusterFrameHeader.HeaderSize + payloadLen;

        if (stagingBuffer.Length < totalFrameBytes)
            throw new ArgumentException($"Staging buffer too small ({stagingBuffer.Length} < {totalFrameBytes})", nameof(stagingBuffer));

        var clusterHeader = new ClusterFrameHeader(
            messageType: (ClusterMessageType)0x71, // MsgResult
            correlationId: correlationId,
            payloadLength: (ulong)payloadLen,
            flags: ClusterFrameFlags.None);

        clusterHeader.WriteTo(stagingBuffer.Slice(0, ClusterFrameHeader.HeaderSize));
        resultHeader.WriteTo(stagingBuffer.Slice(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));

        if (!activationPayload.IsEmpty)
        {
            activationPayload.CopyTo(stagingBuffer.Slice(CombinedHeaderSize, activationPayload.Length));
        }

        int sent = 0;
        var toSend = stagingBuffer.Slice(0, totalFrameBytes);
        while (sent < totalFrameBytes)
        {
            int r = socket.Send(toSend.Slice(sent), SocketFlags.None);
            if (r <= 0) throw new SocketException((int)SocketError.ConnectionReset);
            sent += r;
        }
    }

    /// <summary>
    /// Asynchronously sends an activation result frame returning a non-allocating ValueTask.
    /// </summary>
    public static async ValueTask SendResultAsync(
        Socket socket,
        ulong correlationId,
        FusedActivationResultHeader resultHeader,
        ReadOnlyMemory<byte> activationPayload,
        Memory<byte> stagingBuffer,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(socket);

        int payloadLen = FusedActivationResultHeader.HeaderSize + activationPayload.Length;
        int totalFrameBytes = ClusterFrameHeader.HeaderSize + payloadLen;

        if (stagingBuffer.Length < totalFrameBytes)
            throw new ArgumentException($"Staging buffer too small ({stagingBuffer.Length} < {totalFrameBytes})", nameof(stagingBuffer));

        var clusterHeader = new ClusterFrameHeader(
            messageType: (ClusterMessageType)0x71, // MsgResult
            correlationId: correlationId,
            payloadLength: (ulong)payloadLen,
            flags: ClusterFrameFlags.None);

        clusterHeader.WriteTo(stagingBuffer.Span.Slice(0, ClusterFrameHeader.HeaderSize));
        resultHeader.WriteTo(stagingBuffer.Span.Slice(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));

        if (!activationPayload.IsEmpty)
        {
            activationPayload.Span.CopyTo(stagingBuffer.Span.Slice(CombinedHeaderSize, activationPayload.Length));
        }

        int sent = 0;
        var toSend = stagingBuffer.Slice(0, totalFrameBytes);
        while (sent < totalFrameBytes)
        {
            int r = await socket.SendAsync(toSend.Slice(sent), SocketFlags.None, cancellationToken).ConfigureAwait(false);
            if (r <= 0) throw new SocketException((int)SocketError.ConnectionReset);
            sent += r;
        }
    }

    /// <summary>
    /// Synchronously receives an activation result frame into pre-allocated memory.
    /// Safely handles interleaved 32-byte HeartbeatPing and HeartbeatPong frames without frame desynchronization.
    /// </summary>
    public static bool ReceiveResult(
        Socket socket,
        Span<byte> targetBuffer,
        out ClusterFrameHeader clusterHeader,
        out FusedActivationResultHeader resultHeader,
        out ReadOnlySpan<byte> payloadSpan)
    {
        ArgumentNullException.ThrowIfNull(socket);

        if (targetBuffer.Length < ClusterFrameHeader.HeaderSize)
        {
            clusterHeader = default;
            resultHeader = default;
            payloadSpan = ReadOnlySpan<byte>.Empty;
            return false;
        }

        int received = ReadExact(socket, targetBuffer.Slice(0, ClusterFrameHeader.HeaderSize));
        if (received < ClusterFrameHeader.HeaderSize)
        {
            clusterHeader = default;
            resultHeader = default;
            payloadSpan = ReadOnlySpan<byte>.Empty;
            return false;
        }

        clusterHeader = MemoryMarshal.Read<ClusterFrameHeader>(targetBuffer.Slice(0, ClusterFrameHeader.HeaderSize));
        if (clusterHeader.Magic != ClusterFrameHeader.MagicConstant || clusterHeader.PayloadLength > (ulong)int.MaxValue)
        {
            resultHeader = default;
            payloadSpan = ReadOnlySpan<byte>.Empty;
            return false;
        }

        // Defensive framing: If not a result frame or payload too short for FusedActivationResultHeader,
        // safely drain any payload bytes using targetBuffer (if large enough) or ArrayPool<byte>.Shared
        if (clusterHeader.MessageType != MsgResult || clusterHeader.PayloadLength < (ulong)FusedActivationResultHeader.HeaderSize)
        {
            if (!DrainPayload(socket, targetBuffer, (int)clusterHeader.PayloadLength))
            {
                resultHeader = default;
                payloadSpan = ReadOnlySpan<byte>.Empty;
                return false;
            }

            resultHeader = default;
            payloadSpan = ReadOnlySpan<byte>.Empty;
            return true;
        }

        if (targetBuffer.Length < CombinedHeaderSize)
        {
            resultHeader = default;
            payloadSpan = ReadOnlySpan<byte>.Empty;
            return false;
        }

        int actHdrRead = ReadExact(socket, targetBuffer.Slice(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));
        if (actHdrRead < FusedActivationResultHeader.HeaderSize)
        {
            resultHeader = default;
            payloadSpan = ReadOnlySpan<byte>.Empty;
            return false;
        }

        resultHeader = FusedActivationResultHeader.ReadFrom(targetBuffer.Slice(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));
        int payloadLen = (int)clusterHeader.PayloadLength - FusedActivationResultHeader.HeaderSize;

        if (payloadLen > 0)
        {
            if (targetBuffer.Length < CombinedHeaderSize + payloadLen)
            {
                payloadSpan = ReadOnlySpan<byte>.Empty;
                return false;
            }

            int remaining = ReadExact(socket, targetBuffer.Slice(CombinedHeaderSize, payloadLen));
            if (remaining < payloadLen)
            {
                payloadSpan = ReadOnlySpan<byte>.Empty;
                return false;
            }
        }

        payloadSpan = targetBuffer.Slice(CombinedHeaderSize, payloadLen);
        return true;
    }

    /// <summary>
    /// Asynchronously receives an activation result frame returning a non-allocating ValueTask.
    /// Safely handles interleaved 32-byte HeartbeatPing and HeartbeatPong frames without frame desynchronization.
    /// </summary>
    public static async ValueTask<(ClusterFrameHeader ClusterHdr, FusedActivationResultHeader ResultHdr, int PayloadLength)> ReceiveResultAsync(
        Socket socket,
        Memory<byte> targetBuffer,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(socket);

        if (targetBuffer.Length < ClusterFrameHeader.HeaderSize)
        {
            return (default, default, 0);
        }

        int received = await ReadExactAsync(socket, targetBuffer.Slice(0, ClusterFrameHeader.HeaderSize), cancellationToken).ConfigureAwait(false);
        if (received < ClusterFrameHeader.HeaderSize)
        {
            return (default, default, 0);
        }

        var clusterHeader = MemoryMarshal.Read<ClusterFrameHeader>(targetBuffer.Span.Slice(0, ClusterFrameHeader.HeaderSize));
        if (clusterHeader.Magic != ClusterFrameHeader.MagicConstant || clusterHeader.PayloadLength > (ulong)int.MaxValue)
        {
            return (clusterHeader, default, 0);
        }

        // Defensive framing: If not a result frame or payload too short for FusedActivationResultHeader,
        // safely drain any payload bytes using targetBuffer (if large enough) or ArrayPool<byte>.Shared
        if (clusterHeader.MessageType != MsgResult || clusterHeader.PayloadLength < (ulong)FusedActivationResultHeader.HeaderSize)
        {
            if (!await DrainPayloadAsync(socket, targetBuffer, (int)clusterHeader.PayloadLength, cancellationToken).ConfigureAwait(false))
            {
                return (default, default, 0);
            }

            return (clusterHeader, default, 0);
        }

        if (targetBuffer.Length < CombinedHeaderSize)
        {
            return (default, default, 0);
        }

        int actHdrRead = await ReadExactAsync(socket, targetBuffer.Slice(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize), cancellationToken).ConfigureAwait(false);
        if (actHdrRead < FusedActivationResultHeader.HeaderSize)
        {
            return (default, default, 0);
        }

        var resultHeader = FusedActivationResultHeader.ReadFrom(targetBuffer.Span.Slice(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));
        int payloadLen = (int)clusterHeader.PayloadLength - FusedActivationResultHeader.HeaderSize;

        if (payloadLen > 0)
        {
            if (targetBuffer.Length < CombinedHeaderSize + payloadLen)
            {
                return (default, default, 0);
            }

            int remaining = await ReadExactAsync(socket, targetBuffer.Slice(CombinedHeaderSize, payloadLen), cancellationToken).ConfigureAwait(false);
            if (remaining < payloadLen)
            {
                return (default, default, 0);
            }
        }

        return (clusterHeader, resultHeader, payloadLen);
    }

    /// <summary>
    /// Synchronously reads exactly <paramref name="targetBuffer"/>.Length bytes from a socket.
    /// Guaranteed 0 heap allocations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int ReadExact(Socket socket, Span<byte> targetBuffer)
    {
        int total = 0;
        while (total < targetBuffer.Length)
        {
            int r = socket.Receive(targetBuffer.Slice(total));
            if (r == 0) return total;
            total += r;
        }
        return total;
    }

    /// <summary>
    /// Asynchronously reads exactly <paramref name="targetBuffer"/>.Length bytes from a socket returning a ValueTask.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static async ValueTask<int> ReadExactAsync(Socket socket, Memory<byte> targetBuffer, CancellationToken cancellationToken = default)
    {
        int total = 0;
        while (total < targetBuffer.Length)
        {
            int r = await socket.ReceiveAsync(targetBuffer.Slice(total), SocketFlags.None, cancellationToken).ConfigureAwait(false);
            if (r == 0) return total;
            total += r;
        }
        return total;
    }

    /// <summary>
    /// Asynchronously reads exactly <paramref name="targetBuffer"/>.Length bytes from a Stream returning a ValueTask.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static async ValueTask<int> ReadExactAsync(Stream stream, Memory<byte> targetBuffer, CancellationToken cancellationToken = default)
    {
        int total = 0;
        while (total < targetBuffer.Length)
        {
            int r = await stream.ReadAsync(targetBuffer.Slice(total), cancellationToken).ConfigureAwait(false);
            if (r == 0) return total;
            total += r;
        }
        return total;
    }

    /// <summary>
    /// Safely drains payload bytes of an unexpected or non-result frame to preserve socket framing alignment.
    /// Uses targetBuffer if large enough to avoid pool rentals; otherwise uses ArrayPool{byte}.Shared.
    /// Guaranteed 0 GC allocations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool DrainPayload(Socket socket, Span<byte> targetBuffer, int payloadLength)
    {
        if (payloadLength <= 0) return true;

        if (targetBuffer.Length - ClusterFrameHeader.HeaderSize >= payloadLength)
        {
            int drained = ReadExact(socket, targetBuffer.Slice(ClusterFrameHeader.HeaderSize, payloadLength));
            return drained == payloadLength;
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(Math.Min(payloadLength, 65536));
        try
        {
            int remaining = payloadLength;
            while (remaining > 0)
            {
                int chunk = Math.Min(remaining, rented.Length);
                int drained = ReadExact(socket, rented.AsSpan(0, chunk));
                if (drained < chunk) return false;
                remaining -= chunk;
            }
            return true;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Asynchronously drains payload bytes of an unexpected or non-result frame to preserve socket framing alignment.
    /// Uses targetBuffer if large enough to avoid pool rentals; otherwise uses ArrayPool{byte}.Shared.
    /// Guaranteed 0 GC allocations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static async ValueTask<bool> DrainPayloadAsync(
        Socket socket,
        Memory<byte> targetBuffer,
        int payloadLength,
        CancellationToken cancellationToken)
    {
        if (payloadLength <= 0) return true;

        if (targetBuffer.Length - ClusterFrameHeader.HeaderSize >= payloadLength)
        {
            int drained = await ReadExactAsync(socket, targetBuffer.Slice(ClusterFrameHeader.HeaderSize, payloadLength), cancellationToken).ConfigureAwait(false);
            return drained == payloadLength;
        }

        byte[] rented = ArrayPool<byte>.Shared.Rent(Math.Min(payloadLength, 65536));
        try
        {
            int remaining = payloadLength;
            while (remaining > 0)
            {
                int chunk = Math.Min(remaining, rented.Length);
                int drained = await ReadExactAsync(socket, rented.AsMemory(0, chunk), cancellationToken).ConfigureAwait(false);
                if (drained < chunk) return false;
                remaining -= chunk;
            }
            return true;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }
}
