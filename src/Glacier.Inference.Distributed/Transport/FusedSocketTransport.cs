namespace Glacier.Inference.Distributed;

using System;
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

    /// <summary>
    /// Configures high-performance socket options for ultra-low latency cluster transit.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ConfigureSocket(Socket socket, int bufferSizeBytes = 4 * 1024 * 1024)
    {
        ArgumentNullException.ThrowIfNull(socket);
        socket.NoDelay = true;
        socket.ReceiveBufferSize = bufferSizeBytes;
        socket.SendBufferSize = bufferSizeBytes;
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
    /// </summary>
    public static bool ReceiveResult(
        Socket socket,
        Span<byte> targetBuffer,
        out ClusterFrameHeader clusterHeader,
        out FusedActivationResultHeader resultHeader,
        out ReadOnlySpan<byte> payloadSpan)
    {
        ArgumentNullException.ThrowIfNull(socket);

        int received = ReadExact(socket, targetBuffer.Slice(0, CombinedHeaderSize));
        if (received < CombinedHeaderSize)
        {
            clusterHeader = default;
            resultHeader = default;
            payloadSpan = ReadOnlySpan<byte>.Empty;
            return false;
        }

        clusterHeader = MemoryMarshal.Read<ClusterFrameHeader>(targetBuffer.Slice(0, ClusterFrameHeader.HeaderSize));
        resultHeader = FusedActivationResultHeader.ReadFrom(targetBuffer.Slice(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));
        int payloadLen = (int)clusterHeader.PayloadLength - FusedActivationResultHeader.HeaderSize;

        if (payloadLen > 0)
        {
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
    /// </summary>
    public static async ValueTask<(ClusterFrameHeader ClusterHdr, FusedActivationResultHeader ResultHdr, int PayloadLength)> ReceiveResultAsync(
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
        var resultHeader = FusedActivationResultHeader.ReadFrom(targetBuffer.Span.Slice(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));
        int payloadLen = (int)clusterHeader.PayloadLength - FusedActivationResultHeader.HeaderSize;

        if (payloadLen > 0)
        {
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
}
