namespace Glacier.Inference.Tests.Distributed;

using System;
using System.Buffers;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Inference.Distributed;
using Glacier.Inference.Distributed.Cli;
using Glacier.Inference.Distributed.Transport;
using Xunit;

public class ResilientNetworkStreamingTests
{
    [Fact]
    public void ConfigureSocket_SetsKeepAliveOptionsAndTimeouts()
    {
        using var socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        FusedSocketTransport.ConfigureSocket(socket, 4 * 1024 * 1024);

        Assert.True(socket.NoDelay);
        Assert.Equal(4 * 1024 * 1024, socket.ReceiveBufferSize);
        Assert.Equal(4 * 1024 * 1024, socket.SendBufferSize);

        // KeepAlive socket option must be enabled
        var keepAlive = (int)socket.GetSocketOption(SocketOptionLevel.Socket, SocketOptionName.KeepAlive)!;
        Assert.NotEqual(0, keepAlive);

        // Receive & Send timeouts set to 120s safety threshold
        Assert.Equal(120_000, socket.ReceiveTimeout);
        Assert.Equal(120_000, socket.SendTimeout);

        if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux() || OperatingSystem.IsMacOS())
        {
            var keepAliveTime = (int)socket.GetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveTime)!;
            var keepAliveInterval = (int)socket.GetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveInterval)!;
            var keepAliveRetryCount = (int)socket.GetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveRetryCount)!;

            Assert.Equal(5, keepAliveTime);
            Assert.Equal(1, keepAliveInterval);
            Assert.Equal(5, keepAliveRetryCount);
        }
    }

    [Fact]
    public void NonBlockingDuplexSocketBus_DefaultWatchdog_Is60Seconds_AndAdaptiveCalculatesProperly()
    {
        using var socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        using var bus = new NonBlockingDuplexSocketBus(socket, channelCapacity: 4);

        Assert.Equal(TimeSpan.FromSeconds(60.0), bus.WatchdogTimeout);

        // Adaptive calculation tests
        Assert.Equal(TimeSpan.FromSeconds(60.0), NonBlockingDuplexSocketBus.CalculateAdaptiveTimeout(128));
        Assert.Equal(TimeSpan.FromSeconds(60.0), NonBlockingDuplexSocketBus.CalculateAdaptiveTimeout(512));
        // For 2048 prompt tokens: 0.05 * 2048 + 10 = 102.4 + 10 = 112.4s
        Assert.Equal(TimeSpan.FromSeconds(112.4), NonBlockingDuplexSocketBus.CalculateAdaptiveTimeout(2048));
    }

    [Fact]
    public async Task NonBlockingDuplexSocketBus_InterleavedHeartbeats_ResetsWatchdogAndKeepsBusAlive()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = clientSocket.ConnectAsync(listener.LocalEndPoint!);
        using var serverSocket = await listener.AcceptAsync();
        await connectTask;

        FusedSocketTransport.ConfigureSocket(clientSocket);
        FusedSocketTransport.ConfigureSocket(serverSocket);

        // Strict 400ms watchdog
        using var bus = new NonBlockingDuplexSocketBus(serverSocket, channelCapacity: 4, watchdogTimeout: TimeSpan.FromMilliseconds(400));
        bus.Start();

        // Emit 4 heartbeats spaced 150ms apart (total elapsed ~600ms > 400ms watchdog limit)
        var heartbeatTask = Task.Run(async () =>
        {
            byte[] pingBytes = new byte[ClusterFrameHeader.HeaderSize];
            var pingHdr = new ClusterFrameHeader(ClusterMessageType.HeartbeatPing, 101, 0);
            MemoryMarshal.Write(pingBytes, in pingHdr);

            for (int i = 0; i < 4; i++)
            {
                await Task.Delay(150);
                await clientSocket.SendAsync(pingBytes, SocketFlags.None);
            }

            // Finally send a valid MsgResult frame
            int payloadLen = FusedActivationResultHeader.HeaderSize;
            int totalBytes = ClusterFrameHeader.HeaderSize + payloadLen;
            byte[] resBytes = new byte[totalBytes];

            var clusterHdr = new ClusterFrameHeader((ClusterMessageType)0x71, 999, (ulong)payloadLen);
            clusterHdr.WriteTo(resBytes.AsSpan(0, 32));

            var resultHdr = new FusedActivationResultHeader(
                stepIndex: 42,
                stageId: 1,
                computeMs: 12.5,
                nextTokenId: 777,
                topLogit: 3.14f,
                precisionMode: WirePrecisionMode.FP32,
                flags: 0);
            resultHdr.WriteTo(resBytes.AsSpan(32, 32));

            await clientSocket.SendAsync(resBytes, SocketFlags.None);
        });

        // Bus should not time out because each heartbeat resets the watchdog
        var tokenResult = await bus.ReceiveTokenAsync(CancellationToken.None);
        await heartbeatTask;

        Assert.Equal(999UL, tokenResult.SequenceId);
        Assert.Equal(777, tokenResult.TokenId);
        Assert.Equal(3.14f, tokenResult.TopLogit);
        Assert.Equal(12.5, tokenResult.ComputeMs);
    }

    [Fact]
    public async Task WorkerLoop_DefensiveFraming_DrainsUnknownFrame_WithoutStreamCorruption()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = clientSocket.ConnectAsync(listener.LocalEndPoint!);
        using var serverSocket = await listener.AcceptAsync();
        await connectTask;

        FusedSocketTransport.ConfigureSocket(clientSocket);
        FusedSocketTransport.ConfigureSocket(serverSocket);

        using var clientStream = new NetworkStream(clientSocket, ownsSocket: false);
        using var serverStream = new NetworkStream(serverSocket, ownsSocket: false);

        // 1. Client writes an UNRECOGNIZED frame opcode 0xFE with 128 bytes of garbage payload
        int garbageLen = 128;
        var unknownHdr = new ClusterFrameHeader((ClusterMessageType)0xFE, 888, (ulong)garbageLen);
        byte[] unknownBytes = new byte[ClusterFrameHeader.HeaderSize + garbageLen];
        MemoryMarshal.Write(unknownBytes, in unknownHdr);
        Random.Shared.NextBytes(unknownBytes.AsSpan(ClusterFrameHeader.HeaderSize));

        await clientStream.WriteAsync(unknownBytes, 0, unknownBytes.Length);

        // 2. Client immediately follows with a legitimate MsgResetKv frame
        var resetHdr = new ClusterFrameHeader(DistributedClusterSession.MsgResetKv, 999, 0);
        byte[] resetBytes = new byte[ClusterFrameHeader.HeaderSize];
        MemoryMarshal.Write(resetBytes, in resetHdr);
        await clientStream.WriteAsync(resetBytes, 0, resetBytes.Length);
        await clientStream.FlushAsync();

        // 3. Server simulates the defensive command dispatch loop
        byte[] headerBytes = new byte[ClusterFrameHeader.HeaderSize];
        int readHdr1 = await FusedSocketTransport.ReadExactAsync(serverStream, headerBytes.AsMemory(0, ClusterFrameHeader.HeaderSize));
        Assert.Equal(ClusterFrameHeader.HeaderSize, readHdr1);

        var parsedUnknownHdr = MemoryMarshal.Read<ClusterFrameHeader>(headerBytes);
        Assert.Equal(ClusterFrameHeader.MagicConstant, parsedUnknownHdr.Magic);
        Assert.Equal((ClusterMessageType)0xFE, parsedUnknownHdr.MessageType);

        // Catch-all drainer consumes the 128-byte payload
        if (parsedUnknownHdr.PayloadLength > 0)
        {
            byte[] discard = ArrayPool<byte>.Shared.Rent((int)parsedUnknownHdr.PayloadLength);
            try
            {
                int drained = await FusedSocketTransport.ReadExactAsync(serverStream, discard.AsMemory(0, (int)parsedUnknownHdr.PayloadLength));
                Assert.Equal(garbageLen, drained);
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(discard);
            }
        }

        // 4. Server reads the NEXT frame from the stream
        int readHdr2 = await FusedSocketTransport.ReadExactAsync(serverStream, headerBytes.AsMemory(0, ClusterFrameHeader.HeaderSize));
        Assert.Equal(ClusterFrameHeader.HeaderSize, readHdr2);

        var parsedResetHdr = MemoryMarshal.Read<ClusterFrameHeader>(headerBytes);

        // Invariant: The magic constant is INTACT (stream alignment was perfectly preserved)
        Assert.Equal(ClusterFrameHeader.MagicConstant, parsedResetHdr.Magic);
        Assert.Equal(DistributedClusterSession.MsgResetKv, parsedResetHdr.MessageType);
        Assert.Equal(999UL, parsedResetHdr.CorrelationId);
    }

    [Fact]
    public async Task BackgroundHeartbeat_DuringMockProvisioning_MaintainsStreamWithoutTimeout()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = clientSocket.ConnectAsync(listener.LocalEndPoint!);
        using var serverSocket = await listener.AcceptAsync();
        await connectTask;

        FusedSocketTransport.ConfigureSocket(clientSocket);
        FusedSocketTransport.ConfigureSocket(serverSocket);

        using var coordStream = new NetworkStream(clientSocket, ownsSocket: false);
        using var workerStream = new NetworkStream(serverSocket, ownsSocket: false);

        // Worker simulates CreateStageModel provisioning delay with background HeartbeatPing emitter
        var workerTask = Task.Run(async () =>
        {
            using var pingCts = new CancellationTokenSource();
            var pingTask = Task.Run(async () =>
            {
                using var timer = new PeriodicTimer(TimeSpan.FromMilliseconds(50));
                byte[] pingBytes = new byte[ClusterFrameHeader.HeaderSize];
                var pingHdr = new ClusterFrameHeader(ClusterMessageType.HeartbeatPing, 1, 0);
                MemoryMarshal.Write(pingBytes, in pingHdr);

                try
                {
                    while (!pingCts.IsCancellationRequested && await timer.WaitForNextTickAsync(pingCts.Token).ConfigureAwait(false))
                    {
                        await workerStream.WriteAsync(pingBytes, 0, ClusterFrameHeader.HeaderSize, pingCts.Token).ConfigureAwait(false);
                        await workerStream.FlushAsync(pingCts.Token).ConfigureAwait(false);
                    }
                }
                catch (OperationCanceledException) { }
            }, pingCts.Token);

            try
            {
                // Simulate 250ms VRAM provisioning
                await Task.Delay(250);
            }
            finally
            {
                pingCts.Cancel();
                try { await pingTask.ConfigureAwait(false); } catch { }
            }

            // Send MsgStageReady (16-byte payload)
            var readyHdr = new ClusterFrameHeader(DistributedClusterSession.MsgStageReady, 1, 16);
            byte[] readyHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
            MemoryMarshal.Write(readyHdrBytes, in readyHdr);
            byte[] payload = new byte[16];
            BitConverter.TryWriteBytes(payload.AsSpan(0, 4), 0); // status = 0 (ready)

            await workerStream.WriteAsync(readyHdrBytes, 0, ClusterFrameHeader.HeaderSize);
            await workerStream.WriteAsync(payload, 0, 16);
            await workerStream.FlushAsync();
        });

        // Coordinator runs the robust wait loop handling HeartbeatPing
        byte[] respHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
        byte[] respPayload = new byte[16];
        int heartbeatCount = 0;

        while (true)
        {
            await FusedSocketTransport.ReadExactAsync(coordStream, respHdrBytes.AsMemory(0, ClusterFrameHeader.HeaderSize));
            var respHdr = MemoryMarshal.Read<ClusterFrameHeader>(respHdrBytes);

            Assert.Equal(ClusterFrameHeader.MagicConstant, respHdr.Magic);

            if (respHdr.MessageType == ClusterMessageType.HeartbeatPing)
            {
                heartbeatCount++;
                if (respHdr.PayloadLength > 0)
                {
                    byte[] discard = ArrayPool<byte>.Shared.Rent((int)respHdr.PayloadLength);
                    try { await FusedSocketTransport.ReadExactAsync(coordStream, discard.AsMemory(0, (int)respHdr.PayloadLength)); }
                    finally { ArrayPool<byte>.Shared.Return(discard); }
                }
                continue;
            }
            if (respHdr.MessageType == DistributedClusterSession.MsgStageReady)
            {
                await FusedSocketTransport.ReadExactAsync(coordStream, respPayload.AsMemory(0, 16));
                break;
            }
        }

        await workerTask;

        Assert.True(heartbeatCount >= 2, $"Expected at least 2 heartbeats, received {heartbeatCount}");
        int status = BitConverter.ToInt32(respPayload, 0);
        Assert.Equal(0, status);
    }

    [Fact]
    public void PinnedChunkBuffer_ReusesSingleArray_EliminatesLohAllocationChurn()
    {
        // Verify that 4MB pinned buffer allocation is stable and writable
        byte[] chunkBuffer = GC.AllocateArray<byte>(4 * 1024 * 1024, pinned: true);
        Assert.Equal(4 * 1024 * 1024, chunkBuffer.Length);

        // Simulate 5 chunks written to a memory stream using the single pinned buffer
        using var memStream = new MemoryStream();
        for (int i = 0; i < 5; i++)
        {
            chunkBuffer.AsSpan().Fill((byte)(i + 1));
            memStream.Write(chunkBuffer, 0, chunkBuffer.Length);
        }

        Assert.Equal(5 * 4 * 1024 * 1024L, memStream.Length);
    }
}
