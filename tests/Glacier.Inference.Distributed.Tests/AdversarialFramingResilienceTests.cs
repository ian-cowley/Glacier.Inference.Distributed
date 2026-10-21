namespace Glacier.Inference.Tests.Distributed;

using System;
using System.Buffers;
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

public class AdversarialFramingResilienceTests
{
    [Fact]
    public async Task FusedSocketTransport_InterleavedHeartbeats_DoNotDeadlockOn32ByteExpectation()
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

        // Sender sends 5 HeartbeatPing frames (each strictly 32 bytes, payload=0)
        // followed by 1 MsgResult frame (64 bytes: 32 bytes ClusterFrameHeader + 32 bytes FusedActivationResultHeader)
        var sendTask = Task.Run(async () =>
        {
            byte[] hbBytes = new byte[ClusterFrameHeader.HeaderSize];
            var hbHdr = new ClusterFrameHeader(ClusterMessageType.HeartbeatPing, 100, 0);
            MemoryMarshal.Write(hbBytes, in hbHdr);

            for (int i = 0; i < 5; i++)
            {
                await clientSocket.SendAsync(hbBytes, SocketFlags.None);
                await Task.Delay(10);
            }

            int payloadLen = FusedActivationResultHeader.HeaderSize;
            byte[] resBytes = new byte[ClusterFrameHeader.HeaderSize + payloadLen];
            var resClusterHdr = new ClusterFrameHeader((ClusterMessageType)0x71, 999, (ulong)payloadLen);
            resClusterHdr.WriteTo(resBytes.AsSpan(0, 32));

            var resActHdr = new FusedActivationResultHeader(
                stepIndex: 1,
                stageId: 0,
                computeMs: 5.5,
                nextTokenId: 42,
                topLogit: 10.0f,
                precisionMode: WirePrecisionMode.FP32,
                flags: 0);
            resActHdr.WriteTo(resBytes.AsSpan(32, 32));

            await clientSocket.SendAsync(resBytes, SocketFlags.None);
        });

        // Receiver reads via FusedSocketTransport.ReceiveResultAsync
        byte[] buffer = new byte[1024];
        int heartbeatsReceived = 0;
        ClusterFrameHeader finalClusterHdr = default;
        FusedActivationResultHeader finalResultHdr = default;

        using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(5));

        while (!timeoutCts.IsCancellationRequested)
        {
            var (clusterHdr, resultHdr, _) = await FusedSocketTransport.ReceiveResultAsync(
                serverSocket, buffer.AsMemory(), timeoutCts.Token);

            Assert.Equal(ClusterFrameHeader.MagicConstant, clusterHdr.Magic);

            if (clusterHdr.MessageType == ClusterMessageType.HeartbeatPing)
            {
                heartbeatsReceived++;
                continue;
            }

            if (clusterHdr.MessageType == (ClusterMessageType)0x71)
            {
                finalClusterHdr = clusterHdr;
                finalResultHdr = resultHdr;
                break;
            }
        }

        await sendTask;

        Assert.Equal(5, heartbeatsReceived);
        Assert.Equal(999UL, finalClusterHdr.CorrelationId);
        Assert.Equal(42, finalResultHdr.NextTokenId);
        Assert.Equal(10.0f, finalResultHdr.TopLogit);
    }

    [Fact]
    public async Task FusedSocketTransport_SyncReceiveResult_InterleavedHeartbeats_DoNotDeadlock()
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

        var sendTask = Task.Run(async () =>
        {
            byte[] hbBytes = new byte[ClusterFrameHeader.HeaderSize];
            var hbHdr = new ClusterFrameHeader(ClusterMessageType.HeartbeatPong, 200, 0);
            MemoryMarshal.Write(hbBytes, in hbHdr);

            for (int i = 0; i < 3; i++)
            {
                await clientSocket.SendAsync(hbBytes, SocketFlags.None);
                await Task.Delay(10);
            }

            int payloadLen = FusedActivationResultHeader.HeaderSize;
            byte[] resBytes = new byte[ClusterFrameHeader.HeaderSize + payloadLen];
            var resClusterHdr = new ClusterFrameHeader((ClusterMessageType)0x71, 555, (ulong)payloadLen);
            resClusterHdr.WriteTo(resBytes.AsSpan(0, 32));

            var resActHdr = new FusedActivationResultHeader(
                stepIndex: 2,
                stageId: 1,
                computeMs: 8.0,
                nextTokenId: 1234,
                topLogit: 5.0f,
                precisionMode: WirePrecisionMode.FP32,
                flags: 0);
            resActHdr.WriteTo(resBytes.AsSpan(32, 32));

            await clientSocket.SendAsync(resBytes, SocketFlags.None);
        });

        byte[] buffer = new byte[1024];
        int heartbeatsReceived = 0;
        FusedActivationResultHeader finalResultHdr = default;

        while (true)
        {
            bool ok = FusedSocketTransport.ReceiveResult(
                serverSocket,
                buffer.AsSpan(),
                out var clusterHdr,
                out var resultHdr,
                out _);

            Assert.True(ok);
            Assert.Equal(ClusterFrameHeader.MagicConstant, clusterHdr.Magic);

            if (clusterHdr.MessageType == ClusterMessageType.HeartbeatPong)
            {
                heartbeatsReceived++;
                continue;
            }

            if (clusterHdr.MessageType == (ClusterMessageType)0x71)
            {
                finalResultHdr = resultHdr;
                break;
            }
        }

        await sendTask;

        Assert.Equal(3, heartbeatsReceived);
        Assert.Equal(1234, finalResultHdr.NextTokenId);
    }

    [Fact]
    public async Task UnexpectedFrameInjection_WithPayloadZero_PreservesStreamAlignment()
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

        // Client injects an unexpected frame (Opcode 0xFE) with PayloadLength = 0 (32 bytes)
        // followed immediately by a valid MsgResult frame (64 bytes)
        var sendTask = Task.Run(async () =>
        {
            byte[] unknownBytes = new byte[ClusterFrameHeader.HeaderSize];
            var unknownHdr = new ClusterFrameHeader((ClusterMessageType)0xFE, 111, 0);
            MemoryMarshal.Write(unknownBytes, in unknownHdr);
            await clientSocket.SendAsync(unknownBytes, SocketFlags.None);

            int payloadLen = FusedActivationResultHeader.HeaderSize;
            byte[] resBytes = new byte[ClusterFrameHeader.HeaderSize + payloadLen];
            var resClusterHdr = new ClusterFrameHeader((ClusterMessageType)0x71, 777, (ulong)payloadLen);
            resClusterHdr.WriteTo(resBytes.AsSpan(0, 32));

            var resActHdr = new FusedActivationResultHeader(
                stepIndex: 1,
                stageId: 0,
                computeMs: 1.0,
                nextTokenId: 888,
                topLogit: 2.0f,
                precisionMode: WirePrecisionMode.FP32,
                flags: 0);
            resActHdr.WriteTo(resBytes.AsSpan(32, 32));

            await clientSocket.SendAsync(resBytes, SocketFlags.None);
        });

        byte[] buffer = new byte[1024];
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(3));

        // Attempt to receive first frame (unexpected 0xFE)
        var (hdr1, _, _) = await FusedSocketTransport.ReceiveResultAsync(serverSocket, buffer.AsMemory(), cts.Token);
        Assert.Equal(ClusterFrameHeader.MagicConstant, hdr1.Magic);
        Assert.Equal((ClusterMessageType)0xFE, hdr1.MessageType);

        // Attempt to receive second frame (valid 0x71)
        var (hdr2, res2, _) = await FusedSocketTransport.ReceiveResultAsync(serverSocket, buffer.AsMemory(), cts.Token);

        await sendTask;

        Assert.Equal(ClusterFrameHeader.MagicConstant, hdr2.Magic);
        Assert.Equal((ClusterMessageType)0x71, hdr2.MessageType);
        Assert.Equal(888, res2.NextTokenId);
    }

    [Fact]
    public async Task UnexpectedFrameInjection_WithPayloadNonZero_PreservesStreamAlignment()
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

        // Client injects an unexpected frame (Opcode 0xFE) with 64 bytes of garbage payload
        // followed immediately by a valid MsgResult frame (64 bytes)
        var sendTask = Task.Run(async () =>
        {
            byte[] unknownBytes = new byte[ClusterFrameHeader.HeaderSize + 64];
            var unknownHdr = new ClusterFrameHeader((ClusterMessageType)0xFE, 111, 64);
            MemoryMarshal.Write(unknownBytes, in unknownHdr);
            Random.Shared.NextBytes(unknownBytes.AsSpan(ClusterFrameHeader.HeaderSize));
            await clientSocket.SendAsync(unknownBytes, SocketFlags.None);

            int payloadLen = FusedActivationResultHeader.HeaderSize;
            byte[] resBytes = new byte[ClusterFrameHeader.HeaderSize + payloadLen];
            var resClusterHdr = new ClusterFrameHeader((ClusterMessageType)0x71, 777, (ulong)payloadLen);
            resClusterHdr.WriteTo(resBytes.AsSpan(0, 32));

            var resActHdr = new FusedActivationResultHeader(
                stepIndex: 1,
                stageId: 0,
                computeMs: 1.0,
                nextTokenId: 888,
                topLogit: 2.0f,
                precisionMode: WirePrecisionMode.FP32,
                flags: 0);
            resActHdr.WriteTo(resBytes.AsSpan(32, 32));

            await clientSocket.SendAsync(resBytes, SocketFlags.None);
        });

        byte[] buffer = new byte[1024];
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(3));

        // First frame: unexpected 0xFE with 64 bytes payload should be safely drained
        var (hdr1, _, _) = await FusedSocketTransport.ReceiveResultAsync(serverSocket, buffer.AsMemory(), cts.Token);
        Assert.Equal(ClusterFrameHeader.MagicConstant, hdr1.Magic);
        Assert.Equal((ClusterMessageType)0xFE, hdr1.MessageType);

        // Second frame: valid 0x71 should be cleanly received
        var (hdr2, res2, _) = await FusedSocketTransport.ReceiveResultAsync(serverSocket, buffer.AsMemory(), cts.Token);

        await sendTask;

        Assert.Equal(ClusterFrameHeader.MagicConstant, hdr2.Magic);
        Assert.Equal((ClusterMessageType)0x71, hdr2.MessageType);
        Assert.Equal(888, res2.NextTokenId);
    }

    [Fact]
    public async Task NonBlockingDuplexSocketBus_UnexpectedFrame_PayloadZero_DrainsAndReceivesSubsequentToken()
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

        using var bus = new NonBlockingDuplexSocketBus(serverSocket, channelCapacity: 4, watchdogTimeout: TimeSpan.FromSeconds(2));
        bus.Start();

        var sendTask = Task.Run(async () =>
        {
            // Inject unexpected opcode 0xFE with payload = 0
            byte[] unknownBytes = new byte[ClusterFrameHeader.HeaderSize];
            var unknownHdr = new ClusterFrameHeader((ClusterMessageType)0xFE, 111, 0);
            MemoryMarshal.Write(unknownBytes, in unknownHdr);
            await clientSocket.SendAsync(unknownBytes, SocketFlags.None);

            // Send valid MsgResult
            int payloadLen = FusedActivationResultHeader.HeaderSize;
            byte[] resBytes = new byte[ClusterFrameHeader.HeaderSize + payloadLen];
            var resClusterHdr = new ClusterFrameHeader((ClusterMessageType)0x71, 777, (ulong)payloadLen);
            resClusterHdr.WriteTo(resBytes.AsSpan(0, 32));

            var resActHdr = new FusedActivationResultHeader(
                stepIndex: 1,
                stageId: 0,
                computeMs: 1.0,
                nextTokenId: 888,
                topLogit: 2.0f,
                precisionMode: WirePrecisionMode.FP32,
                flags: 0);
            resActHdr.WriteTo(resBytes.AsSpan(32, 32));

            await clientSocket.SendAsync(resBytes, SocketFlags.None);
        });

        var tokenResult = await bus.ReceiveTokenAsync(CancellationToken.None);
        await sendTask;

        Assert.Equal(888, tokenResult.TokenId);
    }
}
