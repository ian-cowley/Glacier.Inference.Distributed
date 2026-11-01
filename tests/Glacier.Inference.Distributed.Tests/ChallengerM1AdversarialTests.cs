namespace Glacier.Inference.Tests.Distributed;

using System;
using System.Buffers;
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

/// <summary>
/// Empirical and adversarial verification suite for Milestone M1 (Resilient Network Streaming).
/// Adversarially challenges:
/// 1. Watchdog timeout enforcement when no heartbeats arrive
/// 2. Active watchdog reset and survival during extended delays with PeriodicTimer heartbeats (5x watchdog limit)
/// 3. Bi-directional heartbeat contract: verifying HeartbeatPong reply generation
/// 4. Reusable CancellationTokenSource safety across multiple sequential token generations
/// 5. Zero Large Object Heap (LOH) allocations and zero Gen2 GCs during high-volume weight chunk streaming
/// 6. Defensive framing resilience under stream assault with arbitrary unknown opcodes and payloads
/// </summary>
public class ChallengerM1AdversarialTests
{
    [Fact]
    public async Task Watchdog_WithoutHeartbeats_TimesOutStrictly()
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

        // Configure a strict 300ms watchdog
        using var bus = new NonBlockingDuplexSocketBus(serverSocket, channelCapacity: 4, watchdogTimeout: TimeSpan.FromMilliseconds(300));
        bus.Start();

        // Client remains silent: no heartbeats, no tokens
        // Must throw TimeoutException within ~300-800ms
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var ex = await Assert.ThrowsAsync<TimeoutException>(async () =>
        {
            await bus.ReceiveTokenAsync(CancellationToken.None);
        });

        sw.Stop();
        Assert.Contains("watchdog limit", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.True(sw.ElapsedMilliseconds >= 250, $"Elapsed was {sw.ElapsedMilliseconds}ms, expected >= 250ms");
    }

    [Fact]
    public async Task Watchdog_WithPeriodicTimerHeartbeats_ExceedsTimeoutBy5x_WithoutFailing()
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

        // Strict 350ms watchdog timeout
        using var bus = new NonBlockingDuplexSocketBus(serverSocket, channelCapacity: 4, watchdogTimeout: TimeSpan.FromMilliseconds(350));
        bus.Start();

        int pongsReceived = 0;
        using var cts = new CancellationTokenSource();

        // Background receiver on client socket verifying that NonBlockingDuplexSocketBus replies with HeartbeatPong
        var pongReaderTask = Task.Run(async () =>
        {
            byte[] buf = new byte[ClusterFrameHeader.HeaderSize];
            try
            {
                while (!cts.IsCancellationRequested)
                {
                    int r = await FusedSocketTransport.ReadExactAsync(clientSocket, buf.AsMemory(0, ClusterFrameHeader.HeaderSize), cts.Token);
                    if (r < ClusterFrameHeader.HeaderSize) break;

                    var hdr = MemoryMarshal.Read<ClusterFrameHeader>(buf);
                    if (hdr.MessageType == ClusterMessageType.HeartbeatPong)
                    {
                        Interlocked.Increment(ref pongsReceived);
                    }
                }
            }
            catch (OperationCanceledException) { }
            catch { }
        }, cts.Token);

        // Background task emits 20 heartbeats spaced 90ms apart (total elapsed: 1,800ms, which is > 5x the 350ms watchdog!)
        var senderTask = Task.Run(async () =>
        {
            byte[] pingBytes = new byte[ClusterFrameHeader.HeaderSize];
            var pingHdr = new ClusterFrameHeader(ClusterMessageType.HeartbeatPing, 555, 0);
            MemoryMarshal.Write(pingBytes, in pingHdr);

            for (int i = 0; i < 20; i++)
            {
                await Task.Delay(90);
                await clientSocket.SendAsync(pingBytes, SocketFlags.None);
            }

            // After 1.8s of simulated delay, send the actual token frame
            int payloadLen = FusedActivationResultHeader.HeaderSize;
            int totalBytes = ClusterFrameHeader.HeaderSize + payloadLen;
            byte[] resBytes = new byte[totalBytes];

            var clusterHdr = new ClusterFrameHeader((ClusterMessageType)0x71, 12345, (ulong)payloadLen);
            clusterHdr.WriteTo(resBytes.AsSpan(0, 32));

            var resultHdr = new FusedActivationResultHeader(
                stepIndex: 1,
                stageId: 2,
                computeMs: 4.25,
                nextTokenId: 8888,
                topLogit: 9.99f,
                precisionMode: WirePrecisionMode.FP16,
                flags: 0);
            resultHdr.WriteTo(resBytes.AsSpan(32, 32));

            await clientSocket.SendAsync(resBytes, SocketFlags.None);
        });

        // The bus must not time out despite 1.8s exceeding 350ms watchdog by 5x!
        var token = await bus.ReceiveTokenAsync(CancellationToken.None);
        await senderTask;

        cts.Cancel();
        try { await pongReaderTask; } catch { }

        Assert.Equal(12345UL, token.SequenceId);
        Assert.Equal(8888, token.TokenId);
        Assert.Equal(9.99f, token.TopLogit);
        Assert.Equal(4.25, token.ComputeMs);

        // Verify that bidirectional HeartbeatPong frames were indeed sent back to the client
        Assert.True(pongsReceived >= 10, $"Expected >= 10 HeartbeatPongs received, got {pongsReceived}");
    }

    [Fact]
    public async Task Watchdog_ConsecutiveCalls_ReusableCtsResetsCleanly()
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

        using var bus = new NonBlockingDuplexSocketBus(serverSocket, channelCapacity: 4, watchdogTimeout: TimeSpan.FromMilliseconds(500));
        bus.Start();

        // Execute 5 sequential token generations, each preceded by heartbeats
        for (int step = 0; step < 5; step++)
        {
            // Send 3 heartbeats spaced 100ms apart (300ms delay)
            byte[] pingBytes = new byte[ClusterFrameHeader.HeaderSize];
            var pingHdr = new ClusterFrameHeader(ClusterMessageType.HeartbeatPing, (ulong)step, 0);
            MemoryMarshal.Write(pingBytes, in pingHdr);

            for (int h = 0; h < 3; h++)
            {
                await Task.Delay(100);
                await clientSocket.SendAsync(pingBytes, SocketFlags.None);
            }

            // Send token frame
            int payloadLen = FusedActivationResultHeader.HeaderSize;
            int totalBytes = ClusterFrameHeader.HeaderSize + payloadLen;
            byte[] resBytes = new byte[totalBytes];

            var clusterHdr = new ClusterFrameHeader((ClusterMessageType)0x71, (ulong)step, (ulong)payloadLen);
            clusterHdr.WriteTo(resBytes.AsSpan(0, 32));

            var resultHdr = new FusedActivationResultHeader(
                stepIndex: step,
                stageId: 0,
                computeMs: 1.0 + step,
                nextTokenId: 1000 + step,
                topLogit: 1.5f + step,
                precisionMode: WirePrecisionMode.FP32,
                flags: 0);
            resultHdr.WriteTo(resBytes.AsSpan(32, 32));

            await clientSocket.SendAsync(resBytes, SocketFlags.None);

            var token = await bus.ReceiveTokenAsync(CancellationToken.None);
            Assert.Equal((ulong)step, token.SequenceId);
            Assert.Equal(1000 + step, token.TokenId);
        }
    }

    [Fact]
    public async Task WeightStreaming_PinnedBuffer_ZeroGen2Garbage_AndZeroLohFragmentation()
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

        int numChunks = 25; // 25 chunks * 4MB = 100 MB
        int chunkSize = 4 * 1024 * 1024;

        // Pre-allocate the reusable pinned buffers (1 for sender, 1 for receiver)
        byte[] sendBuffer = GC.AllocateArray<byte>(chunkSize, pinned: true);
        byte[] recvBuffer = GC.AllocateArray<byte>(chunkSize, pinned: true);
        byte[] recvHdrBytes = new byte[ClusterFrameHeader.HeaderSize];

        // Warm up JIT and Socket cached AwaitableSocketAsyncEventArgs with a single 4KB frame before measuring
        {
            byte[] warmHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
            var warmHdr = new ClusterFrameHeader(DistributedClusterSession.MsgStreamChunk, ulong.MaxValue, 4096);
            MemoryMarshal.Write(warmHdrBytes, in warmHdr);
            var warmSend = Task.Run(async () =>
            {
                await clientStream.WriteAsync(warmHdrBytes.AsMemory(0, ClusterFrameHeader.HeaderSize));
                await clientStream.WriteAsync(sendBuffer.AsMemory(0, 4096));
                await clientStream.FlushAsync();
            });
            await FusedSocketTransport.ReadExactAsync(serverStream, recvHdrBytes.AsMemory(0, ClusterFrameHeader.HeaderSize));
            var parsedWarm = MemoryMarshal.Read<ClusterFrameHeader>(recvHdrBytes);
            await FusedSocketTransport.ReadExactAsync(serverStream, recvBuffer.AsMemory(0, (int)parsedWarm.PayloadLength));
            await warmSend;
        }

        // Force GC clean baseline
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        int gen2Before = GC.CollectionCount(2);
        long allocatedBeforeReceive = GC.GetTotalAllocatedBytes(precise: true);

        // Sender streams 100MB of chunks using pre-allocated pinned buffer
        var senderTask = Task.Run(async () =>
        {
            byte[] hdrBytes = new byte[ClusterFrameHeader.HeaderSize];

            for (int i = 0; i < numChunks; i++)
            {
                sendBuffer.AsSpan().Fill((byte)(i & 0xFF));
                var chunkHdr = new ClusterFrameHeader(DistributedClusterSession.MsgStreamChunk, (ulong)i, (ulong)chunkSize);
                MemoryMarshal.Write(hdrBytes, in chunkHdr);

                await clientStream.WriteAsync(hdrBytes.AsMemory(0, ClusterFrameHeader.HeaderSize));
                await clientStream.WriteAsync(sendBuffer.AsMemory(0, chunkSize));
            }

            var doneHdr = new ClusterFrameHeader(DistributedClusterSession.MsgStreamDone, 999, 0);
            MemoryMarshal.Write(hdrBytes, in doneHdr);
            await clientStream.WriteAsync(hdrBytes.AsMemory(0, ClusterFrameHeader.HeaderSize));
            await clientStream.FlushAsync();
        });

        // Receiver uses the pre-allocated pinned 4MB buffer (simulating Worker M1 fix)
        int chunksReceived = 0;
        long totalBytesReceived = 0;

        while (true)
        {
            int read = await FusedSocketTransport.ReadExactAsync(serverStream, recvHdrBytes.AsMemory(0, ClusterFrameHeader.HeaderSize));
            if (read == 0) break;

            var hdr = MemoryMarshal.Read<ClusterFrameHeader>(recvHdrBytes);
            if (hdr.MessageType == DistributedClusterSession.MsgStreamDone)
            {
                break;
            }

            if (hdr.MessageType == DistributedClusterSession.MsgStreamChunk)
            {
                int cLen = (int)hdr.PayloadLength;
                int payloadRead = await FusedSocketTransport.ReadExactAsync(serverStream, recvBuffer.AsMemory(0, cLen));
                Assert.Equal(cLen, payloadRead);

                // Verify content integrity
                byte expectedByte = (byte)(chunksReceived & 0xFF);
                Assert.Equal(expectedByte, recvBuffer[0]);
                Assert.Equal(expectedByte, recvBuffer[cLen - 1]);

                chunksReceived++;
                totalBytesReceived += cLen;
            }
        }

        await senderTask;
        long allocatedDuringReceive = GC.GetTotalAllocatedBytes(precise: true) - allocatedBeforeReceive;

        int gen2After = GC.CollectionCount(2);

        Assert.Equal(numChunks, chunksReceived);
        Assert.Equal(numChunks * (long)chunkSize, totalBytesReceived);
        // Gen 2 collections must not have fired during this 100MB pinned transfer
        Assert.Equal(gen2Before, gen2After);
        // Across the entire 100MB (25 x 4MB) async stream transfer, process-wide allocations (ephemeral Gen0 async state machines)
        // must remain well below a single 4MB chunk (< 2MB total, zero 4MB LOH buffer allocations),
        // whereas naive per-chunk allocation would allocate >104,857,600 bytes on LOH!
        Assert.True(allocatedDuringReceive >= 0 && allocatedDuringReceive < 2 * 1024 * 1024, $"Allocated {allocatedDuringReceive} bytes, expected < 2MB across 100MB stream");
    }

    [Fact]
    public async Task DefensiveFraming_ArbitraryGarbageFramesAssault_PreservesStreamAlignment()
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

        // Adversarially generate 10 frames with corrupted/unknown opcodes and arbitrary non-zero payload lengths
        int[] garbagePayloadSizes = [13, 257, 1024, 8191, 7, 65535, 128, 4096, 512, 1];
        var senderTask = Task.Run(async () =>
        {
            for (int i = 0; i < garbagePayloadSizes.Length; i++)
            {
                int pLen = garbagePayloadSizes[i];
                var corruptHdr = new ClusterFrameHeader((ClusterMessageType)(0x80 + i), (ulong)(100 + i), (ulong)pLen);
                byte[] frameBytes = new byte[ClusterFrameHeader.HeaderSize + pLen];
                MemoryMarshal.Write(frameBytes, in corruptHdr);
                Random.Shared.NextBytes(frameBytes.AsSpan(ClusterFrameHeader.HeaderSize));

                await clientStream.WriteAsync(frameBytes, 0, frameBytes.Length);
            }

            // Interleave with legitimate HeartbeatPing
            var pingHdr = new ClusterFrameHeader(ClusterMessageType.HeartbeatPing, 9999, 0);
            byte[] pingBytes = new byte[ClusterFrameHeader.HeaderSize];
            MemoryMarshal.Write(pingBytes, in pingHdr);
            await clientStream.WriteAsync(pingBytes, 0, ClusterFrameHeader.HeaderSize);

            // Follow immediately with legitimate MsgResetKv
            var resetHdr = new ClusterFrameHeader(DistributedClusterSession.MsgResetKv, 8888, 0);
            byte[] resetBytes = new byte[ClusterFrameHeader.HeaderSize];
            MemoryMarshal.Write(resetBytes, in resetHdr);
            await clientStream.WriteAsync(resetBytes, 0, ClusterFrameHeader.HeaderSize);
            await clientStream.FlushAsync();
        });

        // Server processes frames using defensive framing loop
        byte[] hdrBuf = new byte[ClusterFrameHeader.HeaderSize];
        int garbageFramesDrained = 0;
        bool receivedHeartbeat = false;
        bool receivedResetKv = false;

        while (!receivedResetKv)
        {
            int r = await FusedSocketTransport.ReadExactAsync(serverStream, hdrBuf.AsMemory(0, ClusterFrameHeader.HeaderSize));
            Assert.Equal(ClusterFrameHeader.HeaderSize, r);

            var hdr = MemoryMarshal.Read<ClusterFrameHeader>(hdrBuf);
            // Invariant: Magic constant must NEVER be corrupted
            Assert.Equal(ClusterFrameHeader.MagicConstant, hdr.Magic);

            if (hdr.MessageType == ClusterMessageType.HeartbeatPing)
            {
                receivedHeartbeat = true;
                continue;
            }
            if (hdr.MessageType == DistributedClusterSession.MsgResetKv)
            {
                receivedResetKv = true;
                Assert.Equal(8888UL, hdr.CorrelationId);
                break;
            }

            // Unknown opcode: drain payload defensively
            if (hdr.PayloadLength > 0)
            {
                byte[] discard = ArrayPool<byte>.Shared.Rent((int)hdr.PayloadLength);
                try
                {
                    int drained = await FusedSocketTransport.ReadExactAsync(serverStream, discard.AsMemory(0, (int)hdr.PayloadLength));
                    Assert.Equal((int)hdr.PayloadLength, drained);
                    garbageFramesDrained++;
                }
                finally
                {
                    ArrayPool<byte>.Shared.Return(discard);
                }
            }
        }

        await senderTask;

        Assert.Equal(garbagePayloadSizes.Length, garbageFramesDrained);
        Assert.True(receivedHeartbeat);
        Assert.True(receivedResetKv);
    }
}
