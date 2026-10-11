namespace Glacier.Inference.Tests.Distributed;

using System;
using System.Buffers.Binary;
using System.Net;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Inference.Distributed;
using Xunit;

public class FusedZeroAllocationTransportTests
{
    [Fact]
    public void FusedActivationHeader_SizeIsExactly32Bytes_AndLayoutSequential()
    {
        Assert.Equal(32, Unsafe.SizeOf<FusedActivationHeader>());
        Assert.Equal(32, Marshal.SizeOf<FusedActivationHeader>());
        Assert.Equal(32, FusedActivationHeader.HeaderSize);

        // Verify explicit offsets for strict Little-Endian wire packing
        Assert.Equal(0, (int)Marshal.OffsetOf<FusedActivationHeader>(nameof(FusedActivationHeader.Magic)));
        Assert.Equal(4, (int)Marshal.OffsetOf<FusedActivationHeader>(nameof(FusedActivationHeader.SequenceId)));
        Assert.Equal(12, (int)Marshal.OffsetOf<FusedActivationHeader>(nameof(FusedActivationHeader.TokenIndex)));
        Assert.Equal(16, (int)Marshal.OffsetOf<FusedActivationHeader>(nameof(FusedActivationHeader.LayerIndex)));
        Assert.Equal(20, (int)Marshal.OffsetOf<FusedActivationHeader>(nameof(FusedActivationHeader.HiddenDim)));
        Assert.Equal(24, (int)Marshal.OffsetOf<FusedActivationHeader>(nameof(FusedActivationHeader.PrecisionMode)));
        Assert.Equal(25, (int)Marshal.OffsetOf<FusedActivationHeader>(nameof(FusedActivationHeader.Flags)));
        Assert.Equal(26, (int)Marshal.OffsetOf<FusedActivationHeader>(nameof(FusedActivationHeader.BatchTokens)));
        Assert.Equal(28, (int)Marshal.OffsetOf<FusedActivationHeader>(nameof(FusedActivationHeader.Scale)));
    }

    [Fact]
    public void FusedActivationHeader_CombinedWithClusterFrameHeader_IsExactly64BytesAligned()
    {
        Assert.Equal(32, ClusterFrameHeader.HeaderSize);
        Assert.Equal(32, FusedActivationHeader.HeaderSize);
        Assert.Equal(64, FusedSocketTransport.CombinedHeaderSize);
        Assert.Equal(64, ClusterFrameHeader.HeaderSize + FusedActivationHeader.HeaderSize);
    }

    [Fact]
    public void FusedActivationHeader_Serialization_LittleEndianRoundtrip()
    {
        ulong expectedSeq = 0xAABBCCDDEEFF0011UL;
        int expectedTok = 105;
        int expectedLayer = 19;
        int expectedDim = 5120;
        var expectedPrec = WirePrecisionMode.FP16;
        var expectedFlags = FusedActivationFlags.FinalLayer | FusedActivationFlags.IsPrefillChunk;
        ushort expectedBatchTokens = 256;
        float expectedScale = 1.25f;

        var original = new FusedActivationHeader(
            sequenceId: expectedSeq,
            tokenIndex: expectedTok,
            layerIndex: expectedLayer,
            hiddenDim: expectedDim,
            precisionMode: expectedPrec,
            flags: expectedFlags,
            scale: expectedScale,
            batchTokens: expectedBatchTokens);

        byte[] buffer = new byte[32];
        original.WriteTo(buffer.AsSpan());

        // Verify Little-Endian byte pattern for Magic 0x47414354 ("GACT")
        Assert.Equal((byte)0x54, buffer[0]); // 'T'
        Assert.Equal((byte)0x43, buffer[1]); // 'C'
        Assert.Equal((byte)0x41, buffer[2]); // 'A'
        Assert.Equal((byte)0x47, buffer[3]); // 'G'

        // Verify SequenceId
        ulong parsedSeq = BinaryPrimitives.ReadUInt64LittleEndian(buffer.AsSpan(4, 8));
        Assert.Equal(expectedSeq, parsedSeq);

        // Deserialization
        var deserialized = FusedActivationHeader.ReadFrom(buffer.AsSpan());

        Assert.Equal(original, deserialized);
        Assert.Equal(FusedActivationHeader.FusedMagic, deserialized.Magic);
        Assert.Equal(expectedSeq, deserialized.SequenceId);
        Assert.Equal(expectedTok, deserialized.TokenIndex);
        Assert.Equal(expectedLayer, deserialized.LayerIndex);
        Assert.Equal(expectedDim, deserialized.HiddenDim);
        Assert.Equal((byte)expectedPrec, deserialized.PrecisionMode);
        Assert.Equal((byte)expectedFlags, deserialized.Flags);
        Assert.Equal(expectedBatchTokens, deserialized.BatchTokens);
        Assert.Equal(expectedScale, deserialized.Scale);

        // TryRead
        bool ok = FusedActivationHeader.TryRead(buffer.AsSpan(), out var tryReadHdr);
        Assert.True(ok);
        Assert.Equal(original, tryReadHdr);
    }

    [Fact]
    public void FusedActivationResultHeader_SizeIsExactly32Bytes_AndLayoutSequential()
    {
        Assert.Equal(32, Unsafe.SizeOf<FusedActivationResultHeader>());
        Assert.Equal(32, Marshal.SizeOf<FusedActivationResultHeader>());
        Assert.Equal(32, FusedActivationResultHeader.HeaderSize);

        Assert.Equal(0, (int)Marshal.OffsetOf<FusedActivationResultHeader>(nameof(FusedActivationResultHeader.Magic)));
        Assert.Equal(4, (int)Marshal.OffsetOf<FusedActivationResultHeader>(nameof(FusedActivationResultHeader.StepIndex)));
        Assert.Equal(8, (int)Marshal.OffsetOf<FusedActivationResultHeader>(nameof(FusedActivationResultHeader.StageId)));
        Assert.Equal(12, (int)Marshal.OffsetOf<FusedActivationResultHeader>(nameof(FusedActivationResultHeader.ComputeMs)));
        Assert.Equal(20, (int)Marshal.OffsetOf<FusedActivationResultHeader>(nameof(FusedActivationResultHeader.NextTokenId)));
        Assert.Equal(24, (int)Marshal.OffsetOf<FusedActivationResultHeader>(nameof(FusedActivationResultHeader.TopLogit)));
        Assert.Equal(28, (int)Marshal.OffsetOf<FusedActivationResultHeader>(nameof(FusedActivationResultHeader.PrecisionMode)));
        Assert.Equal(29, (int)Marshal.OffsetOf<FusedActivationResultHeader>(nameof(FusedActivationResultHeader.Flags)));
        Assert.Equal(30, (int)Marshal.OffsetOf<FusedActivationResultHeader>(nameof(FusedActivationResultHeader.Reserved)));
    }

    [Fact]
    public void FusedActivationResultHeader_Serialization_Roundtrip()
    {
        int expectedStep = 42;
        int expectedStage = 2;
        double expectedComputeMs = 4.321;
        int expectedToken = 151643;
        float expectedTopLogit = 18.75f;
        var expectedPrec = WirePrecisionMode.FP32;
        byte expectedFlags = (byte)FusedActivationFlags.FinalLayer;

        var original = new FusedActivationResultHeader(
            stepIndex: expectedStep,
            stageId: expectedStage,
            computeMs: expectedComputeMs,
            nextTokenId: expectedToken,
            topLogit: expectedTopLogit,
            precisionMode: expectedPrec,
            flags: expectedFlags);

        byte[] buffer = new byte[32];
        original.WriteTo(buffer.AsSpan());

        // Verify Little-Endian byte pattern for Magic 0x47524553 ("GRES")
        Assert.Equal((byte)0x53, buffer[0]); // 'S'
        Assert.Equal((byte)0x45, buffer[1]); // 'E'
        Assert.Equal((byte)0x52, buffer[2]); // 'R'
        Assert.Equal((byte)0x47, buffer[3]); // 'G'

        // Deserialization
        var deserialized = FusedActivationResultHeader.ReadFrom(buffer.AsSpan());

        Assert.Equal(original, deserialized);
        Assert.Equal(FusedActivationResultHeader.ResultMagic, deserialized.Magic);
        Assert.Equal(expectedStep, deserialized.StepIndex);
        Assert.Equal(expectedStage, deserialized.StageId);
        Assert.Equal(expectedComputeMs, deserialized.ComputeMs);
        Assert.Equal(expectedToken, deserialized.NextTokenId);
        Assert.Equal(expectedTopLogit, deserialized.TopLogit);
        Assert.Equal((byte)expectedPrec, deserialized.PrecisionMode);
        Assert.Equal(expectedFlags, deserialized.Flags);

        // TryRead
        bool ok = FusedActivationResultHeader.TryRead(buffer.AsSpan(), out var tryReadHdr);
        Assert.True(ok);
        Assert.Equal(original, tryReadHdr);
    }

    [Fact]
    public void PinnedSocketBufferPool_RentReturn_IsPinnedAndReusable()
    {
        using var pool = new PinnedSocketBufferPool(bufferSize: 64 * 1024, maxBuffers: 4);

        Assert.Equal(4, pool.AvailableCount);

        var buf1 = pool.Rent();
        Assert.Equal(64 * 1024, buf1.Length);
        Assert.Equal(3, pool.AvailableCount);

        var buf2 = pool.Rent();
        Assert.Equal(2, pool.AvailableCount);

        pool.Return(buf1);
        Assert.Equal(3, pool.AvailableCount);

        pool.Return(buf2);
        Assert.Equal(4, pool.AvailableCount);

        // RentArray should reuse existing pinned array directly
        var arr = pool.RentArray();
        Assert.Equal(64 * 1024, arr.Length);
        Assert.Equal(3, pool.AvailableCount);
        pool.ReturnArray(arr);
        Assert.Equal(4, pool.AvailableCount);
    }

    [Fact]
    public void PinnedBatchBufferArena_AllocatesPinnedSlices_AndResetsSafely()
    {
        var arena = new PinnedBatchBufferArena(maxBatchTokens: 2048, hiddenDim: 5120, vocabSize: 152064);

        Assert.Equal(arena.MaxBytePayload, arena.BatchWireRecvArray.Length);
        Assert.Equal(arena.MaxBytePayload, arena.BatchWireSendArray.Length);
        Assert.Equal(arena.MaxTotalFloats, arena.FloatInputArray.Length);
        Assert.Equal(arena.MaxTotalFloats, arena.FloatOutputArray.Length);
        Assert.True(arena.BatchWireRecvArray.Length > 40 * 1024 * 1024);
        Assert.Equal(152064, arena.LogitsArray.Length);

        // Test mutating and clearing
        arena.BatchWireRecvArray[0] = 0xAA;
        arena.FloatInputArray[0] = 3.14f;

        arena.Reset();

        Assert.Equal((byte)0, arena.BatchWireRecvArray[0]);
        Assert.Equal(0.0f, arena.FloatInputArray[0]);
    }

    [Fact]
    public void FusedSocketTransport_ConfigureSocket_SetsNoDelayAndBufferSizes()
    {
        using var socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        FusedSocketTransport.ConfigureSocket(socket, bufferSizeBytes: 4 * 1024 * 1024);

        Assert.True(socket.NoDelay);
        Assert.True(socket.SendBufferSize >= 4 * 1024 * 1024);
        Assert.True(socket.ReceiveBufferSize >= 4 * 1024 * 1024);
    }

    [Fact]
    public async Task FusedSocketTransport_Loopback_SendReceiveFusedActivation_Roundtrip()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        var serverTask = Task.Run(async () =>
        {
            using var serverSocket = await listener.AcceptAsync();
            FusedSocketTransport.ConfigureSocket(serverSocket);

            byte[] recvBuffer = GC.AllocateArray<byte>(64 * 1024, pinned: true);
            var (clusterHdr, actHdr, payloadLen) = await FusedSocketTransport.ReceiveFusedActivationAsync(serverSocket, recvBuffer);

            Assert.Equal(ClusterFrameHeader.MagicConstant, clusterHdr.Magic);
            Assert.Equal(FusedActivationHeader.FusedMagic, actHdr.Magic);
            Assert.Equal(42UL, actHdr.SequenceId);
            Assert.Equal(7, actHdr.TokenIndex);
            Assert.Equal(3, actHdr.LayerIndex);
            Assert.Equal(4, actHdr.HiddenDim);
            Assert.Equal(WirePrecisionMode.FP32, actHdr.TypedPrecision);
            Assert.Equal(16, payloadLen);

            var receivedFloats = MemoryMarshal.Cast<byte, float>(recvBuffer.AsSpan(FusedSocketTransport.CombinedHeaderSize, payloadLen));
            Assert.Equal(4, receivedFloats.Length);
            Assert.Equal(1.0f, receivedFloats[0]);
            Assert.Equal(2.0f, receivedFloats[1]);
            Assert.Equal(3.0f, receivedFloats[2]);
            Assert.Equal(4.0f, receivedFloats[3]);
        });

        using var clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        await clientSocket.ConnectAsync(listener.LocalEndPoint!);
        FusedSocketTransport.ConfigureSocket(clientSocket);

        byte[] sendBuffer = GC.AllocateArray<byte>(64 * 1024, pinned: true);
        float[] payloadFloats = new float[] { 1.0f, 2.0f, 3.0f, 4.0f };
        byte[] payloadBytes = MemoryMarshal.AsBytes(payloadFloats.AsSpan()).ToArray();

        var actHeader = new FusedActivationHeader(
            sequenceId: 42UL,
            tokenIndex: 7,
            layerIndex: 3,
            hiddenDim: 4,
            precisionMode: WirePrecisionMode.FP32);

        await FusedSocketTransport.SendFusedActivationAsync(
            clientSocket,
            correlationId: 42UL,
            actHeader: actHeader,
            activationPayload: payloadBytes,
            stagingBuffer: sendBuffer);

        await serverTask;
    }

    [Fact]
    public async Task FusedSocketTransport_Loopback_SendReceiveResult_Roundtrip()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        var serverTask = Task.Run(async () =>
        {
            using var serverSocket = await listener.AcceptAsync();
            FusedSocketTransport.ConfigureSocket(serverSocket);

            byte[] recvBuffer = GC.AllocateArray<byte>(64 * 1024, pinned: true);
            var (clusterHdr, resultHdr, payloadLen) = await FusedSocketTransport.ReceiveResultAsync(serverSocket, recvBuffer);

            Assert.Equal(ClusterFrameHeader.MagicConstant, clusterHdr.Magic);
            Assert.Equal(FusedActivationResultHeader.ResultMagic, resultHdr.Magic);
            Assert.Equal(1, resultHdr.StepIndex);
            Assert.Equal(2, resultHdr.StageId);
            Assert.Equal(12345, resultHdr.NextTokenId);
            Assert.Equal(9.5f, resultHdr.TopLogit);
            Assert.Equal(0, payloadLen);
        });

        using var clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        await clientSocket.ConnectAsync(listener.LocalEndPoint!);
        FusedSocketTransport.ConfigureSocket(clientSocket);

        byte[] sendBuffer = GC.AllocateArray<byte>(64 * 1024, pinned: true);

        var resultHeader = new FusedActivationResultHeader(
            stepIndex: 1,
            stageId: 2,
            computeMs: 1.25,
            nextTokenId: 12345,
            topLogit: 9.5f,
            precisionMode: WirePrecisionMode.FP16,
            flags: (byte)FusedActivationFlags.FinalLayer);

        await FusedSocketTransport.SendResultAsync(
            clientSocket,
            correlationId: 99UL,
            resultHeader: resultHeader,
            activationPayload: ReadOnlyMemory<byte>.Empty,
            stagingBuffer: sendBuffer);

        await serverTask;
    }
}
