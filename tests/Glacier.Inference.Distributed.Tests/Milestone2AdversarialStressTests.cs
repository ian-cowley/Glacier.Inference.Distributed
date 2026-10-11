namespace Glacier.Inference.Tests.Distributed;

using System;
using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Inference.Distributed;
using Xunit;

/// <summary>
/// Empirical Challenger Adversarial Stress Suite for Milestone 2:
/// Fused Zero-Allocation Socket Transport and Pinned Buffer Arena.
/// 
/// Stress Dimensions:
/// 1. Zero-Allocation Hot Decode Loop: Assert 0 Gen 0, Gen 1, Gen 2 GC collections during continuous socket transfers.
/// 2. Pinned Buffer Pool Concurrency, High-Frequency Rent/Return Cycles, and Buffer Overrun Resistance.
/// 3. Pinned Batch Buffer Arena under Large Prompt Chunks (up to and beyond 2048 tokens).
/// 4. Framing Overrun & Truncation Guards across socket boundaries.
/// </summary>
public class Milestone2AdversarialStressTests
{
    private const int DeepSeekHiddenDim = 5120;
    private const int Fp16BytesPerToken = DeepSeekHiddenDim * sizeof(short); // 10,240 bytes
    private const int Fp32BytesPerToken = DeepSeekHiddenDim * sizeof(float); // 20,480 bytes
    private const int Int8BytesPerToken = DeepSeekHiddenDim * sizeof(byte) + 4; // 5,124 bytes

    #region 1. Zero-Allocation Hot Decode Loop Stress Tests

    [Fact]
    public async Task HotDecodeLoop_ZeroAlloc_1000ContinuousSocketTokenTransfers_AssertZeroGcCollections()
    {
        // Setup loopback TCP socket pair
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = client.ConnectAsync(listener.LocalEndPoint!);
        using var server = await listener.AcceptAsync();
        await connectTask;

        FusedSocketTransport.ConfigureSocket(client, 4 * 1024 * 1024);
        FusedSocketTransport.ConfigureSocket(server, 4 * 1024 * 1024);

        // Pre-allocated pinned buffers matching coordinator and worker architecture
        byte[] coordSendBuffer = GC.AllocateArray<byte>(64 * 1024, pinned: true);
        byte[] coordRecvBuffer = GC.AllocateArray<byte>(64 * 1024, pinned: true);
        byte[] workerRecvBuffer = GC.AllocateArray<byte>(64 * 1024, pinned: true);
        byte[] workerSendBuffer = GC.AllocateArray<byte>(64 * 1024, pinned: true);

        // Activation payload: FP16 tensor (10,240 bytes for dim 5120)
        byte[] actPayload = GC.AllocateArray<byte>(Fp16BytesPerToken, pinned: true);
        for (int i = 0; i < actPayload.Length; i++) actPayload[i] = (byte)(i & 0xFF);

        const int iterations = 1_000;

        // Warmup: 50 iterations to ensure socket buffers, JIT compilation, and runtime plumbing are warm
        for (int i = 0; i < 50; i++)
        {
            var actHdr = new FusedActivationHeader(
                sequenceId: (ulong)i,
                tokenIndex: i,
                layerIndex: 27,
                hiddenDim: DeepSeekHiddenDim,
                precisionMode: WirePrecisionMode.FP16,
                flags: FusedActivationFlags.FinalLayer);

            FusedSocketTransport.SendFusedActivation(client, (ulong)i, actHdr, actPayload, coordSendBuffer);

            bool ok = FusedSocketTransport.ReceiveFusedActivation(
                server,
                workerRecvBuffer,
                out var clusterHdr,
                out var rxActHdr,
                out var payloadSpan);

            Assert.True(ok);
            Assert.Equal((ulong)i, rxActHdr.SequenceId);

            var resHdr = new FusedActivationResultHeader(
                stepIndex: i,
                stageId: 1,
                computeMs: 1.5,
                nextTokenId: 1000 + i,
                topLogit: 15.0f,
                precisionMode: WirePrecisionMode.FP16,
                flags: (byte)FusedActivationFlags.FinalLayer);

            FusedSocketTransport.SendResult(server, (ulong)i, resHdr, ReadOnlySpan<byte>.Empty, workerSendBuffer);

            bool resOk = FusedSocketTransport.ReceiveResult(
                client,
                coordRecvBuffer,
                out var resClusterHdr,
                out var rxResHdr,
                out var resPayloadSpan);

            Assert.True(resOk);
            Assert.Equal(1000 + i, rxResHdr.NextTokenId);
        }

        // Force full GC collection prior to zero-allocation assertion
        GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);

        int gen0Start = GC.CollectionCount(0);
        int gen1Start = GC.CollectionCount(1);
        int gen2Start = GC.CollectionCount(2);
        long allocBytesStart = GC.GetAllocatedBytesForCurrentThread();

        // Adversarial execution: 1,000 continuous socket token roundtrips
        for (int i = 0; i < iterations; i++)
        {
            var actHdr = new FusedActivationHeader(
                sequenceId: (ulong)i,
                tokenIndex: i,
                layerIndex: 27,
                hiddenDim: DeepSeekHiddenDim,
                precisionMode: WirePrecisionMode.FP16,
                flags: FusedActivationFlags.FinalLayer);

            FusedSocketTransport.SendFusedActivation(client, (ulong)i, actHdr, actPayload, coordSendBuffer);

            bool ok = FusedSocketTransport.ReceiveFusedActivation(
                server,
                workerRecvBuffer,
                out var clusterHdr,
                out var rxActHdr,
                out var payloadSpan);

            if (!ok) throw new InvalidOperationException($"ReceiveFusedActivation failed at step {i}");

            var resHdr = new FusedActivationResultHeader(
                stepIndex: i,
                stageId: 1,
                computeMs: 1.25,
                nextTokenId: 5000 + i,
                topLogit: 12.5f,
                precisionMode: WirePrecisionMode.FP16,
                flags: (byte)FusedActivationFlags.FinalLayer);

            FusedSocketTransport.SendResult(server, (ulong)i, resHdr, ReadOnlySpan<byte>.Empty, workerSendBuffer);

            bool resOk = FusedSocketTransport.ReceiveResult(
                client,
                coordRecvBuffer,
                out var resClusterHdr,
                out var rxResHdr,
                out var resPayloadSpan);

            if (!resOk) throw new InvalidOperationException($"ReceiveResult failed at step {i}");
        }

        long allocBytesEnd = GC.GetAllocatedBytesForCurrentThread();
        int gen0End = GC.CollectionCount(0);
        int gen1End = GC.CollectionCount(1);
        int gen2End = GC.CollectionCount(2);

        // Strict Invariants: Exactly ZERO GC collections during 1,000 hot decode socket steps
        Assert.Equal(0, gen0End - gen0Start);
        Assert.Equal(0, gen1End - gen1Start);
        Assert.Equal(0, gen2End - gen2Start);
        Assert.Equal(0L, allocBytesEnd - allocBytesStart);
    }

    [Fact]
    public async Task HotDecodeLoop_MultiPrecisionSwitching_ZeroAlloc_AssertZeroGcCollections()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = client.ConnectAsync(listener.LocalEndPoint!);
        using var server = await listener.AcceptAsync();
        await connectTask;

        FusedSocketTransport.ConfigureSocket(client);
        FusedSocketTransport.ConfigureSocket(server);

        byte[] coordSend = GC.AllocateArray<byte>(64 * 1024, pinned: true);
        byte[] coordRecv = GC.AllocateArray<byte>(64 * 1024, pinned: true);
        byte[] workerRecv = GC.AllocateArray<byte>(64 * 1024, pinned: true);
        byte[] workerSend = GC.AllocateArray<byte>(64 * 1024, pinned: true);

        byte[] fp32Payload = GC.AllocateArray<byte>(Fp32BytesPerToken, pinned: true);
        byte[] fp16Payload = GC.AllocateArray<byte>(Fp16BytesPerToken, pinned: true);
        byte[] int8Payload = GC.AllocateArray<byte>(Int8BytesPerToken, pinned: true);

        // Warmup
        for (int i = 0; i < 30; i++)
        {
            var prec = (WirePrecisionMode)(i % 3);
            ReadOnlySpan<byte> curPayload = prec switch
            {
                WirePrecisionMode.FP32 => fp32Payload,
                WirePrecisionMode.FP16 => fp16Payload,
                _ => int8Payload
            };

            var hdr = new FusedActivationHeader((ulong)i, i, 0, DeepSeekHiddenDim, prec);
            FusedSocketTransport.SendFusedActivation(client, (ulong)i, hdr, curPayload, coordSend);
            FusedSocketTransport.ReceiveFusedActivation(server, workerRecv, out _, out _, out _);

            var res = new FusedActivationResultHeader(i, 1, 1.0, 100, 1.0f, prec);
            FusedSocketTransport.SendResult(server, (ulong)i, res, ReadOnlySpan<byte>.Empty, workerSend);
            FusedSocketTransport.ReceiveResult(client, coordRecv, out _, out _, out _);
        }

        GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);

        int g0 = GC.CollectionCount(0);
        int g1 = GC.CollectionCount(1);
        int g2 = GC.CollectionCount(2);
        long allocStart = GC.GetAllocatedBytesForCurrentThread();

        for (int i = 0; i < 600; i++)
        {
            var prec = (WirePrecisionMode)(i % 3);
            ReadOnlySpan<byte> curPayload = prec switch
            {
                WirePrecisionMode.FP32 => fp32Payload,
                WirePrecisionMode.FP16 => fp16Payload,
                _ => int8Payload
            };

            var hdr = new FusedActivationHeader((ulong)i, i, 0, DeepSeekHiddenDim, prec);
            FusedSocketTransport.SendFusedActivation(client, (ulong)i, hdr, curPayload, coordSend);
            bool rxOk = FusedSocketTransport.ReceiveFusedActivation(server, workerRecv, out _, out var actHdr, out var pl);
            if (!rxOk || actHdr.PrecisionMode != (byte)prec || pl.Length != curPayload.Length)
            {
                throw new InvalidOperationException($"RX verification failed at step {i}");
            }

            var res = new FusedActivationResultHeader(i, 1, 1.0, 200 + i, 2.0f, prec);
            FusedSocketTransport.SendResult(server, (ulong)i, res, ReadOnlySpan<byte>.Empty, workerSend);
            bool resOk = FusedSocketTransport.ReceiveResult(client, coordRecv, out _, out var rxRes, out _);
            if (!resOk || rxRes.NextTokenId != 200 + i)
            {
                throw new InvalidOperationException($"Result verification failed at step {i}");
            }
        }

        long allocEnd = GC.GetAllocatedBytesForCurrentThread();
        Assert.Equal(0, GC.CollectionCount(0) - g0);
        Assert.Equal(0, GC.CollectionCount(1) - g1);
        Assert.Equal(0, GC.CollectionCount(2) - g2);
        Assert.Equal(0L, allocEnd - allocStart);
    }

    [Fact]
    public async Task HotDecodeLoop_Async_ContinuousTokenTransfers_AssertZeroGcCollections()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = client.ConnectAsync(listener.LocalEndPoint!);
        using var server = await listener.AcceptAsync();
        await connectTask;

        FusedSocketTransport.ConfigureSocket(client, 4 * 1024 * 1024);
        FusedSocketTransport.ConfigureSocket(server, 4 * 1024 * 1024);

        byte[] coordSend = GC.AllocateArray<byte>(64 * 1024, pinned: true);
        byte[] coordRecv = GC.AllocateArray<byte>(64 * 1024, pinned: true);
        byte[] workerRecv = GC.AllocateArray<byte>(64 * 1024, pinned: true);
        byte[] workerSend = GC.AllocateArray<byte>(64 * 1024, pinned: true);
        byte[] actPayload = GC.AllocateArray<byte>(Fp16BytesPerToken, pinned: true);

        // Warmup
        for (int i = 0; i < 20; i++)
        {
            var actHdr = new FusedActivationHeader((ulong)i, i, 1, DeepSeekHiddenDim, WirePrecisionMode.FP16);
            await FusedSocketTransport.SendFusedActivationAsync(client, (ulong)i, actHdr, actPayload, coordSend);
            var (c1, a1, pLen) = await FusedSocketTransport.ReceiveFusedActivationAsync(server, workerRecv);
            Assert.Equal(ClusterFrameHeader.MagicConstant, c1.Magic);

            var resHdr = new FusedActivationResultHeader(i, 1, 0.8, 42, 10.0f, WirePrecisionMode.FP16);
            await FusedSocketTransport.SendResultAsync(server, (ulong)i, resHdr, ReadOnlyMemory<byte>.Empty, workerSend);
            var (c2, r1, _) = await FusedSocketTransport.ReceiveResultAsync(client, coordRecv);
            Assert.Equal(42, r1.NextTokenId);
        }

        GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);

        int g0 = GC.CollectionCount(0);
        int g1 = GC.CollectionCount(1);
        int g2 = GC.CollectionCount(2);

        const int asyncIterations = 500;
        for (int i = 0; i < asyncIterations; i++)
        {
            var actHdr = new FusedActivationHeader((ulong)i, i, 1, DeepSeekHiddenDim, WirePrecisionMode.FP16);
            await FusedSocketTransport.SendFusedActivationAsync(client, (ulong)i, actHdr, actPayload, coordSend);
            var (c1, a1, pLen) = await FusedSocketTransport.ReceiveFusedActivationAsync(server, workerRecv);

            var resHdr = new FusedActivationResultHeader(i, 1, 0.8, 1000 + i, 10.0f, WirePrecisionMode.FP16);
            await FusedSocketTransport.SendResultAsync(server, (ulong)i, resHdr, ReadOnlyMemory<byte>.Empty, workerSend);
            var (c2, r1, _) = await FusedSocketTransport.ReceiveResultAsync(client, coordRecv);

            Assert.Equal(1000 + i, r1.NextTokenId);
        }

        // Under 4MB socket buffer sizes and cached completions, GC collections must remain 0
        Assert.Equal(0, GC.CollectionCount(0) - g0);
        Assert.Equal(0, GC.CollectionCount(1) - g1);
        Assert.Equal(0, GC.CollectionCount(2) - g2);
    }

    #endregion

    #region 2. Pinned Buffer Pool Concurrency, Stress & Overrun Resistance

    [Fact]
    public void PinnedSocketBufferPool_HighFrequencyRentReturn_100kCycles_NoLeaks()
    {
        using var pool = new PinnedSocketBufferPool(bufferSize: 64 * 1024, maxBuffers: 8);

        Assert.Equal(8, pool.AvailableCount);

        for (int i = 0; i < 100_000; i++)
        {
            var mem = pool.Rent();
            pool.Return(mem);
        }

        Assert.Equal(8, pool.AvailableCount);

        for (int i = 0; i < 100_000; i++)
        {
            var arr = pool.RentArray();
            pool.ReturnArray(arr);
        }

        Assert.Equal(8, pool.AvailableCount);
    }

    [Fact]
    public void PinnedSocketBufferPool_HighConcurrency_32Threads_RentReturnStress()
    {
        const int threadCount = 32;
        const int iterationsPerThread = 5_000;
        using var pool = new PinnedSocketBufferPool(bufferSize: 32 * 1024, maxBuffers: threadCount);

        var exceptions = new ConcurrentBag<Exception>();
        var threads = new Thread[threadCount];

        for (int t = 0; t < threadCount; t++)
        {
            int threadId = t;
            threads[t] = new Thread(() =>
            {
                try
                {
                    for (int i = 0; i < iterationsPerThread; i++)
                    {
                        var mem = pool.Rent();
                        // Mutate buffer with thread-specific canary pattern
                        var span = mem.Span;
                        span[0] = (byte)threadId;
                        span[^1] = (byte)(threadId ^ 0xFF);

                        // Read back and assert canary
                        if (span[0] != (byte)threadId || span[^1] != (byte)(threadId ^ 0xFF))
                        {
                            throw new InvalidOperationException($"Canary corruption in thread {threadId}");
                        }

                        pool.Return(mem);
                    }
                }
                catch (Exception ex)
                {
                    exceptions.Add(ex);
                }
            });
        }

        for (int t = 0; t < threadCount; t++) threads[t].Start();
        for (int t = 0; t < threadCount; t++) threads[t].Join();

        Assert.Empty(exceptions);
        Assert.Equal(threadCount, pool.AvailableCount);
    }

    [Fact]
    public void PinnedSocketBufferPool_Exhaustion_ThrowsInvalidOperationException()
    {
        using var pool = new PinnedSocketBufferPool(bufferSize: 1024, maxBuffers: 3);

        var m1 = pool.Rent();
        var m2 = pool.Rent();
        var m3 = pool.Rent();

        Assert.Equal(0, pool.AvailableCount);

        // Exhausted Rent must throw InvalidOperationException
        var ex1 = Assert.Throws<InvalidOperationException>(() => pool.Rent());
        Assert.Contains("exhausted", ex1.Message, StringComparison.OrdinalIgnoreCase);

        var ex2 = Assert.Throws<InvalidOperationException>(() => pool.RentArray());
        Assert.Contains("exhausted", ex2.Message, StringComparison.OrdinalIgnoreCase);

        // Returning one allows Rent to succeed again
        pool.Return(m2);
        Assert.Equal(1, pool.AvailableCount);

        var m4 = pool.Rent();
        Assert.Equal(0, pool.AvailableCount);

        pool.Return(m1);
        pool.Return(m3);
        pool.Return(m4);

        Assert.Equal(3, pool.AvailableCount);
    }

    [Fact]
    public void PinnedSocketBufferPool_ForeignBuffers_SafelyIgnored()
    {
        using var pool = new PinnedSocketBufferPool(bufferSize: 1024, maxBuffers: 4);

        var foreignMem = new Memory<byte>(new byte[1024]);
        var foreignArray = new byte[1024];

        // Returning unpooled buffers must be safely ignored without corrupting AvailableCount
        pool.Return(foreignMem);
        Assert.Equal(4, pool.AvailableCount);

        pool.ReturnArray(foreignArray);
        Assert.Equal(4, pool.AvailableCount);
    }

    [Fact]
    public void PinnedSocketBufferPool_SlicedBufferReturn_CorrectlyMatchesUnderlyingArray()
    {
        using var pool = new PinnedSocketBufferPool(bufferSize: 4096, maxBuffers: 2);

        var mem = pool.Rent();
        Assert.Equal(1, pool.AvailableCount);

        // Slice the memory buffer
        var sliced = mem.Slice(128, 512);

        // Returning the slice must resolve the underlying array and reclaim the buffer
        pool.Return(sliced);
        Assert.Equal(2, pool.AvailableCount);

        // Renting again should succeed without pool exhaustion
        var memAgain = pool.Rent();
        Assert.Equal(1, pool.AvailableCount);
        pool.Return(memAgain);
        Assert.Equal(2, pool.AvailableCount);
    }

    [Theory]
    [InlineData(0, 4)]
    [InlineData(-1, 4)]
    [InlineData(1024, 0)]
    [InlineData(1024, -5)]
    public void PinnedSocketBufferPool_InvalidArguments_ThrowsArgumentOutOfRangeException(int bufferSize, int maxBuffers)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new PinnedSocketBufferPool(bufferSize, maxBuffers));
    }

    [Fact]
    public void PinnedSocketBufferPool_Disposal_ReleasesAndGuardsAccess()
    {
        var pool = new PinnedSocketBufferPool(bufferSize: 1024, maxBuffers: 2);
        pool.Dispose();

        Assert.Equal(0, pool.AvailableCount);
        Assert.Throws<ObjectDisposedException>(() => pool.Rent());
        Assert.Throws<ObjectDisposedException>(() => pool.RentArray());

        // Return and Dispose after disposal must be safe no-ops
        pool.Return(new Memory<byte>(new byte[1024]));
        pool.ReturnArray(new byte[1024]);
        pool.Dispose(); // Idempotent
    }

    #endregion

    #region 3. Pinned Batch Buffer Arena Stress Under Large Prompt Chunks (up to 2048 Tokens)

    [Fact]
    public void PinnedBatchBufferArena_MaxCapacity_2048Tokens_AllBuffersSizedAndPinned()
    {
        const int maxBatch = 2048;
        const int dim = 5120;
        const int vocab = 152064;

        using var arena = new PinnedBatchBufferArena(maxBatchTokens: maxBatch, hiddenDim: dim, vocabSize: vocab);

        int expectedFloats = maxBatch * dim; // 10,485,760
        int expectedPayloadBytes = 64 + (expectedFloats * sizeof(float)); // 41,943,104

        Assert.Equal(expectedFloats, arena.MaxTotalFloats);
        Assert.Equal(expectedPayloadBytes, arena.MaxBytePayload);
        Assert.Equal(expectedPayloadBytes, arena.BatchWireSendArray.Length);
        Assert.Equal(expectedPayloadBytes, arena.BatchWireRecvArray.Length);
        Assert.Equal(expectedFloats, arena.FloatInputArray.Length);
        Assert.Equal(expectedFloats, arena.FloatOutputArray.Length);
        Assert.Equal(vocab, arena.LogitsArray.Length);

        // Verify full spans at exact capacity
        var sendSpan = arena.GetBatchSendSpan(expectedPayloadBytes);
        var recvSpan = arena.GetBatchRecvSpan(expectedPayloadBytes);
        var inSpan = arena.GetFloatInputSpan(expectedFloats);
        var outSpan = arena.GetFloatOutputSpan(expectedFloats);
        var logSpan = arena.GetLogitsSpan(vocab);

        Assert.Equal(expectedPayloadBytes, sendSpan.Length);
        Assert.Equal(expectedPayloadBytes, recvSpan.Length);
        Assert.Equal(expectedFloats, inSpan.Length);
        Assert.Equal(expectedFloats, outSpan.Length);
        Assert.Equal(vocab, logSpan.Length);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(16)]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    [InlineData(2047)]
    [InlineData(2048)]
    public void PinnedBatchBufferArena_ParametricChunkSizes_SpanSlicingAndIntegrity(int numTokens)
    {
        const int dim = 5120;
        using var arena = new PinnedBatchBufferArena(maxBatchTokens: 2048, hiddenDim: dim, vocabSize: 152064);

        int floatCount = numTokens * dim;
        int bytePayload = 64 + (floatCount * sizeof(float));

        var sendSpan = arena.GetBatchSendSpan(bytePayload);
        var inSpan = arena.GetFloatInputSpan(floatCount);

        // Populate input with unique token-indexed values
        for (int t = 0; t < numTokens; t++)
        {
            inSpan[t * dim] = (float)t;
            inSpan[t * dim + dim - 1] = (float)(t + 0.5f);
        }

        // Verify values
        for (int t = 0; t < numTokens; t++)
        {
            Assert.Equal((float)t, inSpan[t * dim]);
            Assert.Equal((float)(t + 0.5f), inSpan[t * dim + dim - 1]);
        }

        Assert.Equal(bytePayload, sendSpan.Length);
    }

    [Fact]
    public void PinnedBatchBufferArena_OverrunProtection_ThrowsOnExceedingCapacity()
    {
        using var arena = new PinnedBatchBufferArena(maxBatchTokens: 2048, hiddenDim: 5120, vocabSize: 152064);

        // Exceeding byte payload
        Assert.Throws<ArgumentOutOfRangeException>(() => arena.GetBatchSendSpan(arena.MaxBytePayload + 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => arena.GetBatchRecvSpan(arena.MaxBytePayload + 1));

        // Exceeding float count (2049 tokens * 5120)
        Assert.Throws<ArgumentOutOfRangeException>(() => arena.GetFloatInputSpan(arena.MaxTotalFloats + 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => arena.GetFloatOutputSpan(arena.MaxTotalFloats + 1));

        // Exceeding vocab count
        Assert.Throws<ArgumentOutOfRangeException>(() => arena.GetLogitsSpan(arena.LogitsArray.Length + 1));
    }

    [Fact]
    public async Task PinnedBatchBufferArena_SocketTransfer_Full2048Tokens_FP16_And_FP32_Integrity()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = client.ConnectAsync(listener.LocalEndPoint!);
        using var server = await listener.AcceptAsync();
        await connectTask;

        FusedSocketTransport.ConfigureSocket(client, 4 * 1024 * 1024);
        FusedSocketTransport.ConfigureSocket(server, 4 * 1024 * 1024);

        using var arena = new PinnedBatchBufferArena(maxBatchTokens: 2048, hiddenDim: DeepSeekHiddenDim, vocabSize: 152064);

        // 1. Test FP16 2048 tokens (~20.97 MB wire payload)
        int fp16PayloadBytes = 2048 * DeepSeekHiddenDim * sizeof(short);
        int totalFp16Frame = FusedSocketTransport.CombinedHeaderSize + fp16PayloadBytes;

        var actHdr = new FusedActivationHeader(
            sequenceId: 100UL,
            tokenIndex: 0,
            layerIndex: 0,
            hiddenDim: DeepSeekHiddenDim,
            precisionMode: WirePrecisionMode.FP16,
            flags: FusedActivationFlags.IsPrefillChunk,
            batchTokens: 2048);

        // Fill payload with synthetic test pattern in arena wire buffer
        var sendSpan = arena.BatchWireSendArray.AsSpan(FusedSocketTransport.CombinedHeaderSize, fp16PayloadBytes);
        for (int i = 0; i < 100; i++) sendSpan[i] = (byte)(i * 3);
        sendSpan[^1] = 0xFE;

        // Perform socket send from arena send buffer
        FusedSocketTransport.SendFusedActivation(
            client,
            correlationId: 100UL,
            actHeader: actHdr,
            activationPayload: sendSpan,
            stagingBuffer: arena.BatchWireSendArray);

        // Receive into arena receive buffer
        bool ok = FusedSocketTransport.ReceiveFusedActivation(
            server,
            arena.BatchWireRecvArray,
            out var rxClusterHdr,
            out var rxActHdr,
            out var rxPayload);

        Assert.True(ok);
        Assert.Equal(ClusterFrameHeader.MagicConstant, rxClusterHdr.Magic);
        Assert.Equal(FusedActivationHeader.FusedMagic, rxActHdr.Magic);
        Assert.Equal(100UL, rxActHdr.SequenceId);
        Assert.Equal((ushort)2048, rxActHdr.BatchTokens);
        Assert.Equal(WirePrecisionMode.FP16, rxActHdr.TypedPrecision);
        Assert.Equal(fp16PayloadBytes, rxPayload.Length);

        // Verify canary values across 20.97 MB payload
        for (int i = 0; i < 100; i++)
        {
            Assert.Equal((byte)(i * 3), rxPayload[i]);
        }
        Assert.Equal((byte)0xFE, rxPayload[^1]);

        // 2. Test FP32 2048 tokens (~41.94 MB wire payload)
        int fp32PayloadBytes = 2048 * DeepSeekHiddenDim * sizeof(float);
        var actHdrFp32 = new FusedActivationHeader(
            sequenceId: 200UL,
            tokenIndex: 0,
            layerIndex: 0,
            hiddenDim: DeepSeekHiddenDim,
            precisionMode: WirePrecisionMode.FP32,
            flags: FusedActivationFlags.IsPrefillChunk,
            batchTokens: 2048);

        var sendSpanFp32 = arena.BatchWireSendArray.AsSpan(FusedSocketTransport.CombinedHeaderSize, fp32PayloadBytes);
        sendSpanFp32[0] = 0xAA;
        sendSpanFp32[1000] = 0xBB;
        sendSpanFp32[^1] = 0xCC;

        FusedSocketTransport.SendFusedActivation(
            client,
            correlationId: 200UL,
            actHeader: actHdrFp32,
            activationPayload: sendSpanFp32,
            stagingBuffer: arena.BatchWireSendArray);

        bool okFp32 = FusedSocketTransport.ReceiveFusedActivation(
            server,
            arena.BatchWireRecvArray,
            out var rxClusterHdrFp32,
            out var rxActHdrFp32,
            out var rxPayloadFp32);

        Assert.True(okFp32);
        Assert.Equal(fp32PayloadBytes, rxPayloadFp32.Length);
        Assert.Equal((byte)0xAA, rxPayloadFp32[0]);
        Assert.Equal((byte)0xBB, rxPayloadFp32[1000]);
        Assert.Equal((byte)0xCC, rxPayloadFp32[^1]);
    }

    [Fact]
    public void PinnedBatchBufferArena_MultiTurnReset_ZeroAllocations()
    {
        using var arena = new PinnedBatchBufferArena(maxBatchTokens: 512, hiddenDim: DeepSeekHiddenDim, vocabSize: 152064);

        // Warmup and baseline check
        var warmupSpan = arena.GetFloatInputSpan(512 * DeepSeekHiddenDim);
        warmupSpan[0] = 99.0f;
        arena.Reset();
        Assert.Equal(0.0f, arena.FloatInputArray[0]);

        GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);

        int g0 = GC.CollectionCount(0);
        int g1 = GC.CollectionCount(1);
        int g2 = GC.CollectionCount(2);
        long allocBefore = GC.GetAllocatedBytesForCurrentThread();

        for (int turn = 0; turn < 50; turn++)
        {
            var inSpan = arena.GetFloatInputSpan(512 * DeepSeekHiddenDim);
            inSpan[0] = (float)turn;
            inSpan[^1] = (float)(turn * 2);

            var sendSpan = arena.GetBatchSendSpan(64 + 512 * DeepSeekHiddenDim * sizeof(float));
            sendSpan[0] = (byte)turn;

            arena.Reset();
        }

        long allocAfter = GC.GetAllocatedBytesForCurrentThread();

        Assert.Equal(0, GC.CollectionCount(0) - g0);
        Assert.Equal(0, GC.CollectionCount(1) - g1);
        Assert.Equal(0, GC.CollectionCount(2) - g2);
        Assert.Equal(0L, allocAfter - allocBefore);
    }

    #endregion

    #region 4. Wire Corruption & Overrun Resistance

    [Fact]
    public async Task FusedSocketTransport_StagingBufferTooSmall_ThrowsArgumentException()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = client.ConnectAsync(listener.LocalEndPoint!);
        using var server = await listener.AcceptAsync();
        await connectTask;

        var actHdr = new FusedActivationHeader(1, 0, 0, DeepSeekHiddenDim, WirePrecisionMode.FP16);
        byte[] payload = new byte[Fp16BytesPerToken];

        // Staging buffer smaller than CombinedHeaderSize (64B) + payload (10,240B)
        byte[] tooSmallBuffer = new byte[100];

        Assert.Throws<ArgumentException>(() =>
            FusedSocketTransport.SendFusedActivation(client, 1, actHdr, payload, tooSmallBuffer));

        var resHdr = new FusedActivationResultHeader(1, 1, 1.0, 100, 1.0f);
        Assert.Throws<ArgumentException>(() =>
            FusedSocketTransport.SendResult(client, 1, resHdr, payload, tooSmallBuffer));
    }

    [Fact]
    public async Task FusedSocketTransport_Receive_CorruptMagic_ReturnsFalse()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = client.ConnectAsync(listener.LocalEndPoint!);
        using var server = await listener.AcceptAsync();
        await connectTask;

        byte[] corruptFrame = new byte[64];
        // Write corrupt ClusterFrameHeader magic (0xDEADBEEF instead of 0x47434C55)
        BinaryPrimitives.WriteUInt32LittleEndian(corruptFrame.AsSpan(0, 4), 0xDEADBEEFU);
        client.Send(corruptFrame);

        byte[] rxBuffer = new byte[64];
        bool ok = FusedSocketTransport.ReceiveFusedActivation(server, rxBuffer, out var c, out var a, out var p);

        Assert.False(ok);
        Assert.NotEqual(ClusterFrameHeader.MagicConstant, c.Magic);
        Assert.Equal(0xDEADBEEFU, c.Magic);
        Assert.Equal(default, a);
        Assert.True(p.IsEmpty);
    }

    [Fact]
    public async Task FusedSocketTransport_Receive_TruncatedHeader_ReturnsFalse()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = client.ConnectAsync(listener.LocalEndPoint!);
        using var server = await listener.AcceptAsync();
        await connectTask;

        // Send only 32 bytes (ClusterFrameHeader) then close connection
        byte[] truncated = new byte[32];
        var hdr = new ClusterFrameHeader((ClusterMessageType)0x70, 1, 100);
        hdr.WriteTo(truncated);
        client.Send(truncated);
        client.Shutdown(SocketShutdown.Both);
        client.Close();

        byte[] rxBuffer = new byte[64];
        bool ok = FusedSocketTransport.ReceiveFusedActivation(server, rxBuffer, out _, out _, out _);

        Assert.False(ok);
    }

    #endregion
}
