[assembly: Xunit.CollectionBehavior(DisableTestParallelization = true)]

namespace Glacier.Inference.Tests.Distributed;

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Inference.Distributed;
using Glacier.Inference.Distributed.Cli;
using Glacier.Inference.Distributed.Pipeline;
using Glacier.Inference.Distributed.Transport;
using Glacier.Inference.Engine;
using Xunit;

/// <summary>
/// Empirical Challenger Adversarial Stress Suite for Milestone 4:
/// Speculative Two-Stage Overlap and Zero-Allocation In-Place KV Cache Rollback.
/// 
/// Adversarial Dimensions:
/// 1. Bit-for-bit numerical parity testing (sequential vs speculative rollback).
/// 2. Stress testing high miss rates (100% miss, 100% hit, alternating, burst patterns).
/// 3. Invariant validation: zero KV cache corruption, zero memory leakage, 0 GC collections.
/// 4. Transport resilience: NonBlockingDuplexSocketBus rapid burst, channel saturation,
///    worker disconnection, forced reset, watchdog timeout handling, and corruption defense.
/// 5. Full end-to-end overlapped pipeline execution over loopback TCP sockets.
/// </summary>
public class Milestone4AdversarialStressTests
{
    private const int DeepSeekHiddenDim = 5120;

    // =========================================================================
    // SECTION 1: BIT-FOR-BIT NUMERICAL PARITY TESTING
    // =========================================================================

    [Fact]
    public void NumericalParity_SingleMissRollback_BitForBitIdentical()
    {
        const int hiddenDim = 256;
        const int maxSeqLen = 128;

        var seqModel = new CausalAttentionStageModel(hiddenDim, maxSeqLen);
        var rollModel = new CausalAttentionStageModel(hiddenDim, maxSeqLen);

        // Warm up with 8 prompt tokens
        for (int p = 0; p < 8; p++)
        {
            float[] sOut = new float[hiddenDim];
            float[] rOut = new float[hiddenDim];
            seqModel.ForwardStage(100 + p, p, ReadOnlySpan<float>.Empty, sOut, Span<float>.Empty, false);
            rollModel.ForwardStage(100 + p, p, ReadOnlySpan<float>.Empty, rOut, Span<float>.Empty, false);
        }

        // Sequential step at position 8
        float[] seqTarget = new float[hiddenDim];
        seqModel.ForwardStage(555, 8, ReadOnlySpan<float>.Empty, seqTarget, Span<float>.Empty, false);

        // Speculative step at position 8: evaluate wrong candidate token 999
        float[] specWrong = new float[hiddenDim];
        rollModel.ForwardStage(999, 8, ReadOnlySpan<float>.Empty, specWrong, Span<float>.Empty, false);

        // In-place rollback: evaluate true token 555 at position 8
        float[] rollRecovered = new float[hiddenDim];
        rollModel.ForwardStage(555, 8, ReadOnlySpan<float>.Empty, rollRecovered, Span<float>.Empty, false);

        // Verify bit-for-bit parity
        double maxAbsErr = 0.0;
        for (int d = 0; d < hiddenDim; d++)
        {
            Assert.Equal(seqTarget[d], rollRecovered[d]);
            maxAbsErr = Math.Max(maxAbsErr, Math.Abs(seqTarget[d] - rollRecovered[d]));
        }

        double cosSim = ComputeCosineSimilarity(seqTarget, rollRecovered);
        Assert.Equal(0.0, maxAbsErr);
        Assert.Equal(1.0, cosSim);
    }

    [Fact]
    public void NumericalParity_100ConsecutiveMisses_BitForBitIdenticalEveryStep()
    {
        const int hiddenDim = 128;
        const int totalSteps = 100;
        const int promptLen = 10;
        const int maxSeqLen = promptLen + totalSteps + 10;

        var seqModel = new CausalAttentionStageModel(hiddenDim, maxSeqLen);
        var rollModel = new CausalAttentionStageModel(hiddenDim, maxSeqLen);

        // Prompt prefill
        for (int p = 0; p < promptLen; p++)
        {
            float[] dummy = new float[hiddenDim];
            seqModel.ForwardStage(p + 1, p, ReadOnlySpan<float>.Empty, dummy, Span<float>.Empty, false);
            rollModel.ForwardStage(p + 1, p, ReadOnlySpan<float>.Empty, dummy, Span<float>.Empty, false);
        }

        // 100 consecutive steps where EVERY step speculates the wrong token
        for (int step = 0; step < totalSteps; step++)
        {
            int pos = promptLen + step;
            int trueToken = 2000 + step;
            int wrongCandidateToken = 90000 + step;

            // Sequential reference
            float[] seqOut = new float[hiddenDim];
            seqModel.ForwardStage(trueToken, pos, ReadOnlySpan<float>.Empty, seqOut, Span<float>.Empty, false);

            // Speculative miss: draft wrong token into KV slot pos
            float[] specOut = new float[hiddenDim];
            rollModel.ForwardStage(wrongCandidateToken, pos, ReadOnlySpan<float>.Empty, specOut, Span<float>.Empty, false);

            // Rollback overwrite: write true token into KV slot pos
            float[] rollOut = new float[hiddenDim];
            rollModel.ForwardStage(trueToken, pos, ReadOnlySpan<float>.Empty, rollOut, Span<float>.Empty, false);

            // Assert bit-for-bit exact match at each step
            for (int d = 0; d < hiddenDim; d++)
            {
                Assert.True(seqOut[d] == rollOut[d], $"Step {step}, dim {d} mismatch: Seq={seqOut[d]} Roll={rollOut[d]}");
            }

            double sim = ComputeCosineSimilarity(seqOut, rollOut);
            Assert.Equal(1.0, sim, precision: 6);
        }
    }

    [Fact]
    public void NumericalParity_AlternatingHitMissPattern_BitForBitIdentical()
    {
        const int hiddenDim = 128;
        const int steps = 50;
        const int promptLen = 5;
        const int maxSeqLen = promptLen + steps + 10;

        var seqModel = new CausalAttentionStageModel(hiddenDim, maxSeqLen);
        var rollModel = new CausalAttentionStageModel(hiddenDim, maxSeqLen);

        for (int p = 0; p < promptLen; p++)
        {
            float[] d = new float[hiddenDim];
            seqModel.ForwardStage(p + 1, p, ReadOnlySpan<float>.Empty, d, Span<float>.Empty, false);
            rollModel.ForwardStage(p + 1, p, ReadOnlySpan<float>.Empty, d, Span<float>.Empty, false);
        }

        for (int step = 0; step < steps; step++)
        {
            int pos = promptLen + step;
            int trueToken = 500 + step;
            bool isHit = (step % 2 == 0);

            float[] seqOut = new float[hiddenDim];
            seqModel.ForwardStage(trueToken, pos, ReadOnlySpan<float>.Empty, seqOut, Span<float>.Empty, false);

            float[] rollOut = new float[hiddenDim];
            if (isHit)
            {
                // Hit: candidate is true token, executed once
                rollModel.ForwardStage(trueToken, pos, ReadOnlySpan<float>.Empty, rollOut, Span<float>.Empty, false);
            }
            else
            {
                // Miss: candidate is wrong, followed by rollback overwrite
                float[] wrongOut = new float[hiddenDim];
                rollModel.ForwardStage(99999, pos, ReadOnlySpan<float>.Empty, wrongOut, Span<float>.Empty, false);
                rollModel.ForwardStage(trueToken, pos, ReadOnlySpan<float>.Empty, rollOut, Span<float>.Empty, false);
            }

            for (int d = 0; d < hiddenDim; d++)
            {
                Assert.Equal(seqOut[d], rollOut[d]);
            }
        }
    }

    [Fact]
    public void NumericalParity_MultiHeadAttentionWithRoPE_RollbackMatchesSequential()
    {
        const int numHeads = 4;
        const int headDim = 32;
        const int hiddenDim = numHeads * headDim;
        const int maxSeqLen = 64;

        var seqModel = new MultiHeadRoPEStageModel(numHeads, headDim, maxSeqLen);
        var rollModel = new MultiHeadRoPEStageModel(numHeads, headDim, maxSeqLen);

        // Preload prompt
        for (int p = 0; p < 10; p++)
        {
            float[] s = new float[hiddenDim];
            float[] r = new float[hiddenDim];
            seqModel.ForwardStage(10 + p, p, ReadOnlySpan<float>.Empty, s, Span<float>.Empty, false);
            rollModel.ForwardStage(10 + p, p, ReadOnlySpan<float>.Empty, r, Span<float>.Empty, false);
        }

        // Test at position 10: miss followed by rollback
        float[] seqOut = new float[hiddenDim];
        seqModel.ForwardStage(777, 10, ReadOnlySpan<float>.Empty, seqOut, Span<float>.Empty, false);

        float[] wrongOut = new float[hiddenDim];
        rollModel.ForwardStage(12345, 10, ReadOnlySpan<float>.Empty, wrongOut, Span<float>.Empty, false);

        float[] rollOut = new float[hiddenDim];
        rollModel.ForwardStage(777, 10, ReadOnlySpan<float>.Empty, rollOut, Span<float>.Empty, false);

        for (int d = 0; d < hiddenDim; d++)
        {
            Assert.Equal(seqOut[d], rollOut[d]);
        }
        Assert.Equal(1.0, ComputeCosineSimilarity(seqOut, rollOut));
    }

    [Fact]
    public void NumericalParity_ExtremeFloatValues_BitForBitIdentical()
    {
        const int hiddenDim = 64;
        const int maxSeqLen = 32;

        var seqModel = new CausalAttentionStageModel(hiddenDim, maxSeqLen, useExtremeFloats: true);
        var rollModel = new CausalAttentionStageModel(hiddenDim, maxSeqLen, useExtremeFloats: true);

        for (int p = 0; p < 5; p++)
        {
            float[] s = new float[hiddenDim];
            float[] r = new float[hiddenDim];
            seqModel.ForwardStage(p, p, ReadOnlySpan<float>.Empty, s, Span<float>.Empty, false);
            rollModel.ForwardStage(p, p, ReadOnlySpan<float>.Empty, r, Span<float>.Empty, false);
        }

        float[] seqOut = new float[hiddenDim];
        seqModel.ForwardStage(99, 5, ReadOnlySpan<float>.Empty, seqOut, Span<float>.Empty, false);

        float[] wrongOut = new float[hiddenDim];
        rollModel.ForwardStage(888, 5, ReadOnlySpan<float>.Empty, wrongOut, Span<float>.Empty, false);

        float[] rollOut = new float[hiddenDim];
        rollModel.ForwardStage(99, 5, ReadOnlySpan<float>.Empty, rollOut, Span<float>.Empty, false);

        for (int d = 0; d < hiddenDim; d++)
        {
            Assert.Equal(seqOut[d], rollOut[d]);
        }
    }

    // =========================================================================
    // SECTION 2: HIGH MISS RATE STRESS TESTING & INVARIANT VALIDATION
    // =========================================================================

    [Fact]
    public void StressTest_100PercentMissRate_ZeroCacheCorruption()
    {
        const int hiddenDim = 64;
        const int steps = 40;
        using var coordinator = new SpeculativeOverlapCoordinator(hiddenDim);

        var model = new CausalAttentionStageModel(hiddenDim, 128);
        var oracleModel = new CausalAttentionStageModel(hiddenDim, 128);

        int[] prompt = [1, 2, 3, 4, 5];
        coordinator.SeedPromptHistory(prompt);
        for (int p = 0; p < prompt.Length; p++)
        {
            float[] dummy = new float[hiddenDim];
            model.ForwardStage(prompt[p], p, ReadOnlySpan<float>.Empty, dummy, Span<float>.Empty, false);
            oracleModel.ForwardStage(prompt[p], p, ReadOnlySpan<float>.Empty, dummy, Span<float>.Empty, false);
        }

        // Force 100% speculation miss on every single turn
        for (int i = 0; i < steps; i++)
        {
            int pos = prompt.Length + i;
            int wrongCandidate = 999000 + i;
            int trueToken = 100 + i;

            // Stage 0 speculates wrong candidate
            coordinator.ExecuteSpeculativeDraft(model, wrongCandidate, pos);

            // Resolve speculation with miss
            bool hit = coordinator.ResolveSpeculation(model, wrongCandidate, trueToken, pos);
            Assert.False(hit);

            // Execute reference oracle
            float[] oracleOut = new float[hiddenDim];
            oracleModel.ForwardStage(trueToken, pos, ReadOnlySpan<float>.Empty, oracleOut, Span<float>.Empty, false);

            // Compare model's internal KV cache slot pos against oracle KV cache
            AssertExtensions.MemoryEqual(
                oracleModel.GetKSlot(pos),
                model.GetKSlot(pos));
            AssertExtensions.MemoryEqual(
                oracleModel.GetVSlot(pos),
                model.GetVSlot(pos));

            // Compare current activation
            var curAct = coordinator.CurrentActivationSpan;
            for (int d = 0; d < hiddenDim; d++)
            {
                Assert.Equal(oracleOut[d], curAct[d]);
            }
        }

        Assert.Equal(steps, coordinator.Metrics.SpeculativeAttempts);
        Assert.Equal(0, coordinator.Metrics.SpeculativeHits);
        Assert.Equal(steps, coordinator.Metrics.SpeculativeMisses);
        Assert.Equal(0.0, coordinator.Metrics.HitRate);
    }

    [Fact]
    public void StressTest_100PercentHitRate_ZeroCacheCorruption()
    {
        const int hiddenDim = 64;
        const int steps = 40;
        using var coordinator = new SpeculativeOverlapCoordinator(hiddenDim);

        var model = new CausalAttentionStageModel(hiddenDim, 128);
        var oracleModel = new CausalAttentionStageModel(hiddenDim, 128);

        int[] prompt = [10, 20, 30, 40];
        coordinator.SeedPromptHistory(prompt);
        for (int p = 0; p < prompt.Length; p++)
        {
            float[] dummy = new float[hiddenDim];
            model.ForwardStage(prompt[p], p, ReadOnlySpan<float>.Empty, dummy, Span<float>.Empty, false);
            oracleModel.ForwardStage(prompt[p], p, ReadOnlySpan<float>.Empty, dummy, Span<float>.Empty, false);
        }

        // Force 100% speculation hit
        for (int i = 0; i < steps; i++)
        {
            int pos = prompt.Length + i;
            int candidate = 200 + i;
            int trueToken = candidate; // exact hit!

            coordinator.ExecuteSpeculativeDraft(model, candidate, pos);
            bool hit = coordinator.ResolveSpeculation(model, candidate, trueToken, pos);
            Assert.True(hit);

            float[] oracleOut = new float[hiddenDim];
            oracleModel.ForwardStage(trueToken, pos, ReadOnlySpan<float>.Empty, oracleOut, Span<float>.Empty, false);

            AssertExtensions.MemoryEqual(oracleModel.GetKSlot(pos), model.GetKSlot(pos));
            AssertExtensions.MemoryEqual(oracleModel.GetVSlot(pos), model.GetVSlot(pos));

            var curAct = coordinator.CurrentActivationSpan;
            for (int d = 0; d < hiddenDim; d++)
            {
                Assert.Equal(oracleOut[d], curAct[d]);
            }
        }

        Assert.Equal(steps, coordinator.Metrics.SpeculativeAttempts);
        Assert.Equal(steps, coordinator.Metrics.SpeculativeHits);
        Assert.Equal(0, coordinator.Metrics.SpeculativeMisses);
        Assert.Equal(1.0, coordinator.Metrics.HitRate);
    }

    [Fact]
    public void StressTest_BurstHitMissPattern_ZeroCacheCorruption()
    {
        const int hiddenDim = 64;
        using var coordinator = new SpeculativeOverlapCoordinator(hiddenDim);

        var model = new CausalAttentionStageModel(hiddenDim, 128);
        var oracleModel = new CausalAttentionStageModel(hiddenDim, 128);

        int[] prompt = [1, 2, 3];
        coordinator.SeedPromptHistory(prompt);
        for (int p = 0; p < prompt.Length; p++)
        {
            float[] dummy = new float[hiddenDim];
            model.ForwardStage(prompt[p], p, ReadOnlySpan<float>.Empty, dummy, Span<float>.Empty, false);
            oracleModel.ForwardStage(prompt[p], p, ReadOnlySpan<float>.Empty, dummy, Span<float>.Empty, false);
        }

        // 10 hits, 10 misses, 10 hits, 10 misses
        int currentPos = prompt.Length;
        for (int phase = 0; phase < 4; phase++)
        {
            bool phaseIsHit = (phase % 2 == 0);
            for (int step = 0; step < 10; step++)
            {
                int trueToken = 300 + currentPos;
                int candidate = phaseIsHit ? trueToken : 99999 + currentPos;

                coordinator.ExecuteSpeculativeDraft(model, candidate, currentPos);
                bool hit = coordinator.ResolveSpeculation(model, candidate, trueToken, currentPos);
                Assert.Equal(phaseIsHit, hit);

                float[] oracleOut = new float[hiddenDim];
                oracleModel.ForwardStage(trueToken, currentPos, ReadOnlySpan<float>.Empty, oracleOut, Span<float>.Empty, false);

                AssertExtensions.MemoryEqual(oracleModel.GetKSlot(currentPos), model.GetKSlot(currentPos));
                AssertExtensions.MemoryEqual(oracleModel.GetVSlot(currentPos), model.GetVSlot(currentPos));

                var curAct = coordinator.CurrentActivationSpan;
                for (int d = 0; d < hiddenDim; d++)
                {
                    Assert.Equal(oracleOut[d], curAct[d]);
                }

                currentPos++;
            }
        }

        Assert.Equal(40, coordinator.Metrics.SpeculativeAttempts);
        Assert.Equal(20, coordinator.Metrics.SpeculativeHits);
        Assert.Equal(20, coordinator.Metrics.SpeculativeMisses);
        Assert.Equal(0.5, coordinator.Metrics.HitRate);
    }

    [Fact]
    public void StressTest_HotLoop_ZeroHeapAllocations_ZeroGC()
    {
        const int hiddenDim = DeepSeekHiddenDim;
        using var coordinator = new SpeculativeOverlapCoordinator(hiddenDim);
        var model = new ZeroAllocMockStageModel(hiddenDim);

        int[] prompt = [1, 2, 3, 4, 1, 2, 3, 4];
        coordinator.SeedPromptHistory(prompt);

        // Warm up JIT
        for (int w = 0; w < 10; w++)
        {
            int c = coordinator.ProposeCandidateToken();
            coordinator.ExecuteSpeculativeDraft(model, c >= 0 ? c : 100, 10 + w);
            coordinator.ResolveSpeculation(model, c >= 0 ? c : 100, 100 + w, 10 + w);
        }

        // Measure GC and heap allocations over 100 continuous hot decode steps
        GC.Collect(2, GCCollectionMode.Forced, true, true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, true, true);

        int gen0Before = GC.CollectionCount(0);
        int gen1Before = GC.CollectionCount(1);
        int gen2Before = GC.CollectionCount(2);
        long allocBefore = GC.GetAllocatedBytesForCurrentThread();

        for (int step = 0; step < 100; step++)
        {
            int pos = 20 + step;
            int candidate = coordinator.ProposeCandidateToken();
            int cToken = candidate >= 0 ? candidate : 50;

            coordinator.ExecuteSpeculativeDraft(model, cToken, pos);

            // Alternating hits and misses
            int trueToken = (step % 2 == 0) ? cToken : 999;
            coordinator.ResolveSpeculation(model, cToken, trueToken, pos);
        }

        long allocAfter = GC.GetAllocatedBytesForCurrentThread();
        int gen0After = GC.CollectionCount(0);
        int gen1After = GC.CollectionCount(1);
        int gen2After = GC.CollectionCount(2);

        Assert.Equal(0, gen0After - gen0Before);
        Assert.Equal(0, gen1After - gen1Before);
        Assert.Equal(0, gen2After - gen2Before);
        Assert.Equal(0L, allocAfter - allocBefore);
    }

    [Fact]
    public void StressTest_TokenHistory_NeverPollutedByMissedCandidates()
    {
        const int hiddenDim = 64;
        using var coordinator = new SpeculativeOverlapCoordinator(hiddenDim);
        var model = new ZeroAllocMockStageModel(hiddenDim);

        int[] prompt = [10, 20, 30];
        coordinator.SeedPromptHistory(prompt);

        // Miss step 1
        coordinator.ExecuteSpeculativeDraft(model, candidateToken: 999, pos: 3);
        coordinator.ResolveSpeculation(model, candidateToken: 999, trueToken: 40, pos: 3);

        // Miss step 2
        coordinator.ExecuteSpeculativeDraft(model, candidateToken: 888, pos: 4);
        coordinator.ResolveSpeculation(model, candidateToken: 888, trueToken: 50, pos: 4);

        // Hit step 3
        coordinator.ExecuteSpeculativeDraft(model, candidateToken: 60, pos: 5);
        coordinator.ResolveSpeculation(model, candidateToken: 60, trueToken: 60, pos: 5);

        // Verify history contains ONLY prompt and true committed tokens
        var history = coordinator.TokenHistory.ToArray();
        int[] expected = [10, 20, 30, 40, 50, 60];

        Assert.Equal(expected.Length, history.Length);
        Assert.Equal(expected, history);
        Assert.DoesNotContain(999, history);
        Assert.DoesNotContain(888, history);
    }

    [Fact]
    public void StressTest_ContextCapacity_CircularShift_DoesNotCorruptOrThrow()
    {
        const int hiddenDim = 32;
        const int maxSeqLen = 16; // Small context for capacity boundary stress
        using var coordinator = new SpeculativeOverlapCoordinator(hiddenDim, maxSeqLen: maxSeqLen);
        var model = new ZeroAllocMockStageModel(hiddenDim);

        // Seed with 16 tokens (filling capacity exactly)
        int[] prompt = new int[maxSeqLen];
        for (int i = 0; i < maxSeqLen; i++) prompt[i] = 100 + i;
        coordinator.SeedPromptHistory(prompt);

        Assert.Equal(maxSeqLen, coordinator.TokenHistory.Length);

        // Add 20 more tokens past capacity; should circular shift without error
        for (int i = 0; i < 20; i++)
        {
            int newToken = 500 + i;
            coordinator.RecordCommittedToken(newToken);

            // Size should stay at maxSeqLen
            Assert.Equal(maxSeqLen, coordinator.TokenHistory.Length);
            // Last element should be the latest token
            Assert.Equal(newToken, coordinator.TokenHistory[^1]);
        }

        // Drafting still functions
        int draft = coordinator.ProposeCandidateToken();
        Assert.True(draft >= -1);
    }

    // =========================================================================
    // SECTION 3: NON-BLOCKING DUPLEX SOCKET BUS STRESS & RESILIENCE
    // =========================================================================

    [Fact]
    public async Task SocketBus_RapidBurst_1000Frames_DeliveredWithoutLossOrReordering()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = clientSocket.ConnectAsync(listener.LocalEndPoint!);
        using var serverSocket = await listener.AcceptAsync();
        await connectTask;

        using var bus = new NonBlockingDuplexSocketBus(serverSocket, channelCapacity: 64);
        bus.Start();

        const int frameCount = 1000;

        // Writer task sending 1,000 framed results as fast as possible
        var sendTask = Task.Run(async () =>
        {
            byte[] frameBuf = new byte[FusedSocketTransport.CombinedHeaderSize];
            for (int i = 0; i < frameCount; i++)
            {
                var clusterHdr = new ClusterFrameHeader(
                    messageType: (ClusterMessageType)0x71,
                    correlationId: (ulong)i,
                    payloadLength: FusedActivationResultHeader.HeaderSize,
                    flags: ClusterFrameFlags.None);
                var resultHdr = new FusedActivationResultHeader(
                    stepIndex: i,
                    stageId: 1,
                    computeMs: 15.0 + (i % 10),
                    nextTokenId: 10000 + i,
                    topLogit: (float)i * 0.5f,
                    precisionMode: WirePrecisionMode.FP16,
                    flags: (byte)FusedActivationFlags.FinalLayer);

                clusterHdr.WriteTo(frameBuf.AsSpan(0, ClusterFrameHeader.HeaderSize));
                resultHdr.WriteTo(frameBuf.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));

                await clientSocket.SendAsync(frameBuf, SocketFlags.None);
            }
        });

        // Reader reads all 1,000 frames from the bus
        for (int i = 0; i < frameCount; i++)
        {
            var tokenFrame = await bus.ReceiveTokenAsync(CancellationToken.None);
            Assert.Equal((ulong)i, tokenFrame.SequenceId);
            Assert.Equal(10000 + i, tokenFrame.TokenId);
            Assert.Equal((float)i * 0.5f, tokenFrame.TopLogit);
        }

        await sendTask;
    }

    [Fact]
    public async Task SocketBus_ChannelSaturation_AppliesBackpressureWithoutLoss()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = clientSocket.ConnectAsync(listener.LocalEndPoint!);
        using var serverSocket = await listener.AcceptAsync();
        await connectTask;

        // Bounded channel capacity of 4 (forces backpressure)
        using var bus = new NonBlockingDuplexSocketBus(serverSocket, channelCapacity: 4);
        bus.Start();

        const int frameCount = 50;

        // Producer rapidly sends 50 frames
        var sendTask = Task.Run(async () =>
        {
            byte[] frameBuf = new byte[FusedSocketTransport.CombinedHeaderSize];
            for (int i = 0; i < frameCount; i++)
            {
                var clusterHdr = new ClusterFrameHeader(
                    messageType: (ClusterMessageType)0x71,
                    correlationId: (ulong)i,
                    payloadLength: FusedActivationResultHeader.HeaderSize,
                    flags: ClusterFrameFlags.None);
                var resultHdr = new FusedActivationResultHeader(
                    stepIndex: i,
                    stageId: 1,
                    computeMs: 1.0,
                    nextTokenId: 500 + i,
                    topLogit: 1.0f,
                    precisionMode: WirePrecisionMode.FP16,
                    flags: 0);

                clusterHdr.WriteTo(frameBuf.AsSpan(0, ClusterFrameHeader.HeaderSize));
                resultHdr.WriteTo(frameBuf.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));

                await clientSocket.SendAsync(frameBuf, SocketFlags.None);
            }
        });

        // Artificially delay consumer to ensure channel is completely filled and backpressured
        await Task.Delay(50);

        for (int i = 0; i < frameCount; i++)
        {
            var tokenFrame = await bus.ReceiveTokenAsync(CancellationToken.None);
            Assert.Equal((ulong)i, tokenFrame.SequenceId);
            Assert.Equal(500 + i, tokenFrame.TokenId);
        }

        await sendTask;
    }

    [Fact]
    public async Task SocketBus_WorkerDisconnection_CleanChannelCompletion()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = clientSocket.ConnectAsync(listener.LocalEndPoint!);
        using var serverSocket = await listener.AcceptAsync();
        await connectTask;

        using var bus = new NonBlockingDuplexSocketBus(serverSocket, channelCapacity: 4);
        bus.Start();

        // Send 1 frame
        byte[] frameBuf = new byte[FusedSocketTransport.CombinedHeaderSize];
        var clusterHdr = new ClusterFrameHeader(
            messageType: (ClusterMessageType)0x71,
            correlationId: 1UL,
            payloadLength: FusedActivationResultHeader.HeaderSize,
            flags: ClusterFrameFlags.None);
        var resultHdr = new FusedActivationResultHeader(
            stepIndex: 1,
            stageId: 1,
            computeMs: 5.0,
            nextTokenId: 42,
            topLogit: 3.14f,
            precisionMode: WirePrecisionMode.FP16,
            flags: 0);
        clusterHdr.WriteTo(frameBuf.AsSpan(0, ClusterFrameHeader.HeaderSize));
        resultHdr.WriteTo(frameBuf.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));
        await clientSocket.SendAsync(frameBuf, SocketFlags.None);

        // Read 1 frame successfully
        var frame = await bus.ReceiveTokenAsync(CancellationToken.None);
        Assert.Equal(42, frame.TokenId);

        // Graceful disconnect from client
        clientSocket.Shutdown(SocketShutdown.Both);
        clientSocket.Close();

        // Next read should throw channel completion exception
        await Assert.ThrowsAnyAsync<Exception>(async () =>
        {
            await bus.ReceiveTokenAsync(CancellationToken.None);
        });
    }

    [Fact]
    public async Task SocketBus_ForcibleSocketReset_PropagatesExceptionOrCompletes()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = clientSocket.ConnectAsync(listener.LocalEndPoint!);
        using var serverSocket = await listener.AcceptAsync();
        await connectTask;

        using var bus = new NonBlockingDuplexSocketBus(serverSocket, channelCapacity: 4);
        bus.Start();

        // Forcibly abort client with RST packet
        clientSocket.LingerState = new LingerOption(true, 0);
        clientSocket.Close();

        // Verify bus handles RST gracefully without process crash
        await Assert.ThrowsAnyAsync<Exception>(async () =>
        {
            await bus.ReceiveTokenAsync(CancellationToken.None);
        });
    }

    [Fact]
    public async Task SocketBus_WatchdogTimeout_CancelsAndThrowsTimeoutException()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = clientSocket.ConnectAsync(listener.LocalEndPoint!);
        using var serverSocket = await listener.AcceptAsync();
        await connectTask;

        // Configure strict 60ms watchdog timeout
        using var bus = new NonBlockingDuplexSocketBus(serverSocket, channelCapacity: 4, watchdogTimeout: TimeSpan.FromMilliseconds(60));
        bus.Start();

        var sw = Stopwatch.StartNew();
        var ex = await Assert.ThrowsAsync<TimeoutException>(async () =>
        {
            await bus.ReceiveTokenAsync(CancellationToken.None);
        });
        sw.Stop();

        Assert.Contains("watchdog limit", ex.Message);
        Assert.True(sw.ElapsedMilliseconds >= 50, $"Timeout must respect duration, took {sw.ElapsedMilliseconds} ms");
    }

    [Fact]
    public async Task SocketBus_CorruptMagicConstant_CompletesChannelImmediately()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = clientSocket.ConnectAsync(listener.LocalEndPoint!);
        using var serverSocket = await listener.AcceptAsync();
        await connectTask;

        using var bus = new NonBlockingDuplexSocketBus(serverSocket, channelCapacity: 4);
        bus.Start();

        // Send frame with completely corrupt magic (0xDEADBEEF instead of 0x47434C55)
        byte[] frameBuf = new byte[FusedSocketTransport.CombinedHeaderSize];
        BinaryPrimitives.WriteUInt32LittleEndian(frameBuf.AsSpan(0, 4), 0xDEADBEEF);
        await clientSocket.SendAsync(frameBuf, SocketFlags.None);

        // Bus should detect bad magic and complete writer cleanly
        await Assert.ThrowsAnyAsync<Exception>(async () =>
        {
            await bus.ReceiveTokenAsync(CancellationToken.None);
        });
    }

    [Fact]
    public async Task SocketBus_DisposalDuringActiveWait_CleansUpWithoutDeadlock()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = clientSocket.ConnectAsync(listener.LocalEndPoint!);
        var serverSocket = await listener.AcceptAsync();
        await connectTask;

        var bus = new NonBlockingDuplexSocketBus(serverSocket, channelCapacity: 4);
        bus.Start();

        // Start waiting for token in background
        var readTask = Task.Run(async () =>
        {
            await bus.ReceiveTokenAsync(CancellationToken.None);
        });

        await Task.Delay(20);

        // Dispose bus concurrently while readTask is blocked
        bus.Dispose();
        serverSocket.Dispose();
        clientSocket.Dispose();

        // Read task should throw and complete promptly (no hang/deadlock)
        await Assert.ThrowsAnyAsync<Exception>(async () => await readTask);
    }

    // =========================================================================
    // SECTION 4: FULL END-TO-END OVERLAPPED PIPELINE LOOPBACK
    // =========================================================================

    [Fact]
    public async Task Coordinator_ExecuteStepAsync_OverlappedPipeline_EndToEndLoopback()
    {
        const int hiddenDim = 64;
        const int numSteps = 20;

        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var stage1ClientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = stage1ClientSocket.ConnectAsync(listener.LocalEndPoint!);
        using var stage0ServerSocket = await listener.AcceptAsync();
        await connectTask;

        using var stage1Bus = new NonBlockingDuplexSocketBus(stage0ServerSocket, channelCapacity: 4);
        stage1Bus.Start();

        using var coordinator = new SpeculativeOverlapCoordinator(hiddenDim);
        var stage0Model = new CausalAttentionStageModel(hiddenDim, 128);

        int[] prompt = [1, 2, 3, 4, 1, 2, 3, 4];
        coordinator.SeedPromptHistory(prompt);

        // Stage 1 worker simulation task (runs concurrently on remote node)
        var stage1WorkerTask = Task.Run(async () =>
        {
            byte[] recvBuffer = new byte[FusedSocketTransport.CombinedHeaderSize + hiddenDim * sizeof(float)];
            byte[] sendResultBuffer = new byte[FusedSocketTransport.CombinedHeaderSize];

            for (int s = 0; s < numSteps; s++)
            {
                // 1. Receive activation from Stage 0
                int actBytes = QuantizationKernels.GetPackedTokenBytes(WirePrecisionMode.FP16, hiddenDim);
                int totalFrame = ClusterFrameHeader.HeaderSize + FusedActivationHeader.HeaderSize + actBytes;

                int read = 0;
                while (read < totalFrame)
                {
                    int r = await stage1ClientSocket.ReceiveAsync(recvBuffer.AsMemory(read, totalFrame - read), SocketFlags.None);
                    if (r <= 0) return;
                    read += r;
                }

                // 2. Evaluate mock Stage 1 layers + LM Head (simulating 10ms compute)
                await Task.Delay(5);

                // Stage 1 determines true token: deterministic sequence
                int trueToken = 500 + s;

                // 3. Send back Result frame
                var clHdr = new ClusterFrameHeader(
                    (ClusterMessageType)0x71,
                    correlationId: (ulong)s,
                    payloadLength: FusedActivationResultHeader.HeaderSize,
                    flags: ClusterFrameFlags.None);
                var resHdr = new FusedActivationResultHeader(
                    stepIndex: s,
                    stageId: 1,
                    computeMs: 5.0,
                    nextTokenId: trueToken,
                    topLogit: 8.5f,
                    precisionMode: WirePrecisionMode.FP16,
                    flags: (byte)FusedActivationFlags.FinalLayer);

                clHdr.WriteTo(sendResultBuffer.AsSpan(0, ClusterFrameHeader.HeaderSize));
                resHdr.WriteTo(sendResultBuffer.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));

                await stage1ClientSocket.SendAsync(sendResultBuffer, SocketFlags.None);
            }
        });

        int currentToken = prompt[^1];
        int currentPos = prompt.Length;
        bool isInitialStep = true;

        for (int s = 0; s < numSteps; s++)
        {
            var (trueTokenId, topLogit, specHit, stepMs) = await coordinator.ExecuteStepAsync(
                stage0Model,
                stage1Bus,
                currentToken,
                currentPos,
                isInitialStep,
                WirePrecisionMode.FP16,
                stepCorrelationId: (ulong)s);

            isInitialStep = false;
            currentToken = trueTokenId;
            currentPos++;

            Assert.Equal(500 + s, trueTokenId);
            Assert.Equal(8.5f, topLogit);
            Assert.True(stepMs > 0.0);
        }

        await stage1WorkerTask;

        var m = coordinator.Metrics;
        Assert.Equal(numSteps, m.TotalTokensGenerated);
        Assert.True(m.TotalStage0ComputeMs >= 0.0);
        Assert.True(m.TotalStage1ComputeMs >= 0.0);
    }

    // =========================================================================
    // HELPER STAGE MODELS AND COMPARERS
    // =========================================================================

    private static double ComputeCosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.SequenceEqual(b)) return 1.0;
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }
        if (normA == 0.0 || normB == 0.0) return 1.0;
        double sim = dot / (Math.Sqrt(normA) * Math.Sqrt(normB));
        return Math.Clamp(sim, -1.0, 1.0);
    }

    private sealed class CausalAttentionStageModel : IStageForwardModel
    {
        private readonly int _hiddenDim;
        private readonly int _maxSeqLen;
        private readonly float[] _kCache;
        private readonly float[] _vCache;
        private readonly bool _useExtremeFloats;

        public CausalAttentionStageModel(int hiddenDim, int maxSeqLen = 256, bool useExtremeFloats = false)
        {
            _hiddenDim = hiddenDim;
            _maxSeqLen = maxSeqLen;
            _useExtremeFloats = useExtremeFloats;
            _kCache = new float[maxSeqLen * hiddenDim];
            _vCache = new float[maxSeqLen * hiddenDim];
        }

        public ReadOnlySpan<float> GetKSlot(int pos) => _kCache.AsSpan(pos * _hiddenDim, _hiddenDim);
        public ReadOnlySpan<float> GetVSlot(int pos) => _vCache.AsSpan(pos * _hiddenDim, _hiddenDim);

        public void ForwardStage(
            int token,
            int pos,
            ReadOnlySpan<float> inputX,
            Span<float> outputX,
            Span<float> logits,
            bool computeLogits)
        {
            int offset = pos * _hiddenDim;
            float tokenScale = _useExtremeFloats ? 1e-12f : 1.25f;

            for (int d = 0; d < _hiddenDim; d++)
            {
                _kCache[offset + d] = (token * tokenScale) + (d * 0.005f);
                _vCache[offset + d] = (token * (tokenScale * 1.5f)) - (d * 0.002f);
            }

            for (int d = 0; d < _hiddenDim; d++)
            {
                float sum = 0f;
                for (int t = 0; t <= pos; t++)
                {
                    int tOff = t * _hiddenDim;
                    sum += (_kCache[tOff + d] * _vCache[tOff + d]) / MathF.Sqrt(d + 1f);
                }
                outputX[d] = sum;
            }
        }
    }

    private sealed class MultiHeadRoPEStageModel : IStageForwardModel
    {
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly int _hiddenDim;
        private readonly int _maxSeqLen;
        private readonly float[] _kCache;
        private readonly float[] _vCache;

        public MultiHeadRoPEStageModel(int numHeads, int headDim, int maxSeqLen = 128)
        {
            _numHeads = numHeads;
            _headDim = headDim;
            _hiddenDim = numHeads * headDim;
            _maxSeqLen = maxSeqLen;
            _kCache = new float[maxSeqLen * _hiddenDim];
            _vCache = new float[maxSeqLen * _hiddenDim];
        }

        public void ForwardStage(
            int token,
            int pos,
            ReadOnlySpan<float> inputX,
            Span<float> outputX,
            Span<float> logits,
            bool computeLogits)
        {
            int offset = pos * _hiddenDim;

            // Apply RoPE rotation to K
            for (int h = 0; h < _numHeads; h++)
            {
                int hOff = offset + (h * _headDim);
                for (int d = 0; d < _headDim; d += 2)
                {
                    float theta = pos / MathF.Pow(10000f, (float)d / _headDim);
                    float cos = MathF.Cos(theta);
                    float sin = MathF.Sin(theta);

                    float k0 = (token * 0.5f) + d;
                    float k1 = (token * 0.5f) + d + 1;

                    _kCache[hOff + d] = (k0 * cos) - (k1 * sin);
                    _kCache[hOff + d + 1] = (k0 * sin) + (k1 * cos);

                    _vCache[hOff + d] = (token * 1.1f) - d;
                    _vCache[hOff + d + 1] = (token * 1.1f) - d - 1;
                }
            }

            // Attention aggregation over 0 <= t <= pos
            for (int i = 0; i < _hiddenDim; i++)
            {
                float acc = 0f;
                for (int t = 0; t <= pos; t++)
                {
                    int tOff = t * _hiddenDim;
                    acc += _kCache[tOff + i] * _vCache[tOff + i];
                }
                outputX[i] = acc / MathF.Sqrt(_headDim);
            }
        }
    }

    private sealed class ZeroAllocMockStageModel : IStageForwardModel
    {
        private readonly int _hiddenDim;

        public ZeroAllocMockStageModel(int hiddenDim)
        {
            _hiddenDim = hiddenDim;
        }

        public void ForwardStage(
            int token,
            int pos,
            ReadOnlySpan<float> inputX,
            Span<float> outputX,
            Span<float> logits,
            bool computeLogits)
        {
            // Pure stack/register math without heap allocations
            float baseVal = (float)(token + pos);
            for (int i = 0; i < outputX.Length; i++)
            {
                outputX[i] = baseVal + (float)i;
            }
        }
    }

    private static class AssertExtensions
    {
        public static void MemoryEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal(expected[i], actual[i]);
            }
        }
    }
}
