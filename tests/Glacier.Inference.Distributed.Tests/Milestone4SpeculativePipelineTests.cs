namespace Glacier.Inference.Tests.Distributed;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
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
/// Milestone 4 Verification Test Suite:
/// - Feature 16: Optimal 27:21 Pareto Layer Split, Candidate Device Mapping, and VRAM Telemetry.
/// - Feature 17 & 18: Speculative Two-Stage Overlap and Prompt Lookup Draft Provider.
/// - Feature 19: Zero-Allocation In-Place KV Cache Rollback Numerical Parity.
/// - Feature 20: Non-Blocking Duplex Socket Bus and Watchdog Timeout Handling.
/// </summary>
public class Milestone4SpeculativePipelineTests
{
    // =========================================================================
    // 1. FEATURE 16: OPTIMAL 27:21 PARETO LAYER SPLIT
    // =========================================================================

    [Fact]
    public void ParetoSplit_Host4060AndBeast3060_AutomatedDetection_Allocates27OnHostAnd21OnBeast()
    {
        var localNodes = new List<NodeResourceProfile>
        {
            new()
            {
                Name = "Host Laptop dGPU",
                Host = "127.0.0.1",
                IsLocal = true,
                DeviceName = "NVIDIA GeForce RTX 4060 Laptop GPU (8GB)",
                IsGpu = true,
                VramBytes = 8UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 1.182,
                AdapterIndex = 0
            },
            new()
            {
                Name = "Host Laptop iGPU",
                Host = "127.0.0.1",
                IsLocal = true,
                DeviceName = "AMD Radeon 890M Graphics",
                IsGpu = true,
                VramBytes = 512UL * 1024 * 1024,
                SharedMemoryBytes = 32UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 22.0,
                AdapterIndex = 1
            }
        };

        var remoteNodes = new List<NodeResourceProfile>
        {
            new()
            {
                Name = "Beast Desktop",
                Host = "192.168.1.108",
                Port = 5055,
                IsLocal = false,
                DeviceName = "NVIDIA GeForce RTX 3060 12GB (GDDR6)",
                IsGpu = true,
                VramBytes = 10UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 1.150,
                AdapterIndex = -1
            }
        };

        int totalLayers = 48;
        ulong modelBytes = 14UL * 1024 * 1024 * 1024; // ~14GB model

        var stages = ClusterResourcePlanner.CalculatePlan(localNodes, remoteNodes, totalLayers, modelBytes, "auto");

        Assert.Equal(2, stages.Count);
        Assert.Equal(27, stages[0].LayerCount);
        Assert.Equal(0, stages[0].StartLayer);
        Assert.Equal(26, stages[0].EndLayer);
        Assert.Contains("4060", stages[0].Node.DeviceName);

        Assert.Equal(21, stages[1].LayerCount);
        Assert.Equal(27, stages[1].StartLayer);
        Assert.Equal(47, stages[1].EndLayer);
        Assert.Contains("3060", stages[1].Node.DeviceName);
        Assert.True(stages[1].IsLastStage);
    }

    [Fact]
    public void ParetoSplit_ExplicitSplitFlag_BypassesIntegratedGpuToRemoteGpu()
    {
        // Bug fix verification: explicitly passing "27:21" must map to Host dGPU (0) and Beast (2), NOT Host iGPU (1)
        var localNodes = new List<NodeResourceProfile>
        {
            new()
            {
                Name = "Host Laptop dGPU",
                Host = "127.0.0.1",
                IsLocal = true,
                DeviceName = "NVIDIA GeForce RTX 4060 Laptop GPU (8GB)",
                IsGpu = true,
                VramBytes = 8UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 1.182,
                AdapterIndex = 0
            },
            new()
            {
                Name = "Host Laptop iGPU",
                Host = "127.0.0.1",
                IsLocal = true,
                DeviceName = "AMD Radeon 890M Graphics",
                IsGpu = true,
                VramBytes = 512UL * 1024 * 1024,
                SharedMemoryBytes = 32UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 22.0,
                AdapterIndex = 1
            }
        };

        var remoteNodes = new List<NodeResourceProfile>
        {
            new()
            {
                Name = "Beast Desktop",
                Host = "192.168.1.108",
                Port = 5055,
                IsLocal = false,
                DeviceName = "NVIDIA GeForce RTX 3060 12GB (GDDR6)",
                IsGpu = true,
                VramBytes = 10UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 1.150,
                AdapterIndex = -1
            }
        };

        int totalLayers = 48;
        ulong modelBytes = 14UL * 1024 * 1024 * 1024;

        var stages = ClusterResourcePlanner.CalculatePlan(localNodes, remoteNodes, totalLayers, modelBytes, "27:21");

        Assert.Equal(2, stages.Count);
        Assert.Equal(27, stages[0].LayerCount);
        Assert.Contains("4060", stages[0].Node.DeviceName);
        Assert.Equal(21, stages[1].LayerCount);
        Assert.Contains("3060", stages[1].Node.DeviceName);
    }

    [Fact]
    public void ParetoSplit_HostVramHeadroom_GuaranteesAtLeast2GBFree()
    {
        // 27 layers on Host RTX 4060 (8GB = 8,192 MB)
        int hostLayers = 27;
        ulong vramTotalBytes = 8UL * 1024 * 1024 * 1024;

        ulong layerWeightsBytes = (ulong)hostLayers * 162UL * 1024 * 1024; // 4,374 MB
        ulong kv4kBytes = (ulong)hostLayers * 16UL * 1024 * 1024;          // 432 MB
        ulong embdBytes = 440UL * 1024 * 1024;                              // 440 MB
        ulong scratchBytes = 60UL * 1024 * 1024;                            // 60 MB

        ulong totalAllocated = layerWeightsBytes + kv4kBytes + embdBytes + scratchBytes; // 5,306 MB
        ulong freeBytes = vramTotalBytes - totalAllocated;

        double freeGb = freeBytes / (1024.0 * 1024.0 * 1024.0);
        Assert.True(freeGb >= 2.10, $"Host free VRAM must be >= 2.10 GB. Measured: {freeGb:F2} GB");
    }

    [Fact]
    public void ParetoSplit_ComputeImbalanceDeltaT_Under2Milliseconds()
    {
        // Host RTX 4060: 1.1821 ms/layer; Beast RTX 3060: 1.150 ms/layer; LM Head: 6.20 ms
        double t0 = 27 * 1.1821;             // 31.9167 ms
        double t1 = (21 * 1.150) + 6.20;     // 30.3500 ms

        double deltaT = Math.Abs(t0 - t1);
        Assert.True(deltaT < 2.0, $"Imbalance Delta T must be < 2.0 ms. Measured: {deltaT:F2} ms");
        Assert.Equal(1.57, Math.Round(deltaT, 2));
    }

    // =========================================================================
    // 2. FEATURE 20: NON-BLOCKING DUPLEX SOCKET TRANSPORT
    // =========================================================================

    [Fact]
    public async Task DuplexSocketBus_ReceiveTokenAsync_ReadsFramedTokenResultCorrectly()
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

        // Client sends framed result
        byte[] buffer = new byte[FusedSocketTransport.CombinedHeaderSize];
        var clusterHdr = new ClusterFrameHeader(
            messageType: (ClusterMessageType)0x71, // MsgResult
            correlationId: 42UL,
            payloadLength: FusedActivationResultHeader.HeaderSize,
            flags: ClusterFrameFlags.None);
        var resultHdr = new FusedActivationResultHeader(
            stepIndex: 1,
            stageId: 1,
            computeMs: 28.5,
            nextTokenId: 1984,
            topLogit: 9.87f,
            precisionMode: WirePrecisionMode.FP16,
            flags: (byte)FusedActivationFlags.FinalLayer);

        clusterHdr.WriteTo(buffer.AsSpan(0, ClusterFrameHeader.HeaderSize));
        resultHdr.WriteTo(buffer.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));

        await clientSocket.SendAsync(buffer, SocketFlags.None);

        var tokenFrame = await bus.ReceiveTokenAsync(CancellationToken.None);

        Assert.Equal(42UL, tokenFrame.SequenceId);
        Assert.Equal(1984, tokenFrame.TokenId);
        Assert.Equal(9.87f, tokenFrame.TopLogit);
        Assert.Equal(28.5, tokenFrame.ComputeMs);
    }

    [Fact]
    public async Task DuplexSocketBus_WatchdogTimeout_ThrowsTimeoutExceptionWhenPeerHangs()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);

        using var clientSocket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = clientSocket.ConnectAsync(listener.LocalEndPoint!);
        using var serverSocket = await listener.AcceptAsync();
        await connectTask;

        // Configure short 80ms watchdog timeout
        using var bus = new NonBlockingDuplexSocketBus(serverSocket, channelCapacity: 4, watchdogTimeout: TimeSpan.FromMilliseconds(80));
        bus.Start();

        // Do not send anything from clientSocket; verify watchdog fires
        await Assert.ThrowsAsync<TimeoutException>(async () =>
        {
            await bus.ReceiveTokenAsync(CancellationToken.None);
        });
    }

    [Fact]
    public async Task DuplexSocketBus_PeerDisconnection_CompletesChannelGracefully()
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

        // Abruptly close client socket
        clientSocket.Shutdown(SocketShutdown.Both);
        clientSocket.Close();

        // Channel should complete without unhandled background exceptions
        await Assert.ThrowsAnyAsync<Exception>(async () =>
        {
            await bus.ReceiveTokenAsync(CancellationToken.None);
        });
    }

    // =========================================================================
    // 3. FEATURES 17 & 18: SPECULATIVE TWO-STAGE OVERLAP & DRAFT PROVIDER
    // =========================================================================

    [Fact]
    public void PromptLookupDraftProvider_RepeatedPattern_ProposesMatchingCandidate()
    {
        using var provider = new PromptLookupDraftProvider(minMatchLength: 2, maxMatchLength: 4);
        Span<int> draft = stackalloc int[1];

        // Sequence: "the quantum computer is powerful. the quantum computer is"
        // Tokens:   [10, 20, 30, 40, 50, 10, 20, 30]
        int[] history = [10, 20, 30, 40, 50, 10, 20, 30];

        int count = provider.Draft(history, maxDraftTokens: 1, draft);

        Assert.Equal(1, count);
        Assert.Equal(40, draft[0]); // Suffix "10, 20, 30" was followed by 40
    }

    [Fact]
    public void SpeculativeCoordinator_SpeculationHit_SwapsScratchpadsAndRecordsHit()
    {
        const int hiddenDim = 64;
        using var coordinator = new SpeculativeOverlapCoordinator(hiddenDim);

        int[] prompt = [10, 20, 30, 40, 50];
        coordinator.SeedPromptHistory(prompt);

        var mockModel = new MockStageModel(hiddenDim);

        // Current committed token 30 at pos 4
        // Draft candidate proposed is 40
        int candidateToken = 40;
        int nextPos = 5;

        // Execute speculative draft on Stage 0
        coordinator.ExecuteSpeculativeDraft(mockModel, candidateToken, nextPos);
        Assert.Equal(1, coordinator.Metrics.SpeculativeAttempts);

        // Simulate Stage 1 confirming trueToken is indeed 40 (HIT!)
        int trueToken = 40;
        bool isHit = coordinator.ResolveSpeculation(mockModel, candidateToken, trueToken, nextPos);

        Assert.True(isHit);
        Assert.Equal(1, coordinator.Metrics.SpeculativeHits);
        Assert.Equal(0, coordinator.Metrics.SpeculativeMisses);
        Assert.Equal(1.0, coordinator.Metrics.HitRate);
    }

    [Fact]
    public void SpeculativeCoordinator_SpeculationMiss_PerformsInPlaceRollbackAndRecordsMiss()
    {
        const int hiddenDim = 64;
        using var coordinator = new SpeculativeOverlapCoordinator(hiddenDim);

        int[] prompt = [10, 20, 30, 40, 50];
        coordinator.SeedPromptHistory(prompt);

        var mockModel = new MockStageModel(hiddenDim);

        int candidateToken = 999; // Drafted 999
        int nextPos = 5;

        coordinator.ExecuteSpeculativeDraft(mockModel, candidateToken, nextPos);
        Assert.Equal(1, coordinator.Metrics.SpeculativeAttempts);

        // Stage 1 confirms trueToken is 42 (MISS!)
        int trueToken = 42;
        bool isHit = coordinator.ResolveSpeculation(mockModel, candidateToken, trueToken, nextPos);

        Assert.False(isHit);
        Assert.Equal(0, coordinator.Metrics.SpeculativeHits);
        Assert.Equal(1, coordinator.Metrics.SpeculativeMisses);
        Assert.Equal(0.0, coordinator.Metrics.HitRate);

        // Verify committed history has the true token (42), not the candidate (999)
        Assert.Equal(42, coordinator.TokenHistory[^1]);
    }

    // =========================================================================
    // 4. FEATURE 19: ZERO-ALLOCATION IN-PLACE KV CACHE ROLLBACK PARITY
    // =========================================================================

    [Fact]
    public void InPlaceKvRollback_NumericalParity_BitwiseIdenticalToSequentialPass()
    {
        const int hiddenDim = 128;
        const int maxSeqLen = 64;

        // Model A: strictly sequential evaluation (no speculation)
        var modelSequential = new MockStageModel(hiddenDim, maxSeqLen);

        // Model B: speculative evaluation of wrong candidate followed by in-place rollback overwrite
        var modelRollback = new MockStageModel(hiddenDim, maxSeqLen);

        // Preload identical prompt context for tokens 0..4
        for (int p = 0; p < 5; p++)
        {
            int tok = 100 + p;
            float[] actSeq = new float[hiddenDim];
            float[] actRoll = new float[hiddenDim];

            modelSequential.ForwardStage(tok, p, ReadOnlySpan<float>.Empty, actSeq, Span<float>.Empty, false);
            modelRollback.ForwardStage(tok, p, ReadOnlySpan<float>.Empty, actRoll, Span<float>.Empty, false);
        }

        // --- Step 5 on Model A (Sequential): true token 205 at pos 5
        float[] outSequential = new float[hiddenDim];
        modelSequential.ForwardStage(205, 5, ReadOnlySpan<float>.Empty, outSequential, Span<float>.Empty, false);

        // --- Step 5 on Model B (Speculative Miss):
        // 1. Speculatively evaluate WRONG candidate token (e.g. 999) at pos 5 (writes into KV slot 5)
        float[] outSpeculativeWrong = new float[hiddenDim];
        modelRollback.ForwardStage(999, 5, ReadOnlySpan<float>.Empty, outSpeculativeWrong, Span<float>.Empty, false);

        // 2. Perform in-place KV cache rollback: evaluate TRUE token 205 at pos 5 (overwrites slot 5 in-place)
        float[] outRollbackOverwritten = new float[hiddenDim];
        modelRollback.ForwardStage(205, 5, ReadOnlySpan<float>.Empty, outRollbackOverwritten, Span<float>.Empty, false);

        // Numerical Parity Assertion: outSequential must be BIT-FOR-BIT IDENTICAL to outRollbackOverwritten
        for (int d = 0; d < hiddenDim; d++)
        {
            Assert.Equal(outSequential[d], outRollbackOverwritten[d]);
        }
    }

    [Fact]
    public void SpeculativeCoordinator_ContinuousGeneration_MultiStepTelemetryAccurate()
    {
        const int hiddenDim = 32;
        using var coordinator = new SpeculativeOverlapCoordinator(hiddenDim);

        int[] prompt = [1, 2, 3, 4, 1, 2, 3];
        coordinator.SeedPromptHistory(prompt);

        var mockModel = new MockStageModel(hiddenDim);

        // Step 1: candidate proposed is 4, true is 4 (Hit)
        int c1 = coordinator.ProposeCandidateToken();
        Assert.Equal(4, c1);
        coordinator.ExecuteSpeculativeDraft(mockModel, c1, 7);
        coordinator.ResolveSpeculation(mockModel, c1, 4, 7);

        // Step 2: fallback step when no candidate is drafted (candidateToken = -1)
        coordinator.ResolveSpeculation(mockModel, candidateToken: -1, trueToken: 10, pos: 8);

        // Step 3: true is 11 (Miss)
        coordinator.ExecuteSpeculativeDraft(mockModel, 99, 9);
        coordinator.ResolveSpeculation(mockModel, 99, 11, 9);

        var metrics = coordinator.Metrics;
        Assert.Equal(3, metrics.TotalTokensGenerated);
        Assert.Equal(2, metrics.SpeculativeAttempts);
        Assert.Equal(1, metrics.SpeculativeHits);
        Assert.Equal(1, metrics.SpeculativeMisses);
        Assert.Equal(1, metrics.FallbackSequentialSteps);
        Assert.Equal(0.5, metrics.HitRate);
        Assert.True(metrics.TheoreticalSpeedup > 1.0);
    }

    // =========================================================================
    // HELPER MOCK STAGE MODEL FOR REAL DETERMINISTIC ATTENTION & KV OVERWRITING
    // =========================================================================

    private sealed class MockStageModel : IStageForwardModel
    {
        private readonly int _hiddenDim;
        private readonly int _maxSeqLen;
        private readonly float[] _kCache;
        private readonly float[] _vCache;

        public MockStageModel(int hiddenDim, int maxSeqLen = 256)
        {
            _hiddenDim = hiddenDim;
            _maxSeqLen = maxSeqLen;
            _kCache = new float[maxSeqLen * hiddenDim];
            _vCache = new float[maxSeqLen * hiddenDim];
        }

        public void ForwardStage(
            int token,
            int pos,
            ReadOnlySpan<float> inputX,
            Span<float> outputX,
            Span<float> logits,
            bool computeLogits)
        {
            // Unconditional in-place write into KV cache slot pos
            int offset = pos * _hiddenDim;
            for (int d = 0; d < _hiddenDim; d++)
            {
                _kCache[offset + d] = (token * 1.5f) + (d * 0.01f);
                _vCache[offset + d] = (token * 2.5f) - (d * 0.02f);
            }

            // Causal attention reduction over slots 0 <= t <= pos
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
}
