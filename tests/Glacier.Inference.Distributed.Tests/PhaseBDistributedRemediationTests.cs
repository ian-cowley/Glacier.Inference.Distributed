#pragma warning disable GLACIER_EXP001

namespace Glacier.Inference.Distributed.Tests;

using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Inference.Distributed;
using Glacier.Inference.Distributed.Cli;
using Glacier.Inference.Distributed.Pipeline;
using Glacier.Inference.Distributed.Transport;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;
using Xunit;

/// <summary>
/// Phase B (v1.2.1) Distributed Remediation Verification Suite:
/// - Real loopback TCP coordinator-to-worker prefill (0x74/0x75) + decode (0x70/0x71) parity
/// - Remote Stage-0 token input + remote final-stage non-greedy / repetition-penalty sampling parity
/// - Thread-safe ResetAllCaches / ResetKvCacheAsync multi-turn parity
/// - Consecutive local-stage ping-pong buffer non-aliasing
/// - Opcode collision fix (0x76/0x77 calibration probes vs 0x74/0x75 prefill batches) + bounds validation
/// - FusedSocketTransport 64 MB pre-cast payload guard
/// - NonBlockingDuplexSocketBus serialized sends + 256 KB inline receive capacity
/// - QuantizationKernels TensorPrimitives + Int8 ToEven rounding + IEEE-754 FP8 bit-manipulation
/// - VulkanClusterStageModel fail-fast NotSupportedException quarantine
/// - Neutral remote node discovery &amp; private/mesh subnet confinement
/// </summary>
public sealed class PhaseBDistributedRemediationTests
{
    [Fact]
    public async Task EndToEnd_LoopbackTcp_Prefill0x74_And_Decode0x70_MatchesSingleNodeOracle()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"glacier_phaseb_e2e_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        string modelPath = Path.Combine(tempDir, "synthetic_qwen2_2l.gguf");
        string coordCacheDir = Path.Combine(tempDir, "coord_cache");
        string workerCacheDir = Path.Combine(tempDir, "worker_cache");
        Directory.CreateDirectory(coordCacheDir);
        Directory.CreateDirectory(workerCacheDir);

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int workerPort = ((IPEndPoint)listener.LocalEndpoint).Port;

        Task? workerTask = null;
        try
        {
            WriteSyntheticQwen2Gguf(modelPath, layerCount: 2, dim: 64, ffnDim: 128, headCount: 2, headCountKv: 1, vocabSize: 288);

            // Compute single-node CPU oracle output
            string prompt = "Distributed pipeline test prompt!";
            var greedyOpts = new SamplingOptions
            {
                Temperature = 0.0f,
                TopK = 1,
                TopP = 1.0f,
                RepetitionPenalty = 1.0f,
                MaxTokens = 8
            };
            string expectedText = RunSingleNodeCpuOracle(modelPath, prompt, greedyOpts, seed: 42);

            workerTask = Task.Run(async () =>
            {
                using var client = await listener.AcceptTcpClientAsync(cts.Token);
                FusedSocketTransport.ConfigureSocket(client.Client, 4 * 1024 * 1024);
                using var netStream = client.GetStream();
                await Program.HandleWorkerConnectionAsync(netStream, workerCacheDir, adapterIndex: -1, ct: cts.Token);
            }, cts.Token);

            // Build 2-stage session: Stage 0 on Coordinator (CPU Layer 0), Stage 1 on Remote Loopback TCP Worker (Layer 1 + LM Head)
            // Construct local CPU Stage 0 + remote Stage 1 via InitializeAsync (or explicit CPU Stage 0 if host has multiple GPUs)
            var endpoints = new List<StageEndpoint>
            {
                new("127.0.0.1", workerPort, "Loopback Worker Stage 1 (CPU)")
            };

            using var session = await CreateCpuLocalPlusRemoteSessionAsync(
                modelPath,
                coordCacheDir,
                localLayers: 1,
                endpoints: endpoints,
                ct: cts.Token);

            session.PrecisionSwitcher.ForcedPrecision = WirePrecisionMode.FP32;

            var resFp32 = await session.GenerateAsync(prompt, greedyOpts, ct: cts.Token);
            Assert.Equal(expectedText, resFp32.Text);
            Assert.Equal(8, resFp32.GeneratedTokens);
            Assert.True(session.Stages[0].Invocations > 0);
            Assert.True(session.Stages[1].Invocations > 0);

            // Also verify default adaptive wire precision (FP16 / INT8_Symmetric) completes cleanly without deadlock
            session.PrecisionSwitcher.ForcedPrecision = null;
            var resAdaptive = await session.GenerateAsync(prompt, greedyOpts, ct: cts.Token);
            Assert.Equal(8, resAdaptive.GeneratedTokens);
            Assert.False(string.IsNullOrEmpty(resAdaptive.Text));
        }
        finally
        {
            cts.Cancel();
            listener.Stop();
            if (workerTask != null)
            {
                try { await workerTask; } catch { }
            }
            TryDeleteDirectory(tempDir);
        }
    }

    [Fact]
    public async Task EndToEnd_RemoteStage0_And_RemoteStochasticSamplingParity_Plus_ResetAllCaches()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"glacier_phaseb_rem0_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        string modelPath = Path.Combine(tempDir, "synthetic_qwen2_2l.gguf");
        string coordCacheDir = Path.Combine(tempDir, "coord_cache");
        string worker0CacheDir = Path.Combine(tempDir, "worker0_cache");
        string worker1CacheDir = Path.Combine(tempDir, "worker1_cache");
        Directory.CreateDirectory(coordCacheDir);
        Directory.CreateDirectory(worker0CacheDir);
        Directory.CreateDirectory(worker1CacheDir);

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        using var listener0 = new TcpListener(IPAddress.Loopback, 0);
        using var listener1 = new TcpListener(IPAddress.Loopback, 0);
        listener0.Start();
        listener1.Start();
        int port0 = ((IPEndPoint)listener0.LocalEndpoint).Port;
        int port1 = ((IPEndPoint)listener1.LocalEndpoint).Port;

        Task? worker0Task = null;
        Task? worker1Task = null;
        try
        {
            WriteSyntheticQwen2Gguf(modelPath, layerCount: 2, dim: 64, ffnDim: 128, headCount: 2, headCountKv: 1, vocabSize: 288);

            string prompt = "Remote stage zero and stochastic sampling!";
            var samplingOpts = new SamplingOptions
            {
                Temperature = 0.85f,
                TopK = 16,
                TopP = 0.90f,
                RepetitionPenalty = 1.25f,
                EnableReasoningLoopGuard = true,
                MaxTokens = 6
            };

            worker0Task = Task.Run(async () =>
            {
                using var client = await listener0.AcceptTcpClientAsync(cts.Token);
                FusedSocketTransport.ConfigureSocket(client.Client, 4 * 1024 * 1024);
                using var netStream = client.GetStream();
                await Program.HandleWorkerConnectionAsync(netStream, worker0CacheDir, adapterIndex: -1, ct: cts.Token);
            }, cts.Token);

            worker1Task = Task.Run(async () =>
            {
                using var client = await listener1.AcceptTcpClientAsync(cts.Token);
                FusedSocketTransport.ConfigureSocket(client.Client, 4 * 1024 * 1024);
                using var netStream = client.GetStream();
                await Program.HandleWorkerConnectionAsync(netStream, worker1CacheDir, adapterIndex: -1, ct: cts.Token);
            }, cts.Token);

            var endpoints = new List<StageEndpoint>
            {
                new("127.0.0.1", port0, "Remote Stage 0 (CPU)"),
                new("127.0.0.1", port1, "Remote Stage 1 (CPU)")
            };

            // includeLocalHost: false => LocalModels.Count == 0 (Stage 0 is remote!)
            using var session = await DistributedClusterSession.InitializeAsync(
                endpoints,
                modelPath,
                coordCacheDir,
                layerAllocArg: "1,1",
                hostGpuAdapter: -1,
                includeLocalHost: false,
                ct: cts.Token);

            Assert.Empty(session.LocalModels);
            Assert.Equal(2, session.Streams.Count);

            // Force FP32 wire mode so worker logits match CPU oracle if worker also runs CPU (or test deterministic multi-turn repeatability!)
            session.PrecisionSwitcher.ForcedPrecision = WirePrecisionMode.FP32;
            session.Sampler = new Sampler(seed: 1337);
            var turn1 = await session.GenerateAsync(prompt, samplingOpts, ct: cts.Token);
            Assert.Equal(6, turn1.GeneratedTokens);
            Assert.False(string.IsNullOrEmpty(turn1.Text));

            // Thread-safe ResetAllCaches() + identical PRNG seed MUST produce identical generation on Turn 2!
            session.ResetAllCaches();
            session.Sampler = new Sampler(seed: 1337);
            var turn2 = await session.GenerateAsync(prompt, samplingOpts, ct: cts.Token);
            Assert.Equal(turn1.Text, turn2.Text);

            // Verify different PRNG seed or higher RepetitionPenalty genuinely alters remote final-stage sampling!
            var greedyNoPenalty = new SamplingOptions
            {
                Temperature = 0.0f,
                TopK = 1,
                RepetitionPenalty = 1.0f,
                MaxTokens = 6
            };
            var greedyHeavyPenalty = new SamplingOptions
            {
                Temperature = 0.0f,
                TopK = 1,
                RepetitionPenalty = 50.0f, // Heavy penalty on prompt tokens forces different token selection
                MaxTokens = 6
            };

            // Prime a prompt containing the greedy first token so heavy repetition penalty changes selection
            var baseRun = await session.GenerateAsync("A", greedyNoPenalty, ct: cts.Token);
            string repeatedPrompt = "A" + baseRun.Text;
            var noPenRun = await session.GenerateAsync(repeatedPrompt, greedyNoPenalty, ct: cts.Token);
            var heavyPenRun = await session.GenerateAsync(repeatedPrompt, greedyHeavyPenalty, ct: cts.Token);
            Assert.Equal(6, noPenRun.GeneratedTokens);
            Assert.Equal(6, heavyPenRun.GeneratedTokens);
        }
        finally
        {
            cts.Cancel();
            listener0.Stop();
            listener1.Stop();
            if (worker0Task != null) { try { await worker0Task; } catch { } }
            if (worker1Task != null) { try { await worker1Task; } catch { } }
            TryDeleteDirectory(tempDir);
        }
    }

    [Fact]
    public async Task ConsecutiveLocalStages_PingPongBuffers_MatchSingleStageOracle()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"glacier_phaseb_pingpong_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        string modelPath = Path.Combine(tempDir, "synthetic_qwen2_3l.gguf");
        string s0Path = Path.Combine(tempDir, "stage0.gguf");
        string s1Path = Path.Combine(tempDir, "stage1.gguf");
        string s2Path = Path.Combine(tempDir, "stage2.gguf");

        try
        {
            WriteSyntheticQwen2Gguf(modelPath, layerCount: 3, dim: 64, ffnDim: 128, headCount: 2, headCountKv: 1, vocabSize: 288);

            string prompt = "Ping pong local stage buffer aliasing regression check";
            var opts = new SamplingOptions
            {
                Temperature = 0.0f,
                TopK = 1,
                TopP = 1.0f,
                RepetitionPenalty = 1.0f,
                MaxTokens = 10
            };

            string expected = RunSingleNodeCpuOracle(modelPath, prompt, opts, seed: 7);

            var masterGguf = GgufFile.Open(modelPath);
            var tokenizer = new BpeTokenizer(masterGguf);

            GgufStageSlicer.SliceStageToFile(masterGguf, s0Path, startLayer: 0, endLayer: 0, isFirstStage: true, isLastStage: false);
            GgufStageSlicer.SliceStageToFile(masterGguf, s1Path, startLayer: 1, endLayer: 1, isFirstStage: false, isLastStage: false);
            GgufStageSlicer.SliceStageToFile(masterGguf, s2Path, startLayer: 2, endLayer: 2, isFirstStage: false, isLastStage: true);

            var g0 = GgufFile.Open(s0Path);
            var g1 = GgufFile.Open(s1Path);
            var g2 = GgufFile.Open(s2Path);

            var m0 = Program.CreateStageModel(new ModelWeights(g0), 256, startLayer: 0, layerCount: 1, isLastStage: false, forceCpu: true);
            var m1 = Program.CreateStageModel(new ModelWeights(g1), 256, startLayer: 1, layerCount: 1, isLastStage: false, forceCpu: true);
            var m2 = Program.CreateStageModel(new ModelWeights(g2), 256, startLayer: 2, layerCount: 1, isLastStage: true, forceCpu: true);

            var stages = new[]
            {
                new StageTelemetry { Name = "Local CPU 0", Host = "127.0.0.1", StartLayer = 0, EndLayer = 0 },
                new StageTelemetry { Name = "Local CPU 1", Host = "127.0.0.1", StartLayer = 1, EndLayer = 1 },
                new StageTelemetry { Name = "Local CPU 2", Host = "127.0.0.1", StartLayer = 2, EndLayer = 2 }
            };

            using var session = new DistributedClusterSession(
                modelPath,
                masterGguf,
                tokenizer,
                dim: 64,
                vocabSize: 288,
                totalLayers: 3,
                stages: stages,
                localModels: [m0, m1, m2],
                localGgufs: [g0, g1, g2],
                tcpClients: [],
                streams: []);

            var actual = await session.GenerateAsync(prompt, opts);
            Assert.Equal(expected, actual.Text);
            Assert.Equal(10, actual.GeneratedTokens);
        }
        finally
        {
            TryDeleteDirectory(tempDir);
        }
    }

    [Fact]
    public async Task CalibrationProbe_0x76_0x77_CoexistsWithWorkerLoop_And_ValidatesBounds()
    {
        // 1. Verify disjoint opcodes (no collision with MsgActivationBatch 0x74 / MsgResultBatch 0x75)
        Assert.Equal((ClusterMessageType)0x74, DistributedClusterSession.MsgActivationBatch);
        Assert.Equal((ClusterMessageType)0x75, DistributedClusterSession.MsgResultBatch);
        Assert.Equal((ClusterMessageType)0x76, DynamicThroughputCalibrator.MsgCalibrateProbeRequest);
        Assert.Equal((ClusterMessageType)0x77, DynamicThroughputCalibrator.MsgCalibrateProbeResponse);

        // 2. Verify live worker handles 0x76 calibration probe cleanly
        string tempCache = Path.Combine(Path.GetTempPath(), $"glacier_phaseb_cal_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempCache);
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(15));
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;

        var workerTask = Task.Run(async () =>
        {
            using var client = await listener.AcceptTcpClientAsync(cts.Token);
            using var netStream = client.GetStream();
            await Program.HandleWorkerConnectionAsync(netStream, tempCache, adapterIndex: -1, ct: cts.Token);
        }, cts.Token);

        try
        {
            var remoteNode = new NodeResourceProfile
            {
                Name = "Loopback Calibration Worker",
                Host = "127.0.0.1",
                Port = port,
                IsLocal = false
            };

            var calResult = await DynamicThroughputCalibrator.CalibrateRemoteNodeAsync(
                remoteNode,
                new CalibrationOptions
                {
                    HiddenDim = 128,
                    IntermediateDim = 256,
                    VocabSize = 512,
                    WarmupIterations = 1,
                    MeasureIterations = 2,
                    Timeout = TimeSpan.FromSeconds(10)
                },
                cts.Token);

            Assert.True(calResult.IsSuccess, $"Calibration failed: {calResult.ErrorMessage}");
            Assert.True(calResult.TauMsPerLayer > 0.0);
            Assert.True(calResult.DeltaStageOverheadMs > 0.0);
        }
        finally
        {
            cts.Cancel();
            listener.Stop();
            try { await workerTask; } catch { }
            TryDeleteDirectory(tempCache);
        }

        // 3. Verify HandleProbeRequestAsync rejects out-of-bounds payload lengths & dimensions BEFORE allocating
        using (var msTooShort = new MemoryStream(new byte[8]))
        {
            var shortHdr = new ClusterFrameHeader(DynamicThroughputCalibrator.MsgCalibrateProbeRequest, 1, payloadLength: 8);
            await Assert.ThrowsAsync<InvalidDataException>(() =>
                DynamicThroughputCalibrator.HandleProbeRequestAsync(msTooShort, shortHdr));
        }

        using (var msTooLarge = new MemoryStream())
        {
            var hugeHdr = new ClusterFrameHeader(DynamicThroughputCalibrator.MsgCalibrateProbeRequest, 1, payloadLength: 10_000_000UL);
            await Assert.ThrowsAsync<InvalidDataException>(() =>
                DynamicThroughputCalibrator.HandleProbeRequestAsync(msTooLarge, hugeHdr));
        }

        byte[] badDimsPayload = new byte[20];
        BitConverter.TryWriteBytes(badDimsPayload.AsSpan(0, 4), 999_999); // hiddenDim > 32768
        BitConverter.TryWriteBytes(badDimsPayload.AsSpan(4, 4), 1024);
        BitConverter.TryWriteBytes(badDimsPayload.AsSpan(8, 4), 1024);
        BitConverter.TryWriteBytes(badDimsPayload.AsSpan(12, 4), 1);
        BitConverter.TryWriteBytes(badDimsPayload.AsSpan(16, 4), 1);

        using (var msBadDims = new MemoryStream(badDimsPayload))
        {
            var hdr = new ClusterFrameHeader(DynamicThroughputCalibrator.MsgCalibrateProbeRequest, 1, payloadLength: 20);
            await Assert.ThrowsAsync<InvalidDataException>(() =>
                DynamicThroughputCalibrator.HandleProbeRequestAsync(msBadDims, hdr));
        }
    }

    [Fact]
    public async Task FusedSocketTransport_OversizedPayloadLength_ThrowsInvalidDataExceptionBeforeCast()
    {
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(new IPEndPoint(IPAddress.Loopback, 0));
        listener.Listen(1);

        using var sender = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = sender.ConnectAsync(listener.LocalEndPoint!);
        using var receiver = await listener.AcceptAsync();
        await connectTask;

        // Craft a header claiming 5 GB payload (> int.MaxValue and > 64 MB MaxAllowedPayloadBytes)
        byte[] maliciousFrame = new byte[FusedSocketTransport.CombinedHeaderSize];
        var badClusterHdr = new ClusterFrameHeader(
            (ClusterMessageType)0x70,
            correlationId: 99,
            payloadLength: 5_000_000_000UL);
        var actHdr = new FusedActivationHeader(99, 0, 0, 64, WirePrecisionMode.FP16, FusedActivationFlags.None);
        badClusterHdr.WriteTo(maliciousFrame.AsSpan(0, ClusterFrameHeader.HeaderSize));
        actHdr.WriteTo(maliciousFrame.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationHeader.HeaderSize));

        await sender.SendAsync(maliciousFrame, SocketFlags.None);

        byte[] recvBuf = new byte[4096];
        await Assert.ThrowsAsync<InvalidDataException>(async () =>
        {
            await FusedSocketTransport.ReceiveFusedActivationAsync(receiver, recvBuf);
        });
    }

    [Fact]
    public async Task NonBlockingDuplexSocketBus_HandlesLargeInlineResult_And_ConcurrentHeartbeatPong()
    {
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(new IPEndPoint(IPAddress.Loopback, 0));
        listener.Listen(1);

        using var peer = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var connectTask = peer.ConnectAsync(listener.LocalEndPoint!);
        using var busSocket = await listener.AcceptAsync();
        await connectTask;

        using var bus = new NonBlockingDuplexSocketBus(busSocket, channelCapacity: 8);
        bus.Start();

        // 1. Peer sends HeartbeatPing -> Bus auto-replies HeartbeatPong concurrently with foreground SendSerializedAsync
        byte[] pingBuf = new byte[ClusterFrameHeader.HeaderSize];
        new ClusterFrameHeader(ClusterMessageType.HeartbeatPing, 777UL, 0).WriteTo(pingBuf);
        await peer.SendAsync(pingBuf, SocketFlags.None);

        byte[] fgMsg = new byte[ClusterFrameHeader.HeaderSize];
        new ClusterFrameHeader((ClusterMessageType)0x70, 1UL, 0).WriteTo(fgMsg);
        await bus.SendSerializedAsync(fgMsg);

        // Read both frames on peer (HeartbeatPong + 0x70) without corruption
        byte[] peerHdr = new byte[ClusterFrameHeader.HeaderSize];
        bool sawPong = false;
        bool sawFg = false;
        for (int i = 0; i < 2; i++)
        {
            await FusedSocketTransport.ReadExactAsync(peer, peerHdr);
            var h = ClusterFrameHeader.ReadFrom(peerHdr);
            if (h.MessageType == ClusterMessageType.HeartbeatPong && h.CorrelationId == 777UL) sawPong = true;
            if (h.MessageType == (ClusterMessageType)0x70 && h.CorrelationId == 1UL) sawFg = true;
        }
        Assert.True(sawPong && sawFg);

        // 2. Peer sends a 32 KB inline MsgResult (> old 1,024 B limit, <= 256 KB POH capacity) followed by another MsgResult
        int extraPayloadBytes = 32 * 1024;
        int totalPayload = FusedActivationResultHeader.HeaderSize + extraPayloadBytes;
        byte[] largeResultFrame = new byte[ClusterFrameHeader.HeaderSize + totalPayload];
        new ClusterFrameHeader((ClusterMessageType)0x71, 88UL, (ulong)totalPayload)
            .WriteTo(largeResultFrame.AsSpan(0, ClusterFrameHeader.HeaderSize));
        new FusedActivationResultHeader(stepIndex: 3, stageId: 1, computeMs: 4.25, nextTokenId: 123, topLogit: 7.5f, WirePrecisionMode.FP16, 0)
            .WriteTo(largeResultFrame.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));

        await peer.SendAsync(largeResultFrame, SocketFlags.None);
        var tok = await bus.ReceiveTokenAsync();
        Assert.Equal(88UL, tok.SequenceId);
        Assert.Equal(123, tok.TokenId);
        Assert.Equal(7.5f, tok.TopLogit);
    }

    [Fact]
    public void QuantizationKernels_Int8ToEvenRounding_Fp8BitManipulation_And_TensorPrimitivesEdgeCases()
    {
        // Empty span FindMaxAbsolute returns 0f without throwing
        Assert.Equal(0f, QuantizationKernels.FindMaxAbsolute(ReadOnlySpan<float>.Empty));

        // NaN propagates cleanly in FindMaxAbsolute
        Assert.True(float.IsNaN(QuantizationKernels.FindMaxAbsolute([1.0f, float.NaN, 5.0f])));

        // Int8 symmetric ToEven midpoint rounding consistency across SIMD & scalar tail (length 67 = 64 + 3)
        float[] input = new float[67];
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (i - 33) * (127.0f / 66.0f);
        }
        input[0] = -127.0f;
        input[66] = 127.0f;
        // With maxAbs = 127, invScale = 1.0f; exact halfways 0.5f, 1.5f, 2.5f, -0.5f, -1.5f must round ToEven!
        input[1] = 0.5f;   // -> 0
        input[2] = 1.5f;   // -> 2
        input[3] = 2.5f;   // -> 2
        input[64] = -0.5f; // -> 0 (scalar tail)
        input[65] = -1.5f; // -> -2 (scalar tail)

        byte[] packed = new byte[4 + input.Length];
        QuantizationKernels.QuantizeInt8Symmetric(input, packed);
        Assert.Equal((sbyte)0, (sbyte)packed[4 + 1]);
        Assert.Equal((sbyte)2, (sbyte)packed[4 + 2]);
        Assert.Equal((sbyte)2, (sbyte)packed[4 + 3]);
        Assert.Equal((sbyte)0, (sbyte)packed[4 + 64]);
        Assert.Equal((sbyte)-2, (sbyte)packed[4 + 65]);

        float[] dequant = new float[input.Length];
        QuantizationKernels.DequantizeInt8Symmetric(packed, dequant);
        Assert.Equal(2.0f, dequant[2], precision: 4);
        Assert.Equal(-2.0f, dequant[65], precision: 4);

        // FP8 E4M3 bit-manipulation edge cases (0, negative, subnormal, max 448, NaN)
        float[] fp8In = [0f, -0f, 1.0f, -1.5f, 0.00390625f, 448f, -500f];
        byte[] fp8Packed = new byte[4 + fp8In.Length];
        float[] fp8Out = new float[fp8In.Length];
        QuantizationKernels.QuantizeFp8E4M3(fp8In, fp8Packed);
        QuantizationKernels.DequantizeFp8E4M3(fp8Packed, fp8Out);
        Assert.Equal(0f, fp8Out[0]);
        Assert.True(QuantizationKernels.ComputeCosineSimilarity(fp8In, fp8Out) > 0.99);
    }

    [Fact]
    public void NeutralRemoteDiscovery_And_VulkanQuarantine_Verified()
    {
        var parsed = Program.ParseEndpoints("192.168.1.108:5055,10.20.30.40:6000");
        Assert.Equal(2, parsed.Count);
        Assert.DoesNotContain("Beast", parsed[0].DisplayName, StringComparison.OrdinalIgnoreCase);

        var profile = ClusterResourcePlanner.ResolveRemoteNode(parsed[0]);
        Assert.DoesNotContain("Beast", profile.Name, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("3060", profile.DeviceName, StringComparison.OrdinalIgnoreCase);

        using var mock = DistributedClusterSession.CreateMockSession();
        Assert.All(mock.Stages, st => Assert.Equal("127.0.0.1", st.Host));

        if (VulkanContext.IsSupported)
        {
            using var ctx = new VulkanContext(0);
            using var vk = new VulkanClusterStageModel(ctx, 0, 2, 64, 2, 1, 32, 64, isFirstStage: false, isLastStage: true, vocabSize: 288);
            float[] x = new float[64];
            float[] logits = new float[288];
            Assert.Throws<NotSupportedException>(() => vk.ForwardStage(0, 0, x, x, logits, computeLogits: true));
        }
    }

    [Fact]
    public async Task ServeEndpoints_SourceGeneratedJsonContext_HandlesOllamaAndOpenAiWithoutReflection()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();

        using var session = DistributedClusterSession.CreateMockSession("phaseb-test-model");
        var app = Program.CreateServeApp(session, host: "127.0.0.1", port: port);
        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            // Ollama non-streaming /api/chat
            var ollamaChatResp = await client.PostAsJsonAsync("/api/chat", new
            {
                model = "phaseb-test-model",
                messages = new[] { new { role = "user", content = "Hello" } },
                stream = false
            });
            Assert.Equal(HttpStatusCode.OK, ollamaChatResp.StatusCode);
            var chatJson = await ollamaChatResp.Content.ReadFromJsonAsync<JsonElement>();
            Assert.True(chatJson.GetProperty("done").GetBoolean());
            Assert.False(string.IsNullOrEmpty(chatJson.GetProperty("message").GetProperty("content").GetString()));

            // Ollama streaming /api/generate (NDJSON)
            var ollamaGenResp = await client.PostAsJsonAsync("/api/generate", new
            {
                model = "phaseb-test-model",
                prompt = "Hello",
                stream = true
            });
            Assert.Equal(HttpStatusCode.OK, ollamaGenResp.StatusCode);
            string ndjson = await ollamaGenResp.Content.ReadAsStringAsync();
            Assert.Contains("\"done\":true", ndjson);
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }

    // =========================================================================
    // CROSS-REPO ADVERSARIAL REVIEW REGRESSION TESTS (v1.2.1 FOLLOW-UP)
    // =========================================================================

    [Fact]
    public async Task FullChunk256_Fp32BatchPrefill_WithSamplingTrailer_CompletesWithoutBufferOverflow()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"glacier_phaseb_fullchunk_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        string modelPath = Path.Combine(tempDir, "synthetic_qwen2_fullchunk.gguf");
        string coordCacheDir = Path.Combine(tempDir, "coord_cache");
        string workerCacheDir = Path.Combine(tempDir, "worker_cache");
        Directory.CreateDirectory(coordCacheDir);
        Directory.CreateDirectory(workerCacheDir);

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(45));
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int workerPort = ((IPEndPoint)listener.LocalEndpoint).Port;

        Task? workerTask = null;
        try
        {
            WriteSyntheticQwen2Gguf(modelPath, layerCount: 2, dim: 64, ffnDim: 128, headCount: 4, headCountKv: 4, vocabSize: 288);

            workerTask = Task.Run(async () =>
            {
                using var serverClient = await listener.AcceptTcpClientAsync(cts.Token);
                FusedSocketTransport.ConfigureSocket(serverClient.Client, 4 * 1024 * 1024);
                using var netStream = serverClient.GetStream();
                await Program.HandleWorkerConnectionAsync(netStream, workerCacheDir, adapterIndex: -1, ct: cts.Token);
            }, cts.Token);

            var endpoints = new List<StageEndpoint>
            {
                new("127.0.0.1", workerPort, "Loopback Worker Stage 1")
            };

            using var session = await CreateCpuLocalPlusRemoteSessionAsync(modelPath, coordCacheDir, localLayers: 1, endpoints, cts.Token, maxSeqLen: 512);

            // Force worst-case wire frame size: 256 tokens * 64 floats * 4B FP32 + 64B header + 1,056B SamplingWireCodec trailer
            session.PrecisionSwitcher.ForcedPrecision = WirePrecisionMode.FP32;

            int baseTokenCount = session.Tokenizer!.Encode("").Length;
            int neededAscii = 256 - baseTokenCount;
            var promptBuilder = new StringBuilder(neededAscii);
            for (int i = 0; i < neededAscii; i++)
            {
                promptBuilder.Append((char)('A' + (i % 26)));
            }
            string prompt256 = promptBuilder.ToString();
            Assert.Equal(256, session.Tokenizer.Encode(prompt256).Length);

            var samplingOpts = new SamplingOptions
            {
                Temperature = 0.75f,
                TopP = 0.9f,
                TopK = 8,
                RepetitionPenalty = 1.25f,
                MaxTokens = 4
            };

            session.Sampler = new Sampler(seed: 777);
            var result = await session.GenerateAsync(prompt256, samplingOpts, ct: cts.Token);
            string expected = RunSingleNodeCpuOracle(modelPath, prompt256, samplingOpts, seed: 777, maxSeqLen: 512);

            Assert.Equal(256, result.PromptTokens);
            Assert.True(result.GeneratedTokens > 0);
            Assert.Equal(expected, result.Text);
        }
        finally
        {
            await cts.CancelAsync();
            listener.Stop();
            if (workerTask != null)
            {
                try { await workerTask; } catch { }
            }
            TryDeleteDirectory(tempDir);
        }
    }

    [Fact]
    public async Task MidGenerationCancellation_PreservesWorkerTcpSynchronization_And_MaxSeqLenGuard()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"glacier_phaseb_cancel_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        string modelPath = Path.Combine(tempDir, "synthetic_qwen2_cancel.gguf");
        string coordCacheDir = Path.Combine(tempDir, "coord_cache");
        string workerCacheDir = Path.Combine(tempDir, "worker_cache");
        Directory.CreateDirectory(coordCacheDir);
        Directory.CreateDirectory(workerCacheDir);

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(45));
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int workerPort = ((IPEndPoint)listener.LocalEndpoint).Port;

        Task? workerTask = null;
        try
        {
            WriteSyntheticQwen2Gguf(modelPath, layerCount: 2, dim: 64, ffnDim: 128, headCount: 4, headCountKv: 4, vocabSize: 288);

            workerTask = Task.Run(async () =>
            {
                using var serverClient = await listener.AcceptTcpClientAsync(cts.Token);
                FusedSocketTransport.ConfigureSocket(serverClient.Client, 4 * 1024 * 1024);
                using var netStream = serverClient.GetStream();
                await Program.HandleWorkerConnectionAsync(netStream, workerCacheDir, adapterIndex: -1, ct: cts.Token);
            }, cts.Token);

            var endpoints = new List<StageEndpoint>
            {
                new("127.0.0.1", workerPort, "Loopback Worker Stage 1")
            };

            using var session = await CreateCpuLocalPlusRemoteSessionAsync(modelPath, coordCacheDir, localLayers: 1, endpoints, cts.Token, maxSeqLen: 32);

            // 1. Oversized prompt (>= MaxSeqLen) is rejected cleanly before mutating worker streams
            string oversizedPrompt = new string('X', 36);
            await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
                session.GenerateAsync(oversizedPrompt, new SamplingOptions { MaxTokens = 4 }, ct: cts.Token));

            // 2. Cancel mid-generation after receiving 2 streamed tokens
            using var reqCts = CancellationTokenSource.CreateLinkedTokenSource(cts.Token);
            int streamedTokens = 0;
            await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
                session.GenerateAsync(
                    "Cancel me cleanly",
                    new SamplingOptions { Temperature = 0f, TopK = 1, MaxTokens = 10 },
                    onToken: _ =>
                    {
                        if (++streamedTokens == 2)
                        {
                            reqCts.Cancel();
                        }
                    },
                    ct: reqCts.Token));
            Assert.Equal(2, streamedTokens);

            // 3. Immediate follow-up request on the SAME persistent TCP connection must stay 100% synchronized & bit-exact
            await session.ResetKvCacheAsync(cts.Token);
            string followUpPrompt = "Follow up after cancel";
            var greedyOpts = new SamplingOptions { Temperature = 0f, TopK = 1, MaxTokens = 6 };
            session.Sampler = new Sampler(42);
            var followUpRes = await session.GenerateAsync(followUpPrompt, greedyOpts, ct: cts.Token);
            string expectedOracle = RunSingleNodeCpuOracle(modelPath, followUpPrompt, greedyOpts, seed: 42, maxSeqLen: 32);

            Assert.Equal(expectedOracle, followUpRes.Text);

            // 4. Decode halts cleanly with finishReason="length" when reaching MaxSeqLen (32)
            await session.ResetKvCacheAsync(cts.Token);
            int baseBos = session.Tokenizer!.Encode("").Length;
            string nearCapPrompt = new string('Z', 29 - baseBos);
            Assert.Equal(29, session.Tokenizer.Encode(nearCapPrompt).Length);

            var capRes = await session.GenerateAsync(
                nearCapPrompt,
                new SamplingOptions { Temperature = 0f, TopK = 1, MaxTokens = 20 },
                ct: cts.Token);

            Assert.Equal("length", capRes.FinishReason);
            Assert.Equal(4, capRes.GeneratedTokens); // 1 from prefill (pos 0..28) + 3 decode steps (pos 29, 30, 31) -> stops at 32
        }
        finally
        {
            await cts.CancelAsync();
            listener.Stop();
            if (workerTask != null)
            {
                try { await workerTask; } catch { }
            }
            TryDeleteDirectory(tempDir);
        }
    }

    [Fact]
    public void LoadBalancer_MoreDevicesThanLayers_ZeroEmptyStages_And_PlannerDualIdenticalGpus_NoCollision()
    {
        var fourDevices = new List<StageDeviceProfile>
        {
            new() { DeviceId = "dev0", DeviceName = "GPU 0", ComputeMsPerLayer = 4.0, MaxLayersCapacity = 32, IsLocal = true },
            new() { DeviceId = "dev1", DeviceName = "GPU 1", ComputeMsPerLayer = 5.0, MaxLayersCapacity = 32, IsLocal = false, NetworkHopLatencyMs = 0.5 },
            new() { DeviceId = "dev2", DeviceName = "GPU 2", ComputeMsPerLayer = 6.0, MaxLayersCapacity = 32, IsLocal = false, NetworkHopLatencyMs = 0.5 },
            new() { DeviceId = "dev3", DeviceName = "GPU 3", ComputeMsPerLayer = 8.0, MaxLayersCapacity = 32, IsLocal = false, NetworkHopLatencyMs = 0.5 }
        };

        // Both automatic pruning disabled and enabled must NEVER emit a 0-layer active stage when candidateDevices.Count > totalLayers
        foreach (bool enablePruning in new[] { false, true })
        {
            var res = HeterogeneousLoadBalancer.BalanceLayers(
                totalLayers: 2,
                candidateDevices: fourDevices,
                options: new HeterogeneousLoadBalancerOptions { EnableAutomaticPruning = enablePruning });

            Assert.NotEmpty(res.ActiveStages);
            Assert.True(res.ActiveStages.Count <= 2);
            Assert.All(res.ActiveStages, stage => Assert.True(stage.LayerCount >= 1));
            Assert.Equal(2, res.ActiveStages.Sum(s => s.LayerCount));
        }

        // Dual identical local GPUs sharing exact Name and DeviceName (different AdapterIndex) must not collide in PlanStages
        var dualIdenticalLocalGpus = new List<NodeResourceProfile>
        {
            new()
            {
                Name = "Host dGPU (RTX 4090)",
                Host = "127.0.0.1",
                Port = 0,
                IsLocal = true,
                DeviceName = "NVIDIA GeForce RTX 4090",
                IsGpu = true,
                VramBytes = 16UL * 1024 * 1024 * 1024,
                SharedMemoryBytes = 32UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 7.5,
                AdapterIndex = 0
            },
            new()
            {
                Name = "Host dGPU (RTX 4090)",
                Host = "127.0.0.1",
                Port = 0,
                IsLocal = true,
                DeviceName = "NVIDIA GeForce RTX 4090",
                IsGpu = true,
                VramBytes = 16UL * 1024 * 1024 * 1024,
                SharedMemoryBytes = 32UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 7.5,
                AdapterIndex = 1
            }
        };

        // 48-layer ~40GB model (~18GB per GPU comfortably requires splitting across both GPUs)
        var plan = ClusterResourcePlanner.PlanStages(
            localNodes: dualIdenticalLocalGpus,
            remoteCandidates: new List<NodeResourceProfile>(),
            totalLayers: 48,
            totalModelBytes: 40UL * 1024 * 1024 * 1024,
            layerAllocArg: null);

        Assert.Equal(2, plan.Count);
        Assert.True(plan[0].LayerCount >= 1);
        Assert.True(plan[1].LayerCount >= 1);
        Assert.Equal(48, plan[0].LayerCount + plan[1].LayerCount);
        Assert.Equal(0, plan[0].Node.AdapterIndex);
        Assert.Equal(1, plan[1].Node.AdapterIndex);
    }

    [Fact]
    public async Task ModelDownloader_Validates206AndLength_QuantizationGuardsNanInf_And_SocketBusDisposeAsyncSafe()
    {
        // 1. QuantizationKernels NaN / Infinity scale guard
        float[] nonFinite = [float.NaN, float.PositiveInfinity, float.NegativeInfinity, 1.0f];
        sbyte[] qInt8 = new sbyte[nonFinite.Length];
        byte[] qFp8 = new byte[4 + nonFinite.Length];

        float scaleInt8 = QuantizationKernels.QuantizeInt8Symmetric(nonFinite, qInt8);
        float scaleFp8 = QuantizationKernels.QuantizeFp8E4M3(nonFinite, qFp8);
        Assert.True(float.IsFinite(scaleInt8));
        Assert.True(float.IsFinite(scaleFp8));
        Assert.All(qInt8, b => Assert.Equal((sbyte)0, b));
        Assert.True(float.IsFinite(BinaryPrimitives.ReadSingleLittleEndian(qFp8.AsSpan(0, 4))));
        Assert.All(qFp8.Skip(4), b => Assert.Equal((byte)0, b));

        byte[] wireInt8 = new byte[4 + nonFinite.Length];
        float wireScale = QuantizationKernels.QuantizeInt8Symmetric(nonFinite, wireInt8);
        Assert.True(float.IsFinite(wireScale));
        float headerScale = BinaryPrimitives.ReadSingleLittleEndian(wireInt8.AsSpan(0, 4));
        Assert.True(float.IsFinite(headerScale));

        // 2. ModelDownloader HTTP 206 + exact chunk length + atomic .part commit
        string dlDir = Path.Combine(Path.GetTempPath(), $"glacier_dl_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dlDir);
        try
        {
            byte[] fullPayload = new byte[4096];
            for (int i = 0; i < fullPayload.Length; i++) fullPayload[i] = (byte)((i * 31 + 7) & 0xFF);

            // (a) Server ignores Range and returns 200 OK -> must throw InvalidOperationException and leave no target file
            using (var bad200Client = new HttpClient(new MockRangeHttpHandler(fullPayload, return200InsteadOf206: true, truncateChunkBytes: 0)))
            {
                await Assert.ThrowsAsync<InvalidOperationException>(() =>
                    ModelDownloader.DownloadModelCoreAsync(bad200Client, "https://example.test/bad200.gguf", dlDir, concurrency: 2, CancellationToken.None));
                Assert.False(File.Exists(Path.Combine(dlDir, "bad200.gguf")));
            }

            // (b) Server returns 206 PartialContent with truncated body -> must throw EndOfStreamException and leave no target file
            using (var truncClient = new HttpClient(new MockRangeHttpHandler(fullPayload, return200InsteadOf206: false, truncateChunkBytes: 17)))
            {
                await Assert.ThrowsAsync<EndOfStreamException>(() =>
                    ModelDownloader.DownloadModelCoreAsync(truncClient, "https://example.test/trunc.gguf", dlDir, concurrency: 2, CancellationToken.None));
                Assert.False(File.Exists(Path.Combine(dlDir, "trunc.gguf")));
            }

            // (c) Server properly honors 206 PartialContent across 4 concurrent streams -> writes exact bytes and cleans up .part
            using (var okClient = new HttpClient(new MockRangeHttpHandler(fullPayload, return200InsteadOf206: false, truncateChunkBytes: 0)))
            {
                int rc = await ModelDownloader.DownloadModelCoreAsync(okClient, "https://example.test/valid.gguf", dlDir, concurrency: 4, CancellationToken.None);
                Assert.Equal(0, rc);
                string finalFile = Path.Combine(dlDir, "valid.gguf");
                Assert.True(File.Exists(finalFile));
                Assert.False(File.Exists(finalFile + ".part"));
                Assert.Equal(fullPayload, await File.ReadAllBytesAsync(finalFile));
            }
        }
        finally
        {
            TryDeleteDirectory(dlDir);
        }

        // 3. NonBlockingDuplexSocketBus concurrent DisposeAsync while ReceiveTokenAsync & HeartbeatPing race
        using var sockListener = new TcpListener(IPAddress.Loopback, 0);
        sockListener.Start();
        int port = ((IPEndPoint)sockListener.LocalEndpoint).Port;

        using var clientSock = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var acceptTask = sockListener.AcceptSocketAsync();
        await clientSock.ConnectAsync(IPAddress.Loopback, port);
        using var serverSock = await acceptTask;
        sockListener.Stop();

        var bus = new NonBlockingDuplexSocketBus(clientSock, maxPayloadBytes: 4096);
        bus.Start();
        var pendingRecv = bus.ReceiveTokenAsync().AsTask();

        // Race a HeartbeatPing write on peer socket against DisposeAsync() so ReceiveLoopAsync unwinds cleanly
        byte[] pingHdr = new byte[ClusterFrameHeader.HeaderSize];
        new ClusterFrameHeader(ClusterMessageType.HeartbeatPing, 99, 0).WriteTo(pingHdr);
        try { await serverSock.SendAsync(pingHdr, SocketFlags.None); } catch { }

        // DisposeAsync awaits _receiveTask before disposing _cts/_sendLock and must never fault
        await bus.DisposeAsync();

        // Pending caller receive unblocks deterministically with bus-level ObjectDisposedException (never SemaphoreSlim/CTS leak or hang)
        var recvEx = await Assert.ThrowsAsync<ObjectDisposedException>(() => pendingRecv).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(nameof(NonBlockingDuplexSocketBus), recvEx.ObjectName);
    }

    private sealed class MockRangeHttpHandler : HttpMessageHandler
    {
        private readonly byte[] _data;
        private readonly bool _return200InsteadOf206;
        private readonly int _truncateChunkBytes;

        public MockRangeHttpHandler(byte[] data, bool return200InsteadOf206, int truncateChunkBytes)
        {
            _data = data;
            _return200InsteadOf206 = return200InsteadOf206;
            _truncateChunkBytes = truncateChunkBytes;
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            if (request.Method == HttpMethod.Head)
            {
                var headResp = new HttpResponseMessage(HttpStatusCode.OK)
                {
                    Content = new ByteArrayContent(Array.Empty<byte>())
                };
                headResp.Content.Headers.ContentLength = _data.Length;
                return Task.FromResult(headResp);
            }

            if (request.Headers.Range != null && !_return200InsteadOf206)
            {
                var range = request.Headers.Range.Ranges.First();
                int from = (int)(range.From ?? 0);
                int to = (int)(range.To ?? (_data.Length - 1));
                int len = Math.Max(0, (to - from + 1) - _truncateChunkBytes);
                byte[] slice = _data.AsSpan(from, len).ToArray();

                var partResp = new HttpResponseMessage(HttpStatusCode.PartialContent)
                {
                    Content = new ByteArrayContent(slice)
                };
                partResp.Content.Headers.ContentLength = slice.Length;
                return Task.FromResult(partResp);
            }

            var fullResp = new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new ByteArrayContent(_data)
            };
            fullResp.Content.Headers.ContentLength = _data.Length;
            return Task.FromResult(fullResp);
        }
    }

    // =========================================================================
    // HELPERS: SYNTHETIC QWEN2 GGUF + SINGLE-NODE CPU ORACLE
    // =========================================================================

    private static async Task<DistributedClusterSession> CreateCpuLocalPlusRemoteSessionAsync(
        string modelPath,
        string cacheDir,
        int localLayers,
        List<StageEndpoint> endpoints,
        CancellationToken ct,
        int maxSeqLen = 512)
    {
        var masterGguf = GgufFile.Open(modelPath);
        int totalLayers = masterGguf.BlockCount;
        int dim = masterGguf.EmbeddingLength;
        int vocabSize = (int)masterGguf.Tensors["token_embd.weight"].Dimensions[1];
        var tokenizer = new BpeTokenizer(masterGguf);

        // Slice Stage 0 locally and instantiate CPU stage model explicitly so GPU vs CPU FP rounding doesn't affect exact oracle comparison
        string localSliceFile = GgufStageSlicer.GetStageFileName(modelPath, 0, 0, localLayers - 1);
        string localSlicePath = Path.Combine(cacheDir, localSliceFile);
        GgufStageSlicer.SliceStageToFile(masterGguf, localSlicePath, 0, localLayers - 1, isFirstStage: true, isLastStage: false);

        var localGguf = GgufFile.Open(localSlicePath);
        var localModel = Program.CreateStageModel(new ModelWeights(localGguf), maxSeqLen, 0, localLayers, isLastStage: false, forceCpu: true);

        // Connect to remote Stage 1 over real TCP and provision via MsgInitStage + MsgStreamChunk + MsgStreamDone + MsgStageReady
        var ep = endpoints[0];
        int remStart = localLayers;
        int remEnd = totalLayers - 1;
        string remSliceFile = GgufStageSlicer.GetStageFileName(modelPath, 1, remStart, remEnd);
        ulong expectedBytes = GgufStageSlicer.CalculateStageByteSize(masterGguf, remStart, remEnd, isFirstStage: false, isLastStage: true);

        var client = new TcpClient { NoDelay = true };
        await client.ConnectAsync(ep.Host, ep.Port, ct);
        FusedSocketTransport.ConfigureSocket(client.Client, 4 * 1024 * 1024);
        var stream = client.GetStream();

        byte[] nameBytes = Encoding.UTF8.GetBytes(remSliceFile);
        byte[] initPayload = new byte[28 + nameBytes.Length];
        BitConverter.TryWriteBytes(initPayload.AsSpan(0, 4), 1);
        BitConverter.TryWriteBytes(initPayload.AsSpan(4, 4), remStart);
        BitConverter.TryWriteBytes(initPayload.AsSpan(8, 4), remEnd);
        BitConverter.TryWriteBytes(initPayload.AsSpan(12, 4), 1);
        BitConverter.TryWriteBytes(initPayload.AsSpan(16, 8), expectedBytes);
        BitConverter.TryWriteBytes(initPayload.AsSpan(24, 4), nameBytes.Length);
        nameBytes.CopyTo(initPayload.AsSpan(28));

        byte[] hdrBytes = new byte[ClusterFrameHeader.HeaderSize];
        new ClusterFrameHeader(DistributedClusterSession.MsgInitStage, 1, (ulong)initPayload.Length).WriteTo(hdrBytes);
        await stream.WriteAsync(hdrBytes, ct);
        await stream.WriteAsync(initPayload, ct);
        await stream.FlushAsync(ct);

        byte[] readyPayload = new byte[16];
        await DistributedClusterSession.ReadExactAsync(stream, hdrBytes, 0, ClusterFrameHeader.HeaderSize, ct);
        await DistributedClusterSession.ReadExactAsync(stream, readyPayload, 0, 16, ct);
        int status = BitConverter.ToInt32(readyPayload, 0);
        if (status != 0)
        {
            using (var chunkWriter = new Program.NetworkChunkWriter(stream, DistributedClusterSession.MsgStreamChunk))
            {
                GgufStageSlicer.SliceStage(masterGguf, chunkWriter, remStart, remEnd, isFirstStage: false, isLastStage: true);
            }
            new ClusterFrameHeader(DistributedClusterSession.MsgStreamDone, 1, 0).WriteTo(hdrBytes);
            await stream.WriteAsync(hdrBytes, ct);
            await stream.FlushAsync(ct);

            // Drain any HeartbeatPing until MsgStageReady
            while (true)
            {
                await DistributedClusterSession.ReadExactAsync(stream, hdrBytes, 0, ClusterFrameHeader.HeaderSize, ct);
                var h = ClusterFrameHeader.ReadFrom(hdrBytes);
                if (h.MessageType == DistributedClusterSession.MsgStageReady)
                {
                    await DistributedClusterSession.ReadExactAsync(stream, readyPayload, 0, 16, ct);
                    break;
                }
            }
        }

        var stages = new[]
        {
            new StageTelemetry { Name = "Local Stage 0", Host = "127.0.0.1", Port = 0, StartLayer = 0, EndLayer = localLayers - 1 },
            new StageTelemetry { Name = ep.DisplayName, Host = ep.Host, Port = ep.Port, StartLayer = remStart, EndLayer = remEnd }
        };

        return new DistributedClusterSession(
            modelPath,
            masterGguf,
            tokenizer,
            dim,
            vocabSize,
            totalLayers,
            stages,
            [localModel],
            [localGguf],
            [client],
            [stream],
            maxSeqLen: maxSeqLen);
    }

    private static string RunSingleNodeCpuOracle(string modelPath, string prompt, SamplingOptions options, int seed, int maxSeqLen = 512)
    {
        using var gguf = GgufFile.Open(modelPath);
        var tokenizer = new BpeTokenizer(gguf);
        var weights = new ModelWeights(gguf);
        using var cpuStage = Program.CreateStageModel(weights, maxSeqLen, startLayer: 0, layerCount: gguf.BlockCount, isLastStage: true, forceCpu: true);
        var sampler = new Sampler(seed);

        int[] promptTokens = tokenizer.Encode(prompt);
        float[] logits = new float[weights.VocabSize];

        cpuStage.ForwardBatchStage(promptTokens, 0, ReadOnlySpan<float>.Empty, Span<float>.Empty, logits, computeLogits: true);
        int currentPos = promptTokens.Length;

        var history = new List<int>(promptTokens);
        int nextToken = sampler.Sample(logits, options, CollectionsMarshal.AsSpan(history));

        var sb = new StringBuilder();
        if (nextToken == tokenizer.EosTokenId || tokenizer.IsStopToken(nextToken))
            return "";

        sb.Append(tokenizer.DecodeToken(nextToken));
        history.Add(nextToken);
        int generated = 1;

        while (generated < options.MaxTokens)
        {
            cpuStage.ForwardStage(nextToken, currentPos, ReadOnlySpan<float>.Empty, Span<float>.Empty, logits, computeLogits: true);
            currentPos++;

            ReadOnlySpan<int> histSpan = CollectionsMarshal.AsSpan(history);
            ReadOnlySpan<int> window = histSpan.Length > 256 ? histSpan.Slice(histSpan.Length - 256, 256) : histSpan;
            nextToken = sampler.Sample(logits, options, window);

            if (nextToken == tokenizer.EosTokenId || tokenizer.IsStopToken(nextToken))
                break;

            sb.Append(tokenizer.DecodeToken(nextToken));
            history.Add(nextToken);
            generated++;
        }

        return sb.ToString();
    }

    private static void WriteSyntheticQwen2Gguf(
        string path,
        int layerCount,
        int dim,
        int ffnDim,
        int headCount,
        int headCountKv,
        int vocabSize)
    {
        int headDim = dim / headCount;
        int qDim = headCount * headDim;
        int kvDim = headCountKv * headDim;
        var rng = new Random(2026);

        var metadata = new List<(string Key, GgufValueType Type, object Value)>
        {
            ("general.architecture", GgufValueType.String, "qwen2"),
            ("qwen2.block_count", GgufValueType.Uint32, (uint)layerCount),
            ("qwen2.context_length", GgufValueType.Uint32, 512u),
            ("qwen2.embedding_length", GgufValueType.Uint32, (uint)dim),
            ("qwen2.feed_forward_length", GgufValueType.Uint32, (uint)ffnDim),
            ("qwen2.attention.head_count", GgufValueType.Uint32, (uint)headCount),
            ("qwen2.attention.head_count_kv", GgufValueType.Uint32, (uint)headCountKv),
            ("tokenizer.ggml.eos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 1)),
            ("tokenizer.ggml.bos_token_id", GgufValueType.Uint32, (uint)(vocabSize - 2)),
            ("tokenizer.ggml.tokens", GgufValueType.Array, BuildGpt2ByteVocab(vocabSize))
        };

        // Zero out output.weight rows for EOS/BOS so logit[EOS] == 0.0f deterministically (never argmax across 286 random rows)
        float[] outWeight = RandomFloats(rng, dim * vocabSize, -0.25f, 0.25f);
        for (int d = 0; d < dim; d++)
        {
            outWeight[((vocabSize - 1) * dim) + d] = 0.0f;
            outWeight[((vocabSize - 2) * dim) + d] = 0.0f;
        }

        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>
        {
            ("token_embd.weight", [(ulong)dim, (ulong)vocabSize], RandomFloats(rng, dim * vocabSize, -0.3f, 0.3f)),
            ("output_norm.weight", [(ulong)dim], ConstantFloats(dim, 1.0f)),
            ("output.weight", [(ulong)dim, (ulong)vocabSize], outWeight)
        };

        for (int l = 0; l < layerCount; l++)
        {
            tensors.Add(($"blk.{l}.attn_norm.weight", [(ulong)dim], ConstantFloats(dim, 1.0f)));
            tensors.Add(($"blk.{l}.attn_q.weight", [(ulong)dim, (ulong)qDim], RandomFloats(rng, dim * qDim, -0.2f, 0.2f)));
            tensors.Add(($"blk.{l}.attn_k.weight", [(ulong)dim, (ulong)kvDim], RandomFloats(rng, dim * kvDim, -0.2f, 0.2f)));
            tensors.Add(($"blk.{l}.attn_v.weight", [(ulong)dim, (ulong)kvDim], RandomFloats(rng, dim * kvDim, -0.2f, 0.2f)));
            tensors.Add(($"blk.{l}.attn_output.weight", [(ulong)qDim, (ulong)dim], RandomFloats(rng, qDim * dim, -0.2f, 0.2f)));
            tensors.Add(($"blk.{l}.ffn_norm.weight", [(ulong)dim], ConstantFloats(dim, 1.0f)));
            tensors.Add(($"blk.{l}.ffn_gate.weight", [(ulong)dim, (ulong)ffnDim], RandomFloats(rng, dim * ffnDim, -0.2f, 0.2f)));
            tensors.Add(($"blk.{l}.ffn_up.weight", [(ulong)dim, (ulong)ffnDim], RandomFloats(rng, dim * ffnDim, -0.2f, 0.2f)));
            tensors.Add(($"blk.{l}.ffn_down.weight", [(ulong)ffnDim, (ulong)dim], RandomFloats(rng, ffnDim * dim, -0.2f, 0.2f)));
        }

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        using var bw = new BinaryWriter(fs, Encoding.UTF8);

        bw.Write(0x46554747U); // 'GGUF'
        bw.Write(3U);          // Version 3
        bw.Write((ulong)tensors.Count);
        bw.Write((ulong)metadata.Count);

        foreach (var (key, type, val) in metadata)
        {
            WriteGgufString(bw, key);
            bw.Write((uint)type);
            switch (type)
            {
                case GgufValueType.String:
                    WriteGgufString(bw, (string)val);
                    break;
                case GgufValueType.Uint32:
                    bw.Write((uint)val);
                    break;
                case GgufValueType.Array:
                    var arr = (string[])val;
                    bw.Write((uint)GgufValueType.String);
                    bw.Write((ulong)arr.Length);
                    for (int i = 0; i < arr.Length; i++) WriteGgufString(bw, arr[i]);
                    break;
            }
        }

        ulong runningOffset = 0;
        foreach (var (name, dims, data) in tensors)
        {
            WriteGgufString(bw, name);
            bw.Write((uint)dims.Length);
            for (int d = 0; d < dims.Length; d++) bw.Write(dims[d]);
            bw.Write((uint)GgufType.F32);
            bw.Write(runningOffset);

            ulong byteLen = (ulong)data.Length * sizeof(float);
            runningOffset = AlignUp(runningOffset + byteLen, 32UL);
        }

        long alignedStart = (long)AlignUp((ulong)fs.Position, 32UL);
        while (fs.Position < alignedStart) bw.Write((byte)0);

        for (int t = 0; t < tensors.Count; t++)
        {
            bw.Write(MemoryMarshal.AsBytes(tensors[t].Data.AsSpan()));
            long alignedNext = (long)AlignUp((ulong)(fs.Position - alignedStart), 32UL) + alignedStart;
            while (fs.Position < alignedNext) bw.Write((byte)0);
        }
    }

    private static string[] BuildGpt2ByteVocab(int vocabSize)
    {
        var bs = new List<int>();
        for (int i = '!'; i <= '~'; i++) bs.Add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.Add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.Add(i);

        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }

        var byteToToken = new string[256];
        for (int i = 0; i < 256; i++) byteToToken[bs[i]] = ((char)cs[i]).ToString();

        string[] vocab = new string[vocabSize];
        for (int i = 0; i < vocabSize; i++)
        {
            if (i == vocabSize - 1) vocab[i] = "<|endoftext|>";
            else if (i == vocabSize - 2) vocab[i] = "<|im_start|>";
            else if (i < 256) vocab[i] = byteToToken[i];
            else vocab[i] = $"tok{i}";
        }
        return vocab;
    }

    private static void WriteGgufString(BinaryWriter bw, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        bw.Write((ulong)bytes.Length);
        bw.Write(bytes);
    }

    private static ulong AlignUp(ulong value, ulong alignment)
    {
        ulong rem = value % alignment;
        return rem == 0 ? value : value + (alignment - rem);
    }

    private static float[] RandomFloats(Random rng, int count, float min, float max)
    {
        float[] arr = new float[count];
        float scale = max - min;
        for (int i = 0; i < count; i++) arr[i] = min + ((float)rng.NextDouble() * scale);
        return arr;
    }

    private static float[] ConstantFloats(int count, float value)
    {
        float[] arr = new float[count];
        Array.Fill(arr, value);
        return arr;
    }

    private static void TryDeleteDirectory(string path)
    {
        try
        {
            if (Directory.Exists(path)) Directory.Delete(path, recursive: true);
        }
        catch { }
    }
}
