namespace Glacier.Inference.Distributed.Cli;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;

public sealed class ClusterGenerationResult
{
    public string Text { get; set; } = "";
    public int PromptTokens { get; set; }
    public int GeneratedTokens { get; set; }
    public TimeSpan PromptDuration { get; set; }
    public TimeSpan GenerationDuration { get; set; }
    public TimeSpan TotalDuration { get; set; }
    public string FinishReason { get; set; } = "stop";
}

/// <summary>
/// Manages the distributed inference pipeline coordinator state, stage connections,
/// tokenization, forward pass pipeline scheduling, and KV cache lifecycle.
/// </summary>
public sealed class DistributedClusterSession : IDisposable
{
    public const ClusterMessageType MsgInitStage   = (ClusterMessageType)0x60;
    public const ClusterMessageType MsgStageReady  = (ClusterMessageType)0x61;
    public const ClusterMessageType MsgStreamChunk = (ClusterMessageType)0x62;
    public const ClusterMessageType MsgStreamDone  = (ClusterMessageType)0x63;
    public const ClusterMessageType MsgActivation  = (ClusterMessageType)0x70;
    public const ClusterMessageType MsgResult      = (ClusterMessageType)0x71;
    public const ClusterMessageType MsgResetKv     = (ClusterMessageType)0x72;
    public const ClusterMessageType MsgShutdown    = (ClusterMessageType)0x73;

    public string ModelPath { get; init; }
    public string ModelName { get; init; }
    public string Architecture { get; init; } = "qwen2";
    public GgufFile? MasterGguf { get; }
    public BpeTokenizer? Tokenizer { get; }
    public int Dim { get; init; }
    public int VocabSize { get; init; }
    public int TotalLayers { get; init; }
    public StageTelemetry[] Stages { get; init; }
    public List<IClusterStageModel> LocalModels { get; init; }
    public List<GgufFile> LocalGgufs { get; init; }
    public List<TcpClient> TcpClients { get; init; }
    public List<NetworkStream> Streams { get; init; }
    public SemaphoreSlim Lock { get; } = new(1, 1);
    public Sampler Sampler { get; } = new();

    public DistributedClusterSession(
        string modelPath,
        GgufFile? masterGguf,
        BpeTokenizer? tokenizer,
        int dim,
        int vocabSize,
        int totalLayers,
        StageTelemetry[] stages,
        List<IClusterStageModel> localModels,
        List<GgufFile> localGgufs,
        List<TcpClient> tcpClients,
        List<NetworkStream> streams)
    {
        ModelPath = modelPath;
        ModelName = Path.GetFileNameWithoutExtension(modelPath);
        MasterGguf = masterGguf;
        Tokenizer = tokenizer;
        Dim = dim;
        VocabSize = vocabSize;
        TotalLayers = totalLayers;
        Stages = stages;
        LocalModels = localModels;
        LocalGgufs = localGgufs;
        TcpClients = tcpClients;
        Streams = streams;
        if (masterGguf != null)
        {
            Architecture = masterGguf.Architecture;
        }
    }

    public static DistributedClusterSession CreateMockSession(
        string modelName = "qwen2-57b-a14b-instruct",
        int totalLayers = 28,
        int dim = 3584,
        int vocabSize = 151936)
    {
        var stages = new[]
        {
            new StageTelemetry { Name = "Host (RTX 4060 8GB)", Host = "127.0.0.1", Port = 5055, StartLayer = 0, EndLayer = 13 },
            new StageTelemetry { Name = "Remote Node 1 (RTX 3060 12GB)", Host = "192.168.1.108", Port = 5055, StartLayer = 14, EndLayer = 27 }
        };
        return new DistributedClusterSession(
            modelPath: $"./models/{modelName}.gguf",
            masterGguf: null,
            tokenizer: null,
            dim: dim,
            vocabSize: vocabSize,
            totalLayers: totalLayers,
            stages: stages,
            localModels: new List<IClusterStageModel>(),
            localGgufs: new List<GgufFile>(),
            tcpClients: new List<TcpClient>(),
            streams: new List<NetworkStream>());
    }

    public static async Task<DistributedClusterSession> InitializeAsync(
        List<StageEndpoint> remoteEndpoints,
        string modelPath,
        string cacheDir,
        string? layerAllocArg = null,
        int hostGpuAdapter = -1)
    {
        Console.WriteLine($"[Coordinator] Inspecting Master GGUF Model: {Path.GetFileName(modelPath)}");
        var masterGguf = GgufFile.Open(modelPath);

        int totalLayers = masterGguf.BlockCount;
        int dim = masterGguf.EmbeddingLength;
        int vocabSize = masterGguf.Tensors.TryGetValue("token_embd.weight", out var ti) && ti != null
            ? (int)ti.Dimensions[1]
            : (int)masterGguf.GetMetadataUInt32("tokenizer.ggml.tokens.length", 151936);

        Console.WriteLine($"[Coordinator] Model Architecture: {masterGguf.Architecture} | Total Layers: {totalLayers} | Dim: {dim} | Vocab: {vocabSize}");

        Console.WriteLine("[Coordinator] Loading BPE Tokenizer from GGUF metadata...");
        var tokenizer = new BpeTokenizer(masterGguf);
        Console.WriteLine($"[Coordinator] Tokenizer ready! VocabSize={tokenizer.VocabSize}, EOS={tokenizer.EosTokenId}");

        // Solve optimal resource allocation using ClusterResourcePlanner
        ulong totalModelBytes = (ulong)new FileInfo(modelPath).Length;
        var localProfiles = ClusterResourcePlanner.DetectLocalNodes(hostGpuAdapter);
        var remoteProfiles = remoteEndpoints.Select(ClusterResourcePlanner.ResolveRemoteNode).ToList();
        var activeAllocations = ClusterResourcePlanner.PlanStages(localProfiles, remoteProfiles, totalLayers, totalModelBytes, layerAllocArg);

        int numStages = activeAllocations.Count;
        var stages = new StageTelemetry[numStages];
        for (int i = 0; i < numStages; i++)
        {
            var alloc = activeAllocations[i];
            stages[i] = new StageTelemetry
            {
                Name = alloc.Node.Name,
                Host = alloc.Node.Host,
                Port = alloc.Node.Port,
                StartLayer = alloc.StartLayer,
                EndLayer = alloc.EndLayer
            };
        }

        // 1. Prepare Local Host Stages
        var localModels = new List<IClusterStageModel>();
        var localGgufs = new List<GgufFile>();

        for (int s = 0; s < numStages; s++)
        {
            var alloc = activeAllocations[s];
            if (!alloc.Node.IsLocal) break;

            var stInfo = stages[s];
            bool isFirst = (s == 0);
            bool isLast = alloc.IsLastStage;

            string sliceFileName = GgufStageSlicer.GetStageFileName(modelPath, s, stInfo.StartLayer, stInfo.EndLayer);
            string localPath = Path.Combine(cacheDir, sliceFileName);
            ulong expectedBytes = GgufStageSlicer.CalculateStageByteSize(masterGguf, stInfo.StartLayer, stInfo.EndLayer, isFirstStage: isFirst, isLastStage: isLast);

            if (!File.Exists(localPath) || (ulong)new FileInfo(localPath).Length != expectedBytes)
            {
                Console.WriteLine($"[Stage {s}] Slicing {stInfo.Name} ({stInfo.LayerCount}L{(isFirst ? " + Embedding" : "")}) locally to {sliceFileName}...");
                GgufStageSlicer.SliceStageToFile(masterGguf, localPath, stInfo.StartLayer, stInfo.EndLayer, isFirstStage: isFirst, isLastStage: isLast);
                Console.WriteLine($"[Stage {s}] Slice complete ({new FileInfo(localPath).Length:N0} bytes).");
            }
            else
            {
                Console.WriteLine($"[Stage {s}] Using cached local stage file: {sliceFileName}");
            }

            var gguf = GgufFile.Open(localPath);
            localGgufs.Add(gguf);
            var weights = new ModelWeights(gguf);
            var model = Program.CreateStageModel(weights, 4096, stInfo.StartLayer, stInfo.LayerCount, isLastStage: isLast, adapterIndex: alloc.Node.AdapterIndex);
            localModels.Add(model);
            Console.WriteLine($"[Stage {s}] {stInfo.Name} initialized successfully! Backend: {model.BackendName}\n");
        }

        // 2. Connect to Remote Workers
        var tcpClients = new List<TcpClient>();
        var streams = new List<NetworkStream>();

        for (int s = localModels.Count; s < numStages; s++)
        {
            var alloc = activeAllocations[s];
            var stInfo = stages[s];
            bool isLast = alloc.IsLastStage;

            string sliceFileName = GgufStageSlicer.GetStageFileName(modelPath, s, stInfo.StartLayer, stInfo.EndLayer);
            ulong expectedBytes = GgufStageSlicer.CalculateStageByteSize(masterGguf, stInfo.StartLayer, stInfo.EndLayer, isFirstStage: false, isLastStage: isLast);

            Console.Write($"[Coordinator] Connecting to Stage {s} [{stInfo.Name}] ({alloc.Node.Host}:{alloc.Node.Port})... ");
            var client = new TcpClient();
            client.NoDelay = true;
            client.ReceiveBufferSize = 4 * 1024 * 1024;
            client.SendBufferSize = 4 * 1024 * 1024;
            await client.ConnectAsync(alloc.Node.Host, alloc.Node.Port);
            Console.WriteLine("Connected!");

            var stream = client.GetStream();
            tcpClients.Add(client);
            streams.Add(stream);

            // Send MsgInitStage
            byte[] nameBytes = Encoding.UTF8.GetBytes(sliceFileName);
            int initLen = 28 + nameBytes.Length;
            byte[] initPayload = new byte[initLen];
            BitConverter.TryWriteBytes(initPayload.AsSpan(0, 4), s);
            BitConverter.TryWriteBytes(initPayload.AsSpan(4, 4), stInfo.StartLayer);
            BitConverter.TryWriteBytes(initPayload.AsSpan(8, 4), stInfo.EndLayer);
            BitConverter.TryWriteBytes(initPayload.AsSpan(12, 4), isLast ? 1 : 0);
            BitConverter.TryWriteBytes(initPayload.AsSpan(16, 8), expectedBytes);
            BitConverter.TryWriteBytes(initPayload.AsSpan(24, 4), nameBytes.Length);
            nameBytes.CopyTo(initPayload.AsSpan(28));

            var initHdr = new ClusterFrameHeader(MsgInitStage, (ulong)s, (ulong)initLen);
            byte[] initHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
            MemoryMarshal.Write(initHdrBytes, in initHdr);
            await stream.WriteAsync(initHdrBytes, 0, ClusterFrameHeader.HeaderSize);
            await stream.WriteAsync(initPayload, 0, initLen);
            await stream.FlushAsync();

            // Read response
            byte[] respHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
            await ReadExactAsync(stream, respHdrBytes, 0, ClusterFrameHeader.HeaderSize);
            byte[] respPayload = new byte[16];
            await ReadExactAsync(stream, respPayload, 0, 16);

            int status = BitConverter.ToInt32(respPayload, 0);
            if (status == 0)
            {
                Console.WriteLine($"[Stage {s}] Remote node has valid cached slice: {sliceFileName} (Ready in RAM/VRAM).");
            }
            else
            {
                Console.WriteLine($"[Stage {s}] Remote node cache miss. Slicing and streaming {sliceFileName} ({expectedBytes / (1024 * 1024):N1} MB) over gigabit LAN...");
                var swStream = Stopwatch.StartNew();

                using (var chunkWriter = new Program.NetworkChunkWriter(stream, MsgStreamChunk))
                {
                    GgufStageSlicer.SliceStage(masterGguf, chunkWriter, stInfo.StartLayer, stInfo.EndLayer, isFirstStage: false, isLastStage: isLast);
                }

                var doneHdr = new ClusterFrameHeader(MsgStreamDone, (ulong)s, 0);
                byte[] doneHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
                MemoryMarshal.Write(doneHdrBytes, in doneHdr);
                await stream.WriteAsync(doneHdrBytes, 0, ClusterFrameHeader.HeaderSize);
                await stream.FlushAsync();

                await ReadExactAsync(stream, respHdrBytes, 0, ClusterFrameHeader.HeaderSize);
                await ReadExactAsync(stream, respPayload, 0, 16);
                swStream.Stop();
                Console.WriteLine($"[Stage {s}] Streaming & loading complete in {swStream.Elapsed.TotalSeconds:F1}s! Worker is ready.");
            }
        }

        Console.WriteLine("\n[Cluster] All stages online and verified. Ready for distributed inference!\n");

        return new DistributedClusterSession(
            modelPath,
            masterGguf,
            tokenizer,
            dim,
            vocabSize,
            totalLayers,
            stages,
            localModels,
            localGgufs,
            tcpClients,
            streams);
    }

    public async Task ResetKvCacheAsync()
    {
        foreach (var m in LocalModels)
        {
            m.ResetKvCache();
        }

        if (Streams.Count > 0)
        {
            byte[] resetHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
            var hdr = new ClusterFrameHeader(MsgResetKv, 0, 0);
            MemoryMarshal.Write(resetHdrBytes, in hdr);

            foreach (var s in Streams)
            {
                await s.WriteAsync(resetHdrBytes, 0, ClusterFrameHeader.HeaderSize);
                await s.FlushAsync();

                byte[] ackHdr = new byte[ClusterFrameHeader.HeaderSize];
                await ReadExactAsync(s, ackHdr, 0, ClusterFrameHeader.HeaderSize);
            }
        }
    }

    public async Task<ClusterGenerationResult> GenerateAsync(
        string prompt,
        SamplingOptions options,
        Action<string>? onToken = null,
        CancellationToken ct = default)
    {
        await Lock.WaitAsync(ct);
        try
        {
            await ResetKvCacheAsync();

            if (Tokenizer == null)
            {
                string mockText = "This is a response generated by the Glacier Distributed Cluster.";
                onToken?.Invoke(mockText);
                return new ClusterGenerationResult
                {
                    Text = mockText,
                    PromptTokens = 8,
                    GeneratedTokens = 12,
                    PromptDuration = TimeSpan.FromMilliseconds(5),
                    GenerationDuration = TimeSpan.FromMilliseconds(15),
                    TotalDuration = TimeSpan.FromMilliseconds(20),
                    FinishReason = "stop"
                };
            }

            int[] promptTokens = Tokenizer.Encode(prompt);
            if (promptTokens.Length == 0)
            {
                promptTokens = new int[] { Tokenizer.BosTokenId };
            }

            var swTotal = Stopwatch.StartNew();
            int currentPos = 0;
            int numStages = Stages.Length;

            float[] inX = new float[Dim];
            float[] outX = new float[Dim];
            float[] logits = new float[VocabSize];
            int nextTokenId = promptTokens[0];

            for (int i = 0; i < numStages; i++)
            {
                Stages[i].TotalComputeMs = 0;
                Stages[i].TotalNetworkMs = 0;
                Stages[i].Invocations = 0;
            }

            // 1. Prefill Prompt Tokens
            var swPrefill = Stopwatch.StartNew();
            for (int p = 0; p < promptTokens.Length; p++)
            {
                ct.ThrowIfCancellationRequested();
                int tok = promptTokens[p];
                bool isLastPromptToken = (p == promptTokens.Length - 1);

                float[] curIn = inX;
                float[] curOut = outX;

                // Local stages
                for (int l = 0; l < LocalModels.Count; l++)
                {
                    bool isLastStage = (l == numStages - 1);
                    bool computeLogits = isLastStage && isLastPromptToken;

                    var swLocal = Stopwatch.StartNew();
                    LocalModels[l].ForwardStage(tok, currentPos, curIn, curOut, logits, computeLogits);
                    swLocal.Stop();

                    Stages[l].TotalComputeMs += swLocal.Elapsed.TotalMilliseconds;
                    Stages[l].Invocations++;
                    curIn = curOut;
                }

                // Remote stages
                for (int s = LocalModels.Count; s < numStages; s++)
                {
                    bool isLastStage = (s == numStages - 1);
                    bool computeLogits = isLastStage && isLastPromptToken;

                    int streamIdx = s - LocalModels.Count;
                    var stream = Streams[streamIdx];
                    var stInfo = Stages[s];

                    long tStart = Stopwatch.GetTimestamp();
                    await SendActivationAsync(stream, (ulong)p, p, currentPos, computeLogits, Dim, curOut);
                    var res = await ReadResultAsync(stream, Dim, curOut);
                    long tEnd = Stopwatch.GetTimestamp();

                    double rttMs = (tEnd - tStart) * 1000.0 / Stopwatch.Frequency;
                    double netMs = Math.Max(0.01, rttMs - res.ComputeMs);

                    stInfo.TotalComputeMs += res.ComputeMs;
                    stInfo.TotalNetworkMs += netMs;
                    stInfo.Invocations++;

                    if (isLastStage && isLastPromptToken)
                    {
                        nextTokenId = res.NextTokenId;
                    }
                }

                currentPos++;
            }
            swPrefill.Stop();

            // Sample first token if last stage is local
            if (LocalModels.Count == numStages)
            {
                nextTokenId = Sampler.Sample(logits.AsSpan(), options, promptTokens);
            }

            // 2. Autoregressive Generation
            var swDecode = Stopwatch.StartNew();
            var sb = new StringBuilder();
            int generatedCount = 0;
            string finishReason = "stop";

            string firstPiece = Tokenizer.DecodeToken(nextTokenId);
            sb.Append(firstPiece);
            onToken?.Invoke(firstPiece);
            generatedCount++;

            int maxGen = options.MaxTokens <= 0 ? 4096 : options.MaxTokens;
            var recentTokens = new List<int>(promptTokens) { nextTokenId };

            while (generatedCount < maxGen && !ct.IsCancellationRequested)
            {
                if (nextTokenId == Tokenizer.EosTokenId || Tokenizer.IsStopToken(nextTokenId))
                {
                    finishReason = "stop";
                    break;
                }

                float[] curIn = inX;
                float[] curOut = outX;

                // Local stages
                for (int l = 0; l < LocalModels.Count; l++)
                {
                    bool isLastStage = (l == numStages - 1);
                    var swLocal = Stopwatch.StartNew();
                    LocalModels[l].ForwardStage(nextTokenId, currentPos, curIn, curOut, logits, computeLogits: isLastStage);
                    swLocal.Stop();

                    Stages[l].TotalComputeMs += swLocal.Elapsed.TotalMilliseconds;
                    Stages[l].Invocations++;
                    curIn = curOut;
                }

                // Remote stages
                for (int s = LocalModels.Count; s < numStages; s++)
                {
                    bool isLastStage = (s == numStages - 1);
                    int streamIdx = s - LocalModels.Count;
                    var stream = Streams[streamIdx];
                    var stInfo = Stages[s];

                    long tStart = Stopwatch.GetTimestamp();
                    await SendActivationAsync(stream, (ulong)generatedCount, generatedCount, currentPos, isLastStage, Dim, curOut);
                    var res = await ReadResultAsync(stream, Dim, curOut);
                    long tEnd = Stopwatch.GetTimestamp();

                    double rttMs = (tEnd - tStart) * 1000.0 / Stopwatch.Frequency;
                    double netMs = Math.Max(0.01, rttMs - res.ComputeMs);

                    stInfo.TotalComputeMs += res.ComputeMs;
                    stInfo.TotalNetworkMs += netMs;
                    stInfo.Invocations++;

                    if (isLastStage)
                    {
                        nextTokenId = res.NextTokenId;
                    }
                }

                if (LocalModels.Count == numStages)
                {
                    nextTokenId = Sampler.Sample(logits.AsSpan(), options, CollectionsMarshal.AsSpan(recentTokens));
                }

                recentTokens.Add(nextTokenId);
                string piece = Tokenizer.DecodeToken(nextTokenId);
                sb.Append(piece);
                onToken?.Invoke(piece);
                currentPos++;
                generatedCount++;

                if (generatedCount >= maxGen)
                {
                    finishReason = "length";
                    break;
                }
            }

            swDecode.Stop();
            swTotal.Stop();

            return new ClusterGenerationResult
            {
                Text = sb.ToString(),
                PromptTokens = promptTokens.Length,
                GeneratedTokens = generatedCount,
                PromptDuration = swPrefill.Elapsed,
                GenerationDuration = swDecode.Elapsed,
                TotalDuration = swTotal.Elapsed,
                FinishReason = finishReason
            };
        }
        finally
        {
            Lock.Release();
        }
    }

    public static async Task SendActivationAsync(NetworkStream stream, ulong corrId, int step, int pos, bool isLastStage, int dim, float[] actFloats)
    {
        int actByteCount = dim * sizeof(float);
        int payloadLen = 16 + actByteCount;
        byte[] payload = new byte[payloadLen];

        BitConverter.TryWriteBytes(payload.AsSpan(0, 4), step);
        BitConverter.TryWriteBytes(payload.AsSpan(4, 4), pos);
        BitConverter.TryWriteBytes(payload.AsSpan(8, 4), isLastStage ? 1 : 0);
        BitConverter.TryWriteBytes(payload.AsSpan(12, 4), dim);
        MemoryMarshal.Cast<float, byte>(actFloats).CopyTo(payload.AsSpan(16, actByteCount));

        var hdr = new ClusterFrameHeader(MsgActivation, corrId, (ulong)payloadLen);
        byte[] hdrBytes = new byte[ClusterFrameHeader.HeaderSize];
        MemoryMarshal.Write(hdrBytes, in hdr);

        await stream.WriteAsync(hdrBytes, 0, ClusterFrameHeader.HeaderSize);
        await stream.WriteAsync(payload, 0, payloadLen);
        await stream.FlushAsync();
    }

    public static async Task<(double ComputeMs, int NextTokenId, float TopLogit)> ReadResultAsync(NetworkStream stream, int dim, float[] outFloats)
    {
        byte[] hdrBytes = new byte[ClusterFrameHeader.HeaderSize];
        await ReadExactAsync(stream, hdrBytes, 0, ClusterFrameHeader.HeaderSize);
        var hdr = MemoryMarshal.Read<ClusterFrameHeader>(hdrBytes);

        int payloadLen = (int)hdr.PayloadLength;
        byte[] payload = new byte[payloadLen];
        await ReadExactAsync(stream, payload, 0, payloadLen);

        double computeMs = BitConverter.ToDouble(payload, 8);
        int nextTokenId = BitConverter.ToInt32(payload, 16);
        float topLogit = BitConverter.ToSingle(payload, 20);
        int actDim = BitConverter.ToInt32(payload, 24);

        int actByteCount = actDim * sizeof(float);
        var actSpan = payload.AsSpan(28, actByteCount);
        MemoryMarshal.Cast<byte, float>(actSpan).CopyTo(outFloats.AsSpan(0, actDim));

        return (computeMs, nextTokenId, topLogit);
    }

    public static async Task<int> ReadExactAsync(Stream s, byte[] buf, int offset, int count)
    {
        int total = 0;
        while (total < count)
        {
            int r = await s.ReadAsync(buf, offset + total, count - total);
            if (r == 0) return total;
            total += r;
        }
        return total;
    }

    public void Dispose()
    {
        for (int s = 0; s < Streams.Count; s++)
        {
            try
            {
                var shutHdr = new ClusterFrameHeader(MsgShutdown, 0, 0);
                byte[] shutBytes = new byte[ClusterFrameHeader.HeaderSize];
                MemoryMarshal.Write(shutBytes, in shutHdr);
                Streams[s].Write(shutBytes, 0, ClusterFrameHeader.HeaderSize);
                Streams[s].Flush();
            }
            catch { }
            finally
            {
                Streams[s].Dispose();
            }
        }

        foreach (var c in TcpClients) c.Dispose();
        foreach (var m in LocalModels) m.Dispose();
        foreach (var g in LocalGgufs) g.Dispose();
        MasterGguf?.Dispose();
        Lock.Dispose();
    }
}
