namespace Glacier.Inference.Distributed.Cli;

using System;
using System.Buffers;
using System.Buffers.Binary;
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
using Glacier.Inference.Distributed;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Distributed.Pipeline;
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
    public const ClusterMessageType MsgInitStage       = (ClusterMessageType)0x60;
    public const ClusterMessageType MsgStageReady      = (ClusterMessageType)0x61;
    public const ClusterMessageType MsgStreamChunk     = (ClusterMessageType)0x62;
    public const ClusterMessageType MsgStreamDone      = (ClusterMessageType)0x63;
    public const ClusterMessageType MsgActivation      = (ClusterMessageType)0x70;
    public const ClusterMessageType MsgResult          = (ClusterMessageType)0x71;
    public const ClusterMessageType MsgResetKv         = (ClusterMessageType)0x72;
    public const ClusterMessageType MsgShutdown        = (ClusterMessageType)0x73;
    public const ClusterMessageType MsgActivationBatch = (ClusterMessageType)0x74;
    public const ClusterMessageType MsgResultBatch     = (ClusterMessageType)0x75;

    private readonly PinnedBatchBufferArena _batchArena;
    private readonly DynamicPrecisionSwitcher _precisionSwitcher = new();

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
        _batchArena = new PinnedBatchBufferArena(maxBatchTokens: 2048, hiddenDim: Math.Max(dim, 5120), vocabSize: Math.Max(vocabSize, 152064));
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
            FusedSocketTransport.ConfigureSocket(client.Client, 4 * 1024 * 1024);
            await client.ConnectAsync(alloc.Node.Host, alloc.Node.Port);
            FusedSocketTransport.ConfigureSocket(client.Client, 4 * 1024 * 1024);
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
            byte[] respPayload = new byte[16];

            while (true)
            {
                await ReadExactAsync(stream, respHdrBytes, 0, ClusterFrameHeader.HeaderSize);
                var respHdr = MemoryMarshal.Read<ClusterFrameHeader>(respHdrBytes);
                if (respHdr.MessageType == ClusterMessageType.HeartbeatPing)
                {
                    if (respHdr.PayloadLength > 0)
                    {
                        byte[] discard = ArrayPool<byte>.Shared.Rent((int)respHdr.PayloadLength);
                        try { await ReadExactAsync(stream, discard, 0, (int)respHdr.PayloadLength); }
                        finally { ArrayPool<byte>.Shared.Return(discard); }
                    }
                    continue;
                }
                if (respHdr.MessageType == MsgStageReady)
                {
                    await ReadExactAsync(stream, respPayload, 0, 16);
                    break;
                }
                if (respHdr.PayloadLength > 0)
                {
                    byte[] discard = ArrayPool<byte>.Shared.Rent((int)respHdr.PayloadLength);
                    try { await ReadExactAsync(stream, discard, 0, (int)respHdr.PayloadLength); }
                    finally { ArrayPool<byte>.Shared.Return(discard); }
                }
            }

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

                while (true)
                {
                    await ReadExactAsync(stream, respHdrBytes, 0, ClusterFrameHeader.HeaderSize);
                    var respHdr = MemoryMarshal.Read<ClusterFrameHeader>(respHdrBytes);
                    if (respHdr.MessageType == ClusterMessageType.HeartbeatPing)
                    {
                        if (respHdr.PayloadLength > 0)
                        {
                            byte[] discard = ArrayPool<byte>.Shared.Rent((int)respHdr.PayloadLength);
                            try { await ReadExactAsync(stream, discard, 0, (int)respHdr.PayloadLength); }
                            finally { ArrayPool<byte>.Shared.Return(discard); }
                        }
                        continue;
                    }
                    if (respHdr.MessageType == MsgStageReady)
                    {
                        await ReadExactAsync(stream, respPayload, 0, 16);
                        break;
                    }
                    if (respHdr.PayloadLength > 0)
                    {
                        byte[] discard = ArrayPool<byte>.Shared.Rent((int)respHdr.PayloadLength);
                        try { await ReadExactAsync(stream, discard, 0, (int)respHdr.PayloadLength); }
                        finally { ArrayPool<byte>.Shared.Return(discard); }
                    }
                }
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

            // 1. Batched Prompt Prefill
            var swPrefill = Stopwatch.StartNew();
            int numPromptTokens = promptTokens.Length;
            int maxChunk = 256;

            float[] curBatch = GC.AllocateArray<float>(maxChunk * Dim, pinned: true);
            float[] nextBatch = GC.AllocateArray<float>(maxChunk * Dim, pinned: true);

            for (int chunkStart = 0; chunkStart < numPromptTokens; chunkStart += maxChunk)
            {
                ct.ThrowIfCancellationRequested();
                int chunkSize = Math.Min(numPromptTokens - chunkStart, maxChunk);
                bool isLastChunk = (chunkStart + chunkSize == numPromptTokens);
                int totalFloats = chunkSize * Dim;

                ReadOnlySpan<int> chunkTokens = promptTokens.AsSpan(chunkStart, chunkSize);

                // Local stages
                for (int l = 0; l < LocalModels.Count; l++)
                {
                    bool isLastStage = (l == numStages - 1);
                    bool computeLogits = isLastStage && isLastChunk;

                    var swLocal = Stopwatch.StartNew();
                    if (l == 0)
                    {
                        LocalModels[l].ForwardBatchStage(
                            chunkTokens,
                            currentPos,
                            ReadOnlySpan<float>.Empty,
                            isLastStage ? Span<float>.Empty : curBatch.AsSpan(0, totalFloats),
                            logits,
                            computeLogits);
                    }
                    else
                    {
                        LocalModels[l].ForwardBatchStage(
                            ReadOnlySpan<int>.Empty,
                            currentPos,
                            curBatch.AsSpan(0, totalFloats),
                            isLastStage ? Span<float>.Empty : nextBatch.AsSpan(0, totalFloats),
                            logits,
                            computeLogits);

                        (curBatch, nextBatch) = (nextBatch, curBatch);
                    }
                    swLocal.Stop();

                    Stages[l].TotalComputeMs += swLocal.Elapsed.TotalMilliseconds;
                    Stages[l].Invocations += chunkSize;
                }

                // Remote stages
                for (int s = LocalModels.Count; s < numStages; s++)
                {
                    bool isLastStage = (s == numStages - 1);
                    bool computeLogits = isLastStage && isLastChunk;

                    int streamIdx = s - LocalModels.Count;
                    var stream = Streams[streamIdx];
                    var stInfo = Stages[s];

                    long tStart = Stopwatch.GetTimestamp();
                    WirePrecisionMode prefillPrec = _precisionSwitcher.SelectPrecision(chunkSize, currentPos, isDecode: false);

                    await SendActivationBatchAsync(
                        stream,
                        (ulong)chunkStart,
                        chunkSize,
                        currentPos,
                        computeLogits,
                        Dim,
                        curBatch,
                        _batchArena.BatchWireSendArray,
                        prefillPrec);

                    var res = await ReadResultBatchAsync(
                        stream,
                        chunkSize,
                        Dim,
                        computeLogits ? null : nextBatch,
                        _batchArena.BatchWireRecvArray);

                    long tEnd = Stopwatch.GetTimestamp();

                    double rttMs = (tEnd - tStart) * 1000.0 / Stopwatch.Frequency;
                    _precisionSwitcher.UpdateMeasuredLatency(rttMs);
                    double netMs = Math.Max(0.01, rttMs - res.ComputeMs);

                    stInfo.TotalComputeMs += res.ComputeMs;
                    stInfo.TotalNetworkMs += netMs;
                    stInfo.Invocations += chunkSize;

                    if (computeLogits)
                    {
                        nextTokenId = res.NextTokenId;
                    }
                    else if (!isLastStage)
                    {
                        (curBatch, nextBatch) = (nextBatch, curBatch);
                    }
                }

                currentPos += chunkSize;
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
        ClusterFrameHeader hdr;
        while (true)
        {
            await ReadExactAsync(stream, hdrBytes, 0, ClusterFrameHeader.HeaderSize);
            hdr = MemoryMarshal.Read<ClusterFrameHeader>(hdrBytes);
            if (hdr.MessageType == ClusterMessageType.HeartbeatPing)
            {
                if (hdr.PayloadLength > 0)
                {
                    byte[] discard = ArrayPool<byte>.Shared.Rent((int)hdr.PayloadLength);
                    try { await ReadExactAsync(stream, discard, 0, (int)hdr.PayloadLength); }
                    finally { ArrayPool<byte>.Shared.Return(discard); }
                }
                continue;
            }
            if (hdr.MessageType == MsgResult)
            {
                break;
            }
            if (hdr.PayloadLength > 0)
            {
                byte[] discard = ArrayPool<byte>.Shared.Rent((int)hdr.PayloadLength);
                try { await ReadExactAsync(stream, discard, 0, (int)hdr.PayloadLength); }
                finally { ArrayPool<byte>.Shared.Return(discard); }
            }
        }

        int payloadLen = (int)hdr.PayloadLength;
        byte[] payload = new byte[payloadLen];
        await ReadExactAsync(stream, payload, 0, payloadLen);

        // 1. Check for modern FusedActivationResultHeader ("GRES")
        if (payloadLen >= FusedActivationResultHeader.HeaderSize &&
            BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(0, 4)) == FusedActivationResultHeader.ResultMagic)
        {
            var resHdr = FusedActivationResultHeader.ReadFrom(payload.AsSpan(0, FusedActivationResultHeader.HeaderSize));
            int actBytes = payloadLen - FusedActivationResultHeader.HeaderSize;
            if (actBytes > 0)
            {
                var actSpan = payload.AsSpan(FusedActivationResultHeader.HeaderSize, actBytes);
                QuantizationKernels.DequantizeToken(actSpan, outFloats.AsSpan(0, dim), resHdr.TypedPrecision);
            }
            return (resHdr.ComputeMs, resHdr.NextTokenId, resHdr.TopLogit);
        }

        // 2. Fallback to legacy un-fused result format
        double computeMs = payloadLen >= 16 ? BitConverter.ToDouble(payload, 8) : 0.0;
        int nextTokenId = payloadLen >= 20 ? BitConverter.ToInt32(payload, 16) : 0;
        float topLogit = payloadLen >= 24 ? BitConverter.ToSingle(payload, 20) : 0f;
        int actDim = payloadLen >= 28 ? BitConverter.ToInt32(payload, 24) : 0;

        int actByteCount = actDim * sizeof(float);
        if (actDim > 0 && payloadLen >= 28 + actByteCount)
        {
            var actSpan = payload.AsSpan(28, actByteCount);
            MemoryMarshal.Cast<byte, float>(actSpan).CopyTo(outFloats.AsSpan(0, Math.Min(actDim, dim)));
        }

        return (computeMs, nextTokenId, topLogit);
    }

    public static async Task SendActivationBatchAsync(
        NetworkStream stream,
        ulong corrId,
        int numTokens,
        int pos,
        bool isLastStage,
        int dim,
        float[] actFloats,
        byte[] batchBuf,
        WirePrecisionMode precisionMode = WirePrecisionMode.INT8_Symmetric)
    {
        int totalFloats = numTokens * dim;
        int actByteCount = QuantizationKernels.GetPackedBatchBytes(precisionMode, dim, numTokens);
        int payloadLen = FusedActivationHeader.HeaderSize + actByteCount;

        var clusterHdr = new ClusterFrameHeader(MsgActivationBatch, corrId, (ulong)payloadLen);
        var actHdr = new FusedActivationHeader(
            sequenceId: corrId,
            tokenIndex: pos,
            layerIndex: 0,
            hiddenDim: dim,
            precisionMode: precisionMode,
            flags: (isLastStage ? FusedActivationFlags.FinalLayer : FusedActivationFlags.None) | FusedActivationFlags.IsPrefillChunk,
            scale: 1.0f,
            batchTokens: (ushort)numTokens);

        byte[] hdrBytes = new byte[ClusterFrameHeader.HeaderSize];
        clusterHdr.WriteTo(hdrBytes.AsSpan(0, ClusterFrameHeader.HeaderSize));

        actHdr.WriteTo(batchBuf.AsSpan(0, FusedActivationHeader.HeaderSize));
        var payload = batchBuf.AsSpan(FusedActivationHeader.HeaderSize, actByteCount);
        QuantizationKernels.QuantizeBatch(actFloats.AsSpan(0, totalFloats), payload, dim, numTokens, precisionMode);

        await stream.WriteAsync(hdrBytes, 0, ClusterFrameHeader.HeaderSize);
        await stream.WriteAsync(batchBuf, 0, payloadLen);
        await stream.FlushAsync();
    }

    public static async Task<(double ComputeMs, int NextTokenId)> ReadResultBatchAsync(
        NetworkStream stream,
        int numTokens,
        int dim,
        float[]? outFloatsBatch,
        byte[] batchRecv)
    {
        byte[] hdrBytes = new byte[ClusterFrameHeader.HeaderSize];
        ClusterFrameHeader clusterHdr;
        while (true)
        {
            await ReadExactAsync(stream, hdrBytes, 0, ClusterFrameHeader.HeaderSize);
            clusterHdr = MemoryMarshal.Read<ClusterFrameHeader>(hdrBytes);
            if (clusterHdr.MessageType == ClusterMessageType.HeartbeatPing)
            {
                if (clusterHdr.PayloadLength > 0)
                {
                    byte[] discard = ArrayPool<byte>.Shared.Rent((int)clusterHdr.PayloadLength);
                    try { await ReadExactAsync(stream, discard, 0, (int)clusterHdr.PayloadLength); }
                    finally { ArrayPool<byte>.Shared.Return(discard); }
                }
                continue;
            }
            break;
        }

        int payloadLen = (int)clusterHdr.PayloadLength;
        if (payloadLen > 0)
        {
            await ReadExactAsync(stream, batchRecv, 0, payloadLen);
        }

        uint resultMagic = payloadLen >= 4 ? BinaryPrimitives.ReadUInt32LittleEndian(batchRecv.AsSpan(0, 4)) : 0;
        double computeMs = 0;
        int nextTokenId = 0;
        int actByteCount = 0;

        if (resultMagic == FusedActivationResultHeader.ResultMagic)
        {
            var resultHdr = FusedActivationResultHeader.ReadFrom(batchRecv.AsSpan(0, FusedActivationResultHeader.HeaderSize));
            computeMs = resultHdr.ComputeMs;
            nextTokenId = resultHdr.NextTokenId;
            WirePrecisionMode resPrec = resultHdr.TypedPrecision;
            actByteCount = payloadLen - FusedActivationResultHeader.HeaderSize;

            if (outFloatsBatch != null && actByteCount > 0)
            {
                var payloadSpan = batchRecv.AsSpan(FusedActivationResultHeader.HeaderSize, actByteCount);
                QuantizationKernels.DequantizeBatch(payloadSpan, outFloatsBatch.AsSpan(0, numTokens * dim), dim, numTokens, resPrec);
            }
        }
        else
        {
            var legacyPayload = batchRecv.AsSpan(0);
            if (payloadLen >= 20)
            {
                computeMs = BitConverter.ToDouble(legacyPayload.Slice(8, 8));
                nextTokenId = BitConverter.ToInt32(legacyPayload.Slice(16, 4));
            }
            actByteCount = payloadLen - 28;
            int totalFloats = numTokens * dim;
            if (outFloatsBatch != null && actByteCount > 0)
            {
                var payloadSpan = batchRecv.AsSpan(28, actByteCount);
                if (actByteCount == totalFloats * 2)
                {
                    var halfSpan = MemoryMarshal.Cast<byte, Half>(payloadSpan);
                    for (int i = 0; i < totalFloats && i < halfSpan.Length; i++)
                    {
                        outFloatsBatch[i] = (float)halfSpan[i];
                    }
                }
                else
                {
                    MemoryMarshal.Cast<byte, float>(payloadSpan).CopyTo(outFloatsBatch.AsSpan(0, totalFloats));
                }
            }
        }

        return (computeMs, nextTokenId);
    }

    public static async Task<int> ReadExactAsync(Stream s, byte[] buf, int offset, int count)
    {
        int total = 0;
        while (total < count)
        {
            int r = await s.ReadAsync(buf, offset + total, count - total);
            if (r == 0) throw new EndOfStreamException($"Expected {count} bytes but reached EOF after {total} bytes.");
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
        _batchArena?.Dispose();
    }
}
