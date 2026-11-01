namespace Glacier.Inference.Distributed.Cli;

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Inference.Distributed;
using Glacier.Inference.Distributed.Pipeline;
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
/// Compact deterministic wire codec for forwarding full <see cref="SamplingOptions"/>,
/// recent token window (up to 256 tokens), and coordinator PRNG draw to a remote final stage.
/// Eliminates all heap allocations on both coordinator and worker hot paths.
/// </summary>
internal static class SamplingWireCodec
{
    public const int HeaderBytes = 32;
    public const int MaxRecentTokens = 256;
    public const int MaxTrailerBytes = HeaderBytes + (MaxRecentTokens * sizeof(int));

    public static int WriteTrailer(
        Span<byte> destination,
        SamplingOptions options,
        ReadOnlySpan<int> recentTokens,
        float randomUniform01)
    {
        ReadOnlySpan<int> window = recentTokens.Length > MaxRecentTokens
            ? recentTokens.Slice(recentTokens.Length - MaxRecentTokens, MaxRecentTokens)
            : recentTokens;

        int totalBytes = HeaderBytes + (window.Length * sizeof(int));
        if (destination.Length < totalBytes)
            throw new ArgumentException($"Destination span too small ({destination.Length} < {totalBytes}).", nameof(destination));

        BinaryPrimitives.WriteSingleLittleEndian(destination.Slice(0, 4), options.Temperature);
        BinaryPrimitives.WriteSingleLittleEndian(destination.Slice(4, 4), options.TopP);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(8, 4), options.TopK);
        BinaryPrimitives.WriteSingleLittleEndian(destination.Slice(12, 4), options.RepetitionPenalty);
        BinaryPrimitives.WriteSingleLittleEndian(destination.Slice(16, 4), options.LoopRecoveryRepetitionPenalty);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(20, 4), options.ReasoningLoopMinPeriod);
        destination[24] = options.EnableReasoningLoopGuard ? (byte)1 : (byte)0;
        destination[25] = 0;
        BinaryPrimitives.WriteUInt16LittleEndian(destination.Slice(26, 2), (ushort)window.Length);
        BinaryPrimitives.WriteSingleLittleEndian(destination.Slice(28, 4), randomUniform01);

        for (int i = 0; i < window.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(HeaderBytes + (i * sizeof(int)), sizeof(int)), window[i]);
        }

        return totalBytes;
    }

    public static (int TokenId, float TopLogit) SampleWithTrailer(
        Span<float> logits,
        ReadOnlySpan<byte> trailer,
        Span<int> recentTokensScratch)
    {
        if (logits.IsEmpty)
            return (0, 0f);

        if (trailer.Length < HeaderBytes)
        {
            return SampleGreedy(logits);
        }

        float temperature = BinaryPrimitives.ReadSingleLittleEndian(trailer.Slice(0, 4));
        float topP = BinaryPrimitives.ReadSingleLittleEndian(trailer.Slice(4, 4));
        int topK = BinaryPrimitives.ReadInt32LittleEndian(trailer.Slice(8, 4));
        float repetitionPenalty = BinaryPrimitives.ReadSingleLittleEndian(trailer.Slice(12, 4));
        float loopRecoveryPenalty = BinaryPrimitives.ReadSingleLittleEndian(trailer.Slice(16, 4));
        int loopMinPeriod = BinaryPrimitives.ReadInt32LittleEndian(trailer.Slice(20, 4));
        bool enableLoopGuard = trailer[24] != 0;
        int recentCount = BinaryPrimitives.ReadUInt16LittleEndian(trailer.Slice(26, 2));
        float uniform01 = BinaryPrimitives.ReadSingleLittleEndian(trailer.Slice(28, 4));

        if (!float.IsFinite(temperature) || temperature < 0f) temperature = 0f;
        if (!float.IsFinite(topP)) topP = 1.0f;
        topP = Math.Clamp(topP, 0f, 1f);
        if (!float.IsFinite(repetitionPenalty) || repetitionPenalty <= 0f) repetitionPenalty = 1.0f;
        if (!float.IsFinite(loopRecoveryPenalty) || loopRecoveryPenalty <= 0f) loopRecoveryPenalty = 1.35f;
        if (loopMinPeriod < 1) loopMinPeriod = 1;
        if (!float.IsFinite(uniform01)) uniform01 = 0f;
        uniform01 = Math.Clamp(uniform01, 0f, 0.99999994f);

        int maxAvailTokens = (trailer.Length - HeaderBytes) / sizeof(int);
        int count = Math.Min(recentCount, Math.Min(maxAvailTokens, recentTokensScratch.Length));
        for (int i = 0; i < count; i++)
        {
            recentTokensScratch[i] = BinaryPrimitives.ReadInt32LittleEndian(trailer.Slice(HeaderBytes + (i * sizeof(int)), sizeof(int)));
        }
        ReadOnlySpan<int> recentTokens = recentTokensScratch.Slice(0, count);

        Sampler.SanitizeLogits(logits);

        float effectivePenalty = repetitionPenalty;
        ReadOnlySpan<int> penaltyWindow = recentTokens;

        if (enableLoopGuard && !recentTokens.IsEmpty)
        {
            int loopPeriod = Sampler.DetectPeriodicReasoningLoop(recentTokens, minPeriod: loopMinPeriod);
            if (loopPeriod > 0)
            {
                effectivePenalty = Math.Max(effectivePenalty, loopRecoveryPenalty);
                int windowLen = Math.Min(recentTokens.Length, loopPeriod * 3);
                penaltyWindow = recentTokens.Slice(recentTokens.Length - windowLen, windowLen);
            }
        }

        if (effectivePenalty != 1.0f && !penaltyWindow.IsEmpty)
        {
            Sampler.ApplyRepetitionPenalty(logits, penaltyWindow, effectivePenalty);
        }

        if (temperature <= Sampler.GreedyTemperatureThreshold || topK == 1)
        {
            return SampleGreedy(logits);
        }

        int vocabSize = logits.Length;
        int k = topK <= 0 ? vocabSize : Math.Min(topK, vocabSize);
        (int Id, float Logit)[]? rented = null;
        Span<(int Id, float Logit)> candidates = k <= 128
            ? stackalloc (int, float)[k]
            : (rented = ArrayPool<(int Id, float Logit)>.Shared.Rent(k)).AsSpan(0, k);

        try
        {
            SelectTopK(logits, k, candidates);
            float topRawLogit = logits[candidates[0].Id];

            float invTemp = 1.0f / temperature;
            for (int i = 0; i < k; i++)
            {
                candidates[i].Logit *= invTemp;
            }

            float maxLogit = candidates[0].Logit;
            float sumExp = 0f;
            for (int i = 0; i < k; i++)
            {
                float exp = MathF.Exp(candidates[i].Logit - maxLogit);
                candidates[i].Logit = exp;
                sumExp += exp;
            }

            if (!(sumExp > 0f) || !float.IsFinite(sumExp))
            {
                return (candidates[0].Id, topRawLogit);
            }

            float invSum = 1.0f / sumExp;
            float cumProb = 0f;
            int activeCount = 0;
            for (int i = 0; i < k; i++)
            {
                float prob = candidates[i].Logit * invSum;
                candidates[i].Logit = prob;
                cumProb += prob;
                activeCount++;
                if (cumProb >= topP) break;
            }

            float r = uniform01 * cumProb;
            float acc = 0f;
            for (int i = 0; i < activeCount; i++)
            {
                acc += candidates[i].Logit;
                if (r <= acc)
                {
                    int chosenId = candidates[i].Id;
                    return (chosenId, logits[chosenId]);
                }
            }

            int fallbackId = candidates[0].Id;
            return (fallbackId, logits[fallbackId]);
        }
        finally
        {
            if (rented != null)
            {
                ArrayPool<(int Id, float Logit)>.Shared.Return(rented);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static (int TokenId, float TopLogit) SampleGreedy(ReadOnlySpan<float> logits)
    {
        int bestId = 0;
        float bestLogit = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > bestLogit)
            {
                bestLogit = logits[i];
                bestId = i;
            }
        }
        return (bestId, float.IsNegativeInfinity(bestLogit) ? 0f : bestLogit);
    }

    private static void SelectTopK(ReadOnlySpan<float> logits, int k, Span<(int Id, float Logit)> heap)
    {
        for (int i = 0; i < k; i++)
        {
            heap[i] = (i, logits[i]);
        }

        for (int i = (k / 2) - 1; i >= 0; i--)
        {
            SiftDown(heap, i, k);
        }

        float minLogit = heap[0].Logit;
        for (int i = k; i < logits.Length; i++)
        {
            float val = logits[i];
            if (val > minLogit)
            {
                heap[0] = (i, val);
                SiftDown(heap, 0, k);
                minLogit = heap[0].Logit;
            }
        }

        heap.Sort(default(LogitDescendingComparer));
    }

    private readonly struct LogitDescendingComparer : IComparer<(int Id, float Logit)>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Compare((int Id, float Logit) x, (int Id, float Logit) y) => y.Logit.CompareTo(x.Logit);
    }

    private static void SiftDown(Span<(int Id, float Logit)> heap, int i, int n)
    {
        while (true)
        {
            int left = (2 * i) + 1;
            int right = (2 * i) + 2;
            int smallest = i;

            if (left < n && heap[left].Logit < heap[smallest].Logit)
                smallest = left;
            if (right < n && heap[right].Logit < heap[smallest].Logit)
                smallest = right;

            if (smallest != i)
            {
                (heap[i], heap[smallest]) = (heap[smallest], heap[i]);
                i = smallest;
            }
            else
            {
                break;
            }
        }
    }
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

    private const int MaxPrefillChunkTokens = 256;

    private readonly PinnedBatchBufferArena _batchArena;
    private readonly DynamicPrecisionSwitcher _precisionSwitcher = new();
    private readonly float[] _decodeBufA;
    private readonly float[] _decodeBufB;
    private readonly byte[] _frameHdrScratch;

    public string ModelPath { get; init; }
    public string ModelName { get; init; }
    public string Architecture { get; init; } = "qwen2";
    public GgufFile? MasterGguf { get; }
    public BpeTokenizer? Tokenizer { get; }
    public int Dim { get; init; }
    public int VocabSize { get; init; }
    public int TotalLayers { get; init; }
    public int MaxSeqLen { get; init; }
    public StageTelemetry[] Stages { get; init; }
    public List<IClusterStageModel> LocalModels { get; init; }
    public List<GgufFile> LocalGgufs { get; init; }
    public List<TcpClient> TcpClients { get; init; }
    public List<NetworkStream> Streams { get; init; }
    public SemaphoreSlim Lock { get; } = new(1, 1);
    public Sampler Sampler { get; set; } = new();
    public DynamicPrecisionSwitcher PrecisionSwitcher => _precisionSwitcher;

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
        List<NetworkStream> streams,
        int maxSeqLen = 4096)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);
        ModelPath = modelPath;
        ModelName = Path.GetFileNameWithoutExtension(modelPath);
        MasterGguf = masterGguf;
        Tokenizer = tokenizer;
        Dim = Math.Max(1, dim);
        VocabSize = Math.Max(1, vocabSize);
        TotalLayers = totalLayers;
        MaxSeqLen = maxSeqLen;
        Stages = stages;
        LocalModels = localModels;
        LocalGgufs = localGgufs;
        TcpClients = tcpClients;
        Streams = streams;
        if (masterGguf != null)
        {
            Architecture = masterGguf.Architecture;
        }

        bool isMockOnly = masterGguf == null && localModels.Count == 0 && streams.Count == 0;
        int arenaMaxTokens = isMockOnly ? 4 : MaxPrefillChunkTokens;
        _batchArena = new PinnedBatchBufferArena(
            maxBatchTokens: arenaMaxTokens,
            hiddenDim: Math.Max(Dim, 64),
            vocabSize: Math.Max(VocabSize, 64));

        _decodeBufA = GC.AllocateArray<float>(Math.Max(Dim, 64), pinned: true);
        _decodeBufB = GC.AllocateArray<float>(Math.Max(Dim, 64), pinned: true);
        _frameHdrScratch = GC.AllocateArray<byte>(ClusterFrameHeader.HeaderSize * 2, pinned: true);
    }

    public static DistributedClusterSession CreateMockSession(
        string modelName = "qwen2-57b-a14b-instruct",
        int totalLayers = 28,
        int dim = 3584,
        int vocabSize = 151936)
    {
        var stages = new[]
        {
            new StageTelemetry { Name = "Host Stage 0", Host = "127.0.0.1", Port = 5055, StartLayer = 0, EndLayer = 13 },
            new StageTelemetry { Name = "Loopback Worker Stage 1", Host = "127.0.0.1", Port = 5056, StartLayer = 14, EndLayer = 27 }
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
        int hostGpuAdapter = -1,
        bool includeLocalHost = true,
        CancellationToken ct = default)
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
        var localProfiles = includeLocalHost
            ? ClusterResourcePlanner.DetectLocalNodes(hostGpuAdapter)
            : new List<NodeResourceProfile>();
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

        // 2. Connect to Remote Workers (supports remote Stage-0 when LocalModels.Count == 0)
        var tcpClients = new List<TcpClient>();
        var streams = new List<NetworkStream>();

        for (int s = localModels.Count; s < numStages; s++)
        {
            var alloc = activeAllocations[s];
            var stInfo = stages[s];
            bool isFirst = (s == 0);
            bool isLast = alloc.IsLastStage;

            string sliceFileName = GgufStageSlicer.GetStageFileName(modelPath, s, stInfo.StartLayer, stInfo.EndLayer);
            ulong expectedBytes = GgufStageSlicer.CalculateStageByteSize(masterGguf, stInfo.StartLayer, stInfo.EndLayer, isFirstStage: isFirst, isLastStage: isLast);

            Console.Write($"[Coordinator] Connecting to Stage {s} [{stInfo.Name}] ({alloc.Node.Host}:{alloc.Node.Port})... ");
            var client = new TcpClient();
            client.NoDelay = true;
            client.ReceiveBufferSize = 4 * 1024 * 1024;
            client.SendBufferSize = 4 * 1024 * 1024;
            FusedSocketTransport.ConfigureSocket(client.Client, 4 * 1024 * 1024);
            await client.ConnectAsync(alloc.Node.Host, alloc.Node.Port, ct).ConfigureAwait(false);
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
            initHdr.WriteTo(initHdrBytes);
            await stream.WriteAsync(initHdrBytes.AsMemory(0, ClusterFrameHeader.HeaderSize), ct).ConfigureAwait(false);
            await stream.WriteAsync(initPayload.AsMemory(0, initLen), ct).ConfigureAwait(false);
            await stream.FlushAsync(ct).ConfigureAwait(false);

            // Read response
            byte[] respHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
            byte[] respPayload = new byte[16];

            await WaitForStageReadyAsync(stream, respHdrBytes, respPayload, ct).ConfigureAwait(false);

            int status = BitConverter.ToInt32(respPayload, 0);
            if (status == 0)
            {
                Console.WriteLine($"[Stage {s}] Remote node has valid cached slice: {sliceFileName} (Ready in RAM/VRAM).");
            }
            else
            {
                Console.WriteLine($"[Stage {s}] Remote node cache miss. Slicing and streaming {sliceFileName} ({expectedBytes / (1024.0 * 1024.0):N1} MB) over LAN...");
                var swStream = Stopwatch.StartNew();

                using (var chunkWriter = new Program.NetworkChunkWriter(stream, MsgStreamChunk))
                {
                    GgufStageSlicer.SliceStage(masterGguf, chunkWriter, stInfo.StartLayer, stInfo.EndLayer, isFirstStage: isFirst, isLastStage: isLast);
                }

                var doneHdr = new ClusterFrameHeader(MsgStreamDone, (ulong)s, 0);
                byte[] doneHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
                doneHdr.WriteTo(doneHdrBytes);
                await stream.WriteAsync(doneHdrBytes.AsMemory(0, ClusterFrameHeader.HeaderSize), ct).ConfigureAwait(false);
                await stream.FlushAsync(ct).ConfigureAwait(false);

                await WaitForStageReadyAsync(stream, respHdrBytes, respPayload, ct).ConfigureAwait(false);
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

    private static async Task WaitForStageReadyAsync(
        Stream stream,
        byte[] respHdrBytes,
        byte[] respPayload,
        CancellationToken ct)
    {
        while (true)
        {
            await ReadExactAsync(stream, respHdrBytes, 0, ClusterFrameHeader.HeaderSize, ct).ConfigureAwait(false);
            var respHdr = MemoryMarshal.Read<ClusterFrameHeader>(respHdrBytes);
            if (respHdr.PayloadLength > FusedSocketTransport.MaxAllowedPayloadBytes)
                throw new InvalidDataException($"Remote worker sent oversized control frame ({respHdr.PayloadLength} bytes).");

            if (respHdr.MessageType == MsgStageReady)
            {
                if (respHdr.PayloadLength != 16)
                    throw new InvalidDataException($"Invalid MsgStageReady payload length: {respHdr.PayloadLength}.");
                await ReadExactAsync(stream, respPayload, 0, 16, ct).ConfigureAwait(false);
                return;
            }

            if (respHdr.PayloadLength > 0)
            {
                int discardLen = (int)respHdr.PayloadLength;
                byte[] discard = ArrayPool<byte>.Shared.Rent(discardLen);
                try { await ReadExactAsync(stream, discard, 0, discardLen, ct).ConfigureAwait(false); }
                finally { ArrayPool<byte>.Shared.Return(discard); }
            }
        }
    }

    /// <summary>
    /// Thread-safe reset of all local and remote stage KV caches.
    /// Acquires <see cref="Lock"/> to serialize against concurrent <see cref="GenerateAsync"/> calls.
    /// </summary>
    public async Task ResetKvCacheAsync(CancellationToken ct = default)
    {
        await Lock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            ct.ThrowIfCancellationRequested();
            // Execute wire control frame + ack atomically so persistent worker TCP streams never desynchronize
            await ResetKvCacheCoreAsync(CancellationToken.None).ConfigureAwait(false);
        }
        finally
        {
            Lock.Release();
        }
    }

    /// <summary>
    /// Alias for <see cref="ResetKvCacheAsync"/> providing multi-turn cache reset parity.
    /// </summary>
    public Task ResetAllCachesAsync(CancellationToken ct = default) => ResetKvCacheAsync(ct);

    /// <summary>
    /// Synchronous convenience wrapper for <see cref="ResetKvCacheAsync"/>.
    /// </summary>
    public void ResetAllCaches() => ResetKvCacheAsync().GetAwaiter().GetResult();

    private async Task ResetKvCacheCoreAsync(CancellationToken ct = default)
    {
        foreach (var m in LocalModels)
        {
            m.ResetKvCache();
        }

        if (Streams.Count > 0)
        {
            var hdr = new ClusterFrameHeader(MsgResetKv, 0, 0);
            hdr.WriteTo(_frameHdrScratch.AsSpan(0, ClusterFrameHeader.HeaderSize));
            var sendMem = _frameHdrScratch.AsMemory(0, ClusterFrameHeader.HeaderSize);
            var ackBufOffset = ClusterFrameHeader.HeaderSize;

            foreach (var s in Streams)
            {
                await s.WriteAsync(sendMem, ct).ConfigureAwait(false);
                await s.FlushAsync(ct).ConfigureAwait(false);

                while (true)
                {
                    await ReadExactAsync(s, _frameHdrScratch, ackBufOffset, ClusterFrameHeader.HeaderSize, ct).ConfigureAwait(false);
                    var ackHdr = MemoryMarshal.Read<ClusterFrameHeader>(_frameHdrScratch.AsSpan(ackBufOffset, ClusterFrameHeader.HeaderSize));
                    if (ackHdr.PayloadLength > FusedSocketTransport.MaxAllowedPayloadBytes)
                        throw new InvalidDataException($"Oversized frame during ResetKvCache ack: {ackHdr.PayloadLength}.");

                    if (ackHdr.PayloadLength > 0)
                    {
                        int pLen = (int)ackHdr.PayloadLength;
                        byte[] discard = ArrayPool<byte>.Shared.Rent(pLen);
                        try { await ReadExactAsync(s, discard, 0, pLen, ct).ConfigureAwait(false); }
                        finally { ArrayPool<byte>.Shared.Return(discard); }
                    }

                    if (ackHdr.MessageType == MsgResetKv)
                    {
                        break;
                    }
                }
            }
        }
    }

    public async Task<ClusterGenerationResult> GenerateAsync(
        string prompt,
        SamplingOptions options,
        Action<string>? onToken = null,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(options);

        await Lock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            ct.ThrowIfCancellationRequested();

            if (Tokenizer == null)
            {
                await ResetKvCacheCoreAsync(CancellationToken.None).ConfigureAwait(false);
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
                promptTokens = [Tokenizer.BosTokenId];
            }

            if (promptTokens.Length >= MaxSeqLen)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(prompt),
                    $"Prompt token count ({promptTokens.Length}) must be strictly less than MaxSeqLen ({MaxSeqLen}) to avoid overflowing stage KV caches.");
            }

            // Reset local & remote stage KV caches atomically before prefill
            await ResetKvCacheCoreAsync(CancellationToken.None).ConfigureAwait(false);

            var swTotal = Stopwatch.StartNew();
            int currentPos = 0;
            int numStages = Stages.Length;

            float[] logits = _batchArena.LogitsArray;
            int nextTokenId = promptTokens[0];

            for (int i = 0; i < numStages; i++)
            {
                Stages[i].TotalComputeMs = 0;
                Stages[i].TotalNetworkMs = 0;
                Stages[i].Invocations = 0;
            }

            // 1. Batched Prompt Prefill (zero LOH: reuses POH buffers from _batchArena)
            var swPrefill = Stopwatch.StartNew();
            int numPromptTokens = promptTokens.Length;
            int maxChunk = MaxPrefillChunkTokens;

            float[] curBatch = _batchArena.FloatInputArray;
            float[] nextBatch = _batchArena.FloatOutputArray;

            for (int chunkStart = 0; chunkStart < numPromptTokens; chunkStart += maxChunk)
            {
                // Check cancellation strictly BETWEEN complete prefill-chunk round-trips so TCP frames never split mid-write/mid-read
                ct.ThrowIfCancellationRequested();
                int chunkSize = Math.Min(numPromptTokens - chunkStart, maxChunk);
                if (currentPos + chunkSize > MaxSeqLen)
                {
                    throw new ArgumentOutOfRangeException(
                        nameof(prompt),
                        $"Prefill position ({currentPos + chunkSize}) exceeds MaxSeqLen ({MaxSeqLen}).");
                }

                bool isLastChunk = (chunkStart + chunkSize == numPromptTokens);
                int totalFloats = chunkSize * Dim;

                ReadOnlySpan<int> chunkTokens = promptTokens.AsSpan(chunkStart, chunkSize);

                // Local stages with non-aliasing ping-pong buffers
                for (int l = 0; l < LocalModels.Count; l++)
                {
                    bool isLastStage = (l == numStages - 1);
                    bool computeLogits = isLastStage && isLastChunk;

                    var swLocal = Stopwatch.StartNew();
                    LocalModels[l].ForwardBatchStage(
                        l == 0 ? chunkTokens : ReadOnlySpan<int>.Empty,
                        currentPos,
                        l == 0 ? ReadOnlySpan<float>.Empty : curBatch.AsSpan(0, totalFloats),
                        isLastStage ? Span<float>.Empty : nextBatch.AsSpan(0, totalFloats),
                        computeLogits ? logits.AsSpan(0, VocabSize) : Span<float>.Empty,
                        computeLogits);
                    swLocal.Stop();

                    if (!isLastStage)
                    {
                        (curBatch, nextBatch) = (nextBatch, curBatch);
                    }

                    Stages[l].TotalComputeMs += swLocal.Elapsed.TotalMilliseconds;
                    Stages[l].Invocations += chunkSize;
                }

                // Remote stages (atomic send + receive per chunk to preserve persistent worker TCP framing)
                for (int s = LocalModels.Count; s < numStages; s++)
                {
                    bool isFirstStage = (s == 0);
                    bool isLastStage = (s == numStages - 1);
                    bool computeLogits = isLastStage && isLastChunk;

                    int streamIdx = s - LocalModels.Count;
                    var stream = Streams[streamIdx];
                    var stInfo = Stages[s];

                    long tStart = Stopwatch.GetTimestamp();
                    WirePrecisionMode prefillPrec = _precisionSwitcher.SelectPrecision(chunkSize, currentPos, isDecode: false);
                    float uniform01 = computeLogits && !(options.Temperature <= Sampler.GreedyTemperatureThreshold || options.TopK == 1)
                        ? (float)Sampler.Rng.NextDouble()
                        : 0f;

                    await SendPrefillStageFrameAsync(
                        stream,
                        corrId: (ulong)chunkStart,
                        numTokens: chunkSize,
                        pos: currentPos,
                        stageIndex: s,
                        isFirstStage: isFirstStage,
                        computeLogits: computeLogits,
                        dim: Dim,
                        chunkTokens: isFirstStage ? promptTokens.AsMemory(chunkStart, chunkSize) : ReadOnlyMemory<int>.Empty,
                        actFloats: curBatch,
                        sendBuf: _batchArena.BatchWireSendArray,
                        precisionMode: prefillPrec,
                        samplingOptions: computeLogits ? options : null,
                        recentTokens: computeLogits ? promptTokens.AsMemory() : ReadOnlyMemory<int>.Empty,
                        randomUniform01: uniform01,
                        ct: CancellationToken.None).ConfigureAwait(false);

                    var res = await ReadResultBatchAsync(
                        stream,
                        chunkSize,
                        Dim,
                        isLastStage ? null : nextBatch,
                        _batchArena.BatchWireRecvArray,
                        _frameHdrScratch,
                        CancellationToken.None).ConfigureAwait(false);

                    long tEnd = Stopwatch.GetTimestamp();

                    double rttMs = (tEnd - tStart) * 1000.0 / Stopwatch.Frequency;
                    double netMs = Math.Max(0.01, rttMs - res.ComputeMs);
                    _precisionSwitcher.UpdateMeasuredLatency(netMs);

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

            ct.ThrowIfCancellationRequested();

            // Sample first token if last stage is local (using identical 256-token recent window)
            if (LocalModels.Count == numStages)
            {
                ReadOnlySpan<int> promptWindow = promptTokens.Length > SamplingWireCodec.MaxRecentTokens
                    ? promptTokens.AsSpan(promptTokens.Length - SamplingWireCodec.MaxRecentTokens, SamplingWireCodec.MaxRecentTokens)
                    : promptTokens.AsSpan();
                nextTokenId = Sampler.Sample(logits.AsSpan(0, VocabSize), options, promptWindow);
            }

            // 2. Autoregressive Generation
            var swDecode = Stopwatch.StartNew();
            var sb = new StringBuilder();
            int generatedCount = 0;
            string finishReason = "stop";

            int maxGen = options.MaxTokens <= 0 ? MaxSeqLen : options.MaxTokens;
            var recentTokens = new List<int>(promptTokens.Length + Math.Min(maxGen, 512));
            recentTokens.AddRange(promptTokens);

            if (nextTokenId == Tokenizer.EosTokenId || Tokenizer.IsStopToken(nextTokenId))
            {
                swDecode.Stop();
                swTotal.Stop();
                return new ClusterGenerationResult
                {
                    Text = "",
                    PromptTokens = promptTokens.Length,
                    GeneratedTokens = 0,
                    PromptDuration = swPrefill.Elapsed,
                    GenerationDuration = swDecode.Elapsed,
                    TotalDuration = swTotal.Elapsed,
                    FinishReason = "stop"
                };
            }

            string firstPiece = Tokenizer.DecodeToken(nextTokenId);
            sb.Append(firstPiece);
            onToken?.Invoke(firstPiece);
            recentTokens.Add(nextTokenId);
            generatedCount++;

            if (generatedCount >= maxGen)
            {
                finishReason = "length";
            }

            while (generatedCount < maxGen)
            {
                // Honor cancellation strictly BETWEEN complete token round-trips so worker TCP streams remain synchronized
                ct.ThrowIfCancellationRequested();

                if (currentPos >= MaxSeqLen)
                {
                    finishReason = "length";
                    break;
                }

                float[] curIn = _decodeBufA;
                float[] curOut = _decodeBufB;

                // Local stages with strict ping-pong non-aliasing buffers
                for (int l = 0; l < LocalModels.Count; l++)
                {
                    bool isLastStage = (l == numStages - 1);
                    var swLocal = Stopwatch.StartNew();
                    LocalModels[l].ForwardStage(
                        nextTokenId,
                        currentPos,
                        l == 0 ? ReadOnlySpan<float>.Empty : curIn.AsSpan(0, Dim),
                        isLastStage ? Span<float>.Empty : curOut.AsSpan(0, Dim),
                        isLastStage ? logits.AsSpan(0, VocabSize) : Span<float>.Empty,
                        computeLogits: isLastStage);
                    swLocal.Stop();

                    Stages[l].TotalComputeMs += swLocal.Elapsed.TotalMilliseconds;
                    Stages[l].Invocations++;

                    if (!isLastStage)
                    {
                        (curIn, curOut) = (curOut, curIn);
                    }
                }

                // Remote stages (atomic send + receive per token step)
                for (int s = LocalModels.Count; s < numStages; s++)
                {
                    bool isFirstStage = (s == 0);
                    bool isLastStage = (s == numStages - 1);
                    int streamIdx = s - LocalModels.Count;
                    var stream = Streams[streamIdx];
                    var stInfo = Stages[s];

                    long tStart = Stopwatch.GetTimestamp();
                    WirePrecisionMode decodePrec = _precisionSwitcher.SelectPrecision(1, currentPos, isDecode: true);
                    float uniform01 = isLastStage && !(options.Temperature <= Sampler.GreedyTemperatureThreshold || options.TopK == 1)
                        ? (float)Sampler.Rng.NextDouble()
                        : 0f;

                    await SendDecodeStageFrameAsync(
                        stream,
                        corrId: (ulong)generatedCount,
                        pos: currentPos,
                        stageIndex: s,
                        isFirstStage: isFirstStage,
                        isLastStage: isLastStage,
                        dim: Dim,
                        tokenId: nextTokenId,
                        actFloats: curIn,
                        sendBuf: _batchArena.BatchWireSendArray,
                        precisionMode: decodePrec,
                        samplingOptions: isLastStage ? options : null,
                        recentTokens: isLastStage ? recentTokens : null,
                        randomUniform01: uniform01,
                        ct: CancellationToken.None).ConfigureAwait(false);

                    var res = await ReadResultFusedAsync(
                        stream,
                        Dim,
                        isLastStage ? null : curOut,
                        _batchArena.BatchWireRecvArray,
                        _frameHdrScratch,
                        CancellationToken.None).ConfigureAwait(false);

                    long tEnd = Stopwatch.GetTimestamp();

                    double rttMs = (tEnd - tStart) * 1000.0 / Stopwatch.Frequency;
                    double netMs = Math.Max(0.01, rttMs - res.ComputeMs);
                    _precisionSwitcher.UpdateMeasuredLatency(netMs);

                    stInfo.TotalComputeMs += res.ComputeMs;
                    stInfo.TotalNetworkMs += netMs;
                    stInfo.Invocations++;

                    if (isLastStage)
                    {
                        nextTokenId = res.NextTokenId;
                    }
                    else
                    {
                        (curIn, curOut) = (curOut, curIn);
                    }
                }

                if (LocalModels.Count == numStages)
                {
                    ReadOnlySpan<int> allRecent = CollectionsMarshal.AsSpan(recentTokens);
                    ReadOnlySpan<int> recentWindow = allRecent.Length > SamplingWireCodec.MaxRecentTokens
                        ? allRecent.Slice(allRecent.Length - SamplingWireCodec.MaxRecentTokens, SamplingWireCodec.MaxRecentTokens)
                        : allRecent;
                    nextTokenId = Sampler.Sample(logits.AsSpan(0, VocabSize), options, recentWindow);
                }

                currentPos++;

                if (nextTokenId == Tokenizer.EosTokenId || Tokenizer.IsStopToken(nextTokenId))
                {
                    finishReason = "stop";
                    break;
                }

                recentTokens.Add(nextTokenId);
                string piece = Tokenizer.DecodeToken(nextTokenId);
                sb.Append(piece);
                onToken?.Invoke(piece);
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

    private static async Task SendPrefillStageFrameAsync(
        Stream stream,
        ulong corrId,
        int numTokens,
        int pos,
        int stageIndex,
        bool isFirstStage,
        bool computeLogits,
        int dim,
        ReadOnlyMemory<int> chunkTokens,
        float[] actFloats,
        byte[] sendBuf,
        WirePrecisionMode precisionMode,
        SamplingOptions? samplingOptions,
        ReadOnlyMemory<int> recentTokens,
        float randomUniform01,
        CancellationToken ct)
    {
        int dataBytes = isFirstStage
            ? numTokens * sizeof(int)
            : QuantizationKernels.GetPackedBatchBytes(precisionMode, dim, numTokens);

        int trailerOffset = ClusterFrameHeader.HeaderSize + FusedActivationHeader.HeaderSize + dataBytes;
        int trailerBytes = 0;

        if (computeLogits && samplingOptions != null)
        {
            trailerBytes = SamplingWireCodec.WriteTrailer(
                sendBuf.AsSpan(trailerOffset),
                samplingOptions,
                recentTokens.Span,
                randomUniform01);
        }

        FusedActivationFlags flags = FusedActivationFlags.IsPrefillChunk;
        if (computeLogits) flags |= FusedActivationFlags.FinalLayer;
        if (isFirstStage) flags |= FusedActivationFlags.Stage0TokenInput;
        if (trailerBytes > 0) flags |= FusedActivationFlags.HasSamplingTrailer;

        int payloadLen = FusedActivationHeader.HeaderSize + dataBytes + trailerBytes;
        int totalFrameBytes = ClusterFrameHeader.HeaderSize + payloadLen;

        var clusterHdr = new ClusterFrameHeader(MsgActivationBatch, corrId, (ulong)payloadLen);
        var actHdr = new FusedActivationHeader(
            sequenceId: corrId,
            tokenIndex: pos,
            layerIndex: stageIndex,
            hiddenDim: dim,
            precisionMode: precisionMode,
            flags: flags,
            scale: 1.0f,
            batchTokens: (ushort)numTokens);

        clusterHdr.WriteTo(sendBuf.AsSpan(0, ClusterFrameHeader.HeaderSize));
        actHdr.WriteTo(sendBuf.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationHeader.HeaderSize));

        var dataSpan = sendBuf.AsSpan(FusedSocketTransport.CombinedHeaderSize, dataBytes);
        if (isFirstStage)
        {
            ReadOnlySpan<int> toks = chunkTokens.Span;
            for (int i = 0; i < numTokens; i++)
            {
                BinaryPrimitives.WriteInt32LittleEndian(dataSpan.Slice(i * sizeof(int), sizeof(int)), toks[i]);
            }
        }
        else
        {
            QuantizationKernels.QuantizeBatch(actFloats.AsSpan(0, numTokens * dim), dataSpan, dim, numTokens, precisionMode);
        }

        await stream.WriteAsync(sendBuf.AsMemory(0, totalFrameBytes), ct).ConfigureAwait(false);
        await stream.FlushAsync(ct).ConfigureAwait(false);
    }

    private static async Task SendDecodeStageFrameAsync(
        Stream stream,
        ulong corrId,
        int pos,
        int stageIndex,
        bool isFirstStage,
        bool isLastStage,
        int dim,
        int tokenId,
        float[] actFloats,
        byte[] sendBuf,
        WirePrecisionMode precisionMode,
        SamplingOptions? samplingOptions,
        List<int>? recentTokens,
        float randomUniform01,
        CancellationToken ct)
    {
        int dataBytes = isFirstStage
            ? sizeof(int)
            : QuantizationKernels.GetPackedTokenBytes(precisionMode, dim);

        int trailerOffset = FusedSocketTransport.CombinedHeaderSize + dataBytes;
        int trailerBytes = 0;

        if (isLastStage && samplingOptions != null)
        {
            ReadOnlySpan<int> recentSpan = recentTokens != null ? CollectionsMarshal.AsSpan(recentTokens) : ReadOnlySpan<int>.Empty;
            trailerBytes = SamplingWireCodec.WriteTrailer(
                sendBuf.AsSpan(trailerOffset),
                samplingOptions,
                recentSpan,
                randomUniform01);
        }

        FusedActivationFlags flags = FusedActivationFlags.None;
        if (isLastStage) flags |= FusedActivationFlags.FinalLayer;
        if (isFirstStage) flags |= FusedActivationFlags.Stage0TokenInput;
        if (trailerBytes > 0) flags |= FusedActivationFlags.HasSamplingTrailer;

        int payloadLen = FusedActivationHeader.HeaderSize + dataBytes + trailerBytes;
        int totalFrameBytes = ClusterFrameHeader.HeaderSize + payloadLen;

        var clusterHdr = new ClusterFrameHeader(MsgActivation, corrId, (ulong)payloadLen);
        var actHdr = new FusedActivationHeader(
            sequenceId: corrId,
            tokenIndex: pos,
            layerIndex: stageIndex,
            hiddenDim: dim,
            precisionMode: precisionMode,
            flags: flags,
            scale: 1.0f,
            batchTokens: 1);

        clusterHdr.WriteTo(sendBuf.AsSpan(0, ClusterFrameHeader.HeaderSize));
        actHdr.WriteTo(sendBuf.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationHeader.HeaderSize));

        var dataSpan = sendBuf.AsSpan(FusedSocketTransport.CombinedHeaderSize, dataBytes);
        if (isFirstStage)
        {
            BinaryPrimitives.WriteInt32LittleEndian(dataSpan, tokenId);
        }
        else
        {
            QuantizationKernels.QuantizeToken(actFloats.AsSpan(0, dim), dataSpan, precisionMode);
        }

        await stream.WriteAsync(sendBuf.AsMemory(0, totalFrameBytes), ct).ConfigureAwait(false);
        await stream.FlushAsync(ct).ConfigureAwait(false);
    }

    public static async Task SendActivationAsync(NetworkStream stream, ulong corrId, int step, int pos, bool isLastStage, int dim, float[] actFloats)
    {
        int actByteCount = QuantizationKernels.GetPackedTokenBytes(WirePrecisionMode.FP32, dim);
        int payloadLen = FusedActivationHeader.HeaderSize + actByteCount;
        int totalBytes = ClusterFrameHeader.HeaderSize + payloadLen;
        byte[] frame = ArrayPool<byte>.Shared.Rent(totalBytes);
        try
        {
            var clusterHdr = new ClusterFrameHeader(MsgActivation, corrId, (ulong)payloadLen);
            var actHdr = new FusedActivationHeader(
                sequenceId: corrId,
                tokenIndex: pos,
                layerIndex: step,
                hiddenDim: dim,
                precisionMode: WirePrecisionMode.FP32,
                flags: isLastStage ? FusedActivationFlags.FinalLayer : FusedActivationFlags.None,
                scale: 1.0f,
                batchTokens: 1);

            clusterHdr.WriteTo(frame.AsSpan(0, ClusterFrameHeader.HeaderSize));
            actHdr.WriteTo(frame.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationHeader.HeaderSize));
            QuantizationKernels.QuantizeToken(actFloats.AsSpan(0, dim), frame.AsSpan(FusedSocketTransport.CombinedHeaderSize, actByteCount), WirePrecisionMode.FP32);

            await stream.WriteAsync(frame.AsMemory(0, totalBytes)).ConfigureAwait(false);
            await stream.FlushAsync().ConfigureAwait(false);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(frame);
        }
    }

    public static async Task<(double ComputeMs, int NextTokenId, float TopLogit)> ReadResultAsync(NetworkStream stream, int dim, float[] outFloats)
    {
        byte[] hdrScratch = ArrayPool<byte>.Shared.Rent(ClusterFrameHeader.HeaderSize);
        byte[] recvScratch = ArrayPool<byte>.Shared.Rent(FusedActivationResultHeader.HeaderSize + (dim * sizeof(float)) + 64);
        try
        {
            return await ReadResultFusedAsync(stream, dim, outFloats, recvScratch, hdrScratch, CancellationToken.None).ConfigureAwait(false);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(recvScratch);
            ArrayPool<byte>.Shared.Return(hdrScratch);
        }
    }

    private static async Task<(double ComputeMs, int NextTokenId, float TopLogit)> ReadResultFusedAsync(
        Stream stream,
        int dim,
        float[]? outFloats,
        byte[] recvBuf,
        byte[] hdrBytes,
        CancellationToken ct)
    {
        ClusterFrameHeader hdr;
        while (true)
        {
            await ReadExactAsync(stream, hdrBytes, 0, ClusterFrameHeader.HeaderSize, ct).ConfigureAwait(false);
            hdr = MemoryMarshal.Read<ClusterFrameHeader>(hdrBytes.AsSpan(0, ClusterFrameHeader.HeaderSize));
            if (hdr.PayloadLength > FusedSocketTransport.MaxAllowedPayloadBytes)
                throw new InvalidDataException($"Remote worker returned oversized MsgResult payload ({hdr.PayloadLength} bytes).");

            if (hdr.MessageType == MsgResult)
            {
                break;
            }

            if (hdr.PayloadLength > 0)
            {
                int discardLen = (int)hdr.PayloadLength;
                byte[] discard = ArrayPool<byte>.Shared.Rent(discardLen);
                try { await ReadExactAsync(stream, discard, 0, discardLen, ct).ConfigureAwait(false); }
                finally { ArrayPool<byte>.Shared.Return(discard); }
            }
        }

        int payloadLen = (int)hdr.PayloadLength;
        byte[]? rented = null;
        byte[] targetBuf = payloadLen <= recvBuf.Length
            ? recvBuf
            : (rented = ArrayPool<byte>.Shared.Rent(payloadLen));

        try
        {
            if (payloadLen > 0)
            {
                await ReadExactAsync(stream, targetBuf, 0, payloadLen, ct).ConfigureAwait(false);
            }

            if (payloadLen >= FusedActivationResultHeader.HeaderSize &&
                BinaryPrimitives.ReadUInt32LittleEndian(targetBuf.AsSpan(0, 4)) == FusedActivationResultHeader.ResultMagic)
            {
                var resHdr = FusedActivationResultHeader.ReadFrom(targetBuf.AsSpan(0, FusedActivationResultHeader.HeaderSize));
                int actBytes = payloadLen - FusedActivationResultHeader.HeaderSize;
                if (outFloats != null && actBytes > 0)
                {
                    var actSpan = targetBuf.AsSpan(FusedActivationResultHeader.HeaderSize, actBytes);
                    QuantizationKernels.DequantizeToken(actSpan, outFloats.AsSpan(0, dim), resHdr.TypedPrecision);
                }
                return (resHdr.ComputeMs, resHdr.NextTokenId, resHdr.TopLogit);
            }

            double computeMs = payloadLen >= 16 ? BitConverter.ToDouble(targetBuf, 8) : 0.0;
            int nextTokenId = payloadLen >= 20 ? BitConverter.ToInt32(targetBuf, 16) : 0;
            float topLogit = payloadLen >= 24 ? BitConverter.ToSingle(targetBuf, 20) : 0f;
            int actDim = payloadLen >= 28 ? BitConverter.ToInt32(targetBuf, 24) : 0;

            int actByteCount = actDim * sizeof(float);
            if (outFloats != null && actDim > 0 && payloadLen >= 28 + actByteCount)
            {
                var actSpan = targetBuf.AsSpan(28, actByteCount);
                MemoryMarshal.Cast<byte, float>(actSpan).CopyTo(outFloats.AsSpan(0, Math.Min(actDim, dim)));
            }

            return (computeMs, nextTokenId, topLogit);
        }
        finally
        {
            if (rented != null)
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }
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
        int totalFrameBytes = ClusterFrameHeader.HeaderSize + payloadLen;

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

        if (batchBuf.Length >= totalFrameBytes)
        {
            clusterHdr.WriteTo(batchBuf.AsSpan(0, ClusterFrameHeader.HeaderSize));
            actHdr.WriteTo(batchBuf.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationHeader.HeaderSize));
            QuantizationKernels.QuantizeBatch(
                actFloats.AsSpan(0, totalFloats),
                batchBuf.AsSpan(FusedSocketTransport.CombinedHeaderSize, actByteCount),
                dim,
                numTokens,
                precisionMode);

            await stream.WriteAsync(batchBuf.AsMemory(0, totalFrameBytes)).ConfigureAwait(false);
            await stream.FlushAsync().ConfigureAwait(false);
            return;
        }

        byte[] hdrBytes = ArrayPool<byte>.Shared.Rent(ClusterFrameHeader.HeaderSize);
        try
        {
            clusterHdr.WriteTo(hdrBytes.AsSpan(0, ClusterFrameHeader.HeaderSize));
            actHdr.WriteTo(batchBuf.AsSpan(0, FusedActivationHeader.HeaderSize));
            var payload = batchBuf.AsSpan(FusedActivationHeader.HeaderSize, actByteCount);
            QuantizationKernels.QuantizeBatch(actFloats.AsSpan(0, totalFloats), payload, dim, numTokens, precisionMode);

            await stream.WriteAsync(hdrBytes.AsMemory(0, ClusterFrameHeader.HeaderSize)).ConfigureAwait(false);
            await stream.WriteAsync(batchBuf.AsMemory(0, payloadLen)).ConfigureAwait(false);
            await stream.FlushAsync().ConfigureAwait(false);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(hdrBytes);
        }
    }

    public static async Task<(double ComputeMs, int NextTokenId)> ReadResultBatchAsync(
        NetworkStream stream,
        int numTokens,
        int dim,
        float[]? outFloatsBatch,
        byte[] batchRecv)
    {
        byte[] hdrScratch = ArrayPool<byte>.Shared.Rent(ClusterFrameHeader.HeaderSize);
        try
        {
            return await ReadResultBatchAsync(stream, numTokens, dim, outFloatsBatch, batchRecv, hdrScratch, CancellationToken.None).ConfigureAwait(false);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(hdrScratch);
        }
    }

    private static async Task<(double ComputeMs, int NextTokenId)> ReadResultBatchAsync(
        Stream stream,
        int numTokens,
        int dim,
        float[]? outFloatsBatch,
        byte[] batchRecv,
        byte[] hdrBytes,
        CancellationToken ct)
    {
        ClusterFrameHeader clusterHdr;
        while (true)
        {
            await ReadExactAsync(stream, hdrBytes, 0, ClusterFrameHeader.HeaderSize, ct).ConfigureAwait(false);
            clusterHdr = MemoryMarshal.Read<ClusterFrameHeader>(hdrBytes.AsSpan(0, ClusterFrameHeader.HeaderSize));
            if (clusterHdr.PayloadLength > FusedSocketTransport.MaxAllowedPayloadBytes)
                throw new InvalidDataException($"Remote worker returned oversized MsgResultBatch payload ({clusterHdr.PayloadLength} bytes).");

            if (clusterHdr.MessageType == MsgResultBatch || clusterHdr.MessageType == MsgResult)
            {
                break;
            }

            if (clusterHdr.PayloadLength > 0)
            {
                int discardLen = (int)clusterHdr.PayloadLength;
                byte[] discard = ArrayPool<byte>.Shared.Rent(discardLen);
                try { await ReadExactAsync(stream, discard, 0, discardLen, ct).ConfigureAwait(false); }
                finally { ArrayPool<byte>.Shared.Return(discard); }
            }
        }

        int payloadLen = (int)clusterHdr.PayloadLength;
        byte[]? rented = null;
        byte[] targetBuf = payloadLen <= batchRecv.Length
            ? batchRecv
            : (rented = ArrayPool<byte>.Shared.Rent(payloadLen));

        try
        {
            if (payloadLen > 0)
            {
                await ReadExactAsync(stream, targetBuf, 0, payloadLen, ct).ConfigureAwait(false);
            }

            uint resultMagic = payloadLen >= 4 ? BinaryPrimitives.ReadUInt32LittleEndian(targetBuf.AsSpan(0, 4)) : 0;
            double computeMs = 0;
            int nextTokenId = 0;

            if (resultMagic == FusedActivationResultHeader.ResultMagic && payloadLen >= FusedActivationResultHeader.HeaderSize)
            {
                var resultHdr = FusedActivationResultHeader.ReadFrom(targetBuf.AsSpan(0, FusedActivationResultHeader.HeaderSize));
                computeMs = resultHdr.ComputeMs;
                nextTokenId = resultHdr.NextTokenId;
                WirePrecisionMode resPrec = resultHdr.TypedPrecision;
                int actByteCount = payloadLen - FusedActivationResultHeader.HeaderSize;

                if (outFloatsBatch != null && actByteCount > 0)
                {
                    var payloadSpan = targetBuf.AsSpan(FusedActivationResultHeader.HeaderSize, actByteCount);
                    QuantizationKernels.DequantizeBatch(payloadSpan, outFloatsBatch.AsSpan(0, numTokens * dim), dim, numTokens, resPrec);
                }
            }
            else
            {
                var legacyPayload = targetBuf.AsSpan(0, payloadLen);
                if (payloadLen >= 20)
                {
                    computeMs = BitConverter.ToDouble(legacyPayload.Slice(8, 8));
                    nextTokenId = BitConverter.ToInt32(legacyPayload.Slice(16, 4));
                }
                int actByteCount = payloadLen - 28;
                int totalFloats = numTokens * dim;
                if (outFloatsBatch != null && actByteCount > 0)
                {
                    var payloadSpan = targetBuf.AsSpan(28, actByteCount);
                    if (actByteCount == totalFloats * 2)
                    {
                        QuantizationKernels.UnpackFp16(payloadSpan, outFloatsBatch.AsSpan(0, totalFloats));
                    }
                    else
                    {
                        MemoryMarshal.Cast<byte, float>(payloadSpan).CopyTo(outFloatsBatch.AsSpan(0, totalFloats));
                    }
                }
            }

            return (computeMs, nextTokenId);
        }
        finally
        {
            if (rented != null)
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }
    }

    public static Task<int> ReadExactAsync(Stream s, byte[] buf, int offset, int count)
        => ReadExactAsync(s, buf, offset, count, CancellationToken.None);

    public static async Task<int> ReadExactAsync(Stream s, byte[] buf, int offset, int count, CancellationToken ct)
    {
        int total = 0;
        while (total < count)
        {
            int r = await s.ReadAsync(buf.AsMemory(offset + total, count - total), ct).ConfigureAwait(false);
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
                shutHdr.WriteTo(_frameHdrScratch.AsSpan(0, ClusterFrameHeader.HeaderSize));
                Streams[s].Write(_frameHdrScratch, 0, ClusterFrameHeader.HeaderSize);
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
        _batchArena.Dispose();
    }
}
