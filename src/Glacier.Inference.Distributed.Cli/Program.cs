namespace Glacier.Inference.Distributed.Cli;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Inference.Gguf;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Sampling;
using Glacier.Inference.Tokenizer;

public record StageEndpoint(string Host, int Port, string DisplayName);

public class StageTelemetry
{
    public string Name { get; set; } = "";
    public string Host { get; set; } = "";
    public int Port { get; set; }
    public int StartLayer { get; set; }
    public int EndLayer { get; set; }
    public int LayerCount => Math.Max(0, EndLayer - StartLayer + 1);
    public double TotalComputeMs { get; set; }
    public double TotalNetworkMs { get; set; }
    public int Invocations { get; set; }
}

public interface IClusterStageModel : IDisposable
{
    string BackendName { get; }
    void ForwardStage(int token, int pos, ReadOnlySpan<float> inputX, Span<float> outputX, Span<float> logits, bool computeLogits);
    void ResetKvCache();
}

public sealed class CpuClusterStageModel : IClusterStageModel
{
    private readonly ICpuModel _model;
    private readonly KVCache _kv;
    public string BackendName { get; }

    public CpuClusterStageModel(ICpuModel model, KVCache kv, string backendName)
    {
        _model = model;
        _kv = kv;
        BackendName = backendName;
    }

    public void ForwardStage(int token, int pos, ReadOnlySpan<float> inputX, Span<float> outputX, Span<float> logits, bool computeLogits)
    {
        _model.ForwardStage(token, pos, inputX, outputX, logits, computeLogits, _kv);
    }

    public void ResetKvCache()
    {
        _kv.Reset();
    }

    public void Dispose()
    {
        _model.Dispose();
        _kv.Dispose();
    }
}

public sealed class D3D12ClusterStageModel : IClusterStageModel
{
    private readonly Qwen2D3D12Model _model;
    private readonly D3D12Context _ctx;
    public string BackendName { get; }

    public D3D12ClusterStageModel(D3D12Context ctx, Qwen2D3D12Model model, string backendName)
    {
        _ctx = ctx;
        _model = model;
        BackendName = backendName;
    }

    public void ForwardStage(int token, int pos, ReadOnlySpan<float> inputX, Span<float> outputX, Span<float> logits, bool computeLogits)
    {
        _model.ForwardStage(token, pos, inputX, outputX, logits, computeLogits);
    }

    public void ResetKvCache()
    {
        _model.ResetKvCache();
    }

    public void Dispose()
    {
        _model.Dispose();
        _ctx.Dispose();
    }
}

public static class Program
{
    private const int DefaultPort = 5055;

    // Cluster wire message opcodes for distributed LLM inference
    private const ClusterMessageType MsgInitStage   = (ClusterMessageType)0x60;
    private const ClusterMessageType MsgStageReady  = (ClusterMessageType)0x61;
    private const ClusterMessageType MsgStreamChunk = (ClusterMessageType)0x62;
    private const ClusterMessageType MsgStreamDone  = (ClusterMessageType)0x63;
    private const ClusterMessageType MsgActivation  = (ClusterMessageType)0x70;
    private const ClusterMessageType MsgResult      = (ClusterMessageType)0x71;
    private const ClusterMessageType MsgResetKv     = (ClusterMessageType)0x72;
    private const ClusterMessageType MsgShutdown    = (ClusterMessageType)0x73;

    public static async Task<int> Main(string[] args)
    {
        string role = "coordinator";
        int port = DefaultPort;
        int maxTokens = -1; // Unlimited by default (generates until natural EOS)
        string? modelPath = null;
        string? downloadTarget = null;
        string? destDir = null;
        string? cacheDir = null;
        int downloadStreams = 8;
        string prompt = "Explain quantum computers to a layman.";
        string stagesArg = "192.168.1.108:5055,192.168.1.101:5056,192.168.1.200:5057";
        string? layerAlloc = null;
        int hostGpu = -1;
        bool interactive = false;

        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--help" || args[i] == "-h" || args[i] == "help")
            {
                PrintUsage();
                return 0;
            }
            if ((args[i] == "download" || args[i] == "--download") && i + 1 < args.Length)
            {
                downloadTarget = args[++i];
            }
            else if (args[i] == "--dest" && i + 1 < args.Length) destDir = args[++i];
            else if (args[i] == "--cache" && i + 1 < args.Length) cacheDir = args[++i];
            else if (args[i] == "--streams" && i + 1 < args.Length && int.TryParse(args[++i], out int s)) downloadStreams = s;
            else if (args[i] == "--role" && i + 1 < args.Length) role = args[++i].ToLowerInvariant();
            else if (args[i] == "--port" && i + 1 < args.Length && int.TryParse(args[++i], out int p)) port = p;
            else if (args[i] == "--tokens" && i + 1 < args.Length && int.TryParse(args[++i], out int t)) maxTokens = t;
            else if (args[i] == "--model" && i + 1 < args.Length) modelPath = args[++i];
            else if (args[i] == "--prompt" && i + 1 < args.Length) prompt = args[++i];
            else if ((args[i] == "--stages" || args[i] == "--nodes") && i + 1 < args.Length) stagesArg = args[++i];
            else if ((args[i] == "--layers" || args[i] == "--alloc") && i + 1 < args.Length) layerAlloc = args[++i];
            else if ((args[i] == "--host-gpu" || args[i] == "--adapter") && i + 1 < args.Length && int.TryParse(args[++i], out int hg)) hostGpu = hg;
            else if (args[i] == "--interactive" || args[i] == "-i" || args[i] == "chat") interactive = true;
        }

        // If user invoked model downloader
        if (!string.IsNullOrEmpty(downloadTarget))
        {
            return await ModelDownloader.DownloadModelAsync(downloadTarget, destDir, downloadStreams);
        }

        // Default model path fallback search
        if (string.IsNullOrEmpty(modelPath) && role == "coordinator")
        {
            string defaultQwen = Path.Combine(AppContext.BaseDirectory, "models", "qwen2-57b-a14b-instruct-q3_k_m.gguf");
            string parentQwen = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "Glacier.Inference", "models", "qwen2-57b-a14b-instruct-q3_k_m.gguf"));
            if (File.Exists(defaultQwen)) modelPath = defaultQwen;
            else if (File.Exists(parentQwen)) modelPath = parentQwen;
        }

        cacheDir ??= Path.Combine(AppContext.BaseDirectory, "models", "cache");
        if (!Directory.Exists(cacheDir)) Directory.CreateDirectory(cacheDir);

        Console.WriteLine("================================================================================");
        Console.WriteLine("  GLACIER HETEROGENEOUS CLUSTER LLM RUNTIME (PURE C# .NET 10)");
        Console.WriteLine($"  Role: {role.ToUpperInvariant()} | Machine: {Environment.MachineName}");
        Console.WriteLine($"  OS: {RuntimeInformation.OSDescription} ({RuntimeInformation.ProcessArchitecture})");
        Console.WriteLine($"  Hardware Threads: {Environment.ProcessorCount} Cores | Vector512: {System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated}");
        Console.WriteLine("================================================================================\n");

        if (role == "worker")
        {
            return await RunWorkerAsync(port, cacheDir);
        }
        else
        {
            if (string.IsNullOrEmpty(modelPath) || !File.Exists(modelPath))
            {
                Console.WriteLine($"[Error] Model file not found: {modelPath ?? "(none)"}");
                Console.WriteLine("Please specify --model <path_to_gguf>");
                return 1;
            }

            var endpoints = ParseEndpoints(stagesArg);
            return await RunCoordinatorAsync(endpoints, modelPath, prompt, maxTokens, cacheDir, interactive, layerAlloc, hostGpu);
        }
    }

    private static void PrintUsage()
    {
        Console.WriteLine(@"Glacier Distributed Cluster CLI (glacier-distributed) - Pure C# .NET 10

Usage:
  glacier-distributed download <preset|url> [--dest <dir>] [--streams <int>]
  glacier-distributed --role worker [--port <int>] [--cache <dir>]
  glacier-distributed --role coordinator --model <path> [options]

Coordinator Options:
  --model <path>                  Path to GGUF model file (e.g. Qwen2-57B or Mixtral-8x22B)
  --stages <ip:port,...>          Comma-separated list of candidate worker endpoints
  --layers <auto|list>            Layer allocation policy (e.g. 'auto' or '8,14,6' or '7,7,7,7')
  --host-gpu <int>                Direct3D 12 adapter index for Host (default: auto fastest VRAM)
  --prompt <string>               Text prompt for inference generation
  --tokens <int>                  Maximum new tokens to generate (default: -1 unlimited)
  --interactive, -i               Keep session alive for multi-turn chat
  --cache <dir>                   Directory to store stage slices (default: ./models/cache)

Worker Options:
  --port <int>                    TCP port to listen on (default: 5055)
  --cache <dir>                   Directory to store received stage slices (default: ./models/cache)
");
    }

    private static List<StageEndpoint> ParseEndpoints(string stagesArg)
    {
        var list = new List<StageEndpoint>();
        if (string.IsNullOrWhiteSpace(stagesArg)) return list;

        var parts = stagesArg.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        int idx = 1;
        foreach (var p in parts)
        {
            string host = p;
            int port = DefaultPort;
            if (p.Contains(':'))
            {
                var hp = p.Split(':');
                host = hp[0];
                int.TryParse(hp[1], out port);
            }

            string name = host switch
            {
                "192.168.1.108" => "Beast (RTX 3060 12GB)",
                "192.168.1.101" => "MinisForum64 (Radeon 680M)",
                "192.168.1.200" => "Fedora Linux (Zen 5 AVX-512)",
                "127.0.0.1" or "localhost" => "Localhost Worker",
                _ => $"Node-{idx} ({host})"
            };

            list.Add(new StageEndpoint(host, port, name));
            idx++;
        }
        return list;
    }

    // =========================================================================
    // WORKER DAEMON IMPLEMENTATION
    // =========================================================================
    private static async Task<int> RunWorkerAsync(int port, string cacheDir)
    {
        Console.WriteLine($"[Worker] Starting Glacier Worker Daemon on port {port}...");
        Console.WriteLine($"[Worker] Cache directory: {Path.GetFullPath(cacheDir)}");

        var listener = new TcpListener(IPAddress.Any, port);
        listener.Server.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.NoDelay, true);
        listener.Start();
        Console.WriteLine($"[Worker] Listening on 0.0.0.0:{port} (NoDelay=true). Awaiting coordinator...\n");

        byte[] headerBytes = new byte[ClusterFrameHeader.HeaderSize];

        while (true)
        {
            try
            {
                using var client = await listener.AcceptTcpClientAsync();
                client.NoDelay = true;
                client.ReceiveBufferSize = 4 * 1024 * 1024;
                client.SendBufferSize = 4 * 1024 * 1024;
                Console.WriteLine($"[Worker] Coordinator connected from {client.Client.RemoteEndPoint}");

                using var stream = client.GetStream();
                GgufFile? stageGguf = null;
                ModelWeights? stageWeights = null;
                IClusterStageModel? stageModel = null;
                int currentStageId = -1;
                int startLayer = 0;
                int endLayer = 0;
                int layerCount = 0;
                int dim = 0;
                bool isLastStage = false;

                try
                {
                    while (true)
                    {
                        int read = await ReadExactAsync(stream, headerBytes, 0, ClusterFrameHeader.HeaderSize);
                        if (read == 0) break;

                        var header = MemoryMarshal.Read<ClusterFrameHeader>(headerBytes);
                        if (header.Magic != ClusterFrameHeader.MagicConstant)
                        {
                            Console.WriteLine($"[Worker] Invalid frame magic: 0x{header.Magic:X8}");
                            break;
                        }

                        // -------------------------------------------------------------
                        // 1. STAGE INITIALIZATION & CACHE VERIFICATION
                        // -------------------------------------------------------------
                        if (header.MessageType == MsgInitStage)
                        {
                            byte[] initPayload = new byte[header.PayloadLength];
                            await ReadExactAsync(stream, initPayload, 0, initPayload.Length);

                            currentStageId = BitConverter.ToInt32(initPayload, 0);
                            startLayer = BitConverter.ToInt32(initPayload, 4);
                            endLayer = BitConverter.ToInt32(initPayload, 8);
                            isLastStage = BitConverter.ToInt32(initPayload, 12) != 0;
                            ulong expectedBytes = BitConverter.ToUInt64(initPayload, 16);
                            int nameLen = BitConverter.ToInt32(initPayload, 24);
                            string sliceFileName = Encoding.UTF8.GetString(initPayload, 28, nameLen);

                            layerCount = endLayer - startLayer + 1;
                            string localSlicePath = Path.Combine(cacheDir, sliceFileName);

                            Console.WriteLine($"[Worker] Stage {currentStageId} Init: Layers {startLayer}..{endLayer} ({layerCount}L), Last={isLastStage}");
                            Console.WriteLine($"[Worker] Target file: {sliceFileName} (Expected: {expectedBytes:N0} bytes)");

                            bool cacheHit = File.Exists(localSlicePath) && (ulong)new FileInfo(localSlicePath).Length == expectedBytes;

                            if (cacheHit)
                            {
                                Console.WriteLine($"[Worker] Cache HIT: {localSlicePath}. Loading stage model directly from disk...");
                                stageGguf = GgufFile.Open(localSlicePath);
                                stageWeights = new ModelWeights(stageGguf);
                                dim = stageWeights.EmbeddingLength;
                                stageModel = CreateStageModel(stageWeights, 4096, startLayer, layerCount, isLastStage);

                                Console.WriteLine($"[Worker] Stage {currentStageId} successfully loaded! Backend: {stageModel.BackendName} | Dim: {dim}");

                                // Send StageReady (status: 0 = ready)
                                SendStageReady(stream, status: 0, startLayer, endLayer, dim);
                            }
                            else
                            {
                                Console.WriteLine($"[Worker] Cache MISS for {sliceFileName}. Requesting stream from coordinator...");
                                // Send StageReady (status: 1 = need stream)
                                SendStageReady(stream, status: 1, startLayer, endLayer, 0);

                                // Receive streamed chunks into .tmp file
                                string tempSlicePath = localSlicePath + ".stream.tmp";
                                if (File.Exists(tempSlicePath)) File.Delete(tempSlicePath);

                                using (var fs = new FileStream(tempSlicePath, FileMode.Create, FileAccess.Write, FileShare.None, 4 * 1024 * 1024))
                                {
                                    while (true)
                                    {
                                        int chunkHdrRead = await ReadExactAsync(stream, headerBytes, 0, ClusterFrameHeader.HeaderSize);
                                        if (chunkHdrRead == 0) break;

                                        var chunkHdr = MemoryMarshal.Read<ClusterFrameHeader>(headerBytes);
                                        if (chunkHdr.MessageType == MsgStreamDone)
                                        {
                                            break;
                                        }

                                        if (chunkHdr.MessageType == MsgStreamChunk)
                                        {
                                            int cLen = (int)chunkHdr.PayloadLength;
                                            byte[] cBuf = new byte[cLen];
                                            await ReadExactAsync(stream, cBuf, 0, cLen);
                                            await fs.WriteAsync(cBuf, 0, cLen);
                                        }
                                    }
                                }

                                if (File.Exists(localSlicePath)) File.Delete(localSlicePath);
                                File.Move(tempSlicePath, localSlicePath);

                                Console.WriteLine($"[Worker] Stream complete! Saved {sliceFileName} ({new FileInfo(localSlicePath).Length:N0} bytes). Loading...");
                                stageGguf = GgufFile.Open(localSlicePath);
                                stageWeights = new ModelWeights(stageGguf);
                                dim = stageWeights.EmbeddingLength;
                                stageModel = CreateStageModel(stageWeights, 4096, startLayer, layerCount, isLastStage);

                                Console.WriteLine($"[Worker] Stage {currentStageId} successfully loaded & verified! Backend: {stageModel.BackendName} | Dim: {dim}");
                                SendStageReady(stream, status: 0, startLayer, endLayer, dim);
                            }
                        }
                        // -------------------------------------------------------------
                        // 2. FORWARD PASS EXECUTION (ACTIVATION FRAME)
                        // -------------------------------------------------------------
                        else if (header.MessageType == MsgActivation)
                        {
                            if (stageModel == null || stageWeights == null)
                            {
                                Console.WriteLine("[Worker] Received activation before stage initialization!");
                                break;
                            }

                            int payloadLen = (int)header.PayloadLength;
                            byte[] payload = new byte[payloadLen];
                            await ReadExactAsync(stream, payload, 0, payloadLen);

                            int stepIndex = BitConverter.ToInt32(payload, 0);
                            int pos = BitConverter.ToInt32(payload, 4);
                            int isLastStageInt = BitConverter.ToInt32(payload, 8);
                            int actDim = BitConverter.ToInt32(payload, 12);
                            bool computeLogitsThisStage = isLastStageInt != 0;

                            int actByteCount = actDim * sizeof(float);
                            var inSpan = MemoryMarshal.Cast<byte, float>(payload.AsSpan(16, actByteCount));

                            float[] outFloats = new float[actDim];
                            float[] logits = computeLogitsThisStage ? new float[stageWeights.VocabSize] : Array.Empty<float>();

                            var swCompute = Stopwatch.StartNew();
                            stageModel.ForwardStage(
                                token: 0,
                                pos: pos,
                                inputX: inSpan,
                                outputX: outFloats,
                                logits: logits,
                                computeLogits: computeLogitsThisStage);
                            swCompute.Stop();
                            double computeMs = swCompute.Elapsed.TotalMilliseconds;

                            int nextTokenId = 0;
                            float topLogit = 0f;

                            if (computeLogitsThisStage)
                            {
                                // Greedy / argmax sampling for deterministic validation
                                for (int v = 0; v < logits.Length; v++)
                                {
                                    if (logits[v] > topLogit || v == 0)
                                    {
                                        topLogit = logits[v];
                                        nextTokenId = v;
                                    }
                                }
                            }

                            // Send back MsgResult
                            int respLen = 28 + actByteCount;
                            byte[] resultPayload = new byte[respLen];
                            BitConverter.TryWriteBytes(resultPayload.AsSpan(0, 4), stepIndex);
                            BitConverter.TryWriteBytes(resultPayload.AsSpan(4, 4), currentStageId);
                            BitConverter.TryWriteBytes(resultPayload.AsSpan(8, 8), computeMs);
                            BitConverter.TryWriteBytes(resultPayload.AsSpan(16, 4), nextTokenId);
                            BitConverter.TryWriteBytes(resultPayload.AsSpan(20, 4), topLogit);
                            BitConverter.TryWriteBytes(resultPayload.AsSpan(24, 4), actDim);
                            MemoryMarshal.Cast<float, byte>(outFloats).CopyTo(resultPayload.AsSpan(28, actByteCount));

                            var respHdr = new ClusterFrameHeader(MsgResult, header.CorrelationId, (ulong)respLen);
                            byte[] respHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
                            MemoryMarshal.Write(respHdrBytes, in respHdr);

                            await stream.WriteAsync(respHdrBytes, 0, ClusterFrameHeader.HeaderSize);
                            await stream.WriteAsync(resultPayload, 0, respLen);
                            await stream.FlushAsync();
                        }
                        // -------------------------------------------------------------
                        // 3. RESET KV CACHE (FOR MULTI-TURN CHAT CONVERSATION)
                        // -------------------------------------------------------------
                        else if (header.MessageType == MsgResetKv)
                        {
                            Console.WriteLine($"[Worker] Resetting KV cache for stage {currentStageId}...");
                            stageModel?.ResetKvCache();

                            // Ack
                            var ackHdr = new ClusterFrameHeader(MsgResetKv, header.CorrelationId, 0);
                            byte[] ackHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
                            MemoryMarshal.Write(ackHdrBytes, in ackHdr);
                            await stream.WriteAsync(ackHdrBytes, 0, ClusterFrameHeader.HeaderSize);
                            await stream.FlushAsync();
                        }
                        // -------------------------------------------------------------
                        // 4. SHUTDOWN
                        // -------------------------------------------------------------
                        else if (header.MessageType == MsgShutdown)
                        {
                            Console.WriteLine("[Worker] Received shutdown frame. Resetting session for next run.");
                            break;
                        }
                    }
                }
                finally
                {
                    stageModel?.Dispose();
                    stageGguf?.Dispose();
                }

                Console.WriteLine("[Worker] Session complete with coordinator. Worker stays alive for next run.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Worker] Error: {ex.Message}");
            }
        }
    }

    private static void SendStageReady(NetworkStream stream, int status, int startLayer, int endLayer, int dim)
    {
        byte[] payload = new byte[16];
        BitConverter.TryWriteBytes(payload.AsSpan(0, 4), status);
        BitConverter.TryWriteBytes(payload.AsSpan(4, 4), startLayer);
        BitConverter.TryWriteBytes(payload.AsSpan(8, 4), endLayer);
        BitConverter.TryWriteBytes(payload.AsSpan(12, 4), dim);

        var hdr = new ClusterFrameHeader(MsgStageReady, 0, 16);
        byte[] hdrBytes = new byte[ClusterFrameHeader.HeaderSize];
        MemoryMarshal.Write(hdrBytes, in hdr);

        stream.Write(hdrBytes, 0, ClusterFrameHeader.HeaderSize);
        stream.Write(payload, 0, 16);
        stream.Flush();
    }

    // =========================================================================
    // COORDINATOR IMPLEMENTATION (HOST NODE 0)
    // =========================================================================
    private static async Task<int> RunCoordinatorAsync(
        List<StageEndpoint> remoteEndpoints,
        string modelPath,
        string initialPrompt,
        int maxTokens,
        string cacheDir,
        bool interactive,
        string? layerAllocArg,
        int hostGpuAdapter)
    {
        Console.WriteLine($"[Coordinator] Inspecting Master GGUF Model: {Path.GetFileName(modelPath)}");
        using var masterGguf = GgufFile.Open(modelPath);

        int totalLayers = masterGguf.BlockCount;
        int dim = masterGguf.EmbeddingLength;
        int vocabSize = masterGguf.Tensors.TryGetValue("token_embd.weight", out var ti) && ti != null
            ? (int)ti.Dimensions[1]
            : (int)masterGguf.GetMetadataUInt32("tokenizer.ggml.tokens.length", 151936);

        Console.WriteLine($"[Coordinator] Model Architecture: {masterGguf.Architecture} | Total Layers: {totalLayers} | Dim: {dim} | Vocab: {vocabSize}");

        // Load Tokenizer
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

        // ---------------------------------------------------------------------
        // 1. PREPARE LOCAL HOST STAGES (MULTI-GPU SUPPORT: RTX 4060 + 890M)
        // ---------------------------------------------------------------------
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
            var model = CreateStageModel(weights, 4096, stInfo.StartLayer, stInfo.LayerCount, isLastStage: isLast, adapterIndex: alloc.Node.AdapterIndex);
            localModels.Add(model);
            Console.WriteLine($"[Stage {s}] {stInfo.Name} initialized successfully! Backend: {model.BackendName}\n");
        }

        // ---------------------------------------------------------------------
        // 2. CONNECT TO REMOTE WORKERS & HANDSHAKE CACHE / STREAMING
        // ---------------------------------------------------------------------
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

                // Slice directly to remote stream
                using (var chunkWriter = new NetworkChunkWriter(stream, MsgStreamChunk))
                {
                    GgufStageSlicer.SliceStage(masterGguf, chunkWriter, stInfo.StartLayer, stInfo.EndLayer, isFirstStage: false, isLastStage: isLast);
                }

                // Send MsgStreamDone
                var doneHdr = new ClusterFrameHeader(MsgStreamDone, (ulong)s, 0);
                byte[] doneHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
                MemoryMarshal.Write(doneHdrBytes, in doneHdr);
                await stream.WriteAsync(doneHdrBytes, 0, ClusterFrameHeader.HeaderSize);
                await stream.FlushAsync();

                // Await final MsgStageReady
                await ReadExactAsync(stream, respHdrBytes, 0, ClusterFrameHeader.HeaderSize);
                await ReadExactAsync(stream, respPayload, 0, 16);
                swStream.Stop();
                Console.WriteLine($"[Stage {s}] Streaming & loading complete in {swStream.Elapsed.TotalSeconds:F1}s! Worker is ready.");
            }
        }

        Console.WriteLine("\n[Cluster] All stages online and verified. Ready for inference!\n");

        int currentPos = 0;
        string activePrompt = initialPrompt;

        while (true)
        {
            if (string.IsNullOrWhiteSpace(activePrompt))
            {
                Console.Write("\nYou > ");
                string? input = Console.ReadLine();
                if (string.IsNullOrWhiteSpace(input)) continue;
                if (input.Equals("exit", StringComparison.OrdinalIgnoreCase) || input.Equals("quit", StringComparison.OrdinalIgnoreCase))
                {
                    break;
                }
                if (input.Equals("clear", StringComparison.OrdinalIgnoreCase) || input.Equals("reset", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine("[Cluster] Resetting conversation context on all nodes...");
                    await BroadcastResetKvAsync(streams);
                    foreach (var m in localModels) m.ResetKvCache();
                    currentPos = 0;
                    Console.WriteLine("[Cluster] KV cache cleared! Ready for new conversation.\n");
                    continue;
                }
                activePrompt = input;
            }

            Console.WriteLine($"[Prompt]: \"{activePrompt}\"");
            Console.WriteLine("--------------------------------------------------------------------------------");
            Console.WriteLine($"GENERATING REAL TOKENS ACROSS {numStages}-STAGE PIPELINE ({localModels.Count} Local GPU + {streams.Count} Remote LAN)");
            Console.WriteLine("--------------------------------------------------------------------------------");

            int[] promptTokens = tokenizer.Encode(activePrompt);
            var swTotal = Stopwatch.StartNew();

            float[] inX = new float[dim];
            float[] outX = new float[dim];
            float[] logits = new float[vocabSize];
            int nextTokenId = promptTokens[0];

            // Reset step invocation counters for this turn
            for (int i = 0; i < numStages; i++)
            {
                stages[i].TotalComputeMs = 0;
                stages[i].TotalNetworkMs = 0;
                stages[i].Invocations = 0;
            }

            // -----------------------------------------------------------------
            // A. PREFILL PROMPT TOKENS (TOKENS READ)
            // -----------------------------------------------------------------
            var swPrefill = Stopwatch.StartNew();
            for (int p = 0; p < promptTokens.Length; p++)
            {
                int tok = promptTokens[p];
                bool isLastPromptToken = (p == promptTokens.Length - 1);

                // Local stages (Host dGPU, Host iGPU)
                float[] curIn = inX;
                float[] curOut = outX;

                for (int l = 0; l < localModels.Count; l++)
                {
                    bool isLastStage = (l == numStages - 1);
                    bool computeLogits = isLastStage && isLastPromptToken;

                    var swLocal = Stopwatch.StartNew();
                    localModels[l].ForwardStage(tok, currentPos, curIn, curOut, logits, computeLogits);
                    swLocal.Stop();

                    stages[l].TotalComputeMs += swLocal.Elapsed.TotalMilliseconds;
                    stages[l].Invocations++;
                    curIn = curOut;
                }

                // Remote stages (Beast, etc.)
                for (int s = localModels.Count; s < numStages; s++)
                {
                    bool isLastStage = (s == numStages - 1);
                    bool computeLogits = isLastStage && isLastPromptToken;

                    int streamIdx = s - localModels.Count;
                    var stream = streams[streamIdx];
                    var stInfo = stages[s];

                    long tStart = Stopwatch.GetTimestamp();
                    await SendActivationAsync(stream, (ulong)p, p, currentPos, computeLogits, dim, curOut);
                    var res = await ReadResultAsync(stream, dim, curOut);
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

            // -----------------------------------------------------------------
            // B. AUTOREGRESSIVE GENERATION (TOKENS OUTPUT)
            // -----------------------------------------------------------------
            var swDecode = Stopwatch.StartNew();
            int generatedCount = 0;
            string firstPiece = tokenizer.DecodeToken(nextTokenId);
            Console.Write(firstPiece);
            generatedCount++;

            int maxGen = maxTokens <= 0 ? int.MaxValue : maxTokens;
            while (generatedCount < maxGen)
            {
                if (nextTokenId == tokenizer.EosTokenId || tokenizer.IsStopToken(nextTokenId))
                {
                    break;
                }

                // Local stages (Host dGPU, Host iGPU)
                float[] curIn = inX;
                float[] curOut = outX;

                for (int l = 0; l < localModels.Count; l++)
                {
                    bool isLastStage = (l == numStages - 1);
                    var swLocal = Stopwatch.StartNew();
                    localModels[l].ForwardStage(nextTokenId, currentPos, curIn, curOut, logits, computeLogits: isLastStage);
                    swLocal.Stop();

                    stages[l].TotalComputeMs += swLocal.Elapsed.TotalMilliseconds;
                    stages[l].Invocations++;
                    curIn = curOut;
                }

                // Remote stages (Beast, etc.)
                for (int s = localModels.Count; s < numStages; s++)
                {
                    bool isLastStage = (s == numStages - 1);
                    int streamIdx = s - localModels.Count;
                    var stream = streams[streamIdx];
                    var stInfo = stages[s];

                    long tStart = Stopwatch.GetTimestamp();
                    await SendActivationAsync(stream, (ulong)generatedCount, generatedCount, currentPos, isLastStage, dim, curOut);
                    var res = await ReadResultAsync(stream, dim, curOut);
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

                string piece = tokenizer.DecodeToken(nextTokenId);
                Console.Write(piece);
                currentPos++;
                generatedCount++;
            }
            swDecode.Stop();
            swTotal.Stop();

            double prefillTokensPerSec = promptTokens.Length / Math.Max(0.0001, swPrefill.Elapsed.TotalSeconds);
            double decodeTokensPerSec = generatedCount / Math.Max(0.0001, swDecode.Elapsed.TotalSeconds);
            double decodeMsPerToken = swDecode.Elapsed.TotalMilliseconds / Math.Max(1, generatedCount);

            Console.WriteLine("\n\n================================================================================");
            Console.WriteLine($"  {numStages}-STAGE PIPELINE INFERENCE BENCHMARK SUMMARY");
            Console.WriteLine("================================================================================");
            Console.WriteLine($"  Tokens Read (Prompt Prefill):   {promptTokens.Length} tokens in {swPrefill.Elapsed.TotalMilliseconds:F2} ms ({prefillTokensPerSec:F2} tok/s, {swPrefill.Elapsed.TotalMilliseconds / promptTokens.Length:F2} ms/tok)");
            Console.WriteLine($"  Tokens Output (Generation):     {generatedCount} tokens in {swDecode.Elapsed.TotalMilliseconds:F2} ms ({decodeTokensPerSec:F2} tok/s, {decodeMsPerToken:F2} ms/tok)");
            Console.WriteLine($"  Total Wall Clock Time:          {swTotal.Elapsed.TotalMilliseconds:F2} ms");

            double totalClusterCompute = 0;
            double totalClusterNet = 0;
            Console.WriteLine("\n  Stage Compute & Network Breakdown per Step:");
            for (int i = 0; i < numStages; i++)
            {
                var st = stages[i];
                double avgComp = st.Invocations > 0 ? st.TotalComputeMs / st.Invocations : 0;
                double avgNet = st.Invocations > 0 ? st.TotalNetworkMs / st.Invocations : 0;
                totalClusterCompute += avgComp;
                totalClusterNet += avgNet;

                string netStr = (i < localModels.Count) ? "Local Direct Memory (0.0 ms)" : $"{avgNet:F3} ms Gigabit LAN";
                Console.WriteLine($"    Stage {i} [{st.Name}]:");
                Console.WriteLine($"      Layers {st.StartLayer}..{st.EndLayer} ({st.LayerCount}L) | Compute: {avgComp:F3} ms | Transit: {netStr}");
            }

            double efficiency = (totalClusterCompute / (totalClusterCompute + totalClusterNet)) * 100.0;
            Console.WriteLine($"\n  Total Compute Time:          {totalClusterCompute:F3} ms / token");
            Console.WriteLine($"  Total Network Transit Time:  {totalClusterNet:F3} ms / token");
            Console.WriteLine($"  Cluster Pipeline Efficiency: {efficiency:F1}% compute utilization");
            Console.WriteLine("================================================================================\n");

            activePrompt = "";
            if (!interactive) break;
        }

        // Clean shutdown of remote daemons
        for (int s = 0; s < streams.Count; s++)
        {
            try
            {
                var shutHdr = new ClusterFrameHeader(MsgShutdown, 0, 0);
                byte[] shutBytes = new byte[ClusterFrameHeader.HeaderSize];
                MemoryMarshal.Write(shutBytes, in shutHdr);
                await streams[s].WriteAsync(shutBytes, 0, ClusterFrameHeader.HeaderSize);
                await streams[s].FlushAsync();
            }
            catch { }
        }

        foreach (var m in localModels) m.Dispose();
        foreach (var g in localGgufs) g.Dispose();

        return 0;
    }

    private static async Task SendActivationAsync(NetworkStream stream, ulong corrId, int step, int pos, bool isLastStage, int dim, float[] actFloats)
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

    private static async Task<(double ComputeMs, int NextTokenId, float TopLogit)> ReadResultAsync(NetworkStream stream, int dim, float[] outFloats)
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

    private static async Task BroadcastResetKvAsync(List<NetworkStream> streams)
    {
        byte[] resetHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
        var hdr = new ClusterFrameHeader(MsgResetKv, 0, 0);
        MemoryMarshal.Write(resetHdrBytes, in hdr);

        foreach (var s in streams)
        {
            await s.WriteAsync(resetHdrBytes, 0, ClusterFrameHeader.HeaderSize);
            await s.FlushAsync();

            byte[] ackHdr = new byte[ClusterFrameHeader.HeaderSize];
            await ReadExactAsync(s, ackHdr, 0, ClusterFrameHeader.HeaderSize);
        }
    }

    private static async Task<int> ReadExactAsync(Stream s, byte[] buf, int offset, int count)
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

    /// <summary>
    /// Helper Stream adapter that flushes buffered chunks as discrete MsgStreamChunk frames over TCP.
    /// </summary>
    private sealed class NetworkChunkWriter : Stream
    {
        private readonly NetworkStream _underlying;
        private readonly ClusterMessageType _msgType;
        private readonly byte[] _buffer = new byte[4 * 1024 * 1024];
        private int _buffered = 0;

        public NetworkChunkWriter(NetworkStream underlying, ClusterMessageType msgType)
        {
            _underlying = underlying;
            _msgType = msgType;
        }

        public override void Write(byte[] buffer, int offset, int count)
        {
            while (count > 0)
            {
                int toCopy = Math.Min(count, _buffer.Length - _buffered);
                Buffer.BlockCopy(buffer, offset, _buffer, _buffered, toCopy);
                _buffered += toCopy;
                offset += toCopy;
                count -= toCopy;

                if (_buffered == _buffer.Length)
                {
                    FlushChunk();
                }
            }
        }

        private void FlushChunk()
        {
            if (_buffered == 0) return;

            var hdr = new ClusterFrameHeader(_msgType, 0, (ulong)_buffered);
            byte[] hdrBytes = new byte[ClusterFrameHeader.HeaderSize];
            MemoryMarshal.Write(hdrBytes, in hdr);

            _underlying.Write(hdrBytes, 0, ClusterFrameHeader.HeaderSize);
            _underlying.Write(_buffer, 0, _buffered);
            _underlying.Flush();
            _buffered = 0;
        }

        public override void Flush()
        {
            FlushChunk();
        }

        public override bool CanRead => false;
        public override bool CanSeek => false;
        public override bool CanWrite => true;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();

        protected override void Dispose(bool disposing)
        {
            if (disposing) FlushChunk();
            base.Dispose(disposing);
        }
    }

    public static IClusterStageModel CreateStageModel(
        ModelWeights weights,
        int maxSeqLen,
        int startLayer,
        int layerCount,
        bool isLastStage,
        int adapterIndex = -1)
    {
        // 1. Auto-detect GPU acceleration (Direct3D 12 on Windows)
        if (OperatingSystem.IsWindows() && (weights.ArchitectureFamily == UniversalArchitecture.Qwen || weights.ArchitectureFamily == UniversalArchitecture.Llama || weights.ArchitectureFamily == UniversalArchitecture.Generic))
        {
            var d3dModel = TryCreateD3D12StageModel(weights, maxSeqLen, startLayer, layerCount, isLastStage, adapterIndex);
            if (d3dModel != null) return d3dModel;
        }

        // 2. High-performance CPU SIMD fallback (AVX-512 / AVX2 / ARM AdvSIMD)
        string cpuSimd = System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated ? "AVX-512" :
                         (System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated ? "AVX2" : "Scalar");
        Console.WriteLine($"[AutoCompute] Using CPU SIMD Runtime: {Environment.ProcessorCount} Cores ({cpuSimd})");
        var cpuModel = CpuModelFactory.Create(weights, maxSeqLen, startLayer, layerCount, isLastStage);
        var kv = new KVCache(layerCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen);
        return new CpuClusterStageModel(cpuModel, kv, $"CPU SIMD ({Environment.ProcessorCount} Cores, {cpuSimd})");
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static IClusterStageModel? TryCreateD3D12StageModel(
        ModelWeights weights,
        int maxSeqLen,
        int startLayer,
        int layerCount,
        bool isLastStage,
        int adapterIndex = -1)
    {
        try
        {
            var ctx = new D3D12Context(adapterIndex);
            ulong dedicatedMb = ctx.DedicatedVideoMemory / (1024 * 1024);
            ulong sharedMb = ctx.SharedSystemMemory / (1024 * 1024);
            ulong totalMb = ctx.TotalAccessibleMemory / (1024 * 1024);

            Console.WriteLine($"[AutoCompute] Detected GPU: {ctx.DeviceName}");
            Console.WriteLine($"[AutoCompute] VRAM: {dedicatedMb:N0} MB Dedicated, {sharedMb:N0} MB Shared ({totalMb:N0} MB Total Accessible)");

            var d3dModel = new Qwen2D3D12Model(ctx, weights, maxSeqLen, KvCachePrecision.Auto, startLayer, layerCount, isLastStage);
            string modeStr = d3dModel.TieredMoe ? "Heterogeneous VRAM+CPU Tiered" : "100% In-VRAM GPU Accelerated";
            Console.WriteLine($"[AutoCompute] Direct3D 12 GPU Engine Loaded! Mode: {modeStr}");
            return new D3D12ClusterStageModel(ctx, d3dModel, $"D3D12 GPU ({ctx.DeviceName})");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AutoCompute] Direct3D 12 GPU fallback to CPU SIMD: {ex.Message}");
            return null;
        }
    }
}
