namespace Glacier.Inference.Distributed.Cli;

using System;
using System.Buffers;
using System.Buffers.Binary;
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
using Glacier.Inference.Distributed;
using Glacier.Inference.Distributed.Pipeline;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;
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

    public void ForwardBatchStage(ReadOnlySpan<int> tokens, int startPos, ReadOnlySpan<float> inputXBatch, Span<float> outputXBatch, Span<float> logits, bool computeLogits)
    {
        _model.ForwardBatchStage(tokens, startPos, inputXBatch, outputXBatch, logits, computeLogits, _kv);
    }

    public void ResetKvCache()
    {
        _kv.Reset();
        _model.ResetState();
    }

    public void Dispose()
    {
        _model.Dispose();
        _kv.Dispose();
    }
}

public sealed class D3D12ClusterStageModel : IClusterStageModel
{
    private readonly ID3D12Model _model;
    private readonly D3D12Context _ctx;
    public string BackendName { get; }

    public D3D12ClusterStageModel(D3D12Context ctx, ID3D12Model model, string backendName)
    {
        _ctx = ctx;
        _model = model;
        BackendName = backendName;
    }

    public void ForwardStage(int token, int pos, ReadOnlySpan<float> inputX, Span<float> outputX, Span<float> logits, bool computeLogits)
    {
        _model.ForwardStage(token, pos, inputX, outputX, logits, computeLogits);
    }

    public void ForwardBatchStage(ReadOnlySpan<int> tokens, int startPos, ReadOnlySpan<float> inputXBatch, Span<float> outputXBatch, Span<float> logits, bool computeLogits)
    {
        _model.ForwardBatchStage(tokens, startPos, inputXBatch, outputXBatch, logits, computeLogits);
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

public static partial class Program
{
    private const int DefaultPort = 5055;
    private const int DefaultWorkerMaxBatchTokens = 512;
    private const int DefaultWorkerMaxSeqLen = 4096;

    // Cluster wire message opcodes for distributed LLM inference
    private const ClusterMessageType MsgInitStage       = DistributedClusterSession.MsgInitStage;
    private const ClusterMessageType MsgStageReady      = DistributedClusterSession.MsgStageReady;
    private const ClusterMessageType MsgStreamChunk     = DistributedClusterSession.MsgStreamChunk;
    private const ClusterMessageType MsgStreamDone      = DistributedClusterSession.MsgStreamDone;
    private const ClusterMessageType MsgActivation      = DistributedClusterSession.MsgActivation;
    private const ClusterMessageType MsgResult          = DistributedClusterSession.MsgResult;
    private const ClusterMessageType MsgResetKv         = DistributedClusterSession.MsgResetKv;
    private const ClusterMessageType MsgShutdown        = DistributedClusterSession.MsgShutdown;
    private const ClusterMessageType MsgActivationBatch = DistributedClusterSession.MsgActivationBatch;
    private const ClusterMessageType MsgResultBatch     = DistributedClusterSession.MsgResultBatch;

    public static async Task<int> Main(string[] args)
    {
        string role = "coordinator";
        int port = DefaultPort;
        int httpPort = 8080;
        string httpHost = "0.0.0.0";
        string? apiKey = Environment.GetEnvironmentVariable("GLACIER_API_KEY");
        bool serve = false;
        int maxTokens = -1; // Unlimited by default (generates until natural EOS)
        string? modelPath = null;
        string? downloadTarget = null;
        string? destDir = null;
        string? cacheDir = null;
        int downloadStreams = 8;
        string prompt = "Explain quantum computers to a layman.";
        string stagesArg = "";
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
            else if (args[i] == "serve" || args[i] == "--serve")
            {
                serve = true;
            }
            else if (args[i] == "--http-port" && i + 1 < args.Length && int.TryParse(args[++i], out int hp))
            {
                httpPort = hp;
            }
            else if (args[i] == "--host" && i + 1 < args.Length)
            {
                httpHost = args[++i];
            }
            else if (args[i] == "--api-key" && i + 1 < args.Length)
            {
                apiKey = args[++i];
            }
            else if (args[i] == "--dest" && i + 1 < args.Length) destDir = args[++i];
            else if (args[i] == "--cache" && i + 1 < args.Length) cacheDir = args[++i];
            else if (args[i] == "--streams" && i + 1 < args.Length && int.TryParse(args[++i], out int s)) downloadStreams = s;
            else if (args[i] == "--role" && i + 1 < args.Length) role = args[++i].ToLowerInvariant();
            else if (args[i] == "--port" && i + 1 < args.Length && int.TryParse(args[++i], out int p))
            {
                port = p;
                httpPort = p;
            }
            else if (args[i] == "--tokens" && i + 1 < args.Length && int.TryParse(args[++i], out int t)) maxTokens = t;
            else if (args[i] == "--model" && i + 1 < args.Length) modelPath = args[++i];
            else if (args[i] == "--prompt" && i + 1 < args.Length) prompt = args[++i];
            else if ((args[i] == "--stages" || args[i] == "--nodes") && i + 1 < args.Length) stagesArg = args[++i];
            else if ((args[i] == "--layers" || args[i] == "--alloc") && i + 1 < args.Length) layerAlloc = args[++i];
            else if ((args[i] == "--host-gpu" || args[i] == "--adapter") && i + 1 < args.Length && int.TryParse(args[++i], out int hg)) hostGpu = hg;
            else if (args[i] == "--interactive" || args[i] == "-i" || args[i] == "chat") interactive = true;
        }

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cts.Cancel();
        };

        // If user invoked model downloader
        if (!string.IsNullOrEmpty(downloadTarget))
        {
            return await ModelDownloader.DownloadModelAsync(downloadTarget, destDir, downloadStreams, cts.Token);
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
        Console.WriteLine($"  Role: {role.ToUpperInvariant()} | Mode: {(serve ? "OPENAI / OLLAMA HTTP SERVICE" : (interactive ? "INTERACTIVE REPL" : "BENCHMARK / GENERATE"))}");
        Console.WriteLine($"  OS: {RuntimeInformation.OSDescription} ({RuntimeInformation.ProcessArchitecture})");
        Console.WriteLine($"  Hardware Threads: {Environment.ProcessorCount} Cores | Vector512: {System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated}");
        Console.WriteLine("================================================================================\n");

        if (role == "worker")
        {
            return await RunWorkerAsync(port, cacheDir, hostGpu, cts.Token);
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
            return await RunCoordinatorAsync(
                endpoints,
                modelPath,
                prompt,
                maxTokens,
                cacheDir,
                interactive,
                layerAlloc,
                hostGpu,
                serve,
                httpHost,
                httpPort,
                apiKey,
                cts.Token);
        }
    }

    private static void PrintUsage()
    {
        Console.WriteLine(@"Glacier Distributed Cluster CLI (glacier-distributed) - Pure C# .NET 10

Usage:
  glacier-distributed serve --model <path> [--http-port 8080] [--host 0.0.0.0]
  glacier-distributed download <preset|url> [--dest <dir>] [--streams <int>]
  glacier-distributed --role worker [--port <int>] [--cache <dir>]
  glacier-distributed --role coordinator --model <path> [options]

Serve / OpenAI Compatible Service:
  serve, --serve                  Run distributed cluster as an OpenAI & Ollama compatible HTTP server
  --http-port, --port <int>       HTTP port to listen on (default: 8080)
  --host <ip>                     Host address to bind HTTP server (default: 0.0.0.0)
  --api-key <string>              Require Bearer token or x-api-key authentication (or set GLACIER_API_KEY)

Endpoints Exposed in Serve Mode:
  POST /v1/chat/completions       OpenAI Chat Completions (streaming SSE & non-streaming JSON)
  POST /v1/completions            OpenAI Text Completions (streaming SSE & non-streaming JSON)
  GET  /v1/models                 OpenAI Models list
  GET  /v1/cluster/status         Cluster stage topology, latency, and hardware telemetry
  GET  /health                    Cluster health check
  POST /api/chat                  Ollama Chat endpoint (streaming ndjson & non-streaming JSON)
  POST /api/generate              Ollama Generate endpoint (streaming ndjson & non-streaming JSON)
  GET  /api/tags                  Ollama Tags list

Coordinator Options:
  --model <path>                  Path to GGUF model file (e.g. Qwen2-57B or Mixtral-8x22B)
  --stages <ip:port,...>          Comma-separated list of candidate worker endpoints (e.g. 192.168.1.50:5055)
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

    public static List<StageEndpoint> ParseEndpoints(string stagesArg)
    {
        var list = new List<StageEndpoint>();
        if (string.IsNullOrWhiteSpace(stagesArg)) return list;
        if (stagesArg.Equals("none", StringComparison.OrdinalIgnoreCase) ||
            stagesArg.Equals("local", StringComparison.OrdinalIgnoreCase) ||
            stagesArg.Equals("single", StringComparison.OrdinalIgnoreCase) ||
            stagesArg.Equals("standalone", StringComparison.OrdinalIgnoreCase))
        {
            return list;
        }

        var parts = stagesArg.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        int idx = 1;
        foreach (var p in parts)
        {
            string host = p;
            int port = DefaultPort;
            int lastColon = p.LastIndexOf(':');
            if (lastColon > 0 && p.IndexOf(':') == lastColon)
            {
                host = p.Substring(0, lastColon);
                if (!int.TryParse(p.AsSpan(lastColon + 1), out port) || port <= 0 || port > 65535)
                {
                    port = DefaultPort;
                }
            }

            string name = host switch
            {
                "127.0.0.1" or "localhost" or "::1" => $"Loopback Worker ({host}:{port})",
                _ => $"Worker-{idx} ({host}:{port})"
            };

            list.Add(new StageEndpoint(host, port, name));
            idx++;
        }
        return list;
    }

    // =========================================================================
    // WORKER DAEMON IMPLEMENTATION
    // =========================================================================
    private static async Task<int> RunWorkerAsync(
        int port,
        string cacheDir,
        int adapterIndex = -1,
        CancellationToken ct = default)
    {
        Console.WriteLine($"[Worker] Starting Glacier Worker Daemon on port {port}...");
        Console.WriteLine($"[Worker] Cache directory: {Path.GetFullPath(cacheDir)}");

        var listener = new TcpListener(IPAddress.Any, port);
        listener.Server.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.NoDelay, true);
        listener.Server.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.KeepAlive, true);
        if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux() || OperatingSystem.IsMacOS())
        {
            listener.Server.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveTime, 5);
            listener.Server.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveInterval, 1);
            listener.Server.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveRetryCount, 5);
        }
        listener.Start();
        Console.WriteLine($"[Worker] Listening on 0.0.0.0:{port} (NoDelay=true, KeepAlive=true). Awaiting coordinator...\n");

        try
        {
            while (!ct.IsCancellationRequested)
            {
                try
                {
                    using var client = await listener.AcceptTcpClientAsync(ct).ConfigureAwait(false);
                    FusedSocketTransport.ConfigureSocket(client.Client, 4 * 1024 * 1024);
                    Console.WriteLine($"[Worker] Coordinator connected from {client.Client.RemoteEndPoint}");

                    using var stream = client.GetStream();
                    await HandleWorkerConnectionAsync(stream, cacheDir, adapterIndex, ct).ConfigureAwait(false);

                    Console.WriteLine("[Worker] Session complete with coordinator. Worker stays alive for next run.");
                }
                catch (OperationCanceledException) when (ct.IsCancellationRequested)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[Worker] Error: {ex.Message}");
                }
            }
        }
        finally
        {
            listener.Stop();
        }

        return 0;
    }

    /// <summary>
    /// Handles a single coordinator-to-worker session over <paramref name="stream"/>.
    /// Supports <c>MsgInitStage</c> (<c>0x60</c>), <c>MsgActivationBatch</c> (<c>0x74</c> -> <c>0x75</c>),
    /// <c>MsgActivation</c> (<c>0x70</c> -> <c>0x71</c>), <c>MsgResetKv</c> (<c>0x72</c>),
    /// <c>MsgShutdown</c> (<c>0x73</c>), <c>MsgCalibrateProbeRequest</c> (<c>0x76</c> -> <c>0x77</c>),
    /// and <c>HeartbeatPing</c>/<c>HeartbeatPong</c> with hoisted POH buffers (zero per-step allocations).
    /// </summary>
    public static async Task HandleWorkerConnectionAsync(
        Stream stream,
        string cacheDir,
        int adapterIndex = -1,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheDir);
        Directory.CreateDirectory(cacheDir);

        byte[] headerBytes = GC.AllocateArray<byte>(ClusterFrameHeader.HeaderSize, pinned: true);
        byte[] controlSendBytes = GC.AllocateArray<byte>(ClusterFrameHeader.HeaderSize + 32, pinned: true);
        int[] recentTokensScratch = GC.AllocateArray<int>(SamplingWireCodec.MaxRecentTokens, pinned: true);

        GgufFile? stageGguf = null;
        ModelWeights? stageWeights = null;
        IClusterStageModel? stageModel = null;

        // Hoisted POH stage buffers initialized once per MsgInitStage
        float[] singleIn = [];
        float[] singleOut = [];
        float[] batchIn = [];
        float[] batchOut = [];
        float[] logits = [];
        int[] batchTokensBuf = [];
        byte[] wireRecvBuf = GC.AllocateArray<byte>(64 * 1024, pinned: true);
        byte[] wireSendBuf = GC.AllocateArray<byte>(64 * 1024, pinned: true);

        int currentStageId = -1;
        int startLayer = 0;
        int endLayer = 0;
        int layerCount = 0;
        int dim = 0;
        bool isLastStage = false;

        try
        {
            while (!ct.IsCancellationRequested)
            {
                bool hasHeader = await ReadHeaderOrEofAsync(stream, headerBytes, ct).ConfigureAwait(false);
                if (!hasHeader) break;

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
                    if (header.PayloadLength < 28 || header.PayloadLength > 64 * 1024)
                    {
                        throw new InvalidDataException($"Invalid MsgInitStage payload length: {header.PayloadLength}.");
                    }

                    stageModel?.Dispose();
                    stageModel = null;
                    stageGguf?.Dispose();
                    stageGguf = null;
                    stageWeights = null;

                    int initLen = (int)header.PayloadLength;
                    byte[] initPayload = new byte[initLen];
                    await ReadExactOrThrowAsync(stream, initPayload.AsMemory(0, initLen), ct).ConfigureAwait(false);

                    currentStageId = BitConverter.ToInt32(initPayload, 0);
                    startLayer = BitConverter.ToInt32(initPayload, 4);
                    endLayer = BitConverter.ToInt32(initPayload, 8);
                    isLastStage = BitConverter.ToInt32(initPayload, 12) != 0;
                    ulong expectedBytes = BitConverter.ToUInt64(initPayload, 16);
                    int nameLen = BitConverter.ToInt32(initPayload, 24);
                    if (nameLen <= 0 || 28 + nameLen > initLen)
                    {
                        throw new InvalidDataException($"Invalid slice filename length ({nameLen}) in MsgInitStage.");
                    }

                    string rawSliceFileName = Encoding.UTF8.GetString(initPayload, 28, nameLen);
                    string sliceFileName = Path.GetFileName(rawSliceFileName);
                    if (string.IsNullOrWhiteSpace(sliceFileName))
                    {
                        throw new InvalidDataException("Empty slice filename in MsgInitStage.");
                    }

                    layerCount = endLayer - startLayer + 1;
                    string localSlicePath = Path.Combine(cacheDir, sliceFileName);

                    Console.WriteLine($"[Worker] Stage {currentStageId} Init: Layers {startLayer}..{endLayer} ({layerCount}L), Last={isLastStage}");
                    Console.WriteLine($"[Worker] Target file: {sliceFileName} (Expected: {expectedBytes:N0} bytes)");

                    bool cacheHit = File.Exists(localSlicePath) && (ulong)new FileInfo(localSlicePath).Length == expectedBytes;

                    if (!cacheHit)
                    {
                        Console.WriteLine($"[Worker] Cache MISS for {sliceFileName}. Requesting stream from coordinator...");
                        await SendStageReadyAsync(stream, controlSendBytes, status: 1, startLayer, endLayer, 0, ct).ConfigureAwait(false);

                        string tempSlicePath = localSlicePath + $".{Guid.NewGuid():N}.stream.tmp";
                        byte[] chunkBuffer = GC.AllocateArray<byte>(4 * 1024 * 1024, pinned: true);

                        using (var fs = new FileStream(tempSlicePath, FileMode.Create, FileAccess.Write, FileShare.None, 4 * 1024 * 1024))
                        {
                            while (true)
                            {
                                bool hasChunkHdr = await ReadHeaderOrEofAsync(stream, headerBytes, ct).ConfigureAwait(false);
                                if (!hasChunkHdr)
                                    throw new EndOfStreamException("Connection closed prematurely while streaming stage slice.");

                                var chunkHdr = MemoryMarshal.Read<ClusterFrameHeader>(headerBytes);
                                if (chunkHdr.Magic != ClusterFrameHeader.MagicConstant)
                                {
                                    throw new InvalidDataException($"Invalid frame magic during chunk streaming: 0x{chunkHdr.Magic:X8}");
                                }

                                if (chunkHdr.MessageType == MsgStreamDone)
                                {
                                    break;
                                }

                                if (chunkHdr.PayloadLength > FusedSocketTransport.MaxAllowedPayloadBytes)
                                {
                                    throw new InvalidDataException($"Oversized MsgStreamChunk payload: {chunkHdr.PayloadLength} bytes.");
                                }

                                if (chunkHdr.MessageType == MsgStreamChunk)
                                {
                                    int cLen = (int)chunkHdr.PayloadLength;
                                    if (cLen > chunkBuffer.Length)
                                    {
                                        chunkBuffer = GC.AllocateArray<byte>(cLen, pinned: true);
                                    }
                                    await ReadExactOrThrowAsync(stream, chunkBuffer.AsMemory(0, cLen), ct).ConfigureAwait(false);
                                    await fs.WriteAsync(chunkBuffer.AsMemory(0, cLen), ct).ConfigureAwait(false);
                                }
                                else if (chunkHdr.PayloadLength > 0)
                                {
                                    int pLen = (int)chunkHdr.PayloadLength;
                                    byte[] discard = ArrayPool<byte>.Shared.Rent(pLen);
                                    try { await ReadExactOrThrowAsync(stream, discard.AsMemory(0, pLen), ct).ConfigureAwait(false); }
                                    finally { ArrayPool<byte>.Shared.Return(discard); }
                                }
                            }
                        }

                        if (File.Exists(localSlicePath)) File.Delete(localSlicePath);
                        File.Move(tempSlicePath, localSlicePath);
                        Console.WriteLine($"[Worker] Stream complete! Saved {sliceFileName} ({new FileInfo(localSlicePath).Length:N0} bytes). Loading...");
                    }
                    else
                    {
                        Console.WriteLine($"[Worker] Cache HIT: {localSlicePath}. Loading stage model directly from disk...");
                    }

                    stageGguf = GgufFile.Open(localSlicePath);
                    stageWeights = new ModelWeights(stageGguf);
                    dim = stageWeights.EmbeddingLength;

                    using (var pingCts = CancellationTokenSource.CreateLinkedTokenSource(ct))
                    {
                        var pingTask = Task.Run(async () =>
                        {
                            using var timer = new PeriodicTimer(TimeSpan.FromSeconds(2));
                            byte[] pingBytes = new byte[ClusterFrameHeader.HeaderSize];
                            var pingHeader = new ClusterFrameHeader(ClusterMessageType.HeartbeatPing, (ulong)currentStageId, 0);
                            pingHeader.WriteTo(pingBytes);
                            try
                            {
                                while (!pingCts.IsCancellationRequested && await timer.WaitForNextTickAsync(pingCts.Token).ConfigureAwait(false))
                                {
                                    await stream.WriteAsync(pingBytes.AsMemory(0, ClusterFrameHeader.HeaderSize), pingCts.Token).ConfigureAwait(false);
                                    await stream.FlushAsync(pingCts.Token).ConfigureAwait(false);
                                }
                            }
                            catch (OperationCanceledException) { }
                            catch { }
                        }, CancellationToken.None);

                        try
                        {
                            stageModel = CreateStageModel(stageWeights, 4096, startLayer, layerCount, isLastStage, adapterIndex);
                        }
                        finally
                        {
                            pingCts.Cancel();
                            try { await pingTask.ConfigureAwait(false); } catch { }
                        }
                    }

                    // Hoist all stage buffers once in POH (zero per-step LOH/Gen0 allocations)
                    singleIn = GC.AllocateArray<float>(Math.Max(dim, 64), pinned: true);
                    singleOut = GC.AllocateArray<float>(Math.Max(dim, 64), pinned: true);
                    batchIn = GC.AllocateArray<float>(DefaultWorkerMaxBatchTokens * Math.Max(dim, 64), pinned: true);
                    batchOut = GC.AllocateArray<float>(DefaultWorkerMaxBatchTokens * Math.Max(dim, 64), pinned: true);
                    batchTokensBuf = GC.AllocateArray<int>(DefaultWorkerMaxBatchTokens, pinned: true);
                    logits = isLastStage
                        ? GC.AllocateArray<float>(Math.Max(stageWeights.VocabSize, 64), pinned: true)
                        : [];

                    int minWireBytes = FusedSocketTransport.CombinedHeaderSize
                        + (DefaultWorkerMaxBatchTokens * Math.Max(dim, 64) * sizeof(float))
                        + PinnedBatchBufferArena.MaxSamplingTrailerBytes;
                    if (wireRecvBuf.Length < minWireBytes)
                        wireRecvBuf = GC.AllocateArray<byte>(minWireBytes, pinned: true);
                    if (wireSendBuf.Length < minWireBytes)
                        wireSendBuf = GC.AllocateArray<byte>(minWireBytes, pinned: true);

                    Console.WriteLine($"[Worker] Stage {currentStageId} successfully loaded! Backend: {stageModel.BackendName} | Dim: {dim}");
                    await SendStageReadyAsync(stream, controlSendBytes, status: 0, startLayer, endLayer, dim, ct).ConfigureAwait(false);
                }
                // -------------------------------------------------------------
                // 2a. BATCHED PREFILL EXECUTION (0x74 MsgActivationBatch -> 0x75 MsgResultBatch)
                // -------------------------------------------------------------
                else if (header.MessageType == MsgActivationBatch)
                {
                    if (stageModel == null || stageWeights == null)
                    {
                        Console.WriteLine("[Worker] Received MsgActivationBatch before stage initialization!");
                        break;
                    }

                    if (header.PayloadLength < FusedActivationHeader.HeaderSize ||
                        header.PayloadLength > FusedSocketTransport.MaxAllowedPayloadBytes)
                    {
                        throw new InvalidDataException($"Invalid MsgActivationBatch payload length: {header.PayloadLength}.");
                    }

                    int payloadLen = (int)header.PayloadLength;
                    if (payloadLen > wireRecvBuf.Length)
                    {
                        wireRecvBuf = GC.AllocateArray<byte>(payloadLen, pinned: true);
                    }
                    await ReadExactOrThrowAsync(stream, wireRecvBuf.AsMemory(0, payloadLen), ct).ConfigureAwait(false);

                    var actHdr = FusedActivationHeader.ReadFrom(wireRecvBuf.AsSpan(0, FusedActivationHeader.HeaderSize));
                    int numTokens = actHdr.BatchTokens > 0 ? actHdr.BatchTokens : 1;
                    int pos = actHdr.TokenIndex;
                    if (pos < 0 || pos + numTokens > DefaultWorkerMaxSeqLen)
                    {
                        throw new InvalidDataException(
                            $"MsgActivationBatch position range [{pos}..{pos + numTokens}) exceeds worker KV cache MaxSeqLen ({DefaultWorkerMaxSeqLen}).");
                    }
                    bool computeLogitsThisChunk = isLastStage && (actHdr.TypedFlags & FusedActivationFlags.FinalLayer) != 0;
                    bool isTokenInput = (actHdr.TypedFlags & FusedActivationFlags.Stage0TokenInput) != 0;
                    bool hasSamplingTrailer = (actHdr.TypedFlags & FusedActivationFlags.HasSamplingTrailer) != 0;
                    WirePrecisionMode wirePrec = actHdr.TypedPrecision;

                    int totalFloats = checked(numTokens * dim);
                    if (!isTokenInput && totalFloats > batchIn.Length)
                    {
                        batchIn = GC.AllocateArray<float>(totalFloats, pinned: true);
                    }
                    if (!isLastStage && totalFloats > batchOut.Length)
                    {
                        batchOut = GC.AllocateArray<float>(totalFloats, pinned: true);
                    }
                    if (isTokenInput && numTokens > batchTokensBuf.Length)
                    {
                        batchTokensBuf = GC.AllocateArray<int>(numTokens, pinned: true);
                    }

                    int dataByteCount = isTokenInput
                        ? checked(numTokens * sizeof(int))
                        : QuantizationKernels.GetPackedBatchBytes(wirePrec, dim, numTokens);

                    if (FusedActivationHeader.HeaderSize + dataByteCount > payloadLen)
                    {
                        throw new InvalidDataException(
                            $"Truncated MsgActivationBatch payload ({payloadLen} < {FusedActivationHeader.HeaderSize + dataByteCount}).");
                    }

                    var dataSpan = wireRecvBuf.AsSpan(FusedActivationHeader.HeaderSize, dataByteCount);
                    if (isTokenInput)
                    {
                        for (int i = 0; i < numTokens; i++)
                        {
                            batchTokensBuf[i] = BinaryPrimitives.ReadInt32LittleEndian(dataSpan.Slice(i * sizeof(int), sizeof(int)));
                        }
                    }
                    else
                    {
                        QuantizationKernels.DequantizeBatch(dataSpan, batchIn.AsSpan(0, totalFloats), dim, numTokens, wirePrec);
                    }

                    long tCompute0 = Stopwatch.GetTimestamp();
                    stageModel.ForwardBatchStage(
                        tokens: isTokenInput ? batchTokensBuf.AsSpan(0, numTokens) : ReadOnlySpan<int>.Empty,
                        startPos: pos,
                        inputXBatch: isTokenInput ? ReadOnlySpan<float>.Empty : batchIn.AsSpan(0, totalFloats),
                        outputXBatch: isLastStage ? Span<float>.Empty : batchOut.AsSpan(0, totalFloats),
                        logits: computeLogitsThisChunk ? logits.AsSpan(0, stageWeights.VocabSize) : Span<float>.Empty,
                        computeLogits: computeLogitsThisChunk);
                    double computeMs = (Stopwatch.GetTimestamp() - tCompute0) * 1000.0 / Stopwatch.Frequency;

                    int nextTokenId = 0;
                    float topLogit = 0f;
                    if (computeLogitsThisChunk)
                    {
                        var logitsSpan = logits.AsSpan(0, stageWeights.VocabSize);
                        int trailerStart = FusedActivationHeader.HeaderSize + dataByteCount;
                        if (hasSamplingTrailer && payloadLen > trailerStart)
                        {
                            (nextTokenId, topLogit) = SamplingWireCodec.SampleWithTrailer(
                                logitsSpan,
                                wireRecvBuf.AsSpan(trailerStart, payloadLen - trailerStart),
                                recentTokensScratch);
                        }
                        else
                        {
                            Sampler.SanitizeLogits(logitsSpan);
                            (nextTokenId, topLogit) = SamplingWireCodec.SampleGreedy(logitsSpan);
                        }
                    }

                    int outActBytes = isLastStage ? 0 : QuantizationKernels.GetPackedBatchBytes(wirePrec, dim, numTokens);
                    int respPayloadLen = FusedActivationResultHeader.HeaderSize + outActBytes;
                    int totalRespBytes = ClusterFrameHeader.HeaderSize + respPayloadLen;
                    if (totalRespBytes > wireSendBuf.Length)
                    {
                        wireSendBuf = GC.AllocateArray<byte>(totalRespBytes, pinned: true);
                    }

                    var respClusterHdr = new ClusterFrameHeader(MsgResultBatch, header.CorrelationId, (ulong)respPayloadLen);
                    var respResultHdr = new FusedActivationResultHeader(
                        stepIndex: pos,
                        stageId: currentStageId,
                        computeMs: computeMs,
                        nextTokenId: nextTokenId,
                        topLogit: topLogit,
                        precisionMode: wirePrec,
                        flags: computeLogitsThisChunk ? (byte)FusedActivationFlags.FinalLayer : (byte)0);

                    respClusterHdr.WriteTo(wireSendBuf.AsSpan(0, ClusterFrameHeader.HeaderSize));
                    respResultHdr.WriteTo(wireSendBuf.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));

                    if (outActBytes > 0)
                    {
                        QuantizationKernels.QuantizeBatch(
                            batchOut.AsSpan(0, totalFloats),
                            wireSendBuf.AsSpan(FusedSocketTransport.CombinedHeaderSize, outActBytes),
                            dim,
                            numTokens,
                            wirePrec);
                    }

                    await stream.WriteAsync(wireSendBuf.AsMemory(0, totalRespBytes), ct).ConfigureAwait(false);
                    await stream.FlushAsync(ct).ConfigureAwait(false);
                }
                // -------------------------------------------------------------
                // 2b. SINGLE-TOKEN DECODE EXECUTION (0x70 MsgActivation -> 0x71 MsgResult)
                // -------------------------------------------------------------
                else if (header.MessageType == MsgActivation)
                {
                    if (stageModel == null || stageWeights == null)
                    {
                        Console.WriteLine("[Worker] Received MsgActivation before stage initialization!");
                        break;
                    }

                    if (header.PayloadLength < 16 || header.PayloadLength > FusedSocketTransport.MaxAllowedPayloadBytes)
                    {
                        throw new InvalidDataException($"Invalid MsgActivation payload length: {header.PayloadLength}.");
                    }

                    int payloadLen = (int)header.PayloadLength;
                    if (payloadLen > wireRecvBuf.Length)
                    {
                        wireRecvBuf = GC.AllocateArray<byte>(payloadLen, pinned: true);
                    }
                    await ReadExactOrThrowAsync(stream, wireRecvBuf.AsMemory(0, payloadLen), ct).ConfigureAwait(false);

                    // Modern FusedActivationHeader ("GACT") protocol
                    if (payloadLen >= FusedActivationHeader.HeaderSize &&
                        BinaryPrimitives.ReadUInt32LittleEndian(wireRecvBuf.AsSpan(0, 4)) == FusedActivationHeader.FusedMagic)
                    {
                        var actHdr = FusedActivationHeader.ReadFrom(wireRecvBuf.AsSpan(0, FusedActivationHeader.HeaderSize));
                        int pos = actHdr.TokenIndex;
                        if (pos < 0 || pos >= DefaultWorkerMaxSeqLen)
                        {
                            throw new InvalidDataException(
                                $"MsgActivation position ({pos}) is outside worker KV cache bounds [0..{DefaultWorkerMaxSeqLen}).");
                        }
                        bool computeLogitsThisStage = isLastStage && (actHdr.TypedFlags & FusedActivationFlags.FinalLayer) != 0;
                        bool isTokenInput = (actHdr.TypedFlags & FusedActivationFlags.Stage0TokenInput) != 0;
                        bool hasSamplingTrailer = (actHdr.TypedFlags & FusedActivationFlags.HasSamplingTrailer) != 0;
                        WirePrecisionMode wirePrec = actHdr.TypedPrecision;

                        int tokenId = 0;
                        int dataByteCount = isTokenInput
                            ? sizeof(int)
                            : QuantizationKernels.GetPackedTokenBytes(wirePrec, dim);

                        if (FusedActivationHeader.HeaderSize + dataByteCount > payloadLen)
                        {
                            throw new InvalidDataException(
                                $"Truncated fused MsgActivation payload ({payloadLen} < {FusedActivationHeader.HeaderSize + dataByteCount}).");
                        }

                        var dataSpan = wireRecvBuf.AsSpan(FusedActivationHeader.HeaderSize, dataByteCount);
                        if (isTokenInput)
                        {
                            tokenId = BinaryPrimitives.ReadInt32LittleEndian(dataSpan);
                        }
                        else
                        {
                            QuantizationKernels.DequantizeToken(dataSpan, singleIn.AsSpan(0, dim), wirePrec);
                        }

                        long tCompute0 = Stopwatch.GetTimestamp();
                        stageModel.ForwardStage(
                            token: tokenId,
                            pos: pos,
                            inputX: isTokenInput ? ReadOnlySpan<float>.Empty : singleIn.AsSpan(0, dim),
                            outputX: isLastStage ? Span<float>.Empty : singleOut.AsSpan(0, dim),
                            logits: computeLogitsThisStage ? logits.AsSpan(0, stageWeights.VocabSize) : Span<float>.Empty,
                            computeLogits: computeLogitsThisStage);
                        double computeMs = (Stopwatch.GetTimestamp() - tCompute0) * 1000.0 / Stopwatch.Frequency;

                        int nextTokenId = 0;
                        float topLogit = 0f;
                        if (computeLogitsThisStage)
                        {
                            var logitsSpan = logits.AsSpan(0, stageWeights.VocabSize);
                            int trailerStart = FusedActivationHeader.HeaderSize + dataByteCount;
                            if (hasSamplingTrailer && payloadLen > trailerStart)
                            {
                                (nextTokenId, topLogit) = SamplingWireCodec.SampleWithTrailer(
                                    logitsSpan,
                                    wireRecvBuf.AsSpan(trailerStart, payloadLen - trailerStart),
                                    recentTokensScratch);
                            }
                            else
                            {
                                Sampler.SanitizeLogits(logitsSpan);
                                (nextTokenId, topLogit) = SamplingWireCodec.SampleGreedy(logitsSpan);
                            }
                        }

                        int outActBytes = isLastStage ? 0 : QuantizationKernels.GetPackedTokenBytes(wirePrec, dim);
                        int respPayloadLen = FusedActivationResultHeader.HeaderSize + outActBytes;
                        int totalRespBytes = ClusterFrameHeader.HeaderSize + respPayloadLen;
                        if (totalRespBytes > wireSendBuf.Length)
                        {
                            wireSendBuf = GC.AllocateArray<byte>(totalRespBytes, pinned: true);
                        }

                        var respClusterHdr = new ClusterFrameHeader(MsgResult, header.CorrelationId, (ulong)respPayloadLen);
                        var respResultHdr = new FusedActivationResultHeader(
                            stepIndex: pos,
                            stageId: currentStageId,
                            computeMs: computeMs,
                            nextTokenId: nextTokenId,
                            topLogit: topLogit,
                            precisionMode: wirePrec,
                            flags: computeLogitsThisStage ? (byte)FusedActivationFlags.FinalLayer : (byte)0);

                        respClusterHdr.WriteTo(wireSendBuf.AsSpan(0, ClusterFrameHeader.HeaderSize));
                        respResultHdr.WriteTo(wireSendBuf.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationResultHeader.HeaderSize));

                        if (outActBytes > 0)
                        {
                            QuantizationKernels.QuantizeToken(
                                singleOut.AsSpan(0, dim),
                                wireSendBuf.AsSpan(FusedSocketTransport.CombinedHeaderSize, outActBytes),
                                wirePrec);
                        }

                        await stream.WriteAsync(wireSendBuf.AsMemory(0, totalRespBytes), ct).ConfigureAwait(false);
                        await stream.FlushAsync(ct).ConfigureAwait(false);
                    }
                    else
                    {
                        // Backward-compatible legacy 16-byte header format
                        int stepIndex = BitConverter.ToInt32(wireRecvBuf, 0);
                        int pos = BitConverter.ToInt32(wireRecvBuf, 4);
                        int isLastStageInt = BitConverter.ToInt32(wireRecvBuf, 8);
                        int actDim = BitConverter.ToInt32(wireRecvBuf, 12);
                        if (actDim <= 0 || actDim > 65536 || 16 + (actDim * sizeof(float)) > payloadLen)
                        {
                            throw new InvalidDataException($"Invalid legacy MsgActivation actDim ({actDim}) for payload length {payloadLen}.");
                        }

                        bool computeLogitsThisStage = isLastStage && isLastStageInt != 0;
                        int actByteCount = actDim * sizeof(float);
                        var inSpan = MemoryMarshal.Cast<byte, float>(wireRecvBuf.AsSpan(16, actByteCount));
                        if (actDim > singleOut.Length)
                        {
                            singleOut = GC.AllocateArray<float>(actDim, pinned: true);
                        }

                        long tCompute0 = Stopwatch.GetTimestamp();
                        stageModel.ForwardStage(
                            token: 0,
                            pos: pos,
                            inputX: inSpan,
                            outputX: singleOut.AsSpan(0, actDim),
                            logits: computeLogitsThisStage ? logits.AsSpan(0, stageWeights.VocabSize) : Span<float>.Empty,
                            computeLogits: computeLogitsThisStage);
                        double computeMs = (Stopwatch.GetTimestamp() - tCompute0) * 1000.0 / Stopwatch.Frequency;

                        int nextTokenId = 0;
                        float topLogit = 0f;
                        if (computeLogitsThisStage)
                        {
                            var logitsSpan = logits.AsSpan(0, stageWeights.VocabSize);
                            Sampler.SanitizeLogits(logitsSpan);
                            (nextTokenId, topLogit) = SamplingWireCodec.SampleGreedy(logitsSpan);
                        }

                        int respLen = 28 + actByteCount;
                        int totalRespBytes = ClusterFrameHeader.HeaderSize + respLen;
                        if (totalRespBytes > wireSendBuf.Length)
                        {
                            wireSendBuf = GC.AllocateArray<byte>(totalRespBytes, pinned: true);
                        }

                        var respHdr = new ClusterFrameHeader(MsgResult, header.CorrelationId, (ulong)respLen);
                        respHdr.WriteTo(wireSendBuf.AsSpan(0, ClusterFrameHeader.HeaderSize));

                        var resSpan = wireSendBuf.AsSpan(ClusterFrameHeader.HeaderSize, respLen);
                        BitConverter.TryWriteBytes(resSpan.Slice(0, 4), stepIndex);
                        BitConverter.TryWriteBytes(resSpan.Slice(4, 4), currentStageId);
                        BitConverter.TryWriteBytes(resSpan.Slice(8, 8), computeMs);
                        BitConverter.TryWriteBytes(resSpan.Slice(16, 4), nextTokenId);
                        BitConverter.TryWriteBytes(resSpan.Slice(20, 4), topLogit);
                        BitConverter.TryWriteBytes(resSpan.Slice(24, 4), actDim);
                        MemoryMarshal.Cast<float, byte>(singleOut.AsSpan(0, actDim)).CopyTo(resSpan.Slice(28, actByteCount));

                        await stream.WriteAsync(wireSendBuf.AsMemory(0, totalRespBytes), ct).ConfigureAwait(false);
                        await stream.FlushAsync(ct).ConfigureAwait(false);
                    }
                }
                // -------------------------------------------------------------
                // 3. RESET KV CACHE (FOR MULTI-TURN CHAT CONVERSATION)
                // -------------------------------------------------------------
                else if (header.MessageType == MsgResetKv)
                {
                    if (header.PayloadLength > 0)
                    {
                        if (header.PayloadLength > FusedSocketTransport.MaxAllowedPayloadBytes)
                            throw new InvalidDataException($"Oversized MsgResetKv payload: {header.PayloadLength}.");
                        int pLen = (int)header.PayloadLength;
                        byte[] discard = ArrayPool<byte>.Shared.Rent(pLen);
                        try { await ReadExactOrThrowAsync(stream, discard.AsMemory(0, pLen), ct).ConfigureAwait(false); }
                        finally { ArrayPool<byte>.Shared.Return(discard); }
                    }

                    stageModel?.ResetKvCache();

                    var ackHdr = new ClusterFrameHeader(MsgResetKv, header.CorrelationId, 0);
                    ackHdr.WriteTo(controlSendBytes.AsSpan(0, ClusterFrameHeader.HeaderSize));
                    await stream.WriteAsync(controlSendBytes.AsMemory(0, ClusterFrameHeader.HeaderSize), ct).ConfigureAwait(false);
                    await stream.FlushAsync(ct).ConfigureAwait(false);
                }
                // -------------------------------------------------------------
                // 4. SHUTDOWN
                // -------------------------------------------------------------
                else if (header.MessageType == MsgShutdown)
                {
                    Console.WriteLine("[Worker] Received shutdown frame. Resetting session for next run.");
                    break;
                }
                // -------------------------------------------------------------
                // 4b. THROUGHPUT CALIBRATION PROBE (0x76 -> 0x77)
                // -------------------------------------------------------------
                else if (header.MessageType == DynamicThroughputCalibrator.MsgCalibrateProbeRequest)
                {
                    Console.WriteLine("[Worker] Handling empirical throughput calibration probe...");
                    await DynamicThroughputCalibrator.HandleProbeRequestAsync(stream, header, null, ct).ConfigureAwait(false);
                }
                // -------------------------------------------------------------
                // 5. HEARTBEAT PING & PONG
                // -------------------------------------------------------------
                else if (header.MessageType == ClusterMessageType.HeartbeatPing)
                {
                    if (header.PayloadLength > 0)
                    {
                        if (header.PayloadLength > FusedSocketTransport.MaxAllowedPayloadBytes)
                            throw new InvalidDataException($"Oversized HeartbeatPing payload: {header.PayloadLength}.");
                        int pLen = (int)header.PayloadLength;
                        byte[] discard = ArrayPool<byte>.Shared.Rent(pLen);
                        try { await ReadExactOrThrowAsync(stream, discard.AsMemory(0, pLen), ct).ConfigureAwait(false); }
                        finally { ArrayPool<byte>.Shared.Return(discard); }
                    }

                    var pongHdr = new ClusterFrameHeader(ClusterMessageType.HeartbeatPong, header.CorrelationId, 0);
                    pongHdr.WriteTo(controlSendBytes.AsSpan(0, ClusterFrameHeader.HeaderSize));
                    await stream.WriteAsync(controlSendBytes.AsMemory(0, ClusterFrameHeader.HeaderSize), ct).ConfigureAwait(false);
                    await stream.FlushAsync(ct).ConfigureAwait(false);
                }
                else if (header.MessageType == ClusterMessageType.HeartbeatPong)
                {
                    if (header.PayloadLength > 0)
                    {
                        if (header.PayloadLength > FusedSocketTransport.MaxAllowedPayloadBytes)
                            throw new InvalidDataException($"Oversized HeartbeatPong payload: {header.PayloadLength}.");
                        int pLen = (int)header.PayloadLength;
                        byte[] discard = ArrayPool<byte>.Shared.Rent(pLen);
                        try { await ReadExactOrThrowAsync(stream, discard.AsMemory(0, pLen), ct).ConfigureAwait(false); }
                        finally { ArrayPool<byte>.Shared.Return(discard); }
                    }
                }
                // -------------------------------------------------------------
                // 6. DEFENSIVE CATCH-ALL UNHANDLED FRAME DRAINER
                // -------------------------------------------------------------
                else
                {
                    if (header.PayloadLength > FusedSocketTransport.MaxAllowedPayloadBytes)
                        throw new InvalidDataException($"Oversized unhandled frame (0x{(byte)header.MessageType:X2}): {header.PayloadLength} bytes.");

                    Console.WriteLine($"[Worker] Unhandled message opcode: 0x{(byte)header.MessageType:X2} (CorrelationId={header.CorrelationId}, PayloadLength={header.PayloadLength}). Safely draining payload to preserve stream synchronization...");
                    if (header.PayloadLength > 0)
                    {
                        int pLen = (int)header.PayloadLength;
                        byte[] discard = ArrayPool<byte>.Shared.Rent(pLen);
                        try { await ReadExactOrThrowAsync(stream, discard.AsMemory(0, pLen), ct).ConfigureAwait(false); }
                        finally { ArrayPool<byte>.Shared.Return(discard); }
                    }
                }
            }
        }
        finally
        {
            stageModel?.Dispose();
            stageGguf?.Dispose();
        }
    }

    private static async Task SendStageReadyAsync(
        Stream stream,
        byte[] scratchBuf,
        int status,
        int startLayer,
        int endLayer,
        int dim,
        CancellationToken ct)
    {
        var hdr = new ClusterFrameHeader(MsgStageReady, 0, 16);
        hdr.WriteTo(scratchBuf.AsSpan(0, ClusterFrameHeader.HeaderSize));

        var payload = scratchBuf.AsSpan(ClusterFrameHeader.HeaderSize, 16);
        BitConverter.TryWriteBytes(payload.Slice(0, 4), status);
        BitConverter.TryWriteBytes(payload.Slice(4, 4), startLayer);
        BitConverter.TryWriteBytes(payload.Slice(8, 4), endLayer);
        BitConverter.TryWriteBytes(payload.Slice(12, 4), dim);

        await stream.WriteAsync(scratchBuf.AsMemory(0, ClusterFrameHeader.HeaderSize + 16), ct).ConfigureAwait(false);
        await stream.FlushAsync(ct).ConfigureAwait(false);
    }

    private static async Task<bool> ReadHeaderOrEofAsync(Stream s, byte[] headerBuf, CancellationToken ct)
    {
        int total = 0;
        while (total < ClusterFrameHeader.HeaderSize)
        {
            int r = await s.ReadAsync(headerBuf.AsMemory(total, ClusterFrameHeader.HeaderSize - total), ct).ConfigureAwait(false);
            if (r == 0)
            {
                if (total == 0) return false;
                throw new EndOfStreamException($"Truncated frame header: expected {ClusterFrameHeader.HeaderSize} bytes, got {total}.");
            }
            total += r;
        }
        return true;
    }

    private static async Task ReadExactOrThrowAsync(Stream s, Memory<byte> buffer, CancellationToken ct)
    {
        int total = 0;
        while (total < buffer.Length)
        {
            int r = await s.ReadAsync(buffer.Slice(total), ct).ConfigureAwait(false);
            if (r == 0)
                throw new EndOfStreamException($"Expected {buffer.Length} payload bytes, got {total}.");
            total += r;
        }
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
        int hostGpuAdapter,
        bool serve,
        string httpHost,
        int httpPort,
        string? apiKey,
        CancellationToken ct = default)
    {
        using var session = await DistributedClusterSession.InitializeAsync(
            remoteEndpoints,
            modelPath,
            cacheDir,
            layerAllocArg,
            hostGpuAdapter,
            includeLocalHost: true,
            ct: ct);

        if (serve)
        {
            return await RunServeAsync(session, httpHost, httpPort, apiKey);
        }

        string activePrompt = initialPrompt;
        while (!ct.IsCancellationRequested)
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
                    await session.ResetKvCacheAsync(ct);
                    Console.WriteLine("[Cluster] KV cache cleared! Ready for new conversation.\n");
                    continue;
                }
                activePrompt = input;
            }

            Console.WriteLine($"[Prompt]: \"{activePrompt}\"");
            Console.WriteLine("--------------------------------------------------------------------------------");
            Console.WriteLine($"GENERATING REAL TOKENS ACROSS {session.Stages.Length}-STAGE PIPELINE ({session.LocalModels.Count} Local GPU + {session.Streams.Count} Remote LAN)");
            Console.WriteLine("--------------------------------------------------------------------------------");

            var options = new SamplingOptions
            {
                MaxTokens = maxTokens <= 0 ? 4096 : maxTokens,
                Temperature = 0.7f
            };

            ClusterGenerationResult result;
            try
            {
                result = await session.GenerateAsync(
                    activePrompt,
                    options,
                    onToken: piece => Console.Write(piece),
                    ct: ct);
            }
            catch (OperationCanceledException) when (ct.IsCancellationRequested)
            {
                Console.WriteLine("\n[Cluster] Generation cancelled.");
                break;
            }

            Console.WriteLine("\n\n================================================================================");
            Console.WriteLine("  [BENCHMARK CLASSIFICATION: END-TO-END DISTRIBUTED PIPELINE INFERENCE]");
            Console.WriteLine($"  Topology: {session.Stages.Length}-Stage Heterogeneous Pipeline ({session.LocalModels.Count} Local GPU + {session.Streams.Count} Remote LAN Nodes)");
            Console.WriteLine("================================================================================");
            double prefillTokensPerSec = result.PromptTokens / Math.Max(0.0001, result.PromptDuration.TotalSeconds);
            double prefillMsPerToken = result.PromptDuration.TotalMilliseconds / Math.Max(1, result.PromptTokens);
            double decodeTokensPerSec = result.GeneratedTokens / Math.Max(0.0001, result.GenerationDuration.TotalSeconds);
            double decodeMsPerToken = result.GenerationDuration.TotalMilliseconds / Math.Max(1, result.GeneratedTokens);
            double ttftMs = result.PromptDuration.TotalMilliseconds;

            Console.WriteLine($"| {"Metric",-28} | {"Tokens",-8} | {"Duration (ms)",-14} | {"Throughput",-14} | {"Latency",-12} |");
            Console.WriteLine($"|{new string('-', 30)}|{new string('-', 10)}|{new string('-', 16)}|{new string('-', 16)}|{new string('-', 14)}|");
            Console.WriteLine($"| {"Tokens Read (Prompt Prefill)",-28} | {result.PromptTokens,8} | {$"{result.PromptDuration.TotalMilliseconds:F2} ms",-14} | {$"{prefillTokensPerSec:F2} tok/s",-14} | {$"{prefillMsPerToken:F2} ms/tok",-12} |");
            Console.WriteLine($"| {"Tokens Out (Decode Generation)",-28} | {result.GeneratedTokens,8} | {$"{result.GenerationDuration.TotalMilliseconds:F2} ms",-14} | {$"{decodeTokensPerSec:F2} tok/s",-14} | {$"{decodeMsPerToken:F2} ms/tok",-12} |");
            Console.WriteLine($"| {"Time to First Token (TTFT)",-28} | {1,8} | {$"{ttftMs:F2} ms",-14} | {$"{1000.0 / Math.Max(0.1, ttftMs):F2} tok/s",-14} | {$"{ttftMs:F2} ms",-12} |");
            Console.WriteLine($"| {"Total End-to-End Execution",-28} | {result.PromptTokens + result.GeneratedTokens,8} | {$"{result.TotalDuration.TotalMilliseconds:F2} ms",-14} | {$"{(result.PromptTokens + result.GeneratedTokens) / Math.Max(0.0001, result.TotalDuration.TotalSeconds):F2} tok/s",-14} | {"-",-12} |");
            Console.WriteLine($"|{new string('-', 30)}|{new string('-', 10)}|{new string('-', 16)}|{new string('-', 16)}|{new string('-', 14)}|");

            double totalClusterCompute = 0;
            double totalClusterNet = 0;
            Console.WriteLine("\n  Stage Compute & Network Breakdown per Step:");
            for (int i = 0; i < session.Stages.Length; i++)
            {
                var st = session.Stages[i];
                double avgComp = st.Invocations > 0 ? st.TotalComputeMs / st.Invocations : 0;
                double avgNet = st.Invocations > 0 ? st.TotalNetworkMs / st.Invocations : 0;
                totalClusterCompute += avgComp;
                totalClusterNet += avgNet;

                string netStr = (i < session.LocalModels.Count) ? "Local Direct Memory (0.0 ms)" : $"{avgNet:F3} ms Gigabit LAN";
                Console.WriteLine($"    Stage {i} [{st.Name}]:");
                Console.WriteLine($"      Layers {st.StartLayer}..{st.EndLayer} ({st.LayerCount}L) | Compute: {avgComp:F3} ms | Transit: {netStr}");
            }

            double efficiency = (totalClusterCompute / Math.Max(0.001, totalClusterCompute + totalClusterNet)) * 100.0;
            Console.WriteLine($"\n  Total Compute Time:          {totalClusterCompute:F3} ms / token");
            Console.WriteLine($"  Total Network Transit Time:  {totalClusterNet:F3} ms / token");
            Console.WriteLine($"  Cluster Pipeline Efficiency: {efficiency:F1}% compute utilization");
            Console.WriteLine("================================================================================\n");

            activePrompt = "";
            if (!interactive) break;
        }

        return 0;
    }

    /// <summary>
    /// Helper Stream adapter that flushes buffered chunks as discrete MsgStreamChunk frames over TCP.
    /// </summary>
    public sealed class NetworkChunkWriter : Stream
    {
        private readonly Stream _underlying;
        private readonly ClusterMessageType _msgType;
        private readonly byte[] _buffer = new byte[4 * 1024 * 1024];
        private readonly byte[] _hdrBytes = new byte[ClusterFrameHeader.HeaderSize];
        private int _buffered;

        public NetworkChunkWriter(Stream underlying, ClusterMessageType msgType)
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
            hdr.WriteTo(_hdrBytes);

            _underlying.Write(_hdrBytes, 0, ClusterFrameHeader.HeaderSize);
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
        int adapterIndex = -1,
        bool forceCpu = false)
    {
        // 1. On Windows, check Direct3D 12 first (hardware Wave32/DirectML), unless forceCpu is requested
        if (!forceCpu &&
            OperatingSystem.IsWindows() &&
            D3D12Context.IsSupported &&
            (weights.ArchitectureFamily == UniversalArchitecture.Qwen ||
             weights.ArchitectureFamily == UniversalArchitecture.Llama ||
             weights.ArchitectureFamily == UniversalArchitecture.HybridSsm ||
             weights.ArchitectureFamily == UniversalArchitecture.Generic))
        {
            int safeD3DAdapter = ResolveSafeD3D12AdapterIndex(adapterIndex);
            var d3dModel = TryCreateD3D12StageModel(weights, maxSeqLen, startLayer, layerCount, isLastStage, safeD3DAdapter);
            if (d3dModel != null) return d3dModel;
        }

        // Note: Experimental VulkanClusterStageModel is intentionally NOT auto-selected here
        // until Vulkan compute shader dispatch and weight upload are implemented.

        // 2. High-performance CPU SIMD fallback (AVX-512 / AVX2 / ARM AdvSIMD)
        string cpuSimd = System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated ? "AVX-512" :
                         (System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated ? "AVX2" : "Scalar");
        Console.WriteLine($"[AutoCompute] Using CPU SIMD Runtime: {Environment.ProcessorCount} Cores ({cpuSimd})");
        var cpuModel = CpuModelFactory.Create(weights, maxSeqLen, startLayer, layerCount, isLastStage);
        var kv = new KVCache(layerCount, weights.HeadCountKv, weights.HeadDim, maxSeqLen);
        return new CpuClusterStageModel(cpuModel, kv, $"CPU SIMD ({Environment.ProcessorCount} Cores, {cpuSimd})");
    }

    public static int ResolveSafeD3D12AdapterIndex(int requestedIndex)
    {
        if (!OperatingSystem.IsWindows()) return -1;
        try
        {
            var adapters = D3D12Context.EnumerateAdapters();
            if (adapters.Count == 0) return -1;
            if (requestedIndex < 0) return -1; // Auto-select highest dedicated VRAM
            if (requestedIndex >= adapters.Count)
            {
                int clamped = adapters.Count - 1;
                Console.WriteLine($"[AutoCompute] Warning: Requested D3D12 adapter index {requestedIndex} exceeds available count ({adapters.Count}). Clamping to index {clamped} ({adapters[clamped].Name}) to prevent CPU fallback.");
                return clamped;
            }
            return requestedIndex;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AutoCompute] Warning: D3D12 adapter enumeration failed: {ex.Message}. Defaulting to auto adapter.");
            return -1;
        }
    }

    public static int ResolveSafeVulkanDeviceIndex(int requestedIndex)
    {
        if (!VulkanContext.IsSupported) return 0;
        try
        {
            var devices = VulkanContext.EnumerateDevices();
            if (devices.Count == 0) return 0;
            if (requestedIndex < 0) return 0;
            if (requestedIndex >= devices.Count)
            {
                int clamped = devices.Count - 1;
                Console.WriteLine($"[AutoCompute] Warning: Requested Vulkan device index {requestedIndex} exceeds available count ({devices.Count}). Clamping to index {clamped} ({devices[clamped].Name}) to prevent CPU fallback.");
                return clamped;
            }
            return requestedIndex;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[AutoCompute] Warning: Vulkan device enumeration failed: {ex.Message}. Defaulting to device 0.");
            return 0;
        }
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
        D3D12Context? ctx = null;
        try
        {
            ctx = new D3D12Context(adapterIndex);
            ulong dedicatedMb = ctx.DedicatedVideoMemory / (1024 * 1024);
            ulong sharedMb = ctx.SharedSystemMemory / (1024 * 1024);
            ulong totalMb = ctx.TotalAccessibleMemory / (1024 * 1024);

            Console.WriteLine($"[AutoCompute] Detected GPU: {ctx.DeviceName}");
            Console.WriteLine($"[AutoCompute] VRAM: {dedicatedMb:N0} MB Dedicated, {sharedMb:N0} MB Shared ({totalMb:N0} MB Total Accessible)");

            if (weights.ArchitectureFamily == UniversalArchitecture.HybridSsm)
            {
                var hybridModel = new Qwen3HybridD3D12Model(ctx, weights, maxSeqLen, startLayer, layerCount, isLastStage);
                string hybridMode = hybridModel.TieredMoe ? "Heterogeneous VRAM+CPU Tiered HybridSSM+MoE" : "100% In-VRAM GPU Accelerated HybridSSM+MoE";
                Console.WriteLine($"[AutoCompute] Direct3D 12 HybridSsm GPU Engine Loaded! Mode: {hybridMode}");
                var wrappedHybrid = new D3D12ClusterStageModel(ctx, hybridModel, $"D3D12 GPU HybridSsm ({ctx.DeviceName})");
                ctx = null;
                return wrappedHybrid;
            }

            var d3dModel = new Qwen2D3D12Model(ctx, weights, maxSeqLen, KvCachePrecision.Auto, startLayer, layerCount, isLastStage);
            string modeStr = d3dModel.TieredMoe ? "Heterogeneous VRAM+CPU Tiered" : "100% In-VRAM GPU Accelerated";
            Console.WriteLine($"[AutoCompute] Direct3D 12 GPU Engine Loaded! Mode: {modeStr}");
            var wrapped = new D3D12ClusterStageModel(ctx, d3dModel, $"D3D12 GPU ({ctx.DeviceName})");
            ctx = null;
            return wrapped;
        }
        catch (Exception ex)
        {
            ctx?.Dispose();
            Console.WriteLine($"[AutoCompute] Direct3D 12 GPU fallback to CPU SIMD: {ex.Message}");
            return null;
        }
    }
}
