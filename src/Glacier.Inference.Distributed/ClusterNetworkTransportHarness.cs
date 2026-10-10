namespace Glacier.Inference.Distributed;

using System;
using System.Buffers;
using System.Diagnostics;
using System.IO.Pipelines;
using System.Net;
using System.Net.Sockets;
using System.Security;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Cluster.Shuffle;

/// <summary>
/// High-performance interconnect benchmark harness measuring end-to-end frame transfer latency,
/// sustained network throughput, serialization overhead, and GC memory allocations using Glacier.Cluster protocol frames.
/// Strictly confined to loopback (127.0.0.1) and authorized local subnet fleet bounds (192.168.1.0/24).
/// </summary>
public sealed class ClusterNetworkTransportHarness
{
    /// <summary>
    /// Comprehensive benchmark measurement results.
    /// </summary>
    public sealed class BenchmarkResult
    {
        public string TargetNode { get; init; } = "";
        public int ActivationBytes { get; init; }
        public int TotalFrames { get; init; }
        public double ElapsedMs { get; init; }
        public double ThroughputMBs { get; init; }
        public double FramesPerSecond { get; init; }
        public double MeanLatencyUs { get; init; }
        public double P50LatencyUs { get; init; }
        public double P95LatencyUs { get; init; }
        public double P99LatencyUs { get; init; }
        public double MinLatencyUs { get; init; }
        public double MaxLatencyUs { get; init; }
        public long GcAllocatedBytes { get; init; }
        public double SerializationOverheadNs { get; init; }

        public override string ToString() =>
            $"[Node: {TargetNode}] Frames: {TotalFrames}, Size: {ActivationBytes} B, Throughput: {ThroughputMBs:F2} MB/s ({FramesPerSecond:F0} fps), " +
            $"Mean: {MeanLatencyUs:F1} µs, P50: {P50LatencyUs:F1} µs, P99: {P99LatencyUs:F1} µs, GC Alloc: {GcAllocatedBytes} B";
    }

    /// <summary>
    /// Validates that target host is strictly within allowed local network bounds (127.0.0.1 or 192.168.1.0/24).
    /// Prevents external or unauthorized socket connections.
    /// </summary>
    public static bool IsSubnetAllowed(string host)
    {
        if (string.Equals(host, "127.0.0.1", StringComparison.OrdinalIgnoreCase) ||
            string.Equals(host, "localhost", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (IPAddress.TryParse(host, out var ip))
        {
            if (IPAddress.IsLoopback(ip)) return true;

            var bytes = ip.GetAddressBytes();
            if (bytes.Length == 4 && bytes[0] == 192 && bytes[1] == 168 && bytes[2] == 1)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Enforces subnet bounds or throws <see cref="SecurityException"/>.
    /// </summary>
    public static void ValidateSubnetOrThrow(string host)
    {
        if (!IsSubnetAllowed(host))
        {
            throw new SecurityException($"Target host '{host}' violates local subnet confinement (strictly restricted to loopback and 192.168.1.0/24).");
        }
    }

    /// <summary>
    /// Measures pure serialization and deserialization overhead of ClusterFrameHeader and ActivationHeader.
    /// </summary>
    public static BenchmarkResult MeasureSerializationOverhead(int iterations = 100_000, int hiddenDim = 4096)
    {
        int activationBytes = hiddenDim * sizeof(float);
        byte[] buffer = new byte[ClusterFrameHeader.HeaderSize + ActivationHeader.HeaderSize];

        var actHeader = new ActivationHeader(
            sequenceId: 42,
            tokenIndex: 128,
            layerIndex: 15,
            hiddenDim: hiddenDim,
            expertId: 3,
            flags: ActivationFlags.None);

        var clusterHeader = new ClusterFrameHeader(
            messageType: ClusterMessageType.StageDispatch,
            correlationId: 42,
            payloadLength: (ulong)(ActivationHeader.HeaderSize + activationBytes),
            flags: ClusterFrameFlags.None);

        // Warmup
        for (int i = 0; i < 1_000; i++)
        {
            clusterHeader.WriteTo(buffer.AsSpan(0, ClusterFrameHeader.HeaderSize));
            actHeader.WriteTo(buffer.AsSpan(ClusterFrameHeader.HeaderSize, ActivationHeader.HeaderSize));
            var r1 = ClusterFrameHeader.ReadFrom(buffer.AsSpan(0, ClusterFrameHeader.HeaderSize));
            var r2 = ActivationHeader.ReadFrom(buffer.AsSpan(ClusterFrameHeader.HeaderSize, ActivationHeader.HeaderSize));
        }

        long allocBefore = GC.GetAllocatedBytesForCurrentThread();
        long t0 = Stopwatch.GetTimestamp();

        for (int i = 0; i < iterations; i++)
        {
            clusterHeader.WriteTo(buffer.AsSpan(0, ClusterFrameHeader.HeaderSize));
            actHeader.WriteTo(buffer.AsSpan(ClusterFrameHeader.HeaderSize, ActivationHeader.HeaderSize));
            var r1 = ClusterFrameHeader.ReadFrom(buffer.AsSpan(0, ClusterFrameHeader.HeaderSize));
            var r2 = ActivationHeader.ReadFrom(buffer.AsSpan(ClusterFrameHeader.HeaderSize, ActivationHeader.HeaderSize));
        }

        long elapsedNs = (long)Stopwatch.GetElapsedTime(t0).TotalNanoseconds;
        long allocAfter = GC.GetAllocatedBytesForCurrentThread();

        double overheadPerOpNs = (double)elapsedNs / iterations;

        return new BenchmarkResult
        {
            TargetNode = "In-Process Memory Serialization",
            ActivationBytes = activationBytes,
            TotalFrames = iterations,
            ElapsedMs = elapsedNs / 1_000_000.0,
            ThroughputMBs = 0,
            FramesPerSecond = iterations / (elapsedNs / 1_000_000_000.0),
            MeanLatencyUs = overheadPerOpNs / 1_000.0,
            P50LatencyUs = overheadPerOpNs / 1_000.0,
            P95LatencyUs = overheadPerOpNs / 1_000.0,
            P99LatencyUs = overheadPerOpNs / 1_000.0,
            MinLatencyUs = overheadPerOpNs / 1_000.0,
            MaxLatencyUs = overheadPerOpNs / 1_000.0,
            GcAllocatedBytes = allocAfter - allocBefore,
            SerializationOverheadNs = overheadPerOpNs
        };
    }

    /// <summary>
    /// Executes an in-process activation streaming and roundtrip ping-pong benchmark over <see cref="DuplexPipeFactory.CreateInProcessPair"/>.
    /// Demonstrates zero-allocation System.IO.Pipelines wire framing.
    /// </summary>
    public static async Task<BenchmarkResult> RunInProcessBenchmarkAsync(
        int vectorDimensions = 4096,
        int frameCount = 10_000,
        CancellationToken cancellationToken = default)
    {
        int activationBytes = vectorDimensions * sizeof(float);
        var (clientPipe, serverPipe) = DuplexPipeFactory.CreateInProcessPair();

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

        // Echo server loop
        var serverTask = Task.Run(async () =>
        {
            try
            {
                while (!cts.IsCancellationRequested)
                {
                    var (hdr, buf) = await DistributedPipelineCoordinator.ReadActivationFrameAsync(serverPipe.Input, cts.Token);
                    using (buf)
                    {
                        await DistributedPipelineCoordinator.WriteActivationFrameAsync(
                            serverPipe.Output,
                            hdr,
                            buf.Memory,
                            messageType: ClusterMessageType.PartitionFetchResponse,
                            correlationId: hdr.SequenceId,
                            cancellationToken: cts.Token);
                    }
                }
            }
            catch (OperationCanceledException) { }
        }, cts.Token);

        byte[] testActivation = GC.AllocateArray<byte>(activationBytes, pinned: true);
        var header = new ActivationHeader(
            sequenceId: 1,
            tokenIndex: 0,
            layerIndex: 0,
            hiddenDim: vectorDimensions,
            expertId: 1,
            flags: ActivationFlags.None);

        // Warmup
        for (int i = 0; i < 50; i++)
        {
            await DistributedPipelineCoordinator.WriteActivationFrameAsync(
                clientPipe.Output,
                header,
                testActivation.AsMemory(),
                cancellationToken: cts.Token);

            var (_, respBuf) = await DistributedPipelineCoordinator.ReadActivationFrameAsync(clientPipe.Input, cts.Token);
            respBuf.Dispose();
        }

        long allocBefore = GC.GetAllocatedBytesForCurrentThread();
        var latenciesUs = new double[frameCount];
        var swTotal = Stopwatch.StartNew();

        for (int i = 0; i < frameCount; i++)
        {
            long t0 = Stopwatch.GetTimestamp();

            await DistributedPipelineCoordinator.WriteActivationFrameAsync(
                clientPipe.Output,
                header,
                testActivation.AsMemory(),
                cancellationToken: cts.Token);

            var (_, respBuf) = await DistributedPipelineCoordinator.ReadActivationFrameAsync(clientPipe.Input, cts.Token);
            respBuf.Dispose();

            latenciesUs[i] = Stopwatch.GetElapsedTime(t0).TotalMicroseconds;
        }

        swTotal.Stop();
        long allocAfter = GC.GetAllocatedBytesForCurrentThread();

        cts.Cancel();
        try { await serverTask; } catch (OperationCanceledException) { }

        Array.Sort(latenciesUs);
        int totalFrameBytes = ClusterFrameHeader.HeaderSize + ActivationHeader.HeaderSize + activationBytes;
        double mbTransferred = ((double)totalFrameBytes * frameCount * 2) / (1024 * 1024); // 2x for full duplex roundtrip

        return new BenchmarkResult
        {
            TargetNode = "InProcess DuplexPipe (Loopback)",
            ActivationBytes = activationBytes,
            TotalFrames = frameCount,
            ElapsedMs = swTotal.Elapsed.TotalMilliseconds,
            ThroughputMBs = mbTransferred / swTotal.Elapsed.TotalSeconds,
            FramesPerSecond = frameCount / swTotal.Elapsed.TotalSeconds,
            MeanLatencyUs = swTotal.Elapsed.TotalMicroseconds / frameCount,
            P50LatencyUs = latenciesUs[(int)(frameCount * 0.50)],
            P95LatencyUs = latenciesUs[(int)(frameCount * 0.95)],
            P99LatencyUs = latenciesUs[(int)(frameCount * 0.99)],
            MinLatencyUs = latenciesUs[0],
            MaxLatencyUs = latenciesUs[^1],
            GcAllocatedBytes = allocAfter - allocBefore,
            SerializationOverheadNs = 0
        };
    }

    /// <summary>
    /// Executes end-to-end activation ping-pong over a real local TCP socket (127.0.0.1).
    /// </summary>
    public static async Task<BenchmarkResult> RunLoopbackSocketBenchmarkAsync(
        int port = 0,
        int vectorDimensions = 4096,
        int frameCount = 5_000,
        CancellationToken cancellationToken = default)
    {
        int activationBytes = vectorDimensions * sizeof(float);
        var listener = new TcpListener(IPAddress.Loopback, port);
        listener.Start();
        int assignedPort = ((IPEndPoint)listener.LocalEndpoint).Port;

        using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

        var serverTask = Task.Run(async () =>
        {
            using var serverSocket = await listener.AcceptSocketAsync(cts.Token);
            serverSocket.NoDelay = true;
            await using var duplexPipe = new SocketDuplexPipe(serverSocket);

            try
            {
                while (!cts.IsCancellationRequested)
                {
                    var (hdr, buf) = await DistributedPipelineCoordinator.ReadActivationFrameAsync(duplexPipe.Input, cts.Token);
                    using (buf)
                    {
                        await DistributedPipelineCoordinator.WriteActivationFrameAsync(
                            duplexPipe.Output,
                            hdr,
                            buf.Memory,
                            messageType: ClusterMessageType.PartitionFetchResponse,
                            correlationId: hdr.SequenceId,
                            cancellationToken: cts.Token);
                    }
                }
            }
            catch (OperationCanceledException) { }
            finally
            {
                listener.Stop();
            }
        }, cts.Token);

        using var clientSocket = new Socket(SocketType.Stream, ProtocolType.Tcp) { NoDelay = true };
        await clientSocket.ConnectAsync(IPAddress.Loopback, assignedPort, cts.Token);
        await using var clientPipe = new SocketDuplexPipe(clientSocket);

        byte[] testActivation = GC.AllocateArray<byte>(activationBytes, pinned: true);
        var header = new ActivationHeader(
            sequenceId: 100,
            tokenIndex: 0,
            layerIndex: 0,
            hiddenDim: vectorDimensions,
            expertId: 2,
            flags: ActivationFlags.None);

        // Warmup
        for (int i = 0; i < 20; i++)
        {
            await DistributedPipelineCoordinator.WriteActivationFrameAsync(
                clientPipe.Output,
                header,
                testActivation.AsMemory(),
                cancellationToken: cts.Token);

            var (_, resp) = await DistributedPipelineCoordinator.ReadActivationFrameAsync(clientPipe.Input, cts.Token);
            resp.Dispose();
        }

        long allocBefore = GC.GetAllocatedBytesForCurrentThread();
        var latenciesUs = new double[frameCount];
        var swTotal = Stopwatch.StartNew();

        for (int i = 0; i < frameCount; i++)
        {
            long t0 = Stopwatch.GetTimestamp();

            await DistributedPipelineCoordinator.WriteActivationFrameAsync(
                clientPipe.Output,
                header,
                testActivation.AsMemory(),
                cancellationToken: cts.Token);

            var (_, resp) = await DistributedPipelineCoordinator.ReadActivationFrameAsync(clientPipe.Input, cts.Token);
            resp.Dispose();

            latenciesUs[i] = Stopwatch.GetElapsedTime(t0).TotalMicroseconds;
        }

        swTotal.Stop();
        long allocAfter = GC.GetAllocatedBytesForCurrentThread();

        cts.Cancel();
        try { await serverTask; } catch (OperationCanceledException) { }

        Array.Sort(latenciesUs);
        int totalFrameBytes = ClusterFrameHeader.HeaderSize + ActivationHeader.HeaderSize + activationBytes;
        double mbTransferred = ((double)totalFrameBytes * frameCount * 2) / (1024 * 1024);

        return new BenchmarkResult
        {
            TargetNode = $"TCP Loopback (127.0.0.1:{assignedPort})",
            ActivationBytes = activationBytes,
            TotalFrames = frameCount,
            ElapsedMs = swTotal.Elapsed.TotalMilliseconds,
            ThroughputMBs = mbTransferred / swTotal.Elapsed.TotalSeconds,
            FramesPerSecond = frameCount / swTotal.Elapsed.TotalSeconds,
            MeanLatencyUs = swTotal.Elapsed.TotalMicroseconds / frameCount,
            P50LatencyUs = latenciesUs[(int)(frameCount * 0.50)],
            P95LatencyUs = latenciesUs[(int)(frameCount * 0.95)],
            P99LatencyUs = latenciesUs[(int)(frameCount * 0.99)],
            MinLatencyUs = latenciesUs[0],
            MaxLatencyUs = latenciesUs[^1],
            GcAllocatedBytes = allocAfter - allocBefore,
            SerializationOverheadNs = 0
        };
    }

    /// <summary>
    /// Executes end-to-end frame transfer benchmark over physical TCP link to a fleet node.
    /// Safely handles offline or unreachable nodes.
    /// </summary>
    public static async Task<BenchmarkResult?> RunPhysicalNodeBenchmarkAsync(
        string targetHost,
        ushort port,
        int vectorDimensions = 4096,
        int frameCount = 1_000,
        int timeoutMs = 2_000,
        CancellationToken cancellationToken = default)
    {
        ValidateSubnetOrThrow(targetHost);

        int activationBytes = vectorDimensions * sizeof(float);
        int totalFrameSize = ClusterFrameHeader.HeaderSize + ActivationHeader.HeaderSize + activationBytes;
        byte[] frameBuffer = GC.AllocateArray<byte>(totalFrameSize, pinned: true);

        var header = new ActivationHeader(
            sequenceId: 1,
            tokenIndex: 0,
            layerIndex: 0,
            hiddenDim: vectorDimensions,
            expertId: 1,
            flags: ActivationFlags.None);

        var clusterHeader = new ClusterFrameHeader(
            messageType: ClusterMessageType.StageDispatch,
            correlationId: 1,
            payloadLength: (ulong)(ActivationHeader.HeaderSize + activationBytes),
            flags: ClusterFrameFlags.None);

        clusterHeader.WriteTo(frameBuffer.AsSpan(0, ClusterFrameHeader.HeaderSize));
        header.WriteTo(frameBuffer.AsSpan(ClusterFrameHeader.HeaderSize, ActivationHeader.HeaderSize));

        using var clientSocket = new Socket(SocketType.Stream, ProtocolType.Tcp) { NoDelay = true };

        using var connectCts = new CancellationTokenSource(timeoutMs);
        using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, connectCts.Token);

        try
        {
            await clientSocket.ConnectAsync(targetHost, port, linkedCts.Token);
        }
        catch
        {
            // Node is offline or not running an activation listener
            return null;
        }

        await using var duplexPipe = new SocketDuplexPipe(clientSocket);

        // Warmup
        for (int i = 0; i < 10; i++)
        {
            await duplexPipe.Output.WriteAsync(frameBuffer.AsMemory(), cancellationToken);
            await duplexPipe.Output.FlushAsync(cancellationToken);
        }

        long allocBefore = GC.GetAllocatedBytesForCurrentThread();
        var latenciesUs = new double[frameCount];
        var swTotal = Stopwatch.StartNew();

        for (int i = 0; i < frameCount; i++)
        {
            long t0 = Stopwatch.GetTimestamp();
            await duplexPipe.Output.WriteAsync(frameBuffer.AsMemory(), cancellationToken);
            await duplexPipe.Output.FlushAsync(cancellationToken);
            latenciesUs[i] = Stopwatch.GetElapsedTime(t0).TotalMicroseconds;
        }

        swTotal.Stop();
        long allocAfter = GC.GetAllocatedBytesForCurrentThread();

        Array.Sort(latenciesUs);
        double mbTotal = ((double)totalFrameSize * frameCount) / (1024 * 1024);
        double throughputMBs = mbTotal / swTotal.Elapsed.TotalSeconds;

        return new BenchmarkResult
        {
            TargetNode = $"{targetHost}:{port}",
            ActivationBytes = activationBytes,
            TotalFrames = frameCount,
            ElapsedMs = swTotal.Elapsed.TotalMilliseconds,
            ThroughputMBs = throughputMBs,
            FramesPerSecond = frameCount / swTotal.Elapsed.TotalSeconds,
            MeanLatencyUs = swTotal.Elapsed.TotalMicroseconds / frameCount,
            P50LatencyUs = latenciesUs[(int)(frameCount * 0.50)],
            P95LatencyUs = latenciesUs[(int)(frameCount * 0.95)],
            P99LatencyUs = latenciesUs[(int)(frameCount * 0.99)],
            MinLatencyUs = latenciesUs[0],
            MaxLatencyUs = latenciesUs[^1],
            GcAllocatedBytes = allocAfter - allocBefore,
            SerializationOverheadNs = 0
        };
    }

    /// <summary>
    /// Executes an empirical multi-stage pipeline benchmark (Stage 0 -> Stage 1 -> Stage 2) for D=4096 and FFN=11008.
    /// Returns the average execution latency in milliseconds per token.
    /// </summary>
    public static double BenchmarkPipelineExecution(int hiddenDim = 4096, int ffnDim = 11008, int iterations = 100)
    {
        using var stage0 = new AttentionRootStageWorker(hiddenDim, 2, 8, "Stage 0: Root Attention & Router");
        using var stage1 = CpuWarmExpertPoolWorker.CreateSynthetic(1, hiddenDim, ffnDim, 8, Gguf.GgufType.F32, "Stage 1: Warm Expert Pool (AVX-512)", multithreaded: false);
        using var stage2 = new AccumulatorStageWorker(hiddenDim, "Stage 2: Residual Accumulator");
        using var coordinator = new DistributedPipelineCoordinator(stage0, stage1, stage2, hiddenDim);

        byte[] inputBytes = new byte[hiddenDim * sizeof(float)];
        Array.Fill(inputBytes, (byte)1);
        int[] selectedExperts = [0, 2];
        float[] expertWeights = [0.6f, 0.4f];

        // Warmup
        for (int i = 0; i < 10; i++)
        {
            using var buf = coordinator.ExecuteStep(1, 0, 0, inputBytes.AsSpan(), selectedExperts.AsSpan(), expertWeights.AsSpan());
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            using var buf = coordinator.ExecuteStep((ulong)i, 0, 0, inputBytes.AsSpan(), selectedExperts.AsSpan(), expertWeights.AsSpan());
        }
        sw.Stop();

        return sw.Elapsed.TotalMilliseconds / iterations;
    }
}
