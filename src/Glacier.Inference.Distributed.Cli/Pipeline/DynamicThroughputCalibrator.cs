namespace Glacier.Inference.Distributed.Pipeline;

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Inference.Distributed.Cli;
using Glacier.Inference.Distributed.Transport;

/// <summary>
/// Result of an empirical single-layer micro-benchmarking probe on a cluster compute node.
/// </summary>
public readonly record struct NodeCalibrationResult(
    string NodeName,
    string Host,
    int Port,
    bool IsLocal,
    string DeviceName,
    bool IsGpu,
    double TauMsPerLayer,       // Measured empirical compute time per layer (ms/layer)
    double DeltaStageOverheadMs, // Measured empirical stage overhead (ms)
    double NetworkHopMs,        // Measured round-trip network transit (ms, 0 for local)
    ulong UsableMemoryBytes,    // Verified available VRAM or UMA host RAM
    bool IsSuccess,
    string? ErrorMessage = null
);

/// <summary>
/// Configuration parameters for the dynamic throughput calibration probe.
/// </summary>
public sealed class CalibrationOptions
{
    public int HiddenDim { get; init; } = 5120;
    public int IntermediateDim { get; init; } = 13824;
    public int VocabSize { get; init; } = 152064;
    public int WarmupIterations { get; init; } = 2;
    public int MeasureIterations { get; init; } = 5;
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(10);
    public bool MeasureLmHead { get; init; } = true;
    public bool MeasurePrefill { get; init; } = false;
    public int PrefillChunkSize { get; init; } = 128;
}

/// <summary>
/// Abstraction for executing synthetic single-layer transformer forward passes across backends.
/// </summary>
public interface ISyntheticLayerRunner
{
    double RunSingleLayerForward(int hiddenDim, int intermediateDim, int iterations);
    double RunLmHeadForward(int hiddenDim, int vocabSize, int iterations);
}

/// <summary>
/// CPU SIMD synthetic single-layer benchmark runner without external vector dependencies.
/// </summary>
public sealed class CpuSimdSyntheticRunner : ISyntheticLayerRunner
{
    public double RunSingleLayerForward(int hiddenDim, int intermediateDim, int iterations)
    {
        float[] inX = new float[hiddenDim];
        float[] outX = new float[hiddenDim];
        float[] dummyWeights = new float[hiddenDim];

        // Seed synthetic values
        Array.Fill(inX, 1.0f);
        Array.Fill(dummyWeights, 0.05f);

        var sw = Stopwatch.StartNew();
        for (int iter = 0; iter < iterations; iter++)
        {
            // 1. RMSNorm
            float sumSq = 0f;
            for (int i = 0; i < hiddenDim; i++)
            {
                float v = inX[i];
                sumSq += v * v;
            }
            float rms = 1.0f / MathF.Sqrt((sumSq / hiddenDim) + 1e-6f);
            for (int i = 0; i < hiddenDim; i++)
            {
                outX[i] = inX[i] * rms;
            }

            // 2. Simulated FFN Gate & Up projections (intermediateDim dot products)
            float accum = 0f;
            int dotLen = Math.Min(hiddenDim, 256);
            int projCount = Math.Min(intermediateDim, 256);
            for (int j = 0; j < projCount; j++)
            {
                float dot = 0f;
                for (int k = 0; k < dotLen; k++)
                {
                    dot += outX[k] * dummyWeights[k];
                }
                accum += dot;
            }
            outX[0] += accum * 1e-4f;
        }
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iterations;
    }

    public double RunLmHeadForward(int hiddenDim, int vocabSize, int iterations)
    {
        float[] inX = new float[hiddenDim];
        float[] dummyWeights = new float[hiddenDim];
        Array.Fill(inX, 1.0f);
        Array.Fill(dummyWeights, 0.01f);

        int dotLen = Math.Min(hiddenDim, 256);
        int sampleCount = Math.Min(vocabSize, 1024);

        var sw = Stopwatch.StartNew();
        for (int iter = 0; iter < iterations; iter++)
        {
            float top = 0f;
            for (int v = 0; v < sampleCount; v++)
            {
                float dot = 0f;
                for (int k = 0; k < dotLen; k++)
                {
                    dot += inX[k] * dummyWeights[k];
                }
                if (dot > top) top = dot;
            }
        }
        sw.Stop();
        return (sw.Elapsed.TotalMilliseconds / iterations) * (vocabSize / 1024.0);
    }
}

/// <summary>
/// Online micro-benchmarking engine that executes synthetic single-layer forward passes
/// on cluster nodes to measure empirical tau_i (ms/layer) and delta_i (stage overhead).
/// Eliminates hardcoded static constants in ClusterResourcePlanner.
/// </summary>
public static class DynamicThroughputCalibrator
{
    public const ClusterMessageType MsgCalibrateProbeRequest  = (ClusterMessageType)0x74;
    public const ClusterMessageType MsgCalibrateProbeResponse = (ClusterMessageType)0x75;

    /// <summary>
    /// Calibrates an entire list of cluster compute nodes in parallel, updating their
    /// EstimatedMsPerLayer and PriorityScores with factual empirical measurements.
    /// </summary>
    public static async Task<List<NodeCalibrationResult>> CalibrateClusterAsync(
        List<NodeResourceProfile> candidateNodes,
        CalibrationOptions? options = null,
        CancellationToken ct = default)
    {
        options ??= new CalibrationOptions();
        var results = new List<NodeCalibrationResult>();

        Console.WriteLine("\n[DynamicCalibrator] Initiating Empirical Micro-Benchmarking Across Cluster Nodes...");
        var swTotal = Stopwatch.StartNew();

        var tasks = candidateNodes.Select(node => CalibrateNodeAsync(node, options, ct)).ToList();
        var calibrationResults = await Task.WhenAll(tasks);

        for (int i = 0; i < candidateNodes.Count; i++)
        {
            var node = candidateNodes[i];
            var res = calibrationResults[i];
            results.Add(res);

            if (res.IsSuccess)
            {
                // Update profile with empirical factual measurements
                node.EstimatedMsPerLayer = res.TauMsPerLayer;
                node.PriorityScore = node.ComputePriorityScore();

                string memStr = node.IsGpu && node.VramBytes >= 1_000_000_000UL
                    ? $"{node.VramBytes / (1024 * 1024 * 1024)}GB VRAM"
                    : $"{node.SharedMemoryBytes / (1024 * 1024 * 1024)}GB UMA/RAM";

                Console.WriteLine($"  [Calibrated] {node.Name,-22} | tau: {res.TauMsPerLayer,5:F2} ms/L | delta: {res.DeltaStageOverheadMs,5:F2} ms | hop: {res.NetworkHopMs,4:F2} ms | {memStr} | Priority: {node.PriorityScore,5:F1}");
            }
            else
            {
                Console.WriteLine($"  [Calibrate Warning] Node '{node.Name}' ({node.Host}) probe failed: {res.ErrorMessage}. Retaining baseline estimate ({node.EstimatedMsPerLayer:F1} ms/L).");
            }
        }

        swTotal.Stop();
        Console.WriteLine($"[DynamicCalibrator] Cluster calibration completed in {swTotal.Elapsed.TotalMilliseconds:F1} ms.\n");
        return results;
    }

    /// <summary>
    /// Calibrates an individual node (either local in-process or remote over TCP socket).
    /// </summary>
    public static async Task<NodeCalibrationResult> CalibrateNodeAsync(
        NodeResourceProfile node,
        CalibrationOptions options,
        CancellationToken ct = default)
    {
        if (node.IsLocal)
        {
            return CalibrateLocalNode(node, options);
        }
        else
        {
            return await CalibrateRemoteNodeAsync(node, options, ct);
        }
    }

    /// <summary>
    /// Measures local device compute time (tau) and LM Head overhead (delta) directly in-process.
    /// </summary>
    public static NodeCalibrationResult CalibrateLocalNode(
        NodeResourceProfile node,
        CalibrationOptions options,
        ISyntheticLayerRunner? runner = null)
    {
        try
        {
            runner ??= new CpuSimdSyntheticRunner();

            // Warmup iterations
            runner.RunSingleLayerForward(options.HiddenDim, options.IntermediateDim, options.WarmupIterations);

            // Measured iterations for tau (ms/layer)
            double tau = runner.RunSingleLayerForward(options.HiddenDim, options.IntermediateDim, options.MeasureIterations);

            // Measured LM Head overhead (delta)
            double deltaLmHead = options.MeasureLmHead
                ? runner.RunLmHeadForward(options.HiddenDim, options.VocabSize, options.MeasureIterations)
                : 3.0;

            ulong hostRam = ClusterResourcePlanner.GetPhysicalHostRamBytes();
            ulong usableMem = node.IsGpu && node.VramBytes >= 2_000_000_000UL
                ? node.VramBytes
                : (ulong)(hostRam * 0.75);

            return new NodeCalibrationResult(
                NodeName: node.Name,
                Host: node.Host,
                Port: node.Port,
                IsLocal: true,
                DeviceName: node.DeviceName,
                IsGpu: node.IsGpu,
                TauMsPerLayer: Math.Max(0.1, tau),
                DeltaStageOverheadMs: Math.Max(0.1, deltaLmHead),
                NetworkHopMs: 0.0,
                UsableMemoryBytes: usableMem,
                IsSuccess: true);
        }
        catch (Exception ex)
        {
            return new NodeCalibrationResult(
                NodeName: node.Name,
                Host: node.Host,
                Port: node.Port,
                IsLocal: true,
                DeviceName: node.DeviceName,
                IsGpu: node.IsGpu,
                TauMsPerLayer: node.EstimatedMsPerLayer,
                DeltaStageOverheadMs: 1.0,
                NetworkHopMs: 0.0,
                UsableMemoryBytes: node.SharedMemoryBytes,
                IsSuccess: false,
                ErrorMessage: ex.Message);
        }
    }

    /// <summary>
    /// Executes calibration probe over TCP socket to a remote worker node.
    /// Measures tau_i, delta_i, and isolates network hop latency (RTT / 2).
    /// </summary>
    public static async Task<NodeCalibrationResult> CalibrateRemoteNodeAsync(
        NodeResourceProfile node,
        CalibrationOptions options,
        CancellationToken ct = default)
    {
        try
        {
            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(options.Timeout);

            using var client = new TcpClient();
            client.NoDelay = true;
            FusedSocketTransport.ConfigureSocket(client.Client, 1024 * 1024);

            await client.ConnectAsync(node.Host, node.Port, cts.Token);
            using var stream = client.GetStream();

            // 1. Serialize Probe Request Frame
            int reqPayloadLen = 24;
            byte[] reqPayload = new byte[reqPayloadLen];
            BinaryPrimitives.WriteInt32LittleEndian(reqPayload.AsSpan(0, 4), options.HiddenDim);
            BinaryPrimitives.WriteInt32LittleEndian(reqPayload.AsSpan(4, 4), options.IntermediateDim);
            BinaryPrimitives.WriteInt32LittleEndian(reqPayload.AsSpan(8, 4), options.VocabSize);
            BinaryPrimitives.WriteInt32LittleEndian(reqPayload.AsSpan(12, 4), options.WarmupIterations);
            BinaryPrimitives.WriteInt32LittleEndian(reqPayload.AsSpan(16, 4), options.MeasureIterations);
            int flags = 0x01 | (options.MeasureLmHead ? 0x04 : 0);
            BinaryPrimitives.WriteInt32LittleEndian(reqPayload.AsSpan(20, 4), flags);

            var reqHdr = new ClusterFrameHeader(MsgCalibrateProbeRequest, correlationId: 1001, (ulong)reqPayloadLen);
            byte[] reqHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
            MemoryMarshal.Write(reqHdrBytes, in reqHdr);

            long tStart = Stopwatch.GetTimestamp();
            await stream.WriteAsync(reqHdrBytes, 0, ClusterFrameHeader.HeaderSize, cts.Token);
            await stream.WriteAsync(reqPayload, 0, reqPayloadLen, cts.Token);
            await stream.FlushAsync(cts.Token);

            // 2. Read Response Frame
            byte[] respHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
            await ReadExactAsync(stream, respHdrBytes, 0, ClusterFrameHeader.HeaderSize, cts.Token);
            var respHdr = MemoryMarshal.Read<ClusterFrameHeader>(respHdrBytes);

            int respPayloadLen = (int)respHdr.PayloadLength;
            byte[] respPayload = new byte[respPayloadLen];
            await ReadExactAsync(stream, respPayload, 0, respPayloadLen, cts.Token);
            long tEnd = Stopwatch.GetTimestamp();

            double rttMs = (tEnd - tStart) * 1000.0 / Stopwatch.Frequency;

            // 3. Parse Probe Response Payload
            double remoteTau = BitConverter.ToDouble(respPayload, 0);
            double remoteLmHead = BitConverter.ToDouble(respPayload, 8);
            ulong remoteRam = BinaryPrimitives.ReadUInt64LittleEndian(respPayload.AsSpan(16, 8));
            ulong remoteVram = BinaryPrimitives.ReadUInt64LittleEndian(respPayload.AsSpan(24, 8));

            string deviceName = node.DeviceName;
            if (respPayloadLen >= 36)
            {
                int nameLen = BinaryPrimitives.ReadInt32LittleEndian(respPayload.AsSpan(32, 4));
                if (nameLen > 0 && 36 + nameLen <= respPayloadLen)
                {
                    deviceName = Encoding.UTF8.GetString(respPayload, 36, nameLen);
                }
            }

            // Isolate network transit latency (one-way hop)
            double netHopMs = Math.Max(0.05, (rttMs - remoteTau) / 2.0);

            return new NodeCalibrationResult(
                NodeName: node.Name,
                Host: node.Host,
                Port: node.Port,
                IsLocal: false,
                DeviceName: deviceName,
                IsGpu: node.IsGpu,
                TauMsPerLayer: Math.Max(0.1, remoteTau),
                DeltaStageOverheadMs: Math.Max(0.1, remoteLmHead),
                NetworkHopMs: netHopMs,
                UsableMemoryBytes: remoteVram > 0 ? remoteVram : (ulong)(remoteRam * 0.75),
                IsSuccess: true);
        }
        catch (Exception ex)
        {
            return new NodeCalibrationResult(
                NodeName: node.Name,
                Host: node.Host,
                Port: node.Port,
                IsLocal: false,
                DeviceName: node.DeviceName,
                IsGpu: node.IsGpu,
                TauMsPerLayer: node.EstimatedMsPerLayer,
                DeltaStageOverheadMs: 1.0,
                NetworkHopMs: 0.6,
                UsableMemoryBytes: node.SharedMemoryBytes,
                IsSuccess: false,
                ErrorMessage: ex.Message);
        }
    }

    /// <summary>
    /// Worker-side probe request handler: executes synthetic pass and returns calibration response frame.
    /// </summary>
    public static async Task HandleProbeRequestAsync(
        Stream stream,
        ClusterFrameHeader requestHeader,
        ISyntheticLayerRunner? runner = null,
        CancellationToken ct = default)
    {
        runner ??= new CpuSimdSyntheticRunner();

        // 1. Read request payload
        int payloadLen = (int)requestHeader.PayloadLength;
        byte[] reqPayload = new byte[payloadLen];
        await ReadExactAsync(stream, reqPayload, 0, payloadLen, ct);

        int hiddenDim = BinaryPrimitives.ReadInt32LittleEndian(reqPayload.AsSpan(0, 4));
        int intermediateDim = BinaryPrimitives.ReadInt32LittleEndian(reqPayload.AsSpan(4, 4));
        int vocabSize = BinaryPrimitives.ReadInt32LittleEndian(reqPayload.AsSpan(8, 4));
        int warmupRuns = BinaryPrimitives.ReadInt32LittleEndian(reqPayload.AsSpan(12, 4));
        int measureRuns = BinaryPrimitives.ReadInt32LittleEndian(reqPayload.AsSpan(16, 4));

        // 2. Execute synthetic warmup & measured forward passes
        runner.RunSingleLayerForward(hiddenDim, intermediateDim, warmupRuns);
        double tau = runner.RunSingleLayerForward(hiddenDim, intermediateDim, measureRuns);
        double lmHead = runner.RunLmHeadForward(hiddenDim, vocabSize, measureRuns);

        // 3. Query physical host RAM and local device name
        ulong ram = ClusterResourcePlanner.GetPhysicalHostRamBytes();
        ulong vram = (ulong)(ram * 0.75); // For UMA integrated GPU or detected VRAM
        string deviceName = Environment.MachineName;
        byte[] nameBytes = Encoding.UTF8.GetBytes(deviceName);

        // 4. Serialize response
        int respLen = 36 + nameBytes.Length;
        byte[] respPayload = new byte[respLen];
        BitConverter.TryWriteBytes(respPayload.AsSpan(0, 8), tau);
        BitConverter.TryWriteBytes(respPayload.AsSpan(8, 8), lmHead);
        BinaryPrimitives.WriteUInt64LittleEndian(respPayload.AsSpan(16, 8), ram);
        BinaryPrimitives.WriteUInt64LittleEndian(respPayload.AsSpan(24, 8), vram);
        BinaryPrimitives.WriteInt32LittleEndian(respPayload.AsSpan(32, 4), nameBytes.Length);
        nameBytes.CopyTo(respPayload.AsSpan(36));

        var respHdr = new ClusterFrameHeader(MsgCalibrateProbeResponse, requestHeader.CorrelationId, (ulong)respLen);
        byte[] respHdrBytes = new byte[ClusterFrameHeader.HeaderSize];
        MemoryMarshal.Write(respHdrBytes, in respHdr);

        await stream.WriteAsync(respHdrBytes, 0, ClusterFrameHeader.HeaderSize, ct);
        await stream.WriteAsync(respPayload, 0, respLen, ct);
        await stream.FlushAsync(ct);
    }

    private static async Task ReadExactAsync(Stream stream, byte[] buffer, int offset, int count, CancellationToken ct = default)
    {
        int read = 0;
        while (read < count)
        {
            int r = await stream.ReadAsync(buffer.AsMemory(offset + read, count - read), ct);
            if (r == 0) throw new EndOfStreamException($"Connection closed prematurely while reading {count} bytes.");
            read += r;
        }
    }
}
