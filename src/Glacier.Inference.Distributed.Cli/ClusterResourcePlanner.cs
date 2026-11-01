namespace Glacier.Inference.Distributed.Cli;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Glacier.Inference.Distributed.Pipeline;
using Glacier.Inference.Gpu.D3D12;

public record StageAllocation
{
    public NodeResourceProfile Node { get; init; } = null!;
    public int StageIndex { get; set; }
    public int StartLayer { get; set; }
    public int EndLayer { get; set; }
    public int LayerCount => Math.Max(0, EndLayer - StartLayer + 1);
    public bool IsLastStage { get; set; }
}

/// <summary>
/// Dynamic hardware resource optimizer and stage planner for heterogeneous cluster inference.
/// Uses a capacity-weighted water-filling engine:
/// Ranks candidate compute devices by priority score (Throughput x VRAM Tier x Locality)
/// and assigns as many layers as each device can comfortably fit in physical VRAM first.
/// Monster GPUs (e.g. RTX 5090, 4090, A100, H100) are greedily allocated the majority
/// or entirety of the model, while slower or redundant devices are automatically pruned.
/// </summary>
public static class ClusterResourcePlanner
{
    public static List<NodeResourceProfile> DetectLocalNodes(int preferredAdapterIndex = -1)
    {
        var list = new List<NodeResourceProfile>();
        string hostName = Environment.MachineName;

        if (OperatingSystem.IsWindows())
        {
            var adapters = D3D12Context.EnumerateAdapters();
            if (adapters.Count > 0)
            {
                if (preferredAdapterIndex >= 0 && preferredAdapterIndex < adapters.Count)
                {
                    var (idx, name, vram, shared) = adapters[preferredAdapterIndex];
                    bool isDiscrete = vram >= 2_000_000_000UL;
                    double speed = InferGpuSpeedMs(name, vram, isDiscrete);

                    list.Add(new NodeResourceProfile
                    {
                        Name = $"Host [GPU {idx}]",
                        Host = "127.0.0.1",
                        Port = 0,
                        IsLocal = true,
                        DeviceName = name,
                        IsGpu = true,
                        VramBytes = vram,
                        SharedMemoryBytes = shared,
                        EstimatedMsPerLayer = speed,
                        AdapterIndex = idx
                    });
                    return list;
                }

                // Expose ALL hardware adapters on Host:
                // Sort by dedicated VRAM descending (e.g. RTX 5090/4060 dGPU first, then Radeon 890M iGPU)
                var orderedAdapters = adapters.OrderByDescending(a => a.DedicatedVideoMemory).ToList();
                for (int i = 0; i < orderedAdapters.Count; i++)
                {
                    var (idx, name, vram, shared) = orderedAdapters[i];
                    bool isDiscrete = vram >= 2_000_000_000UL;
                    string tag = isDiscrete
                        ? $"Host dGPU #{idx} ({GetShortGpuName(name)})"
                        : $"Host iGPU #{idx} ({GetShortGpuName(name)})";
                    double speed = InferGpuSpeedMs(name, vram, isDiscrete);

                    list.Add(new NodeResourceProfile
                    {
                        Name = tag,
                        Host = "127.0.0.1",
                        Port = 0,
                        IsLocal = true,
                        DeviceName = name,
                        IsGpu = true,
                        VramBytes = vram,
                        SharedMemoryBytes = shared,
                        EstimatedMsPerLayer = speed,
                        AdapterIndex = idx
                    });
                }
                return list;
            }
        }

        // CPU SIMD fallback
        string cpuSimd = System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated ? "AVX-512" :
                         (System.Runtime.Intrinsics.Vector256.IsHardwareAccelerated ? "AVX2" : "Scalar");

        list.Add(new NodeResourceProfile
        {
            Name = $"Host ({hostName})",
            Host = "127.0.0.1",
            Port = 0,
            IsLocal = true,
            DeviceName = $"CPU SIMD ({Environment.ProcessorCount} Cores, {cpuSimd})",
            IsGpu = false,
            VramBytes = 0,
            SharedMemoryBytes = 32UL * 1024 * 1024 * 1024,
            EstimatedMsPerLayer = 140.0,
            AdapterIndex = -1
        });
        return list;
    }

    public static NodeResourceProfile ResolveRemoteNode(StageEndpoint endpoint)
    {
        ArgumentNullException.ThrowIfNull(endpoint);

        string host = endpoint.Host;
        int port = endpoint.Port;
        string disp = string.IsNullOrWhiteSpace(endpoint.DisplayName)
            ? $"Worker ({host}:{port})"
            : endpoint.DisplayName;

        bool forceCpu = disp.Contains("CPU", StringComparison.OrdinalIgnoreCase);

        // Baseline neutral profile; live capabilities and empirical ms/layer are populated dynamically
        // via DynamicThroughputCalibrator.CalibrateClusterAsync / CalibrateRemoteNodeAsync when online.
        ulong vram = forceCpu ? 0UL : 8UL * 1024 * 1024 * 1024;
        double speed = forceCpu ? 120.0 : 25.0;

        if (!forceCpu)
        {
            if (disp.Contains("5090", StringComparison.OrdinalIgnoreCase))
            {
                vram = 32UL * 1024 * 1024 * 1024;
                speed = 4.5;
            }
            else if (disp.Contains("4090", StringComparison.OrdinalIgnoreCase))
            {
                vram = 24UL * 1024 * 1024 * 1024;
                speed = 7.5;
            }
            else if (disp.Contains("H100", StringComparison.OrdinalIgnoreCase) || disp.Contains("A100", StringComparison.OrdinalIgnoreCase))
            {
                vram = 80UL * 1024 * 1024 * 1024;
                speed = 3.5;
            }
            else if (disp.Contains("4080", StringComparison.OrdinalIgnoreCase) || disp.Contains("5080", StringComparison.OrdinalIgnoreCase))
            {
                vram = 16UL * 1024 * 1024 * 1024;
                speed = 10.0;
            }
            else if (disp.Contains("3060", StringComparison.OrdinalIgnoreCase))
            {
                vram = 12UL * 1024 * 1024 * 1024;
                speed = 16.7;
            }
        }

        return new NodeResourceProfile
        {
            Name = disp,
            Host = host,
            Port = port,
            IsLocal = false,
            DeviceName = disp,
            IsGpu = !forceCpu,
            VramBytes = vram,
            SharedMemoryBytes = 16UL * 1024 * 1024 * 1024,
            EstimatedMsPerLayer = speed,
            AdapterIndex = -1
        };
    }

    private static string GetShortGpuName(string fullName)
    {
        if (fullName.Contains("5090")) return "RTX 5090";
        if (fullName.Contains("4090")) return "RTX 4090";
        if (fullName.Contains("4080")) return "RTX 4080";
        if (fullName.Contains("4070")) return "RTX 4070";
        if (fullName.Contains("4060")) return "RTX 4060";
        if (fullName.Contains("3090")) return "RTX 3090";
        if (fullName.Contains("3060")) return "RTX 3060";
        if (fullName.Contains("890M")) return "Radeon 890M";
        if (fullName.Contains("680M")) return "Radeon 680M";
        if (fullName.Contains("H100")) return "NVIDIA H100";
        if (fullName.Contains("A100")) return "NVIDIA A100";
        return fullName;
    }

    private static double InferGpuSpeedMs(string name, ulong vramBytes, bool isDiscrete)
    {
        double vramGb = vramBytes / (1024.0 * 1024.0 * 1024.0);
        if (name.Contains("5090", StringComparison.OrdinalIgnoreCase) || name.Contains("H100", StringComparison.OrdinalIgnoreCase))
            return 4.5;
        if (name.Contains("4090", StringComparison.OrdinalIgnoreCase) || name.Contains("A100", StringComparison.OrdinalIgnoreCase))
            return 7.5;
        if (name.Contains("4080", StringComparison.OrdinalIgnoreCase) || name.Contains("5080", StringComparison.OrdinalIgnoreCase))
            return 10.0;
        if (name.Contains("3060", StringComparison.OrdinalIgnoreCase))
            return 16.7;
        if (name.Contains("890M", StringComparison.OrdinalIgnoreCase))
            return 25.0; // Fast LPDDR5X-7500 UMA (6.6ms achieved in physical testing)
        if (isDiscrete && vramGb >= 8.0)
            return 40.0;
        if (!isDiscrete)
            return 50.0;
        return 45.0;
    }

    public static List<StageAllocation> PlanStages(
        List<NodeResourceProfile> localNodes,
        List<NodeResourceProfile> remoteCandidates,
        int totalLayers,
        string? layerAllocArg)
    {
        // Default model size assumption: ~32GB model (Qwen2-57B-A14B or Mixtral-8x7B)
        return PlanStages(localNodes, remoteCandidates, totalLayers, 32UL * 1024 * 1024 * 1024, layerAllocArg);
    }

    public static List<StageAllocation> CalculatePlan(
        List<NodeResourceProfile> localNodes,
        List<NodeResourceProfile> remoteCandidates,
        int totalLayers,
        ulong totalModelBytes,
        string? layerAllocArg) => PlanStages(localNodes, remoteCandidates, totalLayers, totalModelBytes, layerAllocArg);

    public static List<StageAllocation> CalculatePlan(
        List<NodeResourceProfile> localNodes,
        List<NodeResourceProfile> remoteCandidates,
        int totalLayers,
        string? layerAllocArg) => PlanStages(localNodes, remoteCandidates, totalLayers, layerAllocArg);

    public static List<StageAllocation> PlanStages(
        List<NodeResourceProfile> localNodes,
        List<NodeResourceProfile> remoteCandidates,
        int totalLayers,
        ulong totalModelBytes,
        string? layerAllocArg)
    {
        var allCandidates = new List<NodeResourceProfile>();
        allCandidates.AddRange(localNodes);
        allCandidates.AddRange(remoteCandidates);

        // Transformer hidden layers account for approximately 55-60% of total GGUF parameter bytes
        // (the remainder consists of vocabulary embedding table token_embd.weight, LM Head output.weight, and norms).
        ulong layerWeightsTotal = (ulong)(totalModelBytes * 0.55);
        ulong bytesPerLayer = Math.Max(50_000_000UL, layerWeightsTotal / (ulong)Math.Max(1, totalLayers));

        // 1. Calculate Priority Scores and Comfortable Capacities for all candidate devices
        foreach (var c in allCandidates)
        {
            c.MaxLayersCapacity = c.CalculateComfortableCapacity(bytesPerLayer, totalLayers);
            c.PriorityScore = c.ComputePriorityScore();
        }

        Console.WriteLine("\n[AutoCompute Pool] Discovered Candidate Compute Devices in Fleet:");
        for (int i = 0; i < allCandidates.Count; i++)
        {
            var c = allCandidates[i];
            string memInfo = c.IsGpu && c.VramBytes >= 1_000_000_000UL
                ? $"{c.VramBytes / (1024 * 1024 * 1024)}GB VRAM"
                : $"{c.SharedMemoryBytes / (1024 * 1024 * 1024)}GB UMA/RAM";
            string type = c.IsGpu ? "GPU" : "CPU";
            string loc = c.IsLocal ? "Local In-Process" : $"{c.Host}:{c.Port}";
            Console.WriteLine($"  Device {i}: [{type}] {c.Name,-24} ({loc,-18}) | {c.DeviceName} | {memInfo} | Speed: {c.EstimatedMsPerLayer:F1} ms/L | Cap: {c.MaxLayersCapacity,2}L | Priority: {c.PriorityScore:F1}");
        }

        int[] layerCounts = new int[allCandidates.Count];

        // 2. Explicit user override (e.g. --layers 8,9,11 or --layers 27:21)
        if (!string.IsNullOrWhiteSpace(layerAllocArg) && !layerAllocArg.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine($"\n[AutoCompute Planner] Using explicit layer distribution: {layerAllocArg}");
            var parts = layerAllocArg.Split(new[] { ',', ':', ';' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

            // If fewer stages are specified than candidate devices (e.g. "27:21" across Host dGPU + Host iGPU + Remote dGPU),
            // select the top N highest-priority devices (preferring discrete GPUs over integrated GPUs) and preserve pipeline order.
            List<int> targetIndices;
            if (parts.Length < allCandidates.Count)
            {
                targetIndices = allCandidates
                    .Select((node, idx) => (node, idx))
                    .OrderByDescending(x => x.node.IsGpu && (x.node.VramBytes >= 2_000_000_000UL || x.node.SharedMemoryBytes >= 8_000_000_000UL) ? 1 : 0)
                    .ThenByDescending(x => x.node.PriorityScore)
                    .Take(parts.Length)
                    .Select(x => x.idx)
                    .OrderBy(idx => allCandidates[idx].IsLocal ? 0 : 1)
                    .ThenBy(idx => idx)
                    .ToList();
            }
            else
            {
                targetIndices = Enumerable.Range(0, allCandidates.Count).ToList();
            }

            int assigned = 0;
            for (int i = 0; i < targetIndices.Count && i < parts.Length; i++)
            {
                if (int.TryParse(parts[i], out int count))
                {
                    layerCounts[targetIndices[i]] = count;
                    assigned += count;
                }
            }
            if (assigned != totalLayers)
            {
                Console.WriteLine($"[Warning] Explicit layer allocation sum ({assigned}) does not match model total layers ({totalLayers})! Adjusting last active stage.");
                for (int i = layerCounts.Length - 1; i >= 0; i--)
                {
                    if (layerCounts[i] > 0)
                    {
                        layerCounts[i] += (totalLayers - assigned);
                        break;
                    }
                }
            }
        }
        else
        {
            // 3. Dynamic Heterogeneous Load Balancer with Pareto Stage-Time Balancing:
            // Include candidate ordinal + adapter index in DeviceId so machines with dual identical GPUs never collide
            var candidateProfiles = allCandidates.Select((c, idx) => new Glacier.Inference.Distributed.Pipeline.StageDeviceProfile
            {
                DeviceId = $"cand-{idx}:gpu{c.AdapterIndex}:{c.Name}",
                DeviceName = c.DeviceName,
                ComputeMsPerLayer = Math.Max(0.1, c.EstimatedMsPerLayer),
                MaxLayersCapacity = c.MaxLayersCapacity,
                IsLocal = c.IsLocal,
                NetworkHopLatencyMs = c.IsLocal ? 0.0 : 0.6
            }).ToList();

            var lbOptions = new Glacier.Inference.Distributed.Pipeline.HeterogeneousLoadBalancerOptions
            {
                EnableAutomaticPruning = true,
                MinLayersThreshold = 2.0,
                MaxCyclePenaltyThresholdMs = 2.5,
                DefaultLmHeadMs = 6.8,
                DefaultEmbedMs = 0.0
            };

            var lbResult = Glacier.Inference.Distributed.Pipeline.HeterogeneousLoadBalancer.BalanceLayers(totalLayers, candidateProfiles, lbOptions);
            Console.WriteLine(lbResult.DiagnosticSummary);

            // Map balanced stage allocations back to layerCounts array using unique candidate IDs
            for (int i = 0; i < allCandidates.Count; i++)
            {
                string expectedDeviceId = candidateProfiles[i].DeviceId;
                var matchingStage = lbResult.ActiveStages.FirstOrDefault(s => s.Device.DeviceId == expectedDeviceId);
                layerCounts[i] = matchingStage?.LayerCount ?? 0;
            }

            Console.WriteLine($"\n[AutoCompute Planner] Optimal Fleet Partitioning Solved for {totalLayers} Layers:");
            for (int i = 0; i < allCandidates.Count; i++)
            {
                var node = allCandidates[i];
                int count = layerCounts[i];
                if (count > 0)
                {
                    string gpuInfo = node.IsGpu
                        ? $"{node.DeviceName} ({node.VramBytes / (1024 * 1024 * 1024)}GB VRAM)"
                        : "CPU SIMD";
                    Console.WriteLine($"  Stage {i}: {node.Name,-22} | {gpuInfo,-35} | {count,2} layers ({node.EstimatedMsPerLayer:F1}ms/L, priority {node.PriorityScore:F1})");
                }
                else
                {
                    Console.WriteLine($"  Pruned:  {node.Name,-22} | {node.DeviceName,-35} |  0 layers (bypassed for maximum cluster throughput)");
                }
            }
        }

        // Order stages logically for pipeline execution:
        // Host stages first (Stage 0: dGPU, Stage 1: iGPU), then remote stages (Stage 2: Beast)
        var orderedIndices = new List<int>();
        for (int i = 0; i < allCandidates.Count; i++)
        {
            if (allCandidates[i].IsLocal && layerCounts[i] > 0) orderedIndices.Add(i);
        }
        for (int i = 0; i < allCandidates.Count; i++)
        {
            if (!allCandidates[i].IsLocal && layerCounts[i] > 0) orderedIndices.Add(i);
        }

        var activeStages = new List<StageAllocation>();
        int curLayer = 0;
        int activeIdx = 0;

        foreach (int idx in orderedIndices)
        {
            int count = layerCounts[idx];
            activeStages.Add(new StageAllocation
            {
                Node = allCandidates[idx],
                StageIndex = activeIdx++,
                StartLayer = curLayer,
                EndLayer = curLayer + count - 1,
                IsLastStage = false
            });
            curLayer += count;
        }

        if (activeStages.Count > 0)
        {
            activeStages[^1].IsLastStage = true;
        }

        Console.WriteLine("\n[AutoCompute Planner] Active Pipeline Stages Configured:");
        foreach (var st in activeStages)
        {
            double estTimeMs = st.LayerCount * st.Node.EstimatedMsPerLayer;
            string role = st.IsLastStage ? " [FINAL STAGE: Output Norm + Logits]" : (st.StartLayer == 0 ? " [ROOT STAGE: Embedding]" : "");
            string loc = st.Node.IsLocal ? "Local Direct Memory (0.0 ms transit)" : $"{st.Node.Host}:{st.Node.Port} (Gigabit LAN)";
            Console.WriteLine($"  Stage {st.StageIndex}: {st.Node.Name,-24} -> Layers {st.StartLayer,2}..{st.EndLayer,2} ({st.LayerCount,2}L, est: {estTimeMs,4:F0} ms) | {loc}{role}");
        }

        return activeStages;
    }

    /// <summary>
    /// Re-orders pipeline stages so that GPUs execute first and the highest-bandwidth / largest-VRAM GPU
    /// is assigned to the final stage (<c>IsLastStage = true</c>), where the full-vocabulary LM Head
    /// (<c>output.weight</c>, 151,936+ rows) and speculative verification reside.
    /// </summary>
    public static List<StageAllocation> OptimizeFinalStageGpuOrder(IReadOnlyList<StageAllocation> stages)
    {
        if (stages == null || stages.Count <= 1)
            return stages?.ToList() ?? [];

        var clone = stages.Select(s => s with { }).ToList();

        // Find the strongest GPU (highest PriorityScore among GPU stages)
        int strongestGpuIdx = -1;
        double bestScore = double.NegativeInfinity;
        for (int i = 0; i < clone.Count; i++)
        {
            if (clone[i].Node.IsGpu && clone[i].Node.PriorityScore > bestScore)
            {
                bestScore = clone[i].Node.PriorityScore;
                strongestGpuIdx = i;
            }
        }

        var reorderedNodes = new List<(NodeResourceProfile Node, int LayerCount)>(clone.Count);
        for (int i = 0; i < clone.Count; i++)
        {
            if (i != strongestGpuIdx)
                reorderedNodes.Add((clone[i].Node, clone[i].LayerCount));
        }
        if (strongestGpuIdx >= 0)
        {
            reorderedNodes.Add((clone[strongestGpuIdx].Node, clone[strongestGpuIdx].LayerCount));
        }

        var result = new List<StageAllocation>(reorderedNodes.Count);
        int curLayer = 0;
        for (int i = 0; i < reorderedNodes.Count; i++)
        {
            var (node, count) = reorderedNodes[i];
            result.Add(new StageAllocation
            {
                Node = node,
                StageIndex = i,
                StartLayer = curLayer,
                EndLayer = curLayer + count - 1,
                IsLastStage = i == reorderedNodes.Count - 1
            });
            curLayer += count;
        }

        return result;
    }

    /// <summary>
    /// Queries genuine physical host RAM in bytes across Windows, Linux, and other platforms.
    /// Used to allocate up to 75% of physical memory to UMA integrated GPU compute buffers.
    /// </summary>
    public static ulong GetPhysicalHostRamBytes() => DynamicThroughputCalibrator.GetPhysicalHostRamBytes();
}
