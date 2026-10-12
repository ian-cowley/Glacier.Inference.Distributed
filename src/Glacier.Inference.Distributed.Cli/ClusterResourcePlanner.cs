namespace Glacier.Inference.Distributed.Cli;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Glacier.Inference.Gpu.D3D12;

public record NodeResourceProfile
{
    public string Name { get; init; } = "";
    public string Host { get; init; } = "";
    public int Port { get; init; }
    public bool IsLocal { get; init; }
    public string DeviceName { get; init; } = "";
    public bool IsGpu { get; init; }
    public ulong VramBytes { get; init; }
    public ulong SharedMemoryBytes { get; init; }
    public double EstimatedMsPerLayer { get; set; }
    public int MaxLayersCapacity { get; set; }
    public int AdapterIndex { get; init; } = -1;
    public double PriorityScore { get; set; }

    /// <summary>
    /// Computes the dynamic priority weight score for fleet stage scheduling.
    /// Monster GPUs (e.g. RTX 5090, 4090, A100, H100) receive the highest priority weights,
    /// followed by high-speed discrete GPUs, fast local UMA, and finally remote/CPU nodes.
    /// </summary>
    public double ComputePriorityScore()
    {
        // 1. Raw Compute Throughput Factor (layers per second)
        double throughput = 1000.0 / Math.Max(1.0, EstimatedMsPerLayer);

        // 2. VRAM Tier & Bandwidth Multiplier
        double vramGb = VramBytes / (1024.0 * 1024.0 * 1024.0);
        double vramMultiplier;

        if (IsGpu)
        {
            if (vramGb >= 30.0)
            {
                // Monster Flagship / Server GPU (RTX 5090 32GB, A100 40/80GB, H100 80GB, RTX 6000 Ada 48GB)
                // Bandwidth: 1.8 - 3.35 TB/s. Absolute top priority to absorb as much as fits!
                vramMultiplier = 6.0;
            }
            else if (vramGb >= 20.0)
            {
                // Ultra Enthusiast (RTX 4090 24GB, RTX 3090 24GB) - Bandwidth: ~1.0 TB/s
                vramMultiplier = 4.0;
            }
            else if (vramGb >= 11.0)
            {
                // Mid/High Discrete (RTX 3060 12GB, RTX 4070 12GB, RTX 4080 16GB) - Bandwidth: 360-700 GB/s
                vramMultiplier = 2.5;
            }
            else if (vramGb >= 6.0)
            {
                // Entry/Mid Laptop Discrete (RTX 4060 Laptop 8GB) - Bandwidth: 256 GB/s
                vramMultiplier = 1.6;
            }
            else
            {
                // Integrated GPU UMA (Radeon 890M / 680M) - Bandwidth: 76-120 GB/s
                vramMultiplier = 1.0;
            }
        }
        else
        {
            // CPU SIMD (AVX-512 / AVX2)
            vramMultiplier = 0.25;
        }

        // 3. Network Locality Bonus: Local in-process execution has 0.0 ms network transit
        double localityMultiplier = IsLocal ? 1.25 : 1.0;

        return throughput * vramMultiplier * localityMultiplier;
    }

    /// <summary>
    /// Calculates the maximum number of model layers this device can comfortably fit
    /// in dedicated physical memory without causing OS paging, stutter, or memory pressure.
    /// </summary>
    public int CalculateComfortableCapacity(ulong bytesPerLayer, int totalModelLayers)
    {
        if (bytesPerLayer == 0) return totalModelLayers;

        if (IsGpu)
        {
            double vramGb = VramBytes / (1024.0 * 1024.0 * 1024.0);
            if (vramGb >= 3.0)
            {
                // Dedicated VRAM device: leave a 1.5GB safety margin for OS compositor (DWM),
                // display buffers, and activation scratchpads to guarantee 100% in-VRAM execution.
                ulong safetyCushion = 1_500_000_000UL;
                ulong usableVram = VramBytes > safetyCushion ? (VramBytes - safetyCushion) : 0;
                int maxFit = (int)(usableVram / bytesPerLayer);
                return Math.Clamp(maxFit, 0, totalModelLayers);
            }
            else
            {
                // Integrated UMA (e.g. Radeon 890M / 680M): uses system RAM.
                // Leave 8GB for OS + system processes.
                ulong osHeadroom = 8UL * 1024 * 1024 * 1024;
                ulong usableShared = SharedMemoryBytes > osHeadroom ? (SharedMemoryBytes - osHeadroom) : 0;
                int maxFit = (int)(usableShared / bytesPerLayer);
                return Math.Clamp(maxFit, 0, Math.Min(16, totalModelLayers));
            }
        }
        else
        {
            // CPU SIMD: can hold as many layers as fit in available system RAM
            ulong osHeadroom = 8UL * 1024 * 1024 * 1024;
            ulong usableShared = SharedMemoryBytes > osHeadroom ? (SharedMemoryBytes - osHeadroom) : 0;
            int maxFit = (int)(usableShared / bytesPerLayer);
            return Math.Clamp(maxFit, 0, totalModelLayers);
        }
    }
}

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
                    string tag = isDiscrete ? $"Host dGPU ({GetShortGpuName(name)})" : $"Host iGPU ({GetShortGpuName(name)})";
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
        string host = endpoint.Host;
        int port = endpoint.Port;
        string disp = endpoint.DisplayName;

        // Known fleet nodes:
        if (host == "192.168.1.108" || disp.Contains("Beast", StringComparison.OrdinalIgnoreCase))
        {
            return new NodeResourceProfile
            {
                Name = "Beast Desktop",
                Host = host,
                Port = port,
                IsLocal = false,
                DeviceName = "NVIDIA GeForce RTX 3060 12GB (GDDR6)",
                IsGpu = true,
                VramBytes = 12UL * 1024 * 1024 * 1024,
                SharedMemoryBytes = 16UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 16.7,
                AdapterIndex = -1
            };
        }

        if (host == "192.168.1.101" || disp.Contains("MinisForum", StringComparison.OrdinalIgnoreCase))
        {
            return new NodeResourceProfile
            {
                Name = "MinisForum64",
                Host = host,
                Port = port,
                IsLocal = false,
                DeviceName = "AMD Radeon 680M Graphics (64GB DDR5 UMA)",
                IsGpu = true,
                VramBytes = 512UL * 1024 * 1024,
                SharedMemoryBytes = 64UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 50.8,
                AdapterIndex = -1
            };
        }

        if (host == "192.168.1.200" || disp.Contains("Linux", StringComparison.OrdinalIgnoreCase))
        {
            bool forceCpu = disp.Contains("CPU", StringComparison.OrdinalIgnoreCase);
            if (!forceCpu)
            {
                return new NodeResourceProfile
                {
                    Name = "Fedora Linux (890M Vulkan)",
                    Host = host,
                    Port = port,
                    IsLocal = false,
                    DeviceName = "AMD Radeon 890M Graphics (RDNA 3.5 Vulkan RADV / 46GB UMA)",
                    IsGpu = true,
                    VramBytes = 512UL * 1024 * 1024,
                    SharedMemoryBytes = 46UL * 1024 * 1024 * 1024,
                    EstimatedMsPerLayer = 25.0, // 25ms/L via Vulkan compute, prioritizing above 680M
                    AdapterIndex = -1
                };
            }

            return new NodeResourceProfile
            {
                Name = "Fedora Linux (CPU)",
                Host = host,
                Port = port,
                IsLocal = false,
                DeviceName = "AMD Ryzen AI 9 HX 370 Zen 5 AVX-512 (46GB DDR5)",
                IsGpu = false,
                VramBytes = 0,
                SharedMemoryBytes = 46UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 145.5,
                AdapterIndex = -1
            };
        }

        // Generic / Server / Monster GPU node auto-detection from display name
        ulong vram = 8UL * 1024 * 1024 * 1024;
        double speed = 25.0;

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

        return new NodeResourceProfile
        {
            Name = disp,
            Host = host,
            Port = port,
            IsLocal = false,
            DeviceName = disp,
            IsGpu = true,
            VramBytes = vram,
            SharedMemoryBytes = 32UL * 1024 * 1024 * 1024,
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

        ulong bytesPerLayer = totalModelBytes / (ulong)Math.Max(1, totalLayers);

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

        // 2. Explicit user override (e.g. --layers 8,9,11 or --layers 7,7,7,7)
        if (!string.IsNullOrWhiteSpace(layerAllocArg) && !layerAllocArg.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine($"\n[AutoCompute Planner] Using explicit layer distribution: {layerAllocArg}");
            var parts = layerAllocArg.Split(new[] { ',', ':', ';' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            int assigned = 0;
            for (int i = 0; i < allCandidates.Count && i < parts.Length; i++)
            {
                if (int.TryParse(parts[i], out int count))
                {
                    layerCounts[i] = count;
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
            // 3. Dynamic Priority & Capacity Water-Filling Engine:
            // High-priority "monster GPUs" (5090 / 4090 / server GPUs) are ranked at the top
            // and receive as many layers as comfortably fit in physical VRAM first.
            var prioritized = allCandidates
                .Select((node, index) => (node, index))
                .OrderByDescending(x => x.node.PriorityScore)
                .ToList();

            int remaining = totalLayers;

            // Greedily fill devices in order of PriorityScore up to their comfortable in-memory capacity
            foreach (var (node, originalIdx) in prioritized)
            {
                if (remaining <= 0) break;
                int alloc = Math.Min(node.MaxLayersCapacity, remaining);
                layerCounts[originalIdx] = alloc;
                remaining -= alloc;
            }

            // Fallback: If model is colossal (e.g. 100B+ MoE) and exceeds all comfortable GPU VRAM caps,
            // spill remaining layers across available system RAM / CPU nodes.
            if (remaining > 0)
            {
                for (int i = 0; i < allCandidates.Count && remaining > 0; i++)
                {
                    layerCounts[i] += 1;
                    remaining -= 1;
                }
            }

            Console.WriteLine($"\n[AutoCompute Planner] Optimal Fleet Partitioning Solved for {totalLayers} Layers:");
            for (int i = 0; i < allCandidates.Count; i++)
            {
                var c = allCandidates[i];
                if (layerCounts[i] == 0)
                {
                    string reason = !c.IsGpu
                        ? "CPU node kept in reserve"
                        : "Pruned (higher priority GPU capacity was sufficient to hold the model)";
                    Console.WriteLine($"  -> Pruning device '{c.Name}' ({c.Host}): {reason}.");
                }
                else
                {
                    Console.WriteLine($"  -> Allocated {layerCounts[i],2} layers to '{c.Name}' (Priority {c.PriorityScore:F1}, Capacity {c.MaxLayersCapacity}L)");
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
}
