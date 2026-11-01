namespace Glacier.Inference.Distributed.Pipeline;

using System;

/// <summary>
/// Hardware execution profile and empirical telemetry for a cluster compute node.
/// </summary>
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
                // Permit up to 75% allocation of total physical host RAM for layer weights.
                ulong hostRam = SharedMemoryBytes > 0 ? SharedMemoryBytes : DynamicThroughputCalibrator.GetPhysicalHostRamBytes();
                ulong usableUma = (ulong)(hostRam * 0.75);
                int maxFit = (int)(usableUma / bytesPerLayer);
                return Math.Clamp(maxFit, 0, totalModelLayers);
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
