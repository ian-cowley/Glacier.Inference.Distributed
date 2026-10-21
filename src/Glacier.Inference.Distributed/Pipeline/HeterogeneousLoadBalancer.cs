namespace Glacier.Inference.Distributed.Pipeline;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

/// <summary>
/// Hardware execution profile and empirical telemetry for a pipeline candidate stage/device.
/// </summary>
public sealed record StageDeviceProfile
{
    /// <summary>
    /// Unique identifier or hostname of the node (e.g. "Host-dGPU", "Beast-3060").
    /// </summary>
    public string DeviceId { get; init; } = "";

    /// <summary>
    /// Human-readable device display name (e.g. "NVIDIA GeForce RTX 4060 Laptop GPU").
    /// </summary>
    public string DeviceName { get; init; } = "";

    /// <summary>
    /// Empirical or calibrated compute latency per transformer layer in milliseconds (\tau_i).
    /// </summary>
    public double ComputeMsPerLayer { get; init; }

    /// <summary>
    /// Maximum number of model layers this device can fit in physical memory without paging (C_i).
    /// </summary>
    public int MaxLayersCapacity { get; init; } = int.MaxValue;

    /// <summary>
    /// Fixed execution overhead for this stage in milliseconds (\delta_i).
    /// If null, default overhead is assigned based on pipeline position (EmbedMs, IntermediateHandoffMs, LmHeadMs).
    /// </summary>
    public double? FixedOverheadMs { get; init; }

    /// <summary>
    /// Indicates whether this stage executes locally on the coordinator host (in-process / shared-memory)
    /// or remotely across the network (socket transport).
    /// </summary>
    public bool IsLocal { get; init; } = true;

    /// <summary>
    /// Network hop latency to/from this node in milliseconds (0.0 for local in-process, ~0.6ms for 1GbE LAN).
    /// </summary>
    public double NetworkHopLatencyMs { get; init; } = 0.0;
}

/// <summary>
/// Configuration options governing the analytical layer balancing and automated node pruning engine.
/// </summary>
public sealed record HeterogeneousLoadBalancerOptions
{
    /// <summary>
    /// Minimum continuous layer allocation required to keep a stage active in the pipeline.
    /// If L_i^* falls below this threshold and network overhead exceeds compute savings, the node is pruned.
    /// Default is 2.0 layers.
    /// </summary>
    public double MinLayersThreshold { get; init; } = 2.0;

    /// <summary>
    /// Enables automated pruning of underperforming or network-bottlenecked nodes.
    /// Default is true.
    /// </summary>
    public bool EnableAutomaticPruning { get; init; } = true;

    /// <summary>
    /// Maximum tolerable cycle time increase (in milliseconds) when deciding whether to prune a sub-threshold node.
    /// If (T_cycle_without - T_cycle_with) &lt;= MaxCyclePenaltyThresholdMs, the node is pruned.
    /// Default is 2.5 ms.
    /// </summary>
    public double MaxCyclePenaltyThresholdMs { get; init; } = 2.5;

    /// <summary>
    /// Default fixed overhead for Stage 0 (token embedding lookup and root normalization) in milliseconds.
    /// Default is 0.5 ms.
    /// </summary>
    public double DefaultEmbedMs { get; init; } = 0.5;

    /// <summary>
    /// Default fixed overhead for intermediate pipeline stages (socket serialization/deserialization) in milliseconds.
    /// Default is 0.6 ms.
    /// </summary>
    public double DefaultIntermediateHandoffMs { get; init; } = 0.6;

    /// <summary>
    /// Default fixed overhead for the final pipeline stage (output RMSNorm + LM Head projection + sampling) in milliseconds.
    /// Default is 3.5 ms.
    /// </summary>
    public double DefaultLmHeadMs { get; init; } = 3.5;

    /// <summary>
    /// Default network hop transit latency for remote nodes across Gigabit Ethernet in milliseconds.
    /// Default is 0.6 ms.
    /// </summary>
    public double DefaultNetworkHopMs { get; init; } = 0.6;
}

/// <summary>
/// Concrete layer allocation plan for an active pipeline stage.
/// </summary>
public sealed record StageAllocationPlan
{
    /// <summary>
    /// Zero-based sequential stage index in the active execution pipeline.
    /// </summary>
    public int StageIndex { get; init; }

    /// <summary>
    /// The compute device assigned to this stage.
    /// </summary>
    public StageDeviceProfile Device { get; init; } = null!;

    /// <summary>
    /// First transformer layer assigned to this stage (inclusive, 0-indexed).
    /// </summary>
    public int StartLayer { get; init; }

    /// <summary>
    /// Last transformer layer assigned to this stage (inclusive, 0-indexed).
    /// </summary>
    public int EndLayer { get; init; }

    /// <summary>
    /// Total number of transformer layers assigned to this stage (L_i).
    /// </summary>
    public int LayerCount { get; init; }

    /// <summary>
    /// Ideal continuous layer quota calculated before integer rounding (L_i^*).
    /// </summary>
    public double ContinuousOptimalLayers { get; init; }

    /// <summary>
    /// Fractional remainder from integer floor used by Hamilton's Largest Remainder Method (r_i).
    /// </summary>
    public double FractionalRemainder { get; init; }

    /// <summary>
    /// Stage overhead applied in milliseconds (\delta_i).
    /// </summary>
    public double OverheadMs { get; init; }

    /// <summary>
    /// Expected execution latency for this stage in milliseconds: T_i = L_i * \tau_i + \delta_i.
    /// </summary>
    public double EstimatedStageLatencyMs { get; init; }

    /// <summary>
    /// Indicates whether this stage executes the root embedding and ingress normalization.
    /// </summary>
    public bool IsRootStage => StartLayer == 0;

    /// <summary>
    /// Indicates whether this stage executes the final LM Head projection and sampling.
    /// </summary>
    public bool IsFinalStage { get; init; }
}

/// <summary>
/// Complete partitioning plan and throughput telemetry produced by the HeterogeneousLoadBalancer.
/// </summary>
public sealed record LoadBalancerResult
{
    /// <summary>
    /// Total transformer layers distributed across active stages (N).
    /// </summary>
    public int TotalLayers { get; init; }

    /// <summary>
    /// Analytical target cycle time derived from the balance equation in milliseconds (T_target).
    /// </summary>
    public double TargetCycleTimeMs { get; init; }

    /// <summary>
    /// Maximum execution latency across all active pipeline stages after integer rounding: max(T_i).
    /// </summary>
    public double BottleneckStageLatencyMs { get; init; }

    /// <summary>
    /// Theoretical maximum cluster generation throughput in tokens per second: 1000.0 / BottleneckStageLatencyMs.
    /// </summary>
    public double EstimatedThroughputTokensPerSec => BottleneckStageLatencyMs > 0 ? (1000.0 / BottleneckStageLatencyMs) : 0.0;

    /// <summary>
    /// Total estimated prefill latency for a single token across all active stages including network hops.
    /// </summary>
    public double EstimatedPrefillLatencyMs { get; init; }

    /// <summary>
    /// Ordered list of active pipeline stage allocation plans.
    /// </summary>
    public IReadOnlyList<StageAllocationPlan> ActiveStages { get; init; } = Array.Empty<StageAllocationPlan>();

    /// <summary>
    /// List of devices pruned from the pipeline by the automated pruning engine.
    /// </summary>
    public IReadOnlyList<StageDeviceProfile> PrunedDevices { get; init; } = Array.Empty<StageDeviceProfile>();

    /// <summary>
    /// Diagnostic summary explaining the allocation, remainders, and pruning rationale.
    /// </summary>
    public string DiagnosticSummary { get; init; } = "";
}

/// <summary>
/// Analytical Heterogeneous Layer Balancer implementing Interface Contract 3.
/// Computes mathematically optimal pipeline stage partitions using closed-form balance equations,
/// bounded capacity water-filling, Hamilton/Hare-Niemeyer Largest Remainder integer rounding,
/// and automated sub-threshold node pruning.
/// </summary>
public static class HeterogeneousLoadBalancer
{
    /// <summary>
    /// Computes the optimal integer layer distribution across candidate cluster devices
    /// satisfying Interface Contract 3:
    ///   T_target = (N + \sum \delta_j / \tau_j) / \sum(1 / \tau_j)
    ///   L_i^* = (T_target - \delta_i) / \tau_i
    /// </summary>
    /// <param name="totalLayers">Total number of transformer layers in the model (N).</param>
    /// <param name="candidateDevices">Ordered candidate compute devices available in the cluster.</param>
    /// <param name="options">Balancing and pruning options (optional).</param>
    /// <returns>A validated <see cref="LoadBalancerResult"/> detailing stage ranges, latencies, and pruning rationale.</returns>
    public static LoadBalancerResult BalanceLayers(
        int totalLayers,
        IReadOnlyList<StageDeviceProfile> candidateDevices,
        HeterogeneousLoadBalancerOptions? options = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(totalLayers);
        ArgumentNullException.ThrowIfNull(candidateDevices);
        if (candidateDevices.Count == 0)
        {
            throw new ArgumentException("Candidate devices list cannot be empty.", nameof(candidateDevices));
        }

        options ??= new HeterogeneousLoadBalancerOptions();

        // 1. Separate valid active candidates from non-positive capacity / zero compute devices
        var activeList = new List<StageDeviceProfile>();
        var prunedList = new List<StageDeviceProfile>();

        foreach (var dev in candidateDevices)
        {
            if (dev.ComputeMsPerLayer <= 0 || dev.MaxLayersCapacity <= 0)
            {
                prunedList.Add(dev);
            }
            else
            {
                activeList.Add(dev);
            }
        }

        if (activeList.Count == 0)
        {
            throw new InvalidOperationException("No candidate devices have valid positive compute throughput and capacity.");
        }

        long totalCapacity = activeList.Sum(d => (long)d.MaxLayersCapacity);
        if (totalCapacity < totalLayers)
        {
            throw new InvalidOperationException(
                $"Total cluster capacity ({totalCapacity} layers) is insufficient to allocate model ({totalLayers} layers).");
        }

        // 2. Iterative pruning and balance loop
        while (true)
        {
            var (targetCycleMs, continuousAllocations, overheads) = SolveContinuous(totalLayers, activeList, options);

            // Automated Pruning Evaluation: check if any device allocates below threshold
            if (options.EnableAutomaticPruning && activeList.Count > 1)
            {
                int pruneCandidateIdx = -1;
                double minAlloc = double.MaxValue;

                for (int i = 0; i < activeList.Count; i++)
                {
                    if (continuousAllocations[i] < options.MinLayersThreshold && continuousAllocations[i] < minAlloc)
                    {
                        minAlloc = continuousAllocations[i];
                        pruneCandidateIdx = i;
                    }
                }

                if (pruneCandidateIdx >= 0)
                {
                    var candidateToPrune = activeList[pruneCandidateIdx];
                    var testActiveList = new List<StageDeviceProfile>(activeList);
                    testActiveList.RemoveAt(pruneCandidateIdx);

                    long remainingCap = testActiveList.Sum(d => (long)d.MaxLayersCapacity);
                    if (remainingCap >= totalLayers)
                    {
                        var (withoutTargetCycleMs, _, _) = SolveContinuous(totalLayers, testActiveList, options);
                        double cyclePenaltyMs = withoutTargetCycleMs - targetCycleMs;

                        // Pruning condition: cycle penalty is tolerable or network hop penalty exceeds compute savings
                        double hopPenaltyMs = candidateToPrune.IsLocal
                            ? 0.0
                            : (candidateToPrune.NetworkHopLatencyMs > 0 ? candidateToPrune.NetworkHopLatencyMs : options.DefaultNetworkHopMs);

                        bool shouldPrune = cyclePenaltyMs <= options.MaxCyclePenaltyThresholdMs ||
                                           (!candidateToPrune.IsLocal && hopPenaltyMs >= cyclePenaltyMs);

                        if (shouldPrune)
                        {
                            activeList.RemoveAt(pruneCandidateIdx);
                            prunedList.Add(candidateToPrune);
                            continue; // Re-solve with updated active set!
                        }
                    }
                }
            }

            // 3. Integer rounding via Hamilton/Hare-Niemeyer Largest Remainder Method
            var integerLayers = ApplyLargestRemainderMethod(totalLayers, activeList, continuousAllocations);

            // 4. Build concrete stage allocation plan
            return BuildResult(totalLayers, activeList, prunedList, targetCycleMs, continuousAllocations, integerLayers, overheads, options);
        }
    }

    /// <summary>
    /// Solves continuous layer allocations with iterative KKT capacity water-filling:
    ///   T_target = (N_rem + \sum \delta_j / \tau_j) / \sum(1 / \tau_j)
    /// </summary>
    private static (double TargetCycleMs, double[] ContinuousAllocations, double[] Overheads) SolveContinuous(
        int totalLayers,
        IReadOnlyList<StageDeviceProfile> activeDevices,
        HeterogeneousLoadBalancerOptions options)
    {
        int count = activeDevices.Count;
        double[] overheads = new double[count];

        for (int i = 0; i < count; i++)
        {
            if (activeDevices[i].FixedOverheadMs.HasValue)
            {
                overheads[i] = activeDevices[i].FixedOverheadMs!.Value;
            }
            else if (i == 0)
            {
                overheads[i] = options.DefaultEmbedMs;
            }
            else if (i == count - 1)
            {
                overheads[i] = options.DefaultLmHeadMs;
            }
            else
            {
                overheads[i] = options.DefaultIntermediateHandoffMs;
            }
        }

        var unconstrained = new HashSet<int>(Enumerable.Range(0, count));
        var fixedAllocations = new Dictionary<int, double>();
        double nRemaining = totalLayers;
        double tTarget = 0.0;

        while (unconstrained.Count > 0)
        {
            double sumConductance = 0.0;
            double sumOverheadRatio = 0.0;

            foreach (int idx in unconstrained)
            {
                double tau = activeDevices[idx].ComputeMsPerLayer;
                sumConductance += 1.0 / tau;
                sumOverheadRatio += overheads[idx] / tau;
            }

            tTarget = (nRemaining + sumOverheadRatio) / sumConductance;

            int worstViolator = -1;
            double maxViolation = 0.0;
            bool isUpperViolation = true;

            foreach (int idx in unconstrained)
            {
                double tau = activeDevices[idx].ComputeMsPerLayer;
                double provisionalL = (tTarget - overheads[idx]) / tau;
                int cap = activeDevices[idx].MaxLayersCapacity;

                if (provisionalL > cap && (provisionalL - cap) > maxViolation)
                {
                    maxViolation = provisionalL - cap;
                    worstViolator = idx;
                    isUpperViolation = true;
                }
                else if (provisionalL < 0 && (-provisionalL) > maxViolation)
                {
                    maxViolation = -provisionalL;
                    worstViolator = idx;
                    isUpperViolation = false;
                }
            }

            if (worstViolator < 0)
            {
                break; // All unconstrained nodes are within bounds [0, C_i]
            }

            unconstrained.Remove(worstViolator);
            double clampedVal = isUpperViolation ? activeDevices[worstViolator].MaxLayersCapacity : 0.0;
            fixedAllocations[worstViolator] = clampedVal;
            nRemaining -= clampedVal;
        }

        double[] continuousAllocations = new double[count];
        for (int i = 0; i < count; i++)
        {
            if (fixedAllocations.TryGetValue(i, out double fixedVal))
            {
                continuousAllocations[i] = fixedVal;
            }
            else
            {
                continuousAllocations[i] = (tTarget - overheads[i]) / activeDevices[i].ComputeMsPerLayer;
            }
        }

        return (tTarget, continuousAllocations, overheads);
    }

    /// <summary>
    /// Preserves exact total layer conservation (\sum L_i == N) using Hamilton's Largest Remainder Method
    /// while strictly respecting individual device capacity constraints (L_i &lt;= C_i).
    /// </summary>
    private static int[] ApplyLargestRemainderMethod(
        int totalLayers,
        IReadOnlyList<StageDeviceProfile> activeDevices,
        double[] continuousAllocations)
    {
        int count = activeDevices.Count;
        int[] floorAllocations = new int[count];
        double[] remainders = new double[count];
        int sumFloors = 0;

        for (int i = 0; i < count; i++)
        {
            double val = continuousAllocations[i];
            int fl = (int)Math.Floor(val);
            fl = Math.Clamp(fl, 0, activeDevices[i].MaxLayersCapacity);
            floorAllocations[i] = fl;
            remainders[i] = Math.Max(0.0, val - fl);
            sumFloors += fl;
        }

        int deficit = totalLayers - sumFloors;
        if (deficit <= 0)
        {
            return floorAllocations;
        }

        // Rank eligible devices (where floor < capacity) by remainder descending,
        // breaking ties with lower compute time \tau_i, then lower device index.
        var eligibleIndices = Enumerable.Range(0, count)
            .Where(i => floorAllocations[i] < activeDevices[i].MaxLayersCapacity)
            .OrderByDescending(i => remainders[i])
            .ThenBy(i => activeDevices[i].ComputeMsPerLayer)
            .ThenBy(i => i)
            .ToList();

        for (int i = 0; i < deficit && i < eligibleIndices.Count; i++)
        {
            floorAllocations[eligibleIndices[i]]++;
        }

        return floorAllocations;
    }

    private static LoadBalancerResult BuildResult(
        int totalLayers,
        IReadOnlyList<StageDeviceProfile> activeDevices,
        IReadOnlyList<StageDeviceProfile> prunedDevices,
        double targetCycleMs,
        double[] continuousAllocations,
        int[] integerLayers,
        double[] overheads,
        HeterogeneousLoadBalancerOptions options)
    {
        int count = activeDevices.Count;
        var stages = new List<StageAllocationPlan>(count);
        int curLayer = 0;
        double maxStageTime = 0.0;
        double totalPrefillLatency = 0.0;

        for (int i = 0; i < count; i++)
        {
            int layerCount = integerLayers[i];
            int startL = curLayer;
            int endL = curLayer + layerCount - 1;
            curLayer += layerCount;

            double stageTime = layerCount * activeDevices[i].ComputeMsPerLayer + overheads[i];
            if (stageTime > maxStageTime) maxStageTime = stageTime;
            totalPrefillLatency += stageTime;

            if (i > 0)
            {
                double hopMs = activeDevices[i].IsLocal
                    ? 0.0
                    : (activeDevices[i].NetworkHopLatencyMs > 0 ? activeDevices[i].NetworkHopLatencyMs : options.DefaultNetworkHopMs);
                totalPrefillLatency += hopMs;
            }

            stages.Add(new StageAllocationPlan
            {
                StageIndex = i,
                Device = activeDevices[i],
                StartLayer = startL,
                EndLayer = endL,
                LayerCount = layerCount,
                ContinuousOptimalLayers = continuousAllocations[i],
                FractionalRemainder = continuousAllocations[i] - Math.Floor(continuousAllocations[i]),
                OverheadMs = overheads[i],
                EstimatedStageLatencyMs = stageTime,
                IsFinalStage = (i == count - 1)
            });
        }

        var sb = new StringBuilder();
        sb.AppendLine($"Partitioning {totalLayers} layers across {count} active stages (Target cycle: {targetCycleMs:F2} ms, Bottleneck: {maxStageTime:F2} ms, Throughput: {(maxStageTime > 0 ? 1000.0/maxStageTime : 0.0):F2} tok/s):");
        foreach (var s in stages)
        {
            string loc = s.Device.IsLocal ? "Local" : "Remote";
            string role = s.IsRootStage ? "[Root/Embed]" : (s.IsFinalStage ? "[Final/LMHead]" : "[Intermediate]");
            sb.AppendLine($"  Stage {s.StageIndex}: {s.Device.DeviceId,-16} ({loc,-6}) | {s.LayerCount,2}L (layers {s.StartLayer,2}..{s.EndLayer,2}) | {s.EstimatedStageLatencyMs:F2} ms | {role} (ideal: {s.ContinuousOptimalLayers:F3}L, rem: {s.FractionalRemainder:F3})");
        }
        if (prunedDevices.Count > 0)
        {
            sb.AppendLine("Pruned Devices:");
            foreach (var p in prunedDevices)
            {
                sb.AppendLine($"  - {p.DeviceId} ({p.DeviceName}): Pruned (tau={p.ComputeMsPerLayer:F1}ms/L, capacity={p.MaxLayersCapacity}L)");
            }
        }

        return new LoadBalancerResult
        {
            TotalLayers = totalLayers,
            TargetCycleTimeMs = targetCycleMs,
            BottleneckStageLatencyMs = maxStageTime,
            EstimatedPrefillLatencyMs = totalPrefillLatency,
            ActiveStages = stages,
            PrunedDevices = prunedDevices.ToList(),
            DiagnosticSummary = sb.ToString()
        };
    }
}
