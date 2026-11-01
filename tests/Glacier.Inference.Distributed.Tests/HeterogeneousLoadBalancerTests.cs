namespace Glacier.Inference.Distributed.Tests;

using System;
using System.Collections.Generic;
using System.Linq;
using Glacier.Inference.Distributed.Pipeline;
using Xunit;

/// <summary>
/// Comprehensive verification suite for Interface Contract 3: HeterogeneousLoadBalancer.
/// Validates closed-form balance equations, Hamiltonian largest remainder integer rounding,
/// capacity constraints, automated node pruning, and fleet calibration scenarios.
/// </summary>
public class HeterogeneousLoadBalancerTests
{
    private static List<StageDeviceProfile> CreateStandardFleet4Gpus()
    {
        return new List<StageDeviceProfile>
        {
            new StageDeviceProfile
            {
                DeviceId = "Host-RTX4060",
                DeviceName = "NVIDIA GeForce RTX 4060 Laptop GPU",
                ComputeMsPerLayer = 2.0,
                MaxLayersCapacity = 24,
                IsLocal = true,
                FixedOverheadMs = 0.5 // Root embedding overhead
            },
            new StageDeviceProfile
            {
                DeviceId = "Host-Radeon890M",
                DeviceName = "AMD Radeon 890M Graphics",
                ComputeMsPerLayer = 8.0,
                MaxLayersCapacity = 32,
                IsLocal = true,
                FixedOverheadMs = 0.6 // In-process intermediate handoff
            },
            new StageDeviceProfile
            {
                DeviceId = "Beast-RTX3060",
                DeviceName = "NVIDIA GeForce RTX 3060 12GB",
                ComputeMsPerLayer = 2.5,
                MaxLayersCapacity = 28,
                IsLocal = false,
                NetworkHopLatencyMs = 0.6,
                FixedOverheadMs = 0.6 // Socket intermediate handoff
            },
            new StageDeviceProfile
            {
                DeviceId = "MinisForum64-Radeon680M",
                DeviceName = "AMD Radeon 680M Graphics",
                ComputeMsPerLayer = 35.0,
                MaxLayersCapacity = 40,
                IsLocal = false,
                NetworkHopLatencyMs = 0.6,
                FixedOverheadMs = 3.5 // Final stage LM Head overhead
            }
        };
    }

    #region 1. Contract 3 Exact Fleet Calibration Tests

    [Fact]
    public void Fleet4Gpu_WithoutPruning_AllocatesExact19_5_15_1()
    {
        // Arrange
        var fleet = CreateStandardFleet4Gpus();
        var options = new HeterogeneousLoadBalancerOptions
        {
            EnableAutomaticPruning = false
        };

        // Act
        var result = HeterogeneousLoadBalancer.BalanceLayers(40, fleet, options);

        // Assert
        Assert.Equal(40, result.TotalLayers);
        Assert.Equal(4, result.ActiveStages.Count);
        Assert.Empty(result.PrunedDevices);

        // Verify exact contract layer allocations: 19, 5, 15, 1
        Assert.Equal(19, result.ActiveStages[0].LayerCount);
        Assert.Equal(5, result.ActiveStages[1].LayerCount);
        Assert.Equal(15, result.ActiveStages[2].LayerCount);
        Assert.Equal(1, result.ActiveStages[3].LayerCount);

        // Verify layer ranges are contiguous
        Assert.Equal(0, result.ActiveStages[0].StartLayer);
        Assert.Equal(18, result.ActiveStages[0].EndLayer);
        Assert.Equal(19, result.ActiveStages[1].StartLayer);
        Assert.Equal(23, result.ActiveStages[1].EndLayer);
        Assert.Equal(24, result.ActiveStages[2].StartLayer);
        Assert.Equal(38, result.ActiveStages[2].EndLayer);
        Assert.Equal(39, result.ActiveStages[3].StartLayer);
        Assert.Equal(39, result.ActiveStages[3].EndLayer);

        // Verify analytical target cycle time: 38.60 ms (+/- 0.05 ms)
        Assert.InRange(result.TargetCycleTimeMs, 38.55, 38.65);

        // Verify bottleneck stage latency: ~40.6 ms (Stage 1)
        Assert.InRange(result.BottleneckStageLatencyMs, 40.5, 40.7);

        // Verify generation throughput tok/s >= 24.5 tok/s
        Assert.InRange(result.EstimatedThroughputTokensPerSec, 24.5, 24.8);
    }

    [Fact]
    public void Fleet4Gpu_WithPruning_PrunesMinisForumAndAllocatesExact20_5_15()
    {
        // Arrange
        var fleet = CreateStandardFleet4Gpus();
        // Remove hardcoded FixedOverheadMs on Beast & MinisForum so final stage role dynamically rebinds
        fleet[2] = fleet[2] with { FixedOverheadMs = null };
        fleet[3] = fleet[3] with { FixedOverheadMs = null };

        var options = new HeterogeneousLoadBalancerOptions
        {
            EnableAutomaticPruning = true,
            MinLayersThreshold = 2.0,
            MaxCyclePenaltyThresholdMs = 2.5,
            DefaultEmbedMs = 0.5,
            DefaultIntermediateHandoffMs = 0.6,
            DefaultLmHeadMs = 3.5
        };

        // Act
        var result = HeterogeneousLoadBalancer.BalanceLayers(40, fleet, options);

        // Assert
        Assert.Equal(40, result.TotalLayers);
        Assert.Equal(3, result.ActiveStages.Count);
        Assert.Single(result.PrunedDevices);
        Assert.Equal("MinisForum64-Radeon680M", result.PrunedDevices[0].DeviceId);

        // Verify exact contract layer allocations across 3 active GPUs: 20, 5, 15
        Assert.Equal(20, result.ActiveStages[0].LayerCount);
        Assert.Equal(5, result.ActiveStages[1].LayerCount);
        Assert.Equal(15, result.ActiveStages[2].LayerCount);

        // Verify contiguous layer ranges
        Assert.Equal(0, result.ActiveStages[0].StartLayer);
        Assert.Equal(19, result.ActiveStages[0].EndLayer);
        Assert.Equal(20, result.ActiveStages[1].StartLayer);
        Assert.Equal(24, result.ActiveStages[1].EndLayer);
        Assert.Equal(25, result.ActiveStages[2].StartLayer);
        Assert.Equal(39, result.ActiveStages[2].EndLayer);

        // Verify Stage 2 inherited final stage LM Head overhead (3.5 ms)
        Assert.True(result.ActiveStages[2].IsFinalStage);
        Assert.Equal(3.5, result.ActiveStages[2].OverheadMs);

        // Verify target cycle time: 40.71 ms (+/- 0.05 ms)
        Assert.InRange(result.TargetCycleTimeMs, 40.65, 40.75);

        // Verify bottleneck stage latency: ~41.0 ms
        Assert.InRange(result.BottleneckStageLatencyMs, 40.9, 41.1);

        // Verify generation throughput: ~24.39 tok/s
        Assert.InRange(result.EstimatedThroughputTokensPerSec, 24.3, 24.5);
    }

    #endregion

    #region 2. Mathematical Integrity & Invariants

    [Theory]
    [InlineData(16)]
    [InlineData(28)]
    [InlineData(32)]
    [InlineData(40)]
    [InlineData(48)]
    [InlineData(64)]
    [InlineData(80)]
    public void LargestRemainderMethod_AlwaysPreservesExactSumOfLayers(int totalLayers)
    {
        var fleet = CreateStandardFleet4Gpus();
        var options = new HeterogeneousLoadBalancerOptions { EnableAutomaticPruning = false };

        var result = HeterogeneousLoadBalancer.BalanceLayers(totalLayers, fleet, options);

        int allocatedSum = result.ActiveStages.Sum(s => s.LayerCount);
        Assert.Equal(totalLayers, allocatedSum);
    }

    [Fact]
    public void HomogeneousCluster_BalancesEqually()
    {
        var uniformFleet = new List<StageDeviceProfile>
        {
            new StageDeviceProfile { DeviceId = "GPU0", ComputeMsPerLayer = 2.0, MaxLayersCapacity = 30, FixedOverheadMs = 0.5 },
            new StageDeviceProfile { DeviceId = "GPU1", ComputeMsPerLayer = 2.0, MaxLayersCapacity = 30, FixedOverheadMs = 0.5 },
            new StageDeviceProfile { DeviceId = "GPU2", ComputeMsPerLayer = 2.0, MaxLayersCapacity = 30, FixedOverheadMs = 0.5 },
            new StageDeviceProfile { DeviceId = "GPU3", ComputeMsPerLayer = 2.0, MaxLayersCapacity = 30, FixedOverheadMs = 0.5 }
        };

        var result = HeterogeneousLoadBalancer.BalanceLayers(40, uniformFleet, new HeterogeneousLoadBalancerOptions { EnableAutomaticPruning = false });

        Assert.Equal(4, result.ActiveStages.Count);
        foreach (var stage in result.ActiveStages)
        {
            Assert.Equal(10, stage.LayerCount);
        }
    }

    [Fact]
    public void SingleNode_AllocatesAllLayers()
    {
        var singleGpu = new List<StageDeviceProfile>
        {
            new StageDeviceProfile { DeviceId = "SoloGPU", ComputeMsPerLayer = 1.5, MaxLayersCapacity = 40 }
        };

        var result = HeterogeneousLoadBalancer.BalanceLayers(32, singleGpu);

        Assert.Single(result.ActiveStages);
        Assert.Equal(32, result.ActiveStages[0].LayerCount);
        Assert.Equal(0, result.ActiveStages[0].StartLayer);
        Assert.Equal(31, result.ActiveStages[0].EndLayer);
    }

    [Fact]
    public void SingleLayerModel_AllocatesToFastestDeviceWithoutException()
    {
        var candidates = new List<StageDeviceProfile>
        {
            new StageDeviceProfile { DeviceId = "FastGPU", ComputeMsPerLayer = 1.0, MaxLayersCapacity = 10 },
            new StageDeviceProfile { DeviceId = "SlowGPU", ComputeMsPerLayer = 10.0, MaxLayersCapacity = 10 }
        };

        var result = HeterogeneousLoadBalancer.BalanceLayers(1, candidates, new HeterogeneousLoadBalancerOptions { EnableAutomaticPruning = false });

        Assert.Equal(1, result.TotalLayers);
        Assert.Equal(1, result.ActiveStages.Sum(s => s.LayerCount));
        Assert.Equal(1, result.ActiveStages[0].LayerCount);
    }

    #endregion

    #region 3. Capacity Constraint & Water-Filling Tests

    [Fact]
    public void CapacityConstraint_ClampsSaturatedNodeAndReBalancesRemaining()
    {
        // Node 0 is extremely fast (1.0 ms) but only has 5 layers of VRAM capacity.
        // Node 1 is slower (4.0 ms) but has plenty of capacity (50 layers).
        var candidates = new List<StageDeviceProfile>
        {
            new StageDeviceProfile { DeviceId = "FastSmallVram", ComputeMsPerLayer = 1.0, MaxLayersCapacity = 5, FixedOverheadMs = 0.5 },
            new StageDeviceProfile { DeviceId = "SlowLargeVram", ComputeMsPerLayer = 4.0, MaxLayersCapacity = 50, FixedOverheadMs = 0.5 }
        };

        var result = HeterogeneousLoadBalancer.BalanceLayers(20, candidates, new HeterogeneousLoadBalancerOptions { EnableAutomaticPruning = false });

        Assert.Equal(20, result.ActiveStages.Sum(s => s.LayerCount));
        // Node 0 must be clamped exactly to its capacity: 5
        Assert.Equal(5, result.ActiveStages[0].LayerCount);
        // Node 1 receives remaining 15 layers
        Assert.Equal(15, result.ActiveStages[1].LayerCount);
    }

    [Fact]
    public void InsufficientClusterCapacity_ThrowsInvalidOperationException()
    {
        var lowCapFleet = new List<StageDeviceProfile>
        {
            new StageDeviceProfile { DeviceId = "GpuA", ComputeMsPerLayer = 2.0, MaxLayersCapacity = 10 },
            new StageDeviceProfile { DeviceId = "GpuB", ComputeMsPerLayer = 2.0, MaxLayersCapacity = 15 }
        };

        // Requires 40 layers, but cluster only fits 25
        Assert.Throws<InvalidOperationException>(() =>
            HeterogeneousLoadBalancer.BalanceLayers(40, lowCapFleet));
    }

    #endregion

    #region 4. Automated Pruning & Penalty Tests

    [Fact]
    public void AutomatedPruning_PrunesSeverelyLaggingRemoteNode()
    {
        var candidates = new List<StageDeviceProfile>
        {
            new StageDeviceProfile { DeviceId = "Host-RTX4090", ComputeMsPerLayer = 1.0, MaxLayersCapacity = 32, IsLocal = true },
            new StageDeviceProfile { DeviceId = "Remote-SlowCpu", ComputeMsPerLayer = 80.0, MaxLayersCapacity = 32, IsLocal = false, NetworkHopLatencyMs = 2.0 }
        };

        var result = HeterogeneousLoadBalancer.BalanceLayers(32, candidates);

        // Remote CPU should be pruned because continuous layers < 2.0 and network hop exceeds savings
        Assert.Single(result.ActiveStages);
        Assert.Equal("Host-RTX4090", result.ActiveStages[0].Device.DeviceId);
        Assert.Equal(32, result.ActiveStages[0].LayerCount);
        Assert.Single(result.PrunedDevices);
        Assert.Equal("Remote-SlowCpu", result.PrunedDevices[0].DeviceId);
    }

    #endregion

    #region 5. Defensive Input Validation

    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public void InvalidTotalLayers_ThrowsArgumentOutOfRangeException(int invalidLayers)
    {
        var fleet = CreateStandardFleet4Gpus();
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            HeterogeneousLoadBalancer.BalanceLayers(invalidLayers, fleet));
    }

    [Fact]
    public void EmptyCandidatesList_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            HeterogeneousLoadBalancer.BalanceLayers(40, Array.Empty<StageDeviceProfile>()));
    }

    #endregion
}
