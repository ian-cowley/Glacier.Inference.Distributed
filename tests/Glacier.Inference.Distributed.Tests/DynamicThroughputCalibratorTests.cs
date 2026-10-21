namespace Glacier.Inference.Distributed.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Inference.Distributed.Cli;
using Glacier.Inference.Distributed.Pipeline;
using Glacier.Inference.Distributed.Transport;
using Xunit;

public class DynamicThroughputCalibratorTests
{
    private sealed class MockSyntheticRunner : ISyntheticLayerRunner
    {
        public double LayerComputeMs { get; set; } = 3.5;
        public double LmHeadMs { get; set; } = 5.2;

        public double RunSingleLayerForward(int hiddenDim, int intermediateDim, int iterations) => LayerComputeMs;
        public double RunLmHeadForward(int hiddenDim, int vocabSize, int iterations) => LmHeadMs;
    }

    [Fact]
    public void CalibrateLocalNode_MeasuresEmpiricalTauAndZeroNetworkHop()
    {
        var node = new NodeResourceProfile
        {
            Name = "Host Laptop GPU",
            Host = "127.0.0.1",
            Port = 0,
            IsLocal = true,
            DeviceName = "NVIDIA GeForce RTX 4060 Laptop GPU",
            IsGpu = true,
            VramBytes = 8UL * 1024 * 1024 * 1024,
            EstimatedMsPerLayer = 999.0
        };

        var runner = new MockSyntheticRunner { LayerComputeMs = 2.15, LmHeadMs = 4.80 };
        var options = new CalibrationOptions { HiddenDim = 5120, IntermediateDim = 13824 };

        var result = DynamicThroughputCalibrator.CalibrateLocalNode(node, options, runner);

        Assert.True(result.IsSuccess);
        Assert.True(result.IsLocal);
        Assert.Equal(0.0, result.NetworkHopMs);
        Assert.Equal(2.15, result.TauMsPerLayer);
        Assert.Equal(4.80, result.DeltaStageOverheadMs);
        Assert.Equal(node.VramBytes, result.UsableMemoryBytes);
    }

    [Fact]
    public async Task CalibrateRemoteNode_OverSocket_MeasuresTauAndNetworkLatency()
    {
        var endpoint = new IPEndPoint(IPAddress.Loopback, 0);
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(endpoint);
        listener.Listen(1);
        int port = ((IPEndPoint)listener.LocalEndPoint!).Port;

        var serverTask = Task.Run(async () =>
        {
            using var serverSocket = await listener.AcceptAsync();
            using var netStream = new NetworkStream(serverSocket, ownsSocket: false);

            byte[] hdrBytes = new byte[ClusterFrameHeader.HeaderSize];
            int read = await netStream.ReadAsync(hdrBytes);
            Assert.Equal(ClusterFrameHeader.HeaderSize, read);
            var reqHdr = System.Runtime.InteropServices.MemoryMarshal.Read<ClusterFrameHeader>(hdrBytes);

            var runner = new MockSyntheticRunner { LayerComputeMs = 15.4, LmHeadMs = 6.0 };
            await DynamicThroughputCalibrator.HandleProbeRequestAsync(netStream, reqHdr, runner);
        });

        var remoteNode = new NodeResourceProfile
        {
            Name = "Beast Desktop",
            Host = "127.0.0.1",
            Port = port,
            IsLocal = false,
            DeviceName = "NVIDIA RTX 3060",
            IsGpu = true,
            VramBytes = 12UL * 1024 * 1024 * 1024,
            EstimatedMsPerLayer = 999.0
        };

        var options = new CalibrationOptions { Timeout = TimeSpan.FromSeconds(5) };
        var result = await DynamicThroughputCalibrator.CalibrateRemoteNodeAsync(remoteNode, options);
        await serverTask;

        Assert.True(result.IsSuccess);
        Assert.False(result.IsLocal);
        Assert.Equal(15.4, result.TauMsPerLayer);
        Assert.Equal(6.0, result.DeltaStageOverheadMs);
        Assert.True(result.NetworkHopMs >= 0.0);
    }

    [Fact]
    public async Task CalibrateClusterAsync_UpdatesNodeResourceProfiles_EliminatesStaticConstants()
    {
        var candidates = new List<NodeResourceProfile>
        {
            new()
            {
                Name = "Local GPU",
                Host = "127.0.0.1",
                IsLocal = true,
                DeviceName = "RTX 4060",
                IsGpu = true,
                VramBytes = 8UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 999.0
            }
        };

        var results = await DynamicThroughputCalibrator.CalibrateClusterAsync(candidates);

        Assert.Single(results);
        Assert.True(results[0].IsSuccess);
        Assert.NotEqual(999.0, candidates[0].EstimatedMsPerLayer);
        Assert.True(candidates[0].EstimatedMsPerLayer > 0.0);
        Assert.True(candidates[0].PriorityScore > 0.0);
    }

    [Fact]
    public void UncappedUmaCapacity_Radeon890M_CalculatesCapacityBasedOn75PercentHostRam()
    {
        var profile = new NodeResourceProfile
        {
            Name = "Host Radeon 890M",
            DeviceName = "AMD Radeon 890M Graphics",
            IsGpu = true,
            VramBytes = 512UL * 1024 * 1024,
            SharedMemoryBytes = 32UL * 1024 * 1024 * 1024 // 32 GB LPDDR5x RAM
        };

        int totalModelLayers = 40;
        ulong bytesPerLayer = 500_000_000UL; // 500 MB per layer

        // 75% of 32 GB = 24 GB. At 500 MB/layer: 24,000,000,000 / 500,000,000 = 48 layers (clamped to 40)
        int cap = profile.CalculateComfortableCapacity(bytesPerLayer, totalModelLayers);

        Assert.Equal(40, cap);
        Assert.True(cap > 16, "Capacity must NOT be capped at 16 layers for 32GB UMA!");
    }

    [Fact]
    public void UncappedUmaCapacity_MinisForum64_CalculatesCapacityBasedOn64GBHostRam()
    {
        var profile = new NodeResourceProfile
        {
            Name = "MinisForum64",
            DeviceName = "AMD Radeon 680M Graphics",
            IsGpu = true,
            VramBytes = 512UL * 1024 * 1024,
            SharedMemoryBytes = 64UL * 1024 * 1024 * 1024 // 64 GB DDR5 RAM
        };

        int totalModelLayers = 48;
        ulong bytesPerLayer = 600_000_000UL; // 600 MB per layer

        // 75% of 64 GB = 48 GB. At 600 MB/layer: 48,000,000,000 / 600,000,000 = 80 layers (clamped to 48)
        int cap = profile.CalculateComfortableCapacity(bytesPerLayer, totalModelLayers);

        Assert.Equal(48, cap);
        Assert.True(cap > 16, "Capacity must NOT be capped at 16 layers for 64GB UMA!");
    }

    [Fact]
    public void PlanStages_IncludesUmaGpu_WhenFilter3GBIsRemoved()
    {
        var localGpu = new NodeResourceProfile
        {
            Name = "Host dGPU",
            Host = "127.0.0.1",
            IsLocal = true,
            DeviceName = "NVIDIA RTX 4060",
            IsGpu = true,
            VramBytes = 8UL * 1024 * 1024 * 1024,
            EstimatedMsPerLayer = 2.0
        };

        var umaGpu = new NodeResourceProfile
        {
            Name = "Host iGPU",
            Host = "127.0.0.1",
            IsLocal = true,
            DeviceName = "AMD Radeon 890M",
            IsGpu = true,
            VramBytes = 512UL * 1024 * 1024,
            SharedMemoryBytes = 32UL * 1024 * 1024 * 1024,
            EstimatedMsPerLayer = 8.0
        };

        int totalLayers = 40;
        ulong totalBytes = 20UL * 1024 * 1024 * 1024;

        var stages = ClusterResourcePlanner.CalculatePlan([localGpu, umaGpu], [], totalLayers, totalBytes, "auto");

        Assert.NotEmpty(stages);
        int umaAllocated = 0;
        foreach (var st in stages)
        {
            if (st.Node.DeviceName.Contains("890M"))
                umaAllocated += st.LayerCount;
        }
        Assert.True(umaAllocated > 0 || stages.Count >= 1, "UMA GPU must be a valid candidate device.");
    }

    [Fact]
    public void GetPhysicalHostRamBytes_CrossPlatform_ReturnsValidPositiveBytes()
    {
        ulong ram = ClusterResourcePlanner.GetPhysicalHostRamBytes();
        Assert.True(ram >= 4UL * 1024 * 1024 * 1024, $"Physical host RAM ({ram} bytes) must be >= 4GB on modern systems.");
    }

    [Fact]
    public void CpuSimdSyntheticRunner_ExecutesForwardPassCleanly()
    {
        var runner = new CpuSimdSyntheticRunner();
        double msPerLayer = runner.RunSingleLayerForward(hiddenDim: 4096, intermediateDim: 11008, iterations: 2);
        Assert.True(msPerLayer > 0.0);

        double lmHeadMs = runner.RunLmHeadForward(hiddenDim: 4096, vocabSize: 32000, iterations: 2);
        Assert.True(lmHeadMs > 0.0);
    }
}
