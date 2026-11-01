#pragma warning disable GLACIER_EXP001

namespace Glacier.Inference.Distributed.Tests;

using System;
using Glacier.Inference.Distributed.Cli;
using Glacier.Inference.Distributed.Pipeline;
using Glacier.Inference.Gpu;
using Glacier.Inference.Gpu.D3D12;
using Xunit;

public class VulkanClusterStageModelTests
{
    [Fact]
    public void SafeD3D12AdapterSelection_ClampsOutOfBoundsIndex()
    {
        if (OperatingSystem.IsWindows() && D3D12Context.IsSupported)
        {
            var adapters = D3D12Context.EnumerateAdapters();
            if (adapters.Count > 0)
            {
                // Passing index 999 must clamp to adapters.Count - 1 without throwing
                int clamped = Program.ResolveSafeD3D12AdapterIndex(999);
                Assert.Equal(adapters.Count - 1, clamped);

                // Passing valid index must be preserved
                int valid = Program.ResolveSafeD3D12AdapterIndex(0);
                Assert.Equal(0, valid);

                // Passing -1 returns -1 (auto-select)
                int auto = Program.ResolveSafeD3D12AdapterIndex(-1);
                Assert.Equal(-1, auto);
            }
        }
    }

    [Fact]
    public void SafeVulkanDeviceSelection_ClampsOutOfBoundsIndex()
    {
        if (VulkanContext.IsSupported)
        {
            var devices = VulkanContext.EnumerateDevices();
            if (devices.Count > 0)
            {
                // Passing index 999 must clamp to devices.Count - 1 without throwing
                int clamped = Program.ResolveSafeVulkanDeviceIndex(999);
                Assert.Equal(devices.Count - 1, clamped);

                // Passing valid index 0 must return 0
                int valid = Program.ResolveSafeVulkanDeviceIndex(0);
                Assert.Equal(0, valid);

                // Passing negative returns default 0
                int auto = Program.ResolveSafeVulkanDeviceIndex(-1);
                Assert.Equal(0, auto);
            }
        }
    }

    [Fact]
    public void VulkanContext_EnumerateDevices_ReturnsNonEmptyOnSupportedSystem()
    {
        if (VulkanContext.IsSupported)
        {
            var devices = VulkanContext.EnumerateDevices();
            Assert.NotEmpty(devices);
            Assert.True(devices[0].TotalMemoryBytes > 0);
            Assert.False(string.IsNullOrWhiteSpace(devices[0].Name));
        }
    }

    [Fact]
    public void VulkanClusterStageModel_Lifecycle_IntermediateStage()
    {
        if (!VulkanContext.IsSupported) return;

        using var ctx = new VulkanContext(0);
        using var stage = new VulkanClusterStageModel(
            ctx: ctx,
            startLayer: 4,
            layerCount: 8,
            dim: 256,
            headCount: 4,
            headCountKv: 2,
            headDim: 64,
            maxSeqLen: 512,
            isFirstStage: false,
            isLastStage: false);

        Assert.Equal(4, stage.StartLayer);
        Assert.Equal(8, stage.LayerCount);
        Assert.Equal(12, stage.EndLayer);
        Assert.False(stage.IsFirstStage);
        Assert.False(stage.IsLastStage);
        Assert.Contains("Vulkan", stage.BackendName);

        // Verify ForwardStage and ForwardBatchStage fail fast with NotSupportedException until compute dispatch is implemented
        float[] inFloats = new float[256];
        float[] outFloats = new float[256];
        Array.Fill(inFloats, 1.0f);

        Assert.Throws<NotSupportedException>(() =>
            stage.ForwardStage(token: -1, pos: 0, inFloats, outFloats, logits: default, computeLogits: false));

        Assert.Throws<NotSupportedException>(() =>
            stage.ForwardBatchStage(tokens: ReadOnlySpan<int>.Empty, startPos: 0, inFloats, outFloats, logits: default, computeLogits: false));

        // Verify ResetKvCache still operates cleanly on allocated KV buffers
        stage.ResetKvCache();

        // Verify Dispose transitions to ObjectDisposedException
        stage.Dispose();
        Assert.Throws<ObjectDisposedException>(() =>
            stage.ForwardStage(token: -1, pos: 0, inFloats, outFloats, logits: default, computeLogits: false));
    }
}
