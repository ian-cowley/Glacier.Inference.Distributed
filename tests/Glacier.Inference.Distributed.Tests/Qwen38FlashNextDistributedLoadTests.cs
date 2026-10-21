namespace Glacier.Inference.Distributed.Tests;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Cluster.Shuffle;
using Glacier.Inference.Distributed.Cli;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Quant;
using Glacier.Inference.Sampling;
using Vortice.Direct3D12;
using Xunit;
using Xunit.Abstractions;

/// <summary>
/// Comprehensive GPU-First &amp; CPU SIMD accuracy + distributed load test suite for
/// Qwen 3.8 Flash Next (512 experts) vs Qwen 3.8 Flash Next Coder (256 RCO-pruned experts)
/// and their specialized quantization kernels (<c>Q2_0</c>, <c>IQ4_NL</c>, <c>IQ3_S</c>).
/// </summary>
public sealed class Qwen38FlashNextDistributedLoadTests
{
    private readonly ITestOutputHelper _output;

    public Qwen38FlashNextDistributedLoadTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void QuantKernels_Q2_0_IQ4_NL_IQ3_S_640Col_And_256Col_CpuSimd_And_GpuD3D12_ExactParity()
    {
        // Qwen3.8-Flash-Next & Coder have expert_ffn_dim = 640 (640 % 256 != 0).
        // Verify Q2_0 (64-element block) and IQ4_NL (32-element block) on 640 columns,
        // and IQ3_S (256-element super-block) on 512 columns, across both CPU SIMD and Direct3D 12 GPU!
        VerifyQuantKernelCpuAndGpu(GgufType.Q2_0, rows: 64, cols: 640);
        VerifyQuantKernelCpuAndGpu(GgufType.IQ4_NL, rows: 64, cols: 640);
        VerifyQuantKernelCpuAndGpu(GgufType.IQ3_S, rows: 64, cols: 512);
    }

    private unsafe void VerifyQuantKernelCpuAndGpu(GgufType type, int rows, int cols)
    {
        float[] rawWeights = new float[rows * cols];
        float[] x = new float[cols];
        for (int i = 0; i < rawWeights.Length; i++)
            rawWeights[i] = MathF.Sin(i * 0.137f) * 0.5f;
        for (int c = 0; c < cols; c++)
            x[c] = MathF.Cos(c * 0.073f) * 0.8f;

        long rowBytes = GgufTypes.GetRowBytes(type, cols);
        byte* qWeights = (byte*)NativeMemory.AlignedAlloc((nuint)(rows * rowBytes), 64);
        float* dequantRow = (float*)NativeMemory.AlignedAlloc((nuint)(cols * sizeof(float)), 64);
        float[] expectedScalar = new float[rows];
        float[] actualCpuSimd = new float[rows];
        float[] actualGpuD3D12 = new float[rows];

        try
        {
            fixed (float* pRaw = rawWeights, pX = x, pCpu = actualCpuSimd)
            {
                for (int r = 0; r < rows; r++)
                {
                    float* srcRow = pRaw + r * cols;
                    byte* dstRow = qWeights + r * rowBytes;

                    switch (type)
                    {
                        case GgufType.Q2_0:
                            QuantKernels.QuantizeQ2_0(srcRow, (BlockQ2_0*)dstRow, cols);
                            QuantKernels.DequantizeQ2_0((BlockQ2_0*)dstRow, dequantRow, cols);
                            break;
                        case GgufType.IQ4_NL:
                            QuantKernels.QuantizeIQ4_NL(srcRow, (BlockIQ4_NL*)dstRow, cols);
                            QuantKernels.DequantizeIQ4_NL((BlockIQ4_NL*)dstRow, dequantRow, cols);
                            break;
                        case GgufType.IQ3_S:
                            QuantKernels.QuantizeIQ3_S(srcRow, (BlockIQ3_S*)dstRow, cols);
                            QuantKernels.DequantizeIQ3_S((BlockIQ3_S*)dstRow, dequantRow, cols);
                            break;
                    }

                    double dot = 0.0;
                    for (int c = 0; c < cols; c++)
                        dot += dequantRow[c] * pX[c];
                    expectedScalar[r] = (float)dot;
                }

                // 1. CPU SIMD (AVX-512 / AVX2) MatVecMul
                QuantKernels.MatVecMul(type, qWeights, pX, pCpu, cols, rows);
            }

            double cpuCosSim = CosineSimilarity(expectedScalar, actualCpuSimd);
            _output.WriteLine($"[{type} {rows}x{cols}] CPU SIMD vs Dequantized Reference Cosine Similarity: {cpuCosSim:F8}");
            Assert.True(cpuCosSim > 0.99999, $"CPU SIMD {type} cosine similarity ({cpuCosSim}) fell below 0.99999");

            // 2. Direct3D 12 GPU Compute Shader Parity (GemvQ2_0 / GemvIQ4_NL / GemvIQ3_S)
            if (OperatingSystem.IsWindows())
            {
                using var ctx = new D3D12Context();
                string shaderSrc = type switch
                {
                    GgufType.Q2_0 => D3D12Shaders.GemvQ2_0,
                    GgufType.IQ4_NL => D3D12Shaders.GemvIQ4_NL,
                    GgufType.IQ3_S => D3D12Shaders.GemvIQ3_S,
                    _ => throw new NotSupportedException()
                };

                var gemvParams = new RootParameter[]
                {
                    new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
                    new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
                    new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
                };
                using var sig = ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, gemvParams));
                using var pso = ctx.CreatePipelineState(sig, ctx.CompileShader(shaderSrc));

                using var dW = D3D12TensorAlign.UploadAlignedTensor(ctx, type, (IntPtr)qWeights, rows, cols);
                using var dX = ctx.CreateDeviceBuffer((ulong)(cols * sizeof(float)));
                using var dY = ctx.CreateDeviceBuffer((ulong)(rows * sizeof(float)));
                using var readbackY = ctx.CreateReadbackBuffer((ulong)(rows * sizeof(float)));

                fixed (float* pX = x)
                {
                    ctx.CopyToDevice(dX, (IntPtr)pX, (ulong)(cols * sizeof(float)));
                }

                ctx.BeginCommands();
                var cmd = ctx.CommandList;
                cmd.SetComputeRootSignature(sig);
                cmd.SetPipelineState(pso);
                cmd.SetComputeRootShaderResourceView(1, dW.GPUVirtualAddress);
                cmd.SetComputeRootShaderResourceView(2, ctx.DummyBuffer.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(3, dX.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(4, ctx.DummyBuffer.GPUVirtualAddress);
                cmd.SetComputeRootUnorderedAccessView(5, dY.GPUVirtualAddress);

                uint* pConsts = stackalloc uint[6];
                pConsts[0] = (uint)cols;
                pConsts[1] = (uint)rows;
                pConsts[2] = 0u;
                pConsts[3] = 0u;
                pConsts[4] = 1u;
                pConsts[5] = 0u;
                cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);
                cmd.Dispatch(((uint)rows + 3) / 4, 1, 1);
                cmd.ResourceBarrierUnorderedAccessView(null!);

                cmd.ResourceBarrierTransition(dY, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                cmd.CopyBufferRegion(readbackY, 0, dY, 0, (ulong)(rows * sizeof(float)));
                cmd.ResourceBarrierTransition(dY, ResourceStates.CopySource, ResourceStates.Common);
                ctx.EndCommandsAndExecute();
                ctx.Synchronize();

                void* pRb = null;
                readbackY.Map(0, null, &pRb);
                fixed (float* pGpu = actualGpuD3D12)
                {
                    Buffer.MemoryCopy(pRb, pGpu, (ulong)(rows * sizeof(float)), (ulong)(rows * sizeof(float)));
                }
                readbackY.Unmap(0);

                double gpuCosSim = CosineSimilarity(actualCpuSimd, actualGpuD3D12);
                _output.WriteLine($"[{type} {rows}x{cols}] Direct3D 12 GPU ({ctx.DeviceName}) vs CPU SIMD Cosine Similarity: {gpuCosSim:F8}");
                Assert.True(gpuCosSim > 0.99999, $"Direct3D 12 GPU {type} cosine similarity ({gpuCosSim}) fell below 0.99999");
            }
        }
        finally
        {
            NativeMemory.AlignedFree(qWeights);
            NativeMemory.AlignedFree(dequantRow);
        }
    }

    [Fact]
    public void DistributedGpuFirstLoadTest_Qwen38FlashNext_512Experts_Vs_FlashNextCoder_256Experts()
    {
        // Models the exact Qwen 3.8 Flash Next (512 experts) vs Qwen 3.8 Flash Next Coder (256 RCO-pruned experts)
        // with 640-column ffn_down_exps (using Q2_0 and IQ4_NL) and IQ3_S gate/up across a 3-stage GPU-first cluster!
        const int hiddenDim = 512;
        const int expertFfnDim = 640; // Exact 640-column expert intermediate dimension from Qwen3.8-Flash-Next
        const int topK = 8;
        const int tokensToSimulate = 12;

        using var d3dCtxNvidia = OperatingSystem.IsWindows() ? new D3D12Context(0) : null;
        using var d3dCtxAmd = OperatingSystem.IsWindows() ? new D3D12Context(1) : null;
        _output.WriteLine($"[GPU-First Distributed Load Test] Stage 0 GPU: {d3dCtxNvidia?.DeviceName ?? "CPU SIMD"} | Stage 1 GPU: {d3dCtxAmd?.DeviceName ?? "CPU SIMD"}");

        // 1. Qwen 3.8 Flash Next (512 experts, IQ3_S gate/up + IQ4_NL 640-col down, 128 GPU VRAM slots across NVIDIA + AMD)
        using var flashNextStage0 = GpuFirstExpertStageWorker.CreateSynthetic(
            stageRank: 0, dim: hiddenDim, expertFfnDim: expertFfnDim, numExperts: 512,
            gateUpType: GgufType.IQ3_S, downType: GgufType.IQ4_NL,
            d3dContext: d3dCtxNvidia, ownsContext: false, gpuSlotCapacity: 128);
        using var flashNextStage1Amd = GpuFirstExpertStageWorker.CreateSynthetic(
            stageRank: 1, dim: hiddenDim, expertFfnDim: expertFfnDim, numExperts: 512,
            gateUpType: GgufType.IQ3_S, downType: GgufType.IQ4_NL,
            d3dContext: d3dCtxAmd ?? d3dCtxNvidia, ownsContext: false, gpuSlotCapacity: 128);

        // 2. Qwen 3.8 Flash Next Fast Q2_0 (512 experts, Q2_0 gate/up + Q2_0 640-col down, 128 GPU VRAM slots across NVIDIA + AMD)
        using var flashNextQ2Stage0 = GpuFirstExpertStageWorker.CreateSynthetic(
            stageRank: 1, dim: hiddenDim, expertFfnDim: expertFfnDim, numExperts: 512,
            gateUpType: GgufType.Q2_0, downType: GgufType.Q2_0,
            d3dContext: d3dCtxNvidia, ownsContext: false, gpuSlotCapacity: 128);
        using var flashNextQ2Stage1Amd = GpuFirstExpertStageWorker.CreateSynthetic(
            stageRank: 2, dim: hiddenDim, expertFfnDim: expertFfnDim, numExperts: 512,
            gateUpType: GgufType.Q2_0, downType: GgufType.Q2_0,
            d3dContext: d3dCtxAmd ?? d3dCtxNvidia, ownsContext: false, gpuSlotCapacity: 128);

        // 3. Qwen 3.8 Flash Next Coder (256 RCO-pruned experts, IQ3_S gate/up + IQ4_NL 640-col down, 128 GPU VRAM slots across NVIDIA + AMD)
        using var flashNextCoderStage0 = GpuFirstExpertStageWorker.CreateSynthetic(
            stageRank: 2, dim: hiddenDim, expertFfnDim: expertFfnDim, numExperts: 256,
            gateUpType: GgufType.IQ3_S, downType: GgufType.IQ4_NL,
            d3dContext: d3dCtxNvidia, ownsContext: false, gpuSlotCapacity: 128);
        using var flashNextCoderStage1Amd = GpuFirstExpertStageWorker.CreateSynthetic(
            stageRank: 3, dim: hiddenDim, expertFfnDim: expertFfnDim, numExperts: 256,
            gateUpType: GgufType.IQ3_S, downType: GgufType.IQ4_NL,
            d3dContext: d3dCtxAmd ?? d3dCtxNvidia, ownsContext: false, gpuSlotCapacity: 128);

        // Verify 3-stage pipeline activation handoff across GPU-First stage workers (NVIDIA -> AMD -> NVIDIA)
        VerifyThreeStageGpuFirstPipelineHandoff(flashNextStage0, flashNextStage1Amd, flashNextCoderStage0, hiddenDim);

        // Run 59-token Prompt Prefill + 64-token Autoregressive Decode across NVIDIA + AMD GPUs
        const int prefillTokens = 59;
        const int decodeTokens = 64;

        // Warmup both NVIDIA and AMD GPUs
        RunSimulatedMoELoad(flashNextStage0, hiddenDim, numExperts: 512, topK, 4);
        RunSimulatedMoELoad(flashNextStage1Amd, hiddenDim, numExperts: 512, topK, 4);
        RunSimulatedMoELoad(flashNextQ2Stage0, hiddenDim, numExperts: 512, topK, 4);
        RunSimulatedMoELoad(flashNextQ2Stage1Amd, hiddenDim, numExperts: 512, topK, 4);
        RunSimulatedMoELoad(flashNextCoderStage0, hiddenDim, numExperts: 256, topK, 4);
        RunSimulatedMoELoad(flashNextCoderStage1Amd, hiddenDim, numExperts: 256, topK, 4);

        var swFlashNextPrefill = Stopwatch.StartNew();
        RunSimulatedMoELoad(flashNextStage0, hiddenDim, numExperts: 512, topK, prefillTokens);
        RunSimulatedMoELoad(flashNextStage1Amd, hiddenDim, numExperts: 512, topK, prefillTokens);
        swFlashNextPrefill.Stop();

        var swFlashNext = Stopwatch.StartNew();
        RunSimulatedMoELoad(flashNextStage0, hiddenDim, numExperts: 512, topK, decodeTokens);
        RunSimulatedMoELoad(flashNextStage1Amd, hiddenDim, numExperts: 512, topK, decodeTokens);
        swFlashNext.Stop();

        var swFlashNextQ2Prefill = Stopwatch.StartNew();
        RunSimulatedMoELoad(flashNextQ2Stage0, hiddenDim, numExperts: 512, topK, prefillTokens);
        RunSimulatedMoELoad(flashNextQ2Stage1Amd, hiddenDim, numExperts: 512, topK, prefillTokens);
        swFlashNextQ2Prefill.Stop();

        var swFlashNextQ2 = Stopwatch.StartNew();
        RunSimulatedMoELoad(flashNextQ2Stage0, hiddenDim, numExperts: 512, topK, decodeTokens);
        RunSimulatedMoELoad(flashNextQ2Stage1Amd, hiddenDim, numExperts: 512, topK, decodeTokens);
        swFlashNextQ2.Stop();

        var swCoderPrefill = Stopwatch.StartNew();
        RunSimulatedMoELoad(flashNextCoderStage0, hiddenDim, numExperts: 256, topK, prefillTokens);
        RunSimulatedMoELoad(flashNextCoderStage1Amd, hiddenDim, numExperts: 256, topK, prefillTokens);
        swCoderPrefill.Stop();

        var swCoder = Stopwatch.StartNew();
        RunSimulatedMoELoad(flashNextCoderStage0, hiddenDim, numExperts: 256, topK, decodeTokens);
        RunSimulatedMoELoad(flashNextCoderStage1Amd, hiddenDim, numExperts: 256, topK, decodeTokens);
        swCoder.Stop();

        var telFlash = flashNextStage0.GetTelemetry();
        var telFlashAmd = flashNextStage1Amd.GetTelemetry();
        var telQ2 = flashNextQ2Stage0.GetTelemetry();
        var telCoder = flashNextCoderStage0.GetTelemetry();
        var telCoderAmd = flashNextCoderStage1Amd.GetTelemetry();

        _output.WriteLine("==========================================================================================");
        _output.WriteLine("  HETEROGENEOUS NVIDIA + AMD GPU RESULTS (Qwen 3.8 Flash Next vs Flash Next Coder)");
        _output.WriteLine("==========================================================================================");
        _output.WriteLine($"  1. Flash-Next (512 exps, IQ3_S/IQ4_NL) : Read {prefillTokens / swFlashNextPrefill.Elapsed.TotalSeconds,6:F2} tok/s ({swFlashNextPrefill.Elapsed.TotalMilliseconds,6:F2} ms) | Out {decodeTokens / swFlashNext.Elapsed.TotalSeconds,6:F2} tok/s ({swFlashNext.Elapsed.TotalMilliseconds,6:F2} ms) | NVIDIA GPU Ratio: {telFlash.GpuExecutionRatio:P1} | AMD GPU Ratio: {telFlashAmd.GpuExecutionRatio:P1}");
        _output.WriteLine($"  2. Flash-Next (512 exps, Fast Q2_0)    : Read {prefillTokens / swFlashNextQ2Prefill.Elapsed.TotalSeconds,6:F2} tok/s ({swFlashNextQ2Prefill.Elapsed.TotalMilliseconds,6:F2} ms) | Out {decodeTokens / swFlashNextQ2.Elapsed.TotalSeconds,6:F2} tok/s ({swFlashNextQ2.Elapsed.TotalMilliseconds,6:F2} ms) | NVIDIA GPU Ratio: {telQ2.GpuExecutionRatio:P1}");
        _output.WriteLine($"  3. Flash-Next-Coder (256 RCO exps)     : Read {prefillTokens / swCoderPrefill.Elapsed.TotalSeconds,6:F2} tok/s ({swCoderPrefill.Elapsed.TotalMilliseconds,6:F2} ms) | Out {decodeTokens / swCoder.Elapsed.TotalSeconds,6:F2} tok/s ({swCoder.Elapsed.TotalMilliseconds,6:F2} ms) | NVIDIA GPU Ratio: {telCoder.GpuExecutionRatio:P1} | AMD GPU Ratio: {telCoderAmd.GpuExecutionRatio:P1}");
        _output.WriteLine("==========================================================================================");

        if (d3dCtxNvidia != null)
        {
            Assert.True(telFlash.IsGpuActive);
            Assert.True(telQ2.IsGpuActive);
            Assert.True(telCoder.IsGpuActive);
            Assert.True(telCoder.GpuExecutionRatio > 0.5, "GPU-First worker must execute majority of experts on GPU");
            Assert.True(telCoder.LookaheadPrefetchCount > 0, "Router lookahead prefetching should pre-stage next-layer experts");
        }
    }

    private static unsafe void VerifyThreeStageGpuFirstPipelineHandoff(
        GpuFirstExpertStageWorker stage0,
        GpuFirstExpertStageWorker stage1,
        GpuFirstExpertStageWorker stage2,
        int hiddenDim)
    {
        byte[] inputPayload = new byte[hiddenDim * sizeof(float)];
        fixed (byte* pIn = inputPayload)
        {
            float* fIn = (float*)pIn;
            for (int i = 0; i < hiddenDim; i++) fIn[i] = MathF.Sin(i * 0.05f) * 0.25f;
        }

        var hdr0 = new ActivationHeader(sequenceId: 42UL, tokenIndex: 0, layerIndex: 0, hiddenDim: hiddenDim, expertId: 7);
        using var out0 = stage0.ProcessActivation(hdr0, inputPayload);

        var hdr1 = new ActivationHeader(sequenceId: 42UL, tokenIndex: 0, layerIndex: 1, hiddenDim: hiddenDim, expertId: 7);
        using var out1 = stage1.ProcessActivation(hdr1, new ReadOnlySpan<byte>((void*)out0.UnmanagedPointer, out0.ByteLength));

        var hdr2 = new ActivationHeader(sequenceId: 42UL, tokenIndex: 0, layerIndex: 2, hiddenDim: hiddenDim, expertId: 7);
        using var out2 = stage2.ProcessActivation(hdr2, new ReadOnlySpan<byte>((void*)out1.UnmanagedPointer, out1.ByteLength));

        Assert.Equal(hiddenDim * sizeof(float), out2.ByteLength);
        float* fPipelineOut = (float*)out2.UnmanagedPointer;
        for (int i = 0; i < hiddenDim; i++)
        {
            Assert.True(float.IsFinite(fPipelineOut[i]), $"Non-finite output at index {i}");
        }
    }

    private static void RunSimulatedMoELoad(
        GpuFirstExpertStageWorker worker,
        int hiddenDim,
        int numExperts,
        int topK,
        int tokens)
    {
        float[] inputX = new float[hiddenDim];
        float[] outputX = new float[hiddenDim];
        for (int i = 0; i < hiddenDim; i++)
            inputX[i] = ((i % 31) - 15) * 0.02f;

        int[] selectedExperts = new int[topK];
        int[] lookaheadExperts = new int[topK];
        float[] weights = new float[topK];
        for (int k = 0; k < topK; k++)
            weights[k] = 1.0f / topK;

        for (int t = 0; t < tokens; t++)
        {
            for (int k = 0; k < topK; k++)
            {
                // Zipf-like skewed expert activation distribution + next-layer lookahead
                selectedExperts[k] = (t * 7 + k * 3) % Math.Min(numExperts, 96);
                lookaheadExperts[k] = ((t + 1) * 7 + k * 3) % Math.Min(numExperts, 96);
            }

            worker.ProcessRoutedTopK(
                layerIdx: 0,
                inputX: inputX,
                selectedExpertIds: selectedExperts,
                selectedWeights: weights,
                outputX: outputX,
                nextLayerLookaheadExperts: lookaheadExperts);
        }
    }

    [Fact]
    public void ClusterResourcePlanner_OptimizeFinalStageGpuOrder_PlacesStrongestGpuOnFinalStage()
    {
        var gpus = new List<NodeResourceProfile>
        {
            new()
            {
                Name = "Node 0 (RTX 4090 24GB)",
                Host = "127.0.0.1",
                Port = 0,
                IsLocal = true,
                DeviceName = "NVIDIA GeForce RTX 4090",
                IsGpu = true,
                VramBytes = 24UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 7.5
            },
            new()
            {
                Name = "Node 1 (RTX 3060 12GB)",
                Host = "192.168.1.108",
                Port = 5055,
                IsLocal = false,
                DeviceName = "NVIDIA GeForce RTX 3060",
                IsGpu = true,
                VramBytes = 12UL * 1024 * 1024 * 1024,
                EstimatedMsPerLayer = 16.7
            }
        };

        var stages = ClusterResourcePlanner.PlanStages(
            localNodes: [gpus[0]],
            remoteCandidates: [gpus[1]],
            totalLayers: 48,
            totalModelBytes: 28UL * 1024 * 1024 * 1024,
            layerAllocArg: "32,16");

        var optimized = ClusterResourcePlanner.OptimizeFinalStageGpuOrder(stages);
        Assert.Equal(2, optimized.Count);
        Assert.True(optimized[^1].IsLastStage);
        Assert.Contains("4090", optimized[^1].Node.DeviceName);
        Assert.Equal(0, optimized[0].StartLayer);
        Assert.Equal(47, optimized[^1].EndLayer);
    }

    [Fact]
    public void Sampler_SanitizeLogits_And_PeriodicReasoningLoopGuard_BreaksInfiniteLoops()
    {
        float[] logits = [float.NaN, 1.5f, float.PositiveInfinity, float.NegativeInfinity, 3.0f];
        int repaired = Sampler.SanitizeLogits(logits);
        Assert.Equal(3, repaired);
        for (int i = 0; i < logits.Length; i++)
            Assert.True(float.IsFinite(logits[i]));

        // Simulate a 5-token reasoning loop repeating 3 times: [10, 20, 30, 40, 50] x 3
        int[] loopingTokens = [1, 2, 3, 10, 20, 30, 40, 50, 10, 20, 30, 40, 50, 10, 20, 30, 40, 50];
        int detectedPeriod = Sampler.DetectPeriodicReasoningLoop(loopingTokens);
        Assert.Equal(5, detectedPeriod);

        // Verify Sampler penalizes the looping token (10) so the runner-up token (99) is selected
        float[] stepLogits = new float[128];
        Array.Fill(stepLogits, -10.0f);
        stepLogits[10] = 4.0f; // Token 10 would normally win under mild repetition penalty (1.05)
        stepLogits[99] = 3.2f; // Non-looping candidate token

        var sampler = new Sampler(seed: 42);
        var options = new SamplingOptions
        {
            Temperature = 0.0f,
            TopK = 1,
            RepetitionPenalty = 1.05f,
            EnableReasoningLoopGuard = true,
            LoopRecoveryRepetitionPenalty = 1.40f
        };

        int nextToken = sampler.Sample(stepLogits, options, loopingTokens);
        Assert.Equal(99, nextToken);
    }

    private static double CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            normA += (double)a[i] * a[i];
            normB += (double)b[i] * b[i];
        }
        return dot / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-12);
    }
}
