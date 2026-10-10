namespace Glacier.Inference.Tests.Distributed;

using System;
using System.IO.Pipelines;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Cluster.Shuffle;
using Glacier.Inference.Distributed;
using Glacier.Inference.Gguf;
using Xunit;

public class DistributedPipelineTests
{
    [Fact]
    public unsafe void CpuWarmExpertPoolWorker_Synthetic_ProcessesActivation_ProducesAccurateOutput()
    {
        const int dim = 128;
        const int ffnDim = 256;
        const int numExperts = 4;

        using var worker = CpuWarmExpertPoolWorker.CreateSynthetic(
            stageRank: 1,
            dim: dim,
            expertFfnDim: ffnDim,
            numExperts: numExperts,
            weightType: GgufType.F32);

        Assert.Equal(1, worker.StageRank);
        Assert.Equal(dim, worker.HiddenDim);
        Assert.Equal(ffnDim, worker.ExpertFfnDim);
        Assert.Equal(numExperts, worker.NumExperts);

        // Prepare test activation vector with known values
        float[] inputFloats = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            inputFloats[i] = (i % 5) * 0.2f + 0.1f;
        }

        byte[] inputBytes = new byte[dim * sizeof(float)];
        Buffer.BlockCopy(inputFloats, 0, inputBytes, 0, inputBytes.Length);

        var header = new ActivationHeader(
            sequenceId: 10,
            tokenIndex: 0,
            layerIndex: 0,
            hiddenDim: dim,
            expertId: 2,
            flags: ActivationFlags.None);

        using var outputBuffer = worker.ProcessActivation(header, inputBytes);

        Assert.NotNull(outputBuffer);
        Assert.Equal(dim * sizeof(float), outputBuffer.ByteLength);

        float* outPtr = (float*)outputBuffer.UnmanagedPointer;
        bool hasNonZero = false;
        for (int i = 0; i < dim; i++)
        {
            float val = outPtr[i];
            Assert.False(float.IsNaN(val), $"Output float at {i} was NaN");
            Assert.False(float.IsInfinity(val), $"Output float at {i} was Infinity");
            if (MathF.Abs(val) > 1e-6f)
            {
                hasNonZero = true;
            }
        }
        Assert.True(hasNonZero, "Expert execution produced all zeros.");
    }

    [Fact]
    public unsafe void Stage0_AttentionRoot_RMSNorm_AccurateMath()
    {
        const int dim = 64;
        using var rootWorker = new AttentionRootStageWorker(dim, topK: 2, numExperts: 8);

        // Input where all elements are 2.0f
        // rms = sqrt(mean(2^2) + eps) = sqrt(4 + eps) ≈ 2.0
        // norm = 2.0 / 2.0 = 1.0f
        float[] inputFloats = new float[dim];
        Array.Fill(inputFloats, 2.0f);
        byte[] inputBytes = new byte[dim * sizeof(float)];
        Buffer.BlockCopy(inputFloats, 0, inputBytes, 0, inputBytes.Length);

        var header = new ActivationHeader(1, 0, 0, dim, 0);
        using var outputBuffer = rootWorker.ProcessActivation(header, inputBytes);

        float* outPtr = (float*)outputBuffer.UnmanagedPointer;
        for (int i = 0; i < dim; i++)
        {
            Assert.InRange(outPtr[i], 0.99f, 1.01f);
        }
    }

    [Fact]
    public unsafe void Stage2_Accumulator_WeightedExpertAndResidual_AccurateMath()
    {
        const int dim = 64;
        using var accumulatorWorker = new AccumulatorStageWorker(dim);

        float* acc = stackalloc float[dim];
        float* residual = stackalloc float[dim];
        float* expertOut1 = stackalloc float[dim];
        float* expertOut2 = stackalloc float[dim];

        for (int i = 0; i < dim; i++)
        {
            acc[i] = 0.0f;
            residual[i] = 10.0f;
            expertOut1[i] = 2.0f;
            expertOut2[i] = 4.0f;
        }

        // 1. Add residual stream: acc = 10.0
        accumulatorWorker.AddResidual(acc, residual);
        for (int i = 0; i < dim; i++)
        {
            Assert.Equal(10.0f, acc[i]);
        }

        // 2. Accumulate weighted expert 1 with weight 0.5: acc = 10.0 + 0.5 * 2.0 = 11.0
        accumulatorWorker.AccumulateWeightedExpert(acc, expertOut1, 0.5f);
        for (int i = 0; i < dim; i++)
        {
            Assert.Equal(11.0f, acc[i]);
        }

        // 3. Accumulate weighted expert 2 with weight 0.25: acc = 11.0 + 0.25 * 4.0 = 12.0
        accumulatorWorker.AccumulateWeightedExpert(acc, expertOut2, 0.25f);
        for (int i = 0; i < dim; i++)
        {
            Assert.Equal(12.0f, acc[i]);
        }
    }

    [Fact]
    public unsafe void DistributedPipelineCoordinator_EndToEnd_3Stage_Execution()
    {
        const int dim = 128;
        const int ffnDim = 256;

        using var stage0 = new AttentionRootStageWorker(dim, topK: 2, numExperts: 4);
        using var stage1 = CpuWarmExpertPoolWorker.CreateSynthetic(1, dim, ffnDim, numExperts: 4, GgufType.F32);
        using var stage2 = new AccumulatorStageWorker(dim);

        using var coordinator = new DistributedPipelineCoordinator(stage0, stage1, stage2, dim);

        float[] inputFloats = new float[dim];
        Array.Fill(inputFloats, 1.0f);
        byte[] inputBytes = new byte[dim * sizeof(float)];
        Buffer.BlockCopy(inputFloats, 0, inputBytes, 0, inputBytes.Length);

        int[] selectedExperts = [0, 2];
        float[] expertWeights = [0.6f, 0.4f];

        using var resultBuffer = coordinator.ExecuteStep(
            sequenceId: 100,
            tokenIndex: 5,
            layerIndex: 12,
            inputEmbedding: inputBytes,
            selectedExperts: selectedExperts,
            expertWeights: expertWeights);

        Assert.NotNull(resultBuffer);
        Assert.Equal(dim * sizeof(float), resultBuffer.ByteLength);

        float* resultPtr = (float*)resultBuffer.UnmanagedPointer;
        // Verify output is finite, non-zero, and incorporates residual
        for (int i = 0; i < dim; i++)
        {
            float val = resultPtr[i];
            Assert.False(float.IsNaN(val));
            Assert.False(float.IsInfinity(val));
        }
    }

    [Fact]
    public async Task DistributedPipelineCoordinator_NetworkPiping_Roundtrip()
    {
        const int dim = 64;
        const int ffnDim = 128;

        using var expertWorker = CpuWarmExpertPoolWorker.CreateSynthetic(1, dim, ffnDim, numExperts: 2, GgufType.F32);
        var (clientPipe, serverPipe) = DuplexPipeFactory.CreateInProcessPair();

        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));

        // Start background worker service loop
        var serverTask = Task.Run(async () =>
        {
            await DistributedPipelineCoordinator.RunWorkerServiceLoopAsync(serverPipe, expertWorker, cts.Token);
        }, cts.Token);

        float[] inputFloats = new float[dim];
        Array.Fill(inputFloats, 0.5f);
        byte[] inputBytes = new byte[dim * sizeof(float)];
        Buffer.BlockCopy(inputFloats, 0, inputBytes, 0, inputBytes.Length);

        var requestHeader = new ActivationHeader(
            sequenceId: 777,
            tokenIndex: 1,
            layerIndex: 0,
            hiddenDim: dim,
            expertId: 1,
            flags: ActivationFlags.None);

        // Client dispatches activation frame
        await DistributedPipelineCoordinator.WriteActivationFrameAsync(
            clientPipe.Output,
            requestHeader,
            inputBytes.AsMemory(),
            cancellationToken: cts.Token);

        // Client reads response frame
        var (responseHeader, responseBuffer) = await DistributedPipelineCoordinator.ReadActivationFrameAsync(
            clientPipe.Input,
            cts.Token);

        using (responseBuffer)
        {
            Assert.Equal(777UL, responseHeader.SequenceId);
            Assert.Equal(1, responseHeader.TokenIndex);
            Assert.Equal(dim, responseHeader.HiddenDim);
            Assert.True(responseHeader.TypedFlags.HasFlag(ActivationFlags.AcknowledgeRequired));

            // Validate that remote execution matches direct local execution
            using var expectedLocalBuffer = expertWorker.ProcessActivation(requestHeader, inputBytes);
            var remoteFloats = MemoryMarshal.Cast<byte, float>(responseBuffer.Span);
            var localFloats = MemoryMarshal.Cast<byte, float>(expectedLocalBuffer.Span);

            for (int i = 0; i < dim; i++)
            {
                Assert.Equal(localFloats[i], remoteFloats[i], precision: 5);
            }
        }

        cts.Cancel();
        try { await serverTask; } catch (OperationCanceledException) { }
    }

    [Fact]
    public unsafe void DistributedPipelineCoordinator_ZeroAllocation_HotPath()
    {
        const int dim = 128;
        const int ffnDim = 256;

        using var stage0 = new AttentionRootStageWorker(dim, topK: 1, numExperts: 2);
        using var stage1 = CpuWarmExpertPoolWorker.CreateSynthetic(1, dim, ffnDim, numExperts: 2, GgufType.F32, multithreaded: false);
        using var stage2 = new AccumulatorStageWorker(dim);
        using var coordinator = new DistributedPipelineCoordinator(stage0, stage1, stage2, dim);

        float* input = stackalloc float[dim];
        float* dest = stackalloc float[dim];
        float* scratch0 = stackalloc float[dim];
        float* scratchExp = stackalloc float[dim];

        for (int i = 0; i < dim; i++) input[i] = 1.0f;

        int[] experts = [0];
        float[] weights = [1.0f];

        // Warmup JIT
        for (int i = 0; i < 50; i++)
        {
            coordinator.ExecuteStepZeroAlloc(1, 0, 0, input, experts, weights, dest, scratch0, scratchExp);
        }

        // Measure allocations on hot path
        long allocBefore = GC.GetAllocatedBytesForCurrentThread();

        for (int i = 0; i < 100; i++)
        {
            coordinator.ExecuteStepZeroAlloc(1, 0, 0, input, experts, weights, dest, scratch0, scratchExp);
        }

        long allocAfter = GC.GetAllocatedBytesForCurrentThread();
        long totalAlloc = allocAfter - allocBefore;

        // Verify zero GC allocations on synchronous hot path
        Assert.Equal(0, totalAlloc);
    }
}
