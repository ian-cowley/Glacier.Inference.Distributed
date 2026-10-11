namespace Glacier.Inference.Tests.Distributed;

using System;
using System.Buffers.Binary;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Inference.Distributed;
using Xunit;

/// <summary>
/// Empirical Challenger Adversarial Stress Suite for Milestone 3:
/// Dynamic Multi-Precision Wire Protocol and SIMD Quantization Kernels.
/// 
/// Stress Dimensions:
/// 1. Numeric Edge Cases: All zero tokens, all NaN/Inf tokens, subnormals, extreme outliers up to 1e6.
/// 2. Boundary Loop Safety: Non-standard dimensions (Dim 1, 2, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 4096, 5120, 8192).
/// 3. Accuracy Metrics: Cosine similarity > 0.9995 across Gaussian, Uniform, and Sparse Outlier distributions; FP16 relative error < 10^-3.
/// 4. DynamicPrecisionSwitcher: Oscillating RTTs, rapid batch size transitions, deep context boundary at 16384, and CLI overrides.
/// 5. E2E Socket Loopback: Dynamic precision switching on every consecutive frame without buffer misalignment or corruption.
/// </summary>
public class Milestone3AdversarialStressTests
{
    private const int Dim5120 = 5120;

    // =========================================================================
    // SECTION 1: NUMERIC EDGE CASES
    // =========================================================================

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(255)]
    [InlineData(4096)]
    [InlineData(5120)]
    [InlineData(8192)]
    public void NumericEdgeCases_AllZeroTokens_AllPrecisionModes_RoundtripsSafely(int dim)
    {
        float[] zeros = new float[dim];
        WirePrecisionMode[] modes = [WirePrecisionMode.FP32, WirePrecisionMode.FP16, WirePrecisionMode.INT8_Symmetric, WirePrecisionMode.FP8_E4M3];

        foreach (var mode in modes)
        {
            int byteCount = QuantizationKernels.GetPackedTokenBytes(mode, dim);
            byte[] wireBuffer = new byte[byteCount];

            float scale = QuantizationKernels.QuantizeToken(zeros, wireBuffer, mode);

            Assert.False(float.IsNaN(scale), $"Scale is NaN for mode {mode}, dim {dim}");
            Assert.False(float.IsInfinity(scale), $"Scale is Infinity for mode {mode}, dim {dim}");
            Assert.True(scale > 0.0f, $"Scale must be positive for mode {mode}, dim {dim}");

            float[] restored = new float[dim];
            QuantizationKernels.DequantizeToken(wireBuffer, restored, mode);

            for (int i = 0; i < dim; i++)
            {
                Assert.Equal(0.0f, restored[i]);
            }

            double sim = QuantizationKernels.ComputeCosineSimilarity(zeros, restored);
            Assert.Equal(1.0, sim);
        }
    }

    [Theory]
    [InlineData(WirePrecisionMode.INT8_Symmetric)]
    [InlineData(WirePrecisionMode.FP8_E4M3)]
    public void NumericEdgeCases_AllNaNTokens_HandledGracefullyWithoutCrashing(WirePrecisionMode mode)
    {
        float[] nans = new float[Dim5120];
        Array.Fill(nans, float.NaN);

        int byteCount = QuantizationKernels.GetPackedTokenBytes(mode, Dim5120);
        byte[] wireBuffer = new byte[byteCount];

        float scale = QuantizationKernels.QuantizeToken(nans, wireBuffer, mode);

        Assert.False(float.IsNaN(scale), $"Returned scale must not be NaN for mode {mode}");
        Assert.Equal(1.0f, scale);

        float[] restored = new float[Dim5120];
        QuantizationKernels.DequantizeToken(wireBuffer, restored, mode);

        // NaN tokens must safely zero out rather than propagating undefined corrupt floats
        for (int i = 0; i < Dim5120; i++)
        {
            Assert.Equal(0.0f, restored[i]);
        }
    }

    [Theory]
    [InlineData(WirePrecisionMode.INT8_Symmetric)]
    [InlineData(WirePrecisionMode.FP8_E4M3)]
    public void NumericEdgeCases_AllInfinityTokens_HandledGracefullyWithoutCrashing(WirePrecisionMode mode)
    {
        float[] infs = new float[Dim5120];
        for (int i = 0; i < Dim5120; i++)
        {
            infs[i] = (i % 2 == 0) ? float.PositiveInfinity : float.NegativeInfinity;
        }

        int byteCount = QuantizationKernels.GetPackedTokenBytes(mode, Dim5120);
        byte[] wireBuffer = new byte[byteCount];

        float scale = QuantizationKernels.QuantizeToken(infs, wireBuffer, mode);

        Assert.False(float.IsNaN(scale), $"Returned scale must not be NaN for mode {mode}");
        Assert.False(float.IsInfinity(scale), $"Returned scale must not be Infinity for mode {mode}");
        Assert.Equal(1.0f, scale);

        float[] restored = new float[Dim5120];
        QuantizationKernels.DequantizeToken(wireBuffer, restored, mode);

        for (int i = 0; i < Dim5120; i++)
        {
            Assert.Equal(0.0f, restored[i]);
        }
    }

    [Fact]
    public void NumericEdgeCases_SubnormalFloats_HandledWithoutUnderflowExceptions()
    {
        float[] subnormals = new float[Dim5120];
        subnormals[0] = 1e-38f;
        subnormals[1] = -1e-38f;
        subnormals[2] = 1e-45f;
        subnormals[3] = -1e-45f;
        for (int i = 4; i < Dim5120; i++) subnormals[i] = 1e-39f;

        byte[] wireBuffer = new byte[5124];
        float scale = QuantizationKernels.QuantizeToken(subnormals, wireBuffer, WirePrecisionMode.INT8_Symmetric);

        Assert.False(float.IsNaN(scale));
        Assert.True(scale > 0.0f);

        float[] restored = new float[Dim5120];
        QuantizationKernels.DequantizeToken(wireBuffer, restored, WirePrecisionMode.INT8_Symmetric);

        for (int i = 0; i < Dim5120; i++)
        {
            Assert.False(float.IsNaN(restored[i]));
            Assert.False(float.IsInfinity(restored[i]));
        }
    }

    [Theory]
    [InlineData(100.0f)]
    [InlineData(10_000.0f)]
    [InlineData(1_000_000.0f)]
    public void NumericEdgeCases_MassiveDynamicRange_ExtremeOutliersUpTo1e6(float outlierMagnitude)
    {
        float[] vector = new float[Dim5120];
        var rng = new Random(42);
        for (int i = 0; i < Dim5120; i++)
        {
            vector[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        // Inject massive outliers
        vector[10] = outlierMagnitude;
        vector[20] = -outlierMagnitude;

        byte[] dest = new byte[5124];
        float scale = QuantizationKernels.QuantizeToken(vector, dest, WirePrecisionMode.INT8_Symmetric);

        Assert.False(float.IsNaN(scale));
        Assert.False(float.IsInfinity(scale));
        Assert.True(scale > 0.0f);

        // Scale must match max magnitude / 127
        float expectedScale = outlierMagnitude / 127.0f;
        Assert.Equal(expectedScale, scale, 3);

        float[] restored = new float[Dim5120];
        QuantizationKernels.DequantizeToken(dest, restored, WirePrecisionMode.INT8_Symmetric);

        // Outlier values must be preserved at highest quantization bins
        Assert.Equal(outlierMagnitude, restored[10], 1);
        Assert.Equal(-outlierMagnitude, restored[20], 1);

        // Double precision cosine similarity should not overflow or produce NaN
        double cosSim = QuantizationKernels.ComputeCosineSimilarity(vector, restored);
        Assert.False(double.IsNaN(cosSim));
        Assert.True(cosSim > 0.99, $"Cosine similarity under outlier was {cosSim:F6}");
    }

    // =========================================================================
    // SECTION 2: NON-STANDARD DIMENSIONS & BOUNDARY LOOP SAFETY
    // =========================================================================

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(7)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(127)]
    [InlineData(128)]
    [InlineData(129)]
    [InlineData(255)]
    [InlineData(256)]
    [InlineData(257)]
    [InlineData(4096)]
    [InlineData(5120)]
    [InlineData(8192)]
    public void BoundaryLoopSafety_ArbitraryDimensions_Int8Symmetric_RoundtripsWithoutOverrun(int dim)
    {
        float[] original = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            original[i] = MathF.Sin(i * 0.1f) * 2.5f;
        }

        int expectedBytes = dim + 4;
        Assert.Equal(expectedBytes, QuantizationKernels.GetPackedTokenBytes(WirePrecisionMode.INT8_Symmetric, dim));

        byte[] wire = new byte[expectedBytes];
        float scale = QuantizationKernels.QuantizeToken(original, wire, WirePrecisionMode.INT8_Symmetric);

        Assert.True(scale > 0.0f);

        float[] restored = new float[dim];
        QuantizationKernels.DequantizeToken(wire, restored, WirePrecisionMode.INT8_Symmetric);

        double cosSim = QuantizationKernels.ComputeCosineSimilarity(original, restored);
        Assert.True(cosSim > 0.999, $"Dim {dim} cosine similarity {cosSim:F6} was below 0.999");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(15)]
    [InlineData(31)]
    [InlineData(65)]
    [InlineData(255)]
    [InlineData(5120)]
    public void BoundaryLoopSafety_ArbitraryDimensions_Fp8E4M3_RoundtripsWithoutOverrun(int dim)
    {
        float[] original = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            original[i] = MathF.Cos(i * 0.2f) * 1.5f;
        }

        int expectedBytes = dim + 4;
        byte[] wire = new byte[expectedBytes];
        float scale = QuantizationKernels.QuantizeToken(original, wire, WirePrecisionMode.FP8_E4M3);

        Assert.True(scale > 0.0f);

        float[] restored = new float[dim];
        QuantizationKernels.DequantizeToken(wire, restored, WirePrecisionMode.FP8_E4M3);

        double cosSim = QuantizationKernels.ComputeCosineSimilarity(original, restored);
        Assert.True(cosSim > 0.99, $"Dim {dim} FP8 cosine similarity {cosSim:F6} was below 0.99");
    }

    [Theory]
    [InlineData(1, 4)]
    [InlineData(7, 3)]
    [InlineData(33, 2)]
    [InlineData(255, 5)]
    [InlineData(5120, 3)]
    public void BoundaryLoopSafety_BatchQuantization_ArbitraryDimsAndBatches(int dim, int batchSize)
    {
        int totalElements = dim * batchSize;
        float[] inBatch = new float[totalElements];
        for (int i = 0; i < totalElements; i++) inBatch[i] = MathF.Sin(i * 0.05f) * 3.0f;

        int batchBytes = QuantizationKernels.GetPackedBatchBytes(WirePrecisionMode.INT8_Symmetric, dim, batchSize);
        byte[] wireBatch = new byte[batchBytes];

        QuantizationKernels.QuantizeBatch(inBatch, wireBatch, dim, batchSize, WirePrecisionMode.INT8_Symmetric);

        float[] outBatch = new float[totalElements];
        QuantizationKernels.DequantizeBatch(wireBatch, outBatch, dim, batchSize, WirePrecisionMode.INT8_Symmetric);

        for (int b = 0; b < batchSize; b++)
        {
            var inTok = inBatch.AsSpan(b * dim, dim);
            var outTok = outBatch.AsSpan(b * dim, dim);
            double sim = QuantizationKernels.ComputeCosineSimilarity(inTok, outTok);
            Assert.True(sim > 0.999, $"Batch {b} in Dim {dim} had similarity {sim:F6}");
        }
    }

    // =========================================================================
    // SECTION 3: ACCURACY METRICS ACROSS DISTRIBUTIONS
    // =========================================================================

    [Fact]
    public void Accuracy_GaussianDistribution_CosineSimilarityAbove9995_AndFp16ErrorBelow1e3()
    {
        float[] gaussian = new float[Dim5120];
        // Box-Muller transform for standard normal N(0, 1)
        var rng = new Random(2026);
        for (int i = 0; i < Dim5120; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            gaussian[i] = (float)(r * Math.Cos(theta));
            if (i + 1 < Dim5120) gaussian[i + 1] = (float)(r * Math.Sin(theta));
        }

        // 1. INT8_Symmetric
        byte[] int8Wire = new byte[5124];
        QuantizationKernels.QuantizeToken(gaussian, int8Wire, WirePrecisionMode.INT8_Symmetric);
        float[] int8Restored = new float[Dim5120];
        QuantizationKernels.DequantizeToken(int8Wire, int8Restored, WirePrecisionMode.INT8_Symmetric);

        double int8CosSim = QuantizationKernels.ComputeCosineSimilarity(gaussian, int8Restored);
        Assert.True(int8CosSim > 0.9995, $"Gaussian INT8 Cosine similarity {int8CosSim:F6} was not > 0.9995");

        // 2. FP16
        byte[] fp16Wire = new byte[Dim5120 * 2];
        QuantizationKernels.QuantizeToken(gaussian, fp16Wire, WirePrecisionMode.FP16);
        float[] fp16Restored = new float[Dim5120];
        QuantizationKernels.DequantizeToken(fp16Wire, fp16Restored, WirePrecisionMode.FP16);

        double fp16CosSim = QuantizationKernels.ComputeCosineSimilarity(gaussian, fp16Restored);
        Assert.True(fp16CosSim > 0.9999, $"Gaussian FP16 Cosine similarity {fp16CosSim:F6} was not > 0.9999");

        float fp16MaxRel = QuantizationKernels.ComputeMaxRelativeError(gaussian, fp16Restored);
        Assert.True(fp16MaxRel < 1e-3f, $"Gaussian FP16 Max relative error {fp16MaxRel:E6} was not < 10^-3");

        // 3. FP32 (Lossless reference)
        byte[] fp32Wire = new byte[Dim5120 * 4];
        QuantizationKernels.QuantizeToken(gaussian, fp32Wire, WirePrecisionMode.FP32);
        float[] fp32Restored = new float[Dim5120];
        QuantizationKernels.DequantizeToken(fp32Wire, fp32Restored, WirePrecisionMode.FP32);

        float fp32MaxRel = QuantizationKernels.ComputeMaxRelativeError(gaussian, fp32Restored);
        Assert.Equal(0.0f, fp32MaxRel);
    }

    [Fact]
    public void Accuracy_UniformDistribution_CosineSimilarityAbove9995_AndFp16ErrorBelow1e3()
    {
        float[] uniform = new float[Dim5120];
        var rng = new Random(777);
        for (int i = 0; i < Dim5120; i++)
        {
            uniform[i] = (float)(rng.NextDouble() * 6.0 - 3.0); // U(-3, 3)
        }

        // 1. INT8_Symmetric
        byte[] int8Wire = new byte[5124];
        QuantizationKernels.QuantizeToken(uniform, int8Wire, WirePrecisionMode.INT8_Symmetric);
        float[] int8Restored = new float[Dim5120];
        QuantizationKernels.DequantizeToken(int8Wire, int8Restored, WirePrecisionMode.INT8_Symmetric);

        double int8CosSim = QuantizationKernels.ComputeCosineSimilarity(uniform, int8Restored);
        Assert.True(int8CosSim > 0.9995, $"Uniform INT8 Cosine similarity {int8CosSim:F6} was not > 0.9995");

        // 2. FP16
        byte[] fp16Wire = new byte[Dim5120 * 2];
        QuantizationKernels.QuantizeToken(uniform, fp16Wire, WirePrecisionMode.FP16);
        float[] fp16Restored = new float[Dim5120];
        QuantizationKernels.DequantizeToken(fp16Wire, fp16Restored, WirePrecisionMode.FP16);

        float fp16MaxRel = QuantizationKernels.ComputeMaxRelativeError(uniform, fp16Restored);
        Assert.True(fp16MaxRel < 1e-3f, $"Uniform FP16 Max relative error {fp16MaxRel:E6} was not < 10^-3");
    }

    [Fact]
    public void Accuracy_SparseOutlierDistribution_CosineSimilarityAbove9995()
    {
        float[] activations = new float[Dim5120];
        // Standard normal N(0, 1) base distribution matching RMSNorm output in LLM hidden states
        var rng = new Random(888);
        for (int i = 0; i < Dim5120; i += 2)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double theta = 2.0 * Math.PI * u2;
            activations[i] = (float)(r * Math.Cos(theta));
            if (i + 1 < Dim5120) activations[i + 1] = (float)(r * Math.Sin(theta));
        }

        // Inject 5 sparse outliers typical in LLM intermediate hidden states (e.g., 6.0f to 8.5f)
        activations[15] = 7.8f;
        activations[142] = -8.2f;
        activations[1024] = 6.9f;
        activations[2048] = -7.4f;
        activations[4096] = 8.5f;

        // INT8_Symmetric
        byte[] int8Wire = new byte[5124];
        QuantizationKernels.QuantizeToken(activations, int8Wire, WirePrecisionMode.INT8_Symmetric);
        float[] int8Restored = new float[Dim5120];
        QuantizationKernels.DequantizeToken(int8Wire, int8Restored, WirePrecisionMode.INT8_Symmetric);

        double int8CosSim = QuantizationKernels.ComputeCosineSimilarity(activations, int8Restored);
        Assert.True(int8CosSim > 0.9995, $"Sparse outlier INT8 Cosine similarity {int8CosSim:F6} was not > 0.9995");

        // FP16
        byte[] fp16Wire = new byte[Dim5120 * 2];
        QuantizationKernels.QuantizeToken(activations, fp16Wire, WirePrecisionMode.FP16);
        float[] fp16Restored = new float[Dim5120];
        QuantizationKernels.DequantizeToken(fp16Wire, fp16Restored, WirePrecisionMode.FP16);

        float fp16MaxRel = QuantizationKernels.ComputeMaxRelativeError(activations, fp16Restored);
        Assert.True(fp16MaxRel < 1e-3f, $"Sparse outlier FP16 Max relative error {fp16MaxRel:E6} was not < 10^-3");
    }

    [Fact]
    public void Accuracy_ExtremeOutlierPaprAnalysis_DocumentsMathematicalInt8Degradation()
    {
        // Low variance activations U(-1, 1) with high peak outliers (PAPR > 18 dB)
        // Mathematically demonstrates the SQNR drop for uniform single-scale INT8
        float[] lowVarHighOutlier = new float[Dim5120];
        var rng = new Random(999);
        for (int i = 0; i < Dim5120; i++) lowVarHighOutlier[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        lowVarHighOutlier[0] = 12.0f;
        lowVarHighOutlier[100] = -12.0f;

        byte[] int8Wire = new byte[5124];
        QuantizationKernels.QuantizeToken(lowVarHighOutlier, int8Wire, WirePrecisionMode.INT8_Symmetric);
        float[] int8Restored = new float[Dim5120];
        QuantizationKernels.DequantizeToken(int8Wire, int8Restored, WirePrecisionMode.INT8_Symmetric);

        double int8CosSim = QuantizationKernels.ComputeCosineSimilarity(lowVarHighOutlier, int8Restored);
        // Under extreme PAPR (> 20 dB), single-scale per-token INT8 drops to ~0.9991-0.9994
        Assert.True(int8CosSim > 0.9985, $"Extreme PAPR INT8 Cosine similarity {int8CosSim:F6} dropped below 0.9985");

        // BUT FP16 remains impervious to high PAPR outliers:
        byte[] fp16Wire = new byte[Dim5120 * 2];
        QuantizationKernels.QuantizeToken(lowVarHighOutlier, fp16Wire, WirePrecisionMode.FP16);
        float[] fp16Restored = new float[Dim5120];
        QuantizationKernels.DequantizeToken(fp16Wire, fp16Restored, WirePrecisionMode.FP16);

        double fp16CosSim = QuantizationKernels.ComputeCosineSimilarity(lowVarHighOutlier, fp16Restored);
        Assert.True(fp16CosSim > 0.9999, $"FP16 Cosine similarity {fp16CosSim:F6} should be > 0.9999");
        float fp16MaxRel = QuantizationKernels.ComputeMaxRelativeError(lowVarHighOutlier, fp16Restored);
        Assert.True(fp16MaxRel < 1e-3f, $"FP16 Max relative error {fp16MaxRel:E6} should be < 10^-3");
    }

    // =========================================================================
    // SECTION 4: DYNAMIC PRECISION SWITCHER STRESS TESTS
    // =========================================================================

    [Fact]
    public void Switcher_OscillatingRttAroundCongestionThreshold_MaintainsDeterministicState()
    {
        var switcher = new DynamicPrecisionSwitcher();
        Assert.Equal(0.5, switcher.CurrentEmaLatencyMs);

        // Oscillate RTT around 2.5 ms threshold: 2.0 ms -> 3.0 ms -> 2.0 ms -> 3.0 ms
        for (int cycle = 0; cycle < 50; cycle++)
        {
            double rtt = (cycle % 2 == 0) ? 2.0 : 3.0;
            switcher.UpdateMeasuredLatency(rtt);

            WirePrecisionMode mode = switcher.SelectPrecision(batchSize: 1, sequencePosition: 100, isDecode: true);

            // Verify mode aligns with current smoothed EMA
            if (switcher.CurrentEmaLatencyMs > switcher.CongestionLatencyThresholdMs)
            {
                Assert.Equal(WirePrecisionMode.INT8_Symmetric, mode);
            }
            else
            {
                Assert.Equal(WirePrecisionMode.FP16, mode);
            }
        }

        Assert.True(switcher.TotalSwitches > 0, "Switcher should have adapted to oscillations");
    }

    [Fact]
    public void Switcher_RapidBatchSizeTransitions_TransitionsCorrectlyWithoutStateCorruption()
    {
        var switcher = new DynamicPrecisionSwitcher();
        int[] batchSizes = [1, 64, 1, 128, 16, 32, 31, 512, 1, 2048, 8, 1];

        foreach (int bs in batchSizes)
        {
            bool isDecode = bs == 1;
            WirePrecisionMode mode = switcher.SelectPrecision(batchSize: bs, sequencePosition: 100, isDecode: isDecode);

            if (isDecode)
            {
                Assert.Equal(WirePrecisionMode.FP16, mode);
            }
            else
            {
                if (bs >= switcher.PrefillBatchThreshold)
                {
                    Assert.Equal(WirePrecisionMode.INT8_Symmetric, mode);
                }
                else
                {
                    Assert.Equal(WirePrecisionMode.FP16, mode);
                }
            }
        }
    }

    [Theory]
    [InlineData(WirePrecisionMode.FP32)]
    [InlineData(WirePrecisionMode.FP16)]
    [InlineData(WirePrecisionMode.INT8_Symmetric)]
    [InlineData(WirePrecisionMode.FP8_E4M3)]
    public void Switcher_CliForcedOverrides_PreemptsAllDynamicHeuristics(WirePrecisionMode forced)
    {
        var switcher = new DynamicPrecisionSwitcher(forced);

        // Try various stressful conditions: huge batch, congestion RTT, deep context
        Assert.Equal(forced, switcher.SelectPrecision(batchSize: 1, sequencePosition: 0, isDecode: true));
        Assert.Equal(forced, switcher.SelectPrecision(batchSize: 1024, sequencePosition: 0, isDecode: false));
        Assert.Equal(forced, switcher.SelectPrecision(batchSize: 1, sequencePosition: 30000, isDecode: true, overrideRttMs: 15.0));

        // Switch forced mode dynamically
        switcher.ForcedPrecision = null;
        Assert.Equal(WirePrecisionMode.INT8_Symmetric, switcher.SelectPrecision(batchSize: 128, sequencePosition: 0, isDecode: false));
    }

    [Fact]
    public void Switcher_DeepContextBoundary_StrictTransitionAt16384()
    {
        var switcher = new DynamicPrecisionSwitcher();
        switcher.UpdateMeasuredLatency(4.0); // Congestion is high

        // Below or at 16384: high latency forces INT8
        WirePrecisionMode at16384 = switcher.SelectPrecision(batchSize: 1, sequencePosition: 16384, isDecode: true);
        Assert.Equal(WirePrecisionMode.INT8_Symmetric, at16384);

        // Above 16384: deep context protection forces FP16 regardless of congestion
        WirePrecisionMode at16385 = switcher.SelectPrecision(batchSize: 1, sequencePosition: 16385, isDecode: true);
        Assert.Equal(WirePrecisionMode.FP16, at16385);
    }

    // =========================================================================
    // SECTION 5: E2E SOCKET TRANSPORT WITH DYNAMIC PRECISION JITTER
    // =========================================================================

    [Fact]
    public async Task E2E_SocketStreaming_DynamicPrecisionSwitchedEveryFrame_ZeroCorruption()
    {
        using var listener = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        listener.Bind(new IPEndPoint(IPAddress.Loopback, 0));
        listener.Listen(1);
        int port = ((IPEndPoint)listener.LocalEndPoint!).Port;

        using var client = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        var acceptTask = listener.AcceptAsync();
        await client.ConnectAsync(IPAddress.Loopback, port);
        using var server = await acceptTask;

        FusedSocketTransport.ConfigureSocket(client, 2 * 1024 * 1024);
        FusedSocketTransport.ConfigureSocket(server, 2 * 1024 * 1024);

        byte[] sendStaging = GC.AllocateArray<byte>(64 * 1024, pinned: true);
        byte[] recvBuffer = GC.AllocateArray<byte>(64 * 1024, pinned: true);

        float[] referenceActivations = new float[Dim5120];
        for (int i = 0; i < Dim5120; i++) referenceActivations[i] = MathF.Sin(i * 0.05f) * 2.0f;

        WirePrecisionMode[] precisionSequence = [
            WirePrecisionMode.FP16,
            WirePrecisionMode.INT8_Symmetric,
            WirePrecisionMode.FP8_E4M3,
            WirePrecisionMode.FP32,
            WirePrecisionMode.INT8_Symmetric,
            WirePrecisionMode.FP16
        ];

        for (int step = 0; step < 24; step++)
        {
            var mode = precisionSequence[step % precisionSequence.Length];
            int tokenBytes = QuantizationKernels.GetPackedTokenBytes(mode, Dim5120);
            byte[] payload = new byte[tokenBytes];

            float scale = QuantizationKernels.QuantizeToken(referenceActivations, payload, mode);

            var actHdr = new FusedActivationHeader(
                sequenceId: (ulong)step,
                tokenIndex: step,
                layerIndex: 27,
                hiddenDim: Dim5120,
                precisionMode: mode,
                flags: FusedActivationFlags.HasScale,
                scale: scale);

            // Send from client
            FusedSocketTransport.SendFusedActivation(client, (ulong)step, in actHdr, payload, sendStaging);

            // Receive on server
            bool ok = FusedSocketTransport.ReceiveFusedActivation(
                server,
                recvBuffer,
                out var clusterHdr,
                out var rxActHdr,
                out var rxPayload);

            Assert.True(ok);
            Assert.Equal(mode, rxActHdr.TypedPrecision);
            Assert.Equal(tokenBytes, rxPayload.Length);
            Assert.Equal((ulong)step, rxActHdr.SequenceId);

            // Dequantize and verify accuracy
            float[] restored = new float[Dim5120];
            QuantizationKernels.DequantizeToken(rxPayload, restored, mode);

            double cosSim = QuantizationKernels.ComputeCosineSimilarity(referenceActivations, restored);
            Assert.True(cosSim > 0.999, $"Step {step} mode {mode} cosine similarity {cosSim:F6} was not > 0.999");
        }
    }
}
