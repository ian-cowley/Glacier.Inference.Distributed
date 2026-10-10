namespace Glacier.Inference.Tests.Distributed;

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.IO.Pipelines;
using System.Net;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Cluster.Shuffle;
using Glacier.Inference.Distributed;
using Glacier.Inference.Gguf;
using Xunit;

/// <summary>
/// Adversarial stress harness and empirical challenge suite for Milestone M3 (Distributed Interconnect & Activation Protocols).
/// Tests framing invariants, subnet security boundaries, zero-allocation hot paths, corrupt packets, and boundary edge cases.
/// </summary>
public class AdversarialInterconnectStressTests
{
    #region 1. ActivationHeader Adversarial Validation

    [Fact]
    public void ActivationHeader_StructSize_IsExactly32Bytes_WithExactFieldOffsets()
    {
        // Struct size must be strictly 32 bytes
        Assert.Equal(32, Unsafe.SizeOf<ActivationHeader>());
        Assert.Equal(32, Marshal.SizeOf<ActivationHeader>());
        Assert.Equal(32, ActivationHeader.HeaderSize);

        // Verify explicit 64-bit alignment and packing offsets
        Assert.Equal(0, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.Magic)));
        Assert.Equal(4, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.SequenceId)));
        Assert.Equal(12, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.TokenIndex)));
        Assert.Equal(16, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.LayerIndex)));
        Assert.Equal(20, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.HiddenDim)));
        Assert.Equal(24, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.ExpertId)));
        Assert.Equal(28, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.Flags)));
    }

    [Theory]
    [InlineData(0x00000000U)]
    [InlineData(0xFFFFFFFFU)]
    [InlineData(0x47434C55U)] // "GCLU" (ClusterFrameHeader magic, should NOT match ActivationHeader)
    [InlineData(0x54434147U)] // "TCAG" (Big-Endian mirror)
    [InlineData(0x12345678U)]
    [InlineData(0xDEADBEEFU)]
    [InlineData(0xCAFEBABEU)]
    public void ActivationHeader_InvalidMagic_ThrowsInvalidOperationException_AndTryReadReturnsFalse(uint corruptMagic)
    {
        byte[] buffer = new byte[32];
        var validHeader = new ActivationHeader(42, 1, 2, 4096, 0);
        validHeader.WriteTo(buffer.AsSpan());

        // Overwrite magic with corrupt value
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(0, 4), corruptMagic);

        Assert.Throws<InvalidOperationException>(() => ActivationHeader.ReadFrom(buffer.AsSpan()));

        bool success = ActivationHeader.TryRead(buffer.AsSpan(), out var header);
        Assert.False(success);
        Assert.Equal(default, header);
    }

    [Fact]
    public void ActivationHeader_FuzzInvalidMagic_AllRejected()
    {
        byte[] buffer = new byte[32];
        var validHeader = new ActivationHeader(12345, 10, 5, 2048, 3);
        validHeader.WriteTo(buffer.AsSpan());

        var rng = new Random(42);
        for (int i = 0; i < 200; i++)
        {
            uint randomMagic = (uint)rng.Next();
            if (randomMagic == ActivationHeader.ActivationMagic) continue;

            BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(0, 4), randomMagic);
            Assert.Throws<InvalidOperationException>(() => ActivationHeader.ReadFrom(buffer.AsSpan()));
            Assert.False(ActivationHeader.TryRead(buffer.AsSpan(), out _));
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(31)]
    public void ActivationHeader_TruncatedSpans_StrictlyRejected(int truncatedLength)
    {
        byte[] truncatedBuffer = new byte[truncatedLength];
        var header = new ActivationHeader(999, 1, 2, 4096, 0);

        // WriteTo must throw ArgumentException
        Assert.Throws<ArgumentException>(() => header.WriteTo(truncatedBuffer.AsSpan()));

        // ReadFrom must throw ArgumentException
        Assert.Throws<ArgumentException>(() => ActivationHeader.ReadFrom(truncatedBuffer.AsSpan()));

        // TryRead must return false without throwing
        bool readSuccess = ActivationHeader.TryRead(truncatedBuffer.AsSpan(), out var readHeader);
        Assert.False(readSuccess);
        Assert.Equal(default, readHeader);
    }

    [Fact]
    public void ActivationHeader_ExactSlice_LeavesSurroundingMemoryUntouched()
    {
        byte[] canaryBuffer = new byte[64];
        Array.Fill(canaryBuffer, (byte)0xEE);

        var header = new ActivationHeader(8888, 77, 33, 4096, 5);

        // Write into slice [16..48)
        header.WriteTo(canaryBuffer.AsSpan(16, 32));

        // Verify pre-slice canaries [0..16) are untouched
        for (int i = 0; i < 16; i++)
        {
            Assert.Equal((byte)0xEE, canaryBuffer[i]);
        }

        // Verify post-slice canaries [48..64) are untouched
        for (int i = 48; i < 64; i++)
        {
            Assert.Equal((byte)0xEE, canaryBuffer[i]);
        }

        // Verify deserialization from exact slice
        var roundtripped = ActivationHeader.ReadFrom(canaryBuffer.AsSpan(16, 32));
        Assert.Equal(header, roundtripped);
    }

    [Fact]
    public void ActivationHeader_BoundaryValues_RoundtripWithBitAccuracy()
    {
        (ulong seq, int tok, int layer, int dim, int exp, uint flags)[] testCases = [
            (0UL, 0, 0, 0, 0, 0U),
            (ulong.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, uint.MaxValue),
            (1UL, -1, -1, -1, -1, 0x80000000U),
            (0x0123456789ABCDEFUL, int.MinValue, int.MinValue, int.MinValue, int.MinValue, 0x55555555U),
            (999999999999UL, 1000000, 80, 16384, 255, (uint)(ActivationFlags.ResidualAccumulate | ActivationFlags.FinalLayer | ActivationFlags.Compressed | ActivationFlags.SharedExpert | ActivationFlags.AcknowledgeRequired)),
        ];

        byte[] span = new byte[32];

        foreach (var tc in testCases)
        {
            var original = new ActivationHeader(tc.seq, tc.tok, tc.layer, tc.dim, tc.exp, tc.flags);
            original.WriteTo(span.AsSpan());

            var restored = ActivationHeader.ReadFrom(span.AsSpan());

            Assert.Equal(original, restored);
            Assert.True(original == restored);
            Assert.False(original != restored);
            Assert.Equal(original.GetHashCode(), restored.GetHashCode());
            Assert.Equal(tc.seq, restored.SequenceId);
            Assert.Equal(tc.tok, restored.TokenIndex);
            Assert.Equal(tc.layer, restored.LayerIndex);
            Assert.Equal(tc.dim, restored.HiddenDim);
            Assert.Equal(tc.exp, restored.ExpertId);
            Assert.Equal(tc.flags, restored.Flags);
        }
    }

    [Fact]
    public void ActivationHeader_EqualityAndInequality_AllFieldsTested()
    {
        var baseHeader = new ActivationHeader(100, 10, 5, 4096, 2, 0);
        var identicalHeader = new ActivationHeader(100, 10, 5, 4096, 2, 0);

        Assert.Equal(baseHeader, identicalHeader);
        Assert.True(baseHeader == identicalHeader);
        Assert.False(baseHeader != identicalHeader);

        // Perturb each individual field and verify inequality
        Assert.NotEqual(baseHeader, new ActivationHeader(101, 10, 5, 4096, 2, 0));
        Assert.NotEqual(baseHeader, new ActivationHeader(100, 11, 5, 4096, 2, 0));
        Assert.NotEqual(baseHeader, new ActivationHeader(100, 10, 6, 4096, 2, 0));
        Assert.NotEqual(baseHeader, new ActivationHeader(100, 10, 5, 8192, 2, 0));
        Assert.NotEqual(baseHeader, new ActivationHeader(100, 10, 5, 4096, 3, 0));
        Assert.NotEqual(baseHeader, new ActivationHeader(100, 10, 5, 4096, 2, 1));
        Assert.NotEqual(baseHeader, new ActivationHeader(100, 10, 5, 4096, 2, 0, magic: 0x12345678));

        Assert.False(baseHeader.Equals(null));
        Assert.False(baseHeader.Equals("not an activation header"));
    }

    [Fact]
    public async Task ActivationHeader_WriteTo_BufferWriter_AdvancesExactly32Bytes()
    {
        var header = new ActivationHeader(1234, 5, 10, 2048, 1);
        var pipe = new Pipe();

        header.WriteTo(pipe.Writer);
        await pipe.Writer.FlushAsync();

        var readResult = await pipe.Reader.ReadAsync();
        Assert.Equal(32, readResult.Buffer.Length);

        Span<byte> span = stackalloc byte[32];
        readResult.Buffer.CopyTo(span);
        var roundtripped = ActivationHeader.ReadFrom(span);
        Assert.Equal(header, roundtripped);
        pipe.Reader.AdvanceTo(readResult.Buffer.End);
    }

    #endregion

    #region 2. Subnet Confinement & Security Enforcement

    [Theory]
    // Public / external IP addresses
    [InlineData("8.8.8.8")]
    [InlineData("8.8.4.4")]
    [InlineData("1.1.1.1")]
    [InlineData("9.9.9.9")]
    [InlineData("142.250.190.46")] // google.com IP
    // Unauthorized private subnets
    [InlineData("10.0.0.1")]
    [InlineData("10.255.255.255")]
    [InlineData("172.16.0.1")]
    [InlineData("172.31.255.254")]
    [InlineData("192.168.0.1")]   // 192.168.0.x is NOT 192.168.1.x
    [InlineData("192.168.2.1")]   // 192.168.2.x is NOT 192.168.1.x
    [InlineData("192.169.1.1")]
    [InlineData("192.168.100.1")]
    // Special / Link-local / Broadcast
    [InlineData("0.0.0.0")]
    [InlineData("255.255.255.255")]
    [InlineData("169.254.1.1")]
    // External domains
    [InlineData("google.com")]
    [InlineData("api.openai.com")]
    [InlineData("attacker.local")]
    // Malformed inputs
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("not-an-ip")]
    [InlineData("192.168.1")]
    [InlineData("192.168.1.1.1")]
    public void SubnetConfinement_StrictlyRejectsExternalAndUnauthorizedHosts_WithSecurityException(string unauthorizedHost)
    {
        Assert.False(
            ClusterNetworkTransportHarness.IsSubnetAllowed(unauthorizedHost),
            $"Host '{unauthorizedHost}' should have been rejected by IsSubnetAllowed.");

        var secEx = Assert.Throws<SecurityException>(() =>
            ClusterNetworkTransportHarness.ValidateSubnetOrThrow(unauthorizedHost));

        Assert.Contains(unauthorizedHost, secEx.Message);
    }

    [Theory]
    // Loopback identifiers
    [InlineData("127.0.0.1")]
    [InlineData("localhost")]
    [InlineData("LOCALHOST")]
    [InlineData("LocalHost")]
    [InlineData("::1")]
    // Fleet node endpoints (192.168.1.0/24)
    [InlineData("192.168.1.111")] // Host Laptop
    [InlineData("192.168.1.108")] // Beast
    [InlineData("192.168.1.200")] // Fedora Linux
    [InlineData("192.168.1.101")] // MinisForum64
    // Boundary endpoints in allowed subnet
    [InlineData("192.168.1.0")]
    [InlineData("192.168.1.1")]
    [InlineData("192.168.1.254")]
    [InlineData("192.168.1.255")]
    public void SubnetConfinement_AllowsFleetAndLoopbackEndpoints(string authorizedHost)
    {
        Assert.True(
            ClusterNetworkTransportHarness.IsSubnetAllowed(authorizedHost),
            $"Host '{authorizedHost}' should have been allowed by IsSubnetAllowed.");

        // Must not throw
        ClusterNetworkTransportHarness.ValidateSubnetOrThrow(authorizedHost);
    }

    [Fact]
    public async Task SubnetConfinement_RunPhysicalNodeBenchmark_RejectsUnauthorizedHost()
    {
        await Assert.ThrowsAsync<SecurityException>(async () =>
        {
            await ClusterNetworkTransportHarness.RunPhysicalNodeBenchmarkAsync("8.8.8.8", 12345);
        });

        await Assert.ThrowsAsync<SecurityException>(async () =>
        {
            await ClusterNetworkTransportHarness.RunPhysicalNodeBenchmarkAsync("10.0.0.1", 12345);
        });

        await Assert.ThrowsAsync<SecurityException>(async () =>
        {
            await ClusterNetworkTransportHarness.RunPhysicalNodeBenchmarkAsync("172.16.0.1", 12345);
        });
    }

    #endregion

    #region 3. Zero-Allocation Hot Path Stress Testing

    [Theory]
    [InlineData(64, 128, 1)]
    [InlineData(128, 256, 2)]
    [InlineData(512, 1024, 4)]
    [InlineData(4096, 11008, 2)] // Production hidden dimension
    public unsafe void DistributedPipelineCoordinator_ExecuteStepZeroAlloc_UnderHighRepetition_GuaranteesZeroAlloc(
        int dim, int ffnDim, int topK)
    {
        const int numExperts = 8;
        using var stage0 = new AttentionRootStageWorker(dim, topK: topK, numExperts: numExperts);
        using var stage1 = CpuWarmExpertPoolWorker.CreateSynthetic(1, dim, ffnDim, numExperts: numExperts, GgufType.F32, multithreaded: false);
        using var stage2 = new AccumulatorStageWorker(dim);
        using var coordinator = new DistributedPipelineCoordinator(stage0, stage1, stage2, dim);

        // Pre-allocate unmanaged buffers
        float* input = (float*)NativeMemory.AlignedAlloc((nuint)(dim * sizeof(float)), 64);
        float* destination = (float*)NativeMemory.AlignedAlloc((nuint)(dim * sizeof(float)), 64);
        float* scratch0 = (float*)NativeMemory.AlignedAlloc((nuint)(dim * sizeof(float)), 64);
        float* scratchExp = (float*)NativeMemory.AlignedAlloc((nuint)(dim * sizeof(float)), 64);

        for (int i = 0; i < dim; i++) input[i] = 1.0f;

        int[] selectedExperts = new int[topK];
        float[] expertWeights = new float[topK];
        for (int k = 0; k < topK; k++)
        {
            selectedExperts[k] = k % numExperts;
            expertWeights[k] = 1.0f / topK;
        }

        try
        {
            // Warmup JIT & tiered compilation
            for (int i = 0; i < 50; i++)
            {
                coordinator.ExecuteStepZeroAlloc(
                    (ulong)i, 0, 0, input, selectedExperts.AsSpan(), expertWeights.AsSpan(),
                    destination, scratch0, scratchExp);
            }

            // High repetition stress measurement: 1,000 hot-path iterations
            long allocBefore = GC.GetAllocatedBytesForCurrentThread();

            for (int i = 0; i < 1_000; i++)
            {
                coordinator.ExecuteStepZeroAlloc(
                    (ulong)i, 0, 0, input, selectedExperts.AsSpan(), expertWeights.AsSpan(),
                    destination, scratch0, scratchExp);
            }

            long allocAfter = GC.GetAllocatedBytesForCurrentThread();
            long totalAlloc = allocAfter - allocBefore;

            Assert.Equal(0, totalAlloc);

            // Verify output is valid (finite, non-zero)
            for (int i = 0; i < dim; i++)
            {
                Assert.False(float.IsNaN(destination[i]), $"Destination float at {i} was NaN");
                Assert.False(float.IsInfinity(destination[i]), $"Destination float at {i} was Infinity");
            }
        }
        finally
        {
            NativeMemory.AlignedFree(input);
            NativeMemory.AlignedFree(destination);
            NativeMemory.AlignedFree(scratch0);
            NativeMemory.AlignedFree(scratchExp);
        }
    }

    [Fact]
    public unsafe void Stage0_AttentionRoot_ZeroAlloc_DirectPointers()
    {
        const int dim = 4096;
        using var stage0 = new AttentionRootStageWorker(dim, topK: 2, numExperts: 8);

        float* inPtr = (float*)NativeMemory.AlignedAlloc((nuint)(dim * sizeof(float)), 64);
        float* outPtr = (float*)NativeMemory.AlignedAlloc((nuint)(dim * sizeof(float)), 64);
        for (int i = 0; i < dim; i++) inPtr[i] = 1.5f;

        var header = new ActivationHeader(1, 0, 0, dim, 0);

        try
        {
            // Warmup
            for (int i = 0; i < 20; i++)
            {
                stage0.ProcessActivation(header, inPtr, outPtr);
            }

            long allocBefore = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 1_000; i++)
            {
                stage0.ProcessActivation(header, inPtr, outPtr);
            }
            long allocAfter = GC.GetAllocatedBytesForCurrentThread();

            Assert.Equal(0, allocAfter - allocBefore);
        }
        finally
        {
            NativeMemory.AlignedFree(inPtr);
            NativeMemory.AlignedFree(outPtr);
        }
    }

    [Fact]
    public unsafe void Stage2_Accumulator_ZeroAlloc_DirectPointers()
    {
        const int dim = 4096;
        using var stage2 = new AccumulatorStageWorker(dim);

        float* acc = (float*)NativeMemory.AlignedAlloc((nuint)(dim * sizeof(float)), 64);
        float* res = (float*)NativeMemory.AlignedAlloc((nuint)(dim * sizeof(float)), 64);
        float* exp = (float*)NativeMemory.AlignedAlloc((nuint)(dim * sizeof(float)), 64);

        for (int i = 0; i < dim; i++)
        {
            acc[i] = 0.0f;
            res[i] = 1.0f;
            exp[i] = 2.0f;
        }

        try
        {
            // Warmup
            for (int i = 0; i < 20; i++)
            {
                stage2.AddResidual(acc, res);
                stage2.AccumulateWeightedExpert(acc, exp, 0.5f);
            }

            long allocBefore = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 1_000; i++)
            {
                stage2.AddResidual(acc, res);
                stage2.AccumulateWeightedExpert(acc, exp, 0.5f);
            }
            long allocAfter = GC.GetAllocatedBytesForCurrentThread();

            Assert.Equal(0, allocAfter - allocBefore);
        }
        finally
        {
            NativeMemory.AlignedFree(acc);
            NativeMemory.AlignedFree(res);
            NativeMemory.AlignedFree(exp);
        }
    }

    [Fact]
    public unsafe void CpuWarmExpertPoolWorker_ZeroAlloc_DirectPointers_LargeFfn()
    {
        const int dim = 1024;
        const int ffnDim = 2048; // Tests stackalloc scratch buffer path
        using var worker = CpuWarmExpertPoolWorker.CreateSynthetic(1, dim, ffnDim, numExperts: 4, GgufType.F32, multithreaded: false);

        float* inPtr = (float*)NativeMemory.AlignedAlloc((nuint)(dim * sizeof(float)), 64);
        float* outPtr = (float*)NativeMemory.AlignedAlloc((nuint)(dim * sizeof(float)), 64);
        for (int i = 0; i < dim; i++) inPtr[i] = 0.5f;

        var header = new ActivationHeader(42, 0, 0, dim, 1);

        try
        {
            // Warmup
            for (int i = 0; i < 20; i++)
            {
                worker.ProcessActivation(header, inPtr, outPtr);
            }

            long allocBefore = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < 500; i++)
            {
                worker.ProcessActivation(header, inPtr, outPtr);
            }
            long allocAfter = GC.GetAllocatedBytesForCurrentThread();

            Assert.Equal(0, allocAfter - allocBefore);
        }
        finally
        {
            NativeMemory.AlignedFree(inPtr);
            NativeMemory.AlignedFree(outPtr);
        }
    }

    #endregion

    #region 4. Pipeline Guardrails, Disposal & Corrupt Framing Tests

    [Fact]
    public unsafe void Coordinator_ExecuteStepZeroAlloc_ArgumentValidation_GuardsPointersAndDimensions()
    {
        const int dim = 64;
        using var stage0 = new AttentionRootStageWorker(dim, topK: 1, numExperts: 2);
        using var stage1 = CpuWarmExpertPoolWorker.CreateSynthetic(1, dim, 128, numExperts: 2, GgufType.F32, multithreaded: false);
        using var stage2 = new AccumulatorStageWorker(dim);
        using var coordinator = new DistributedPipelineCoordinator(stage0, stage1, stage2, dim);

        float* dummy = stackalloc float[dim];
        int[] experts = [0];
        float[] weights = [1.0f];

        // Null pointer checks
        Assert.Throws<ArgumentNullException>(() =>
            coordinator.ExecuteStepZeroAlloc(1, 0, 0, null, experts, weights, dummy, dummy, dummy));

        Assert.Throws<ArgumentNullException>(() =>
            coordinator.ExecuteStepZeroAlloc(1, 0, 0, dummy, experts, weights, null, dummy, dummy));

        Assert.Throws<ArgumentNullException>(() =>
            coordinator.ExecuteStepZeroAlloc(1, 0, 0, dummy, experts, weights, dummy, null, dummy));

        Assert.Throws<ArgumentNullException>(() =>
            coordinator.ExecuteStepZeroAlloc(1, 0, 0, dummy, experts, weights, dummy, dummy, null));

        // Mismatched expert and weight arrays
        int[] mismatchedExperts = [0, 1];
        float[] mismatchedWeights = [1.0f]; // length 1 != 2
        Assert.Throws<ArgumentException>(() =>
            coordinator.ExecuteStepZeroAlloc(1, 0, 0, dummy, mismatchedExperts, mismatchedWeights, dummy, dummy, dummy));
    }

    [Fact]
    public unsafe void Coordinator_DisposedState_ThrowsObjectDisposedException()
    {
        const int dim = 64;
        var stage0 = new AttentionRootStageWorker(dim, topK: 1, numExperts: 2);
        var stage1 = CpuWarmExpertPoolWorker.CreateSynthetic(1, dim, 128, numExperts: 2, GgufType.F32);
        var stage2 = new AccumulatorStageWorker(dim);
        var coordinator = new DistributedPipelineCoordinator(stage0, stage1, stage2, dim);

        coordinator.Dispose();

        float* dummy = stackalloc float[dim];
        int[] experts = [0];
        float[] weights = [1.0f];
        byte[] inputBytes = new byte[dim * sizeof(float)];

        Assert.Throws<ObjectDisposedException>(() =>
            coordinator.ExecuteStep(1, 0, 0, inputBytes, experts, weights));

        Assert.Throws<ObjectDisposedException>(() =>
            coordinator.ExecuteStepZeroAlloc(1, 0, 0, dummy, experts, weights, dummy, dummy, dummy));
    }

    [Fact]
    public async Task WireFraming_TruncatedPayload_ThrowsInvalidOperationException()
    {
        var pipe = new Pipe();

        // Write a ClusterFrameHeader that specifies 100 bytes payload, but write only 16 bytes payload (< 32 required for ActivationHeader)
        var clusterHeader = new ClusterFrameHeader(
            messageType: ClusterMessageType.StageDispatch,
            correlationId: 1,
            payloadLength: 16, // Smaller than 32-byte ActivationHeader
            flags: ClusterFrameFlags.None);

        var span = pipe.Writer.GetSpan(ClusterFrameHeader.HeaderSize + 16);
        clusterHeader.WriteTo(span.Slice(0, ClusterFrameHeader.HeaderSize));
        span.Slice(ClusterFrameHeader.HeaderSize, 16).Clear();
        pipe.Writer.Advance(ClusterFrameHeader.HeaderSize + 16);
        await pipe.Writer.FlushAsync();

        // ReadActivationFrameAsync must reject the frame because payload < 32
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await DistributedPipelineCoordinator.ReadActivationFrameAsync(pipe.Reader);
        });
    }

    [Fact]
    public async Task WireFraming_PipeCompletedPrematurely_ThrowsInvalidOperationException()
    {
        var pipe = new Pipe();

        // Write a ClusterFrameHeader indicating a 1024-byte payload, but complete writer immediately without payload
        var clusterHeader = new ClusterFrameHeader(
            messageType: ClusterMessageType.StageDispatch,
            correlationId: 1,
            payloadLength: 1024,
            flags: ClusterFrameFlags.None);

        var span = pipe.Writer.GetSpan(ClusterFrameHeader.HeaderSize);
        clusterHeader.WriteTo(span.Slice(0, ClusterFrameHeader.HeaderSize));
        pipe.Writer.Advance(ClusterFrameHeader.HeaderSize);
        await pipe.Writer.FlushAsync();
        await pipe.Writer.CompleteAsync();

        // Reader must throw because pipe completed before full frame arrived
        await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await DistributedPipelineCoordinator.ReadActivationFrameAsync(pipe.Reader);
        });
    }

    #endregion
}
