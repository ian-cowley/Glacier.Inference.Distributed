namespace Glacier.Inference.Tests.Distributed;

using System;
using System.Buffers.Binary;
using System.IO.Pipelines;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Inference.Distributed;
using Xunit;

public class ClusterInterconnectTests
{
    [Fact]
    public void ActivationHeader_SizeIsExactly32Bytes_AndLayoutSequential()
    {
        Assert.Equal(32, Unsafe.SizeOf<ActivationHeader>());
        Assert.Equal(32, Marshal.SizeOf<ActivationHeader>());
        Assert.Equal(32, ActivationHeader.HeaderSize);

        // Verify explicit offsets for 64-bit aligned Little-Endian wire packing
        Assert.Equal(0, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.Magic)));
        Assert.Equal(4, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.SequenceId)));
        Assert.Equal(12, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.TokenIndex)));
        Assert.Equal(16, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.LayerIndex)));
        Assert.Equal(20, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.HiddenDim)));
        Assert.Equal(24, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.ExpertId)));
        Assert.Equal(28, (int)Marshal.OffsetOf<ActivationHeader>(nameof(ActivationHeader.Flags)));
    }

    [Fact]
    public void ActivationHeader_Serialization_LittleEndianRoundtrip()
    {
        ulong expectedSeq = 0x1122334455667788UL;
        int expectedTok = 42;
        int expectedLayer = 15;
        int expectedDim = 4096;
        int expectedExp = 7;
        uint expectedFlags = (uint)(ActivationFlags.ResidualAccumulate | ActivationFlags.FinalLayer);

        var original = new ActivationHeader(
            sequenceId: expectedSeq,
            tokenIndex: expectedTok,
            layerIndex: expectedLayer,
            hiddenDim: expectedDim,
            expertId: expectedExp,
            flags: expectedFlags);

        byte[] buffer = new byte[32];
        original.WriteTo(buffer.AsSpan());

        // Verify Little-Endian byte pattern for Magic 0x47414354 ("GACT")
        Assert.Equal((byte)0x54, buffer[0]); // 'T'
        Assert.Equal((byte)0x43, buffer[1]); // 'C'
        Assert.Equal((byte)0x41, buffer[2]); // 'A'
        Assert.Equal((byte)0x47, buffer[3]); // 'G'

        // Verify SequenceId Little-Endian
        ulong parsedSeq = BinaryPrimitives.ReadUInt64LittleEndian(buffer.AsSpan(4, 8));
        Assert.Equal(expectedSeq, parsedSeq);

        // Deserialization
        var deserialized = ActivationHeader.ReadFrom(buffer.AsSpan());

        Assert.Equal(original, deserialized);
        Assert.Equal(ActivationHeader.ActivationMagic, deserialized.Magic);
        Assert.Equal(expectedSeq, deserialized.SequenceId);
        Assert.Equal(expectedTok, deserialized.TokenIndex);
        Assert.Equal(expectedLayer, deserialized.LayerIndex);
        Assert.Equal(expectedDim, deserialized.HiddenDim);
        Assert.Equal(expectedExp, deserialized.ExpertId);
        Assert.Equal(expectedFlags, deserialized.Flags);
        Assert.True(deserialized.TypedFlags.HasFlag(ActivationFlags.ResidualAccumulate));
        Assert.True(deserialized.TypedFlags.HasFlag(ActivationFlags.FinalLayer));
        Assert.False(deserialized.TypedFlags.HasFlag(ActivationFlags.Compressed));
    }

    [Fact]
    public void ActivationHeader_InvalidMagic_ThrowsInvalidOperationException()
    {
        byte[] buffer = new byte[32];
        BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(0, 4), 0xDEADBEEF);

        Assert.Throws<InvalidOperationException>(() => ActivationHeader.ReadFrom(buffer.AsSpan()));
        Assert.False(ActivationHeader.TryRead(buffer.AsSpan(), out _));
    }

    [Fact]
    public void ActivationHeader_SpanTooSmall_ThrowsArgumentException()
    {
        byte[] shortBuffer = new byte[31];
        var header = new ActivationHeader(1, 0, 0, 4096, 0);

        Assert.Throws<ArgumentException>(() => header.WriteTo(shortBuffer.AsSpan()));
        Assert.Throws<ArgumentException>(() => ActivationHeader.ReadFrom(shortBuffer.AsSpan()));
        Assert.False(ActivationHeader.TryRead(shortBuffer.AsSpan(), out _));
    }

    [Fact]
    public void SubnetConfinement_AllowsFleetAndLoopback_RejectsExternal()
    {
        // Loopback
        Assert.True(ClusterNetworkTransportHarness.IsSubnetAllowed("127.0.0.1"));
        Assert.True(ClusterNetworkTransportHarness.IsSubnetAllowed("localhost"));

        // Fleet endpoints on 192.168.1.0/24
        Assert.True(ClusterNetworkTransportHarness.IsSubnetAllowed("192.168.1.111")); // Host Laptop
        Assert.True(ClusterNetworkTransportHarness.IsSubnetAllowed("192.168.1.108")); // Beast
        Assert.True(ClusterNetworkTransportHarness.IsSubnetAllowed("192.168.1.200")); // Linux
        Assert.True(ClusterNetworkTransportHarness.IsSubnetAllowed("192.168.1.101")); // MinisForum64

        // External / unauthorized subnets must fail
        Assert.False(ClusterNetworkTransportHarness.IsSubnetAllowed("8.8.8.8"));
        Assert.False(ClusterNetworkTransportHarness.IsSubnetAllowed("10.0.0.1"));
        Assert.False(ClusterNetworkTransportHarness.IsSubnetAllowed("172.16.0.1"));
        Assert.False(ClusterNetworkTransportHarness.IsSubnetAllowed("192.168.2.1"));

        Assert.Throws<SecurityException>(() => ClusterNetworkTransportHarness.ValidateSubnetOrThrow("8.8.8.8"));
    }

    [Fact]
    public void SerializationOverhead_Microbenchmark_IsZeroAllocation_AndSubMicrosecond()
    {
        var result = ClusterNetworkTransportHarness.MeasureSerializationOverhead(iterations: 10_000, hiddenDim: 4096);

        Assert.NotNull(result);
        Assert.Equal(10_000, result.TotalFrames);
        Assert.Equal(16_384, result.ActivationBytes);
        Assert.Equal(0, result.GcAllocatedBytes);
        Assert.True(result.SerializationOverheadNs < 2000.0, $"Serialization overhead was {result.SerializationOverheadNs:F1} ns (expected <2000 ns).");
    }

    [Fact]
    public async Task InProcess_DuplexPipe_StreamingPingPong_Succeeds()
    {
        var result = await ClusterNetworkTransportHarness.RunInProcessBenchmarkAsync(
            vectorDimensions: 4096,
            frameCount: 200);

        Assert.NotNull(result);
        Assert.Equal(200, result.TotalFrames);
        Assert.Equal(16_384, result.ActivationBytes);
        Assert.True(result.ThroughputMBs > 0);
        Assert.True(result.MeanLatencyUs < 1_000.0, $"Mean latency was {result.MeanLatencyUs:F1} µs.");
    }

    [Fact]
    public async Task Loopback_Socket_StreamingPingPong_Succeeds()
    {
        var result = await ClusterNetworkTransportHarness.RunLoopbackSocketBenchmarkAsync(
            port: 0,
            vectorDimensions: 4096,
            frameCount: 100);

        Assert.NotNull(result);
        Assert.Equal(100, result.TotalFrames);
        Assert.Equal(16_384, result.ActivationBytes);
        Assert.True(result.ThroughputMBs > 0);
        Assert.True(result.MeanLatencyUs < 2_000.0, $"Mean latency was {result.MeanLatencyUs:F1} µs.");
    }
}
