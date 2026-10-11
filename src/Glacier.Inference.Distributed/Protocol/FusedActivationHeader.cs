namespace Glacier.Inference.Distributed;

using System;
using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Cluster.Protocol;

/// <summary>
/// Wire precision modes supported by the dynamic cluster transport protocol.
/// </summary>
public enum WirePrecisionMode : byte
{
    FP32 = 0,
    FP16 = 1,
    INT8_Symmetric = 2,
    FP8_E4M3 = 3
}

/// <summary>
/// Bitflags for fused activation transport frame.
/// </summary>
[Flags]
public enum FusedActivationFlags : byte
{
    None = 0,
    IsSpeculative = 1 << 0,
    IsPrefillChunk = 1 << 1,
    HasScale = 1 << 2,
    FinalLayer = 1 << 3
}

/// <summary>
/// Exactly 32-byte sequentially packed wire header for fused activations.
/// When combined with ClusterFrameHeader (32B), produces a 64-byte naturally aligned header.
/// Encoded in strict Little-Endian format.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 32)]
public readonly struct FusedActivationHeader : IEquatable<FusedActivationHeader>
{
    public const uint FusedMagic = 0x47414354; // "GACT"
    public const int HeaderSize = 32;

    public readonly uint Magic;          // 0..3 (4B)
    public readonly ulong SequenceId;    // 4..11 (8B)
    public readonly int TokenIndex;      // 12..15 (4B)
    public readonly int LayerIndex;      // 16..19 (4B)
    public readonly int HiddenDim;       // 20..23 (4B)
    public readonly byte PrecisionMode;  // 24 (1B)
    public readonly byte Flags;          // 25 (1B)
    public readonly ushort BatchTokens;  // 26..27 (2B) - token count for prefill chunk batching, or 0
    public readonly float Scale;         // 28..31 (4B)

    public ushort Reserved => BatchTokens;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public FusedActivationHeader(
        ulong sequenceId,
        int tokenIndex,
        int layerIndex,
        int hiddenDim,
        WirePrecisionMode precisionMode,
        FusedActivationFlags flags = FusedActivationFlags.None,
        float scale = 1.0f,
        ushort batchTokens = 0)
    {
        Magic = FusedMagic;
        SequenceId = sequenceId;
        TokenIndex = tokenIndex;
        LayerIndex = layerIndex;
        HiddenDim = hiddenDim;
        PrecisionMode = (byte)precisionMode;
        Flags = (byte)flags;
        BatchTokens = batchTokens;
        Scale = scale;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public FusedActivationHeader(
        ulong sequenceId,
        int tokenIndex,
        int layerIndex,
        int hiddenDim,
        byte precisionMode,
        byte flags = 0,
        float scale = 1.0f,
        ushort batchTokens = 0)
    {
        Magic = FusedMagic;
        SequenceId = sequenceId;
        TokenIndex = tokenIndex;
        LayerIndex = layerIndex;
        HiddenDim = hiddenDim;
        PrecisionMode = precisionMode;
        Flags = flags;
        BatchTokens = batchTokens;
        Scale = scale;
    }

    public WirePrecisionMode TypedPrecision => (WirePrecisionMode)PrecisionMode;
    public FusedActivationFlags TypedFlags => (FusedActivationFlags)Flags;

    /// <summary>
    /// Computes the exact byte length of the activation payload following this header.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetPayloadByteCount()
    {
        int tokens = BatchTokens > 0 ? BatchTokens : 1;
        return TypedPrecision switch
        {
            WirePrecisionMode.FP32 => HiddenDim * tokens * 4,
            WirePrecisionMode.FP16 => HiddenDim * tokens * 2,
            WirePrecisionMode.INT8_Symmetric => (HiddenDim + 4) * tokens,
            WirePrecisionMode.FP8_E4M3 => (HiddenDim + 4) * tokens,
            _ => HiddenDim * tokens * 2
        };
    }

    /// <summary>
    /// Returns a copy of this header with the specified precision mode and scale factor.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public FusedActivationHeader WithPrecision(WirePrecisionMode precisionMode, float scale = 1.0f) =>
        new FusedActivationHeader(SequenceId, TokenIndex, LayerIndex, HiddenDim, precisionMode, (FusedActivationFlags)Flags, scale, BatchTokens);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void WriteTo(Span<byte> destination)
    {
        if (destination.Length < HeaderSize)
            throw new ArgumentException("Destination too small for FusedActivationHeader", nameof(destination));

        BinaryPrimitives.WriteUInt32LittleEndian(destination.Slice(0, 4), Magic);
        BinaryPrimitives.WriteUInt64LittleEndian(destination.Slice(4, 8), SequenceId);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(12, 4), TokenIndex);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(16, 4), LayerIndex);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(20, 4), HiddenDim);
        destination[24] = PrecisionMode;
        destination[25] = Flags;
        BinaryPrimitives.WriteUInt16LittleEndian(destination.Slice(26, 2), Reserved);
        BinaryPrimitives.WriteSingleLittleEndian(destination.Slice(28, 4), Scale);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FusedActivationHeader ReadFrom(ReadOnlySpan<byte> source)
    {
        if (source.Length < HeaderSize)
            throw new ArgumentException("Source too small for FusedActivationHeader", nameof(source));

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(0, 4));
        if (magic != FusedMagic)
            throw new InvalidOperationException($"Invalid FusedActivationHeader magic: 0x{magic:X8}, expected 0x{FusedMagic:X8}");

        ulong seq = BinaryPrimitives.ReadUInt64LittleEndian(source.Slice(4, 8));
        int tok = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(12, 4));
        int layer = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(16, 4));
        int dim = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(20, 4));
        byte prec = source[24];
        byte flags = source[25];
        ushort batchTokens = BinaryPrimitives.ReadUInt16LittleEndian(source.Slice(26, 2));
        float scale = BinaryPrimitives.ReadSingleLittleEndian(source.Slice(28, 4));

        return new FusedActivationHeader(seq, tok, layer, dim, (WirePrecisionMode)prec, (FusedActivationFlags)flags, scale, batchTokens);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool TryRead(ReadOnlySpan<byte> source, out FusedActivationHeader header)
    {
        if (source.Length < HeaderSize)
        {
            header = default;
            return false;
        }

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(0, 4));
        if (magic != FusedMagic)
        {
            header = default;
            return false;
        }

        ulong seq = BinaryPrimitives.ReadUInt64LittleEndian(source.Slice(4, 8));
        int tok = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(12, 4));
        int layer = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(16, 4));
        int dim = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(20, 4));
        byte prec = source[24];
        byte flags = source[25];
        ushort batchTokens = BinaryPrimitives.ReadUInt16LittleEndian(source.Slice(26, 2));
        float scale = BinaryPrimitives.ReadSingleLittleEndian(source.Slice(28, 4));

        header = new FusedActivationHeader(seq, tok, layer, dim, (WirePrecisionMode)prec, (FusedActivationFlags)flags, scale, batchTokens);
        return true;
    }

    public bool Equals(FusedActivationHeader other) =>
        Magic == other.Magic &&
        SequenceId == other.SequenceId &&
        TokenIndex == other.TokenIndex &&
        LayerIndex == other.LayerIndex &&
        HiddenDim == other.HiddenDim &&
        PrecisionMode == other.PrecisionMode &&
        Flags == other.Flags &&
        BatchTokens == other.BatchTokens &&
        Scale.Equals(other.Scale);

    public override bool Equals(object? obj) => obj is FusedActivationHeader other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(Magic, SequenceId, TokenIndex, LayerIndex, HiddenDim, PrecisionMode, Scale);

    public static bool operator ==(FusedActivationHeader left, FusedActivationHeader right) => left.Equals(right);
    public static bool operator !=(FusedActivationHeader left, FusedActivationHeader right) => !left.Equals(right);
}

/// <summary>
/// Exactly 32-byte activation result wire header returned from downstream stage to coordinator.
/// Encoded in strict Little-Endian format.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 32)]
public readonly struct FusedActivationResultHeader : IEquatable<FusedActivationResultHeader>
{
    public const uint ResultMagic = 0x47524553; // "GRES"
    public const int HeaderSize = 32;

    public readonly uint Magic;          // 0..3 (4B)
    public readonly int StepIndex;       // 4..7 (4B)
    public readonly int StageId;         // 8..11 (4B)
    public readonly double ComputeMs;    // 12..19 (8B)
    public readonly int NextTokenId;     // 20..23 (4B)
    public readonly float TopLogit;      // 24..27 (4B)
    public readonly byte PrecisionMode;  // 28 (1B)
    public readonly byte Flags;          // 29 (1B)
    public readonly ushort Reserved;     // 30..31 (2B)

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public FusedActivationResultHeader(
        int stepIndex,
        int stageId,
        double computeMs,
        int nextTokenId,
        float topLogit,
        WirePrecisionMode precisionMode = WirePrecisionMode.FP16,
        byte flags = 0)
    {
        Magic = ResultMagic;
        StepIndex = stepIndex;
        StageId = stageId;
        ComputeMs = computeMs;
        NextTokenId = nextTokenId;
        TopLogit = topLogit;
        PrecisionMode = (byte)precisionMode;
        Flags = flags;
        Reserved = 0;
    }

    public WirePrecisionMode TypedPrecision => (WirePrecisionMode)PrecisionMode;
    public FusedActivationFlags TypedFlags => (FusedActivationFlags)Flags;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void WriteTo(Span<byte> destination)
    {
        if (destination.Length < HeaderSize)
            throw new ArgumentException("Destination too small for FusedActivationResultHeader", nameof(destination));

        BinaryPrimitives.WriteUInt32LittleEndian(destination.Slice(0, 4), Magic);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(4, 4), StepIndex);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(8, 4), StageId);
        BinaryPrimitives.WriteDoubleLittleEndian(destination.Slice(12, 8), ComputeMs);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(20, 4), NextTokenId);
        BinaryPrimitives.WriteSingleLittleEndian(destination.Slice(24, 4), TopLogit);
        destination[28] = PrecisionMode;
        destination[29] = Flags;
        BinaryPrimitives.WriteUInt16LittleEndian(destination.Slice(30, 2), Reserved);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static FusedActivationResultHeader ReadFrom(ReadOnlySpan<byte> source)
    {
        if (source.Length < HeaderSize)
            throw new ArgumentException("Source too small for FusedActivationResultHeader", nameof(source));

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(0, 4));
        if (magic != ResultMagic)
            throw new InvalidOperationException($"Invalid FusedActivationResultHeader magic: 0x{magic:X8}, expected 0x{ResultMagic:X8}");

        int step = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(4, 4));
        int stage = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(8, 4));
        double computeMs = BinaryPrimitives.ReadDoubleLittleEndian(source.Slice(12, 8));
        int nextTok = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(20, 4));
        float topLogit = BinaryPrimitives.ReadSingleLittleEndian(source.Slice(24, 4));
        byte prec = source[28];
        byte flags = source[29];

        return new FusedActivationResultHeader(step, stage, computeMs, nextTok, topLogit, (WirePrecisionMode)prec, flags);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool TryRead(ReadOnlySpan<byte> source, out FusedActivationResultHeader header)
    {
        if (source.Length < HeaderSize)
        {
            header = default;
            return false;
        }

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(0, 4));
        if (magic != ResultMagic)
        {
            header = default;
            return false;
        }

        int step = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(4, 4));
        int stage = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(8, 4));
        double computeMs = BinaryPrimitives.ReadDoubleLittleEndian(source.Slice(12, 8));
        int nextTok = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(20, 4));
        float topLogit = BinaryPrimitives.ReadSingleLittleEndian(source.Slice(24, 4));
        byte prec = source[28];
        byte flags = source[29];

        header = new FusedActivationResultHeader(step, stage, computeMs, nextTok, topLogit, (WirePrecisionMode)prec, flags);
        return true;
    }

    public bool Equals(FusedActivationResultHeader other) =>
        Magic == other.Magic &&
        StepIndex == other.StepIndex &&
        StageId == other.StageId &&
        ComputeMs.Equals(other.ComputeMs) &&
        NextTokenId == other.NextTokenId &&
        TopLogit.Equals(other.TopLogit) &&
        PrecisionMode == other.PrecisionMode &&
        Flags == other.Flags;

    public override bool Equals(object? obj) => obj is FusedActivationResultHeader other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(Magic, StepIndex, StageId, ComputeMs, NextTokenId, TopLogit);

    public static bool operator ==(FusedActivationResultHeader left, FusedActivationResultHeader right) => left.Equals(right);
    public static bool operator !=(FusedActivationResultHeader left, FusedActivationResultHeader right) => !left.Equals(right);
}
