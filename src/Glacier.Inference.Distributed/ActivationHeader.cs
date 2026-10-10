namespace Glacier.Inference.Distributed;

using System;
using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// Bitflags representing activation metadata and routing hints.
/// </summary>
[Flags]
public enum ActivationFlags : uint
{
    /// <summary>Default flags: standard activation tensor.</summary>
    None = 0,

    /// <summary>Bit 0: Indicates that the stage should accumulate into an existing residual tensor.</summary>
    ResidualAccumulate = 1 << 0,

    /// <summary>Bit 1: Indicates that this activation is exiting the final transformer layer (heading to LM head).</summary>
    FinalLayer = 1 << 1,

    /// <summary>Bit 2: Indicates that the activation vector is compressed (e.g. FP8 or INT8 quantized).</summary>
    Compressed = 1 << 2,

    /// <summary>Bit 3: Indicates that this is a shared expert activation applied to all tokens.</summary>
    SharedExpert = 1 << 3,

    /// <summary>Bit 4: Request immediate acknowledgment upon receipt.</summary>
    AcknowledgeRequired = 1 << 4,
}

/// <summary>
/// Exactly 32-byte sequentially packed wire header for distributed activation handoff between inference pipeline stages.
/// Encoded in strict Little-Endian format with 64-bit natural machine alignment.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 32)]
public readonly struct ActivationHeader : IEquatable<ActivationHeader>
{
    /// <summary>
    /// Magic constant identifying Glacier Activation frames: 0x47414354 ("GACT").
    /// </summary>
    public const uint ActivationMagic = 0x47414354;

    /// <summary>
    /// Size of the header in bytes (exactly 32 bytes).
    /// </summary>
    public const int HeaderSize = 32;

    public readonly uint Magic;          // Offset 0..3   (4 bytes)
    public readonly ulong SequenceId;    // Offset 4..11  (8 bytes)
    public readonly int TokenIndex;      // Offset 12..15 (4 bytes)
    public readonly int LayerIndex;      // Offset 16..19 (4 bytes)
    public readonly int HiddenDim;       // Offset 20..23 (4 bytes)
    public readonly int ExpertId;        // Offset 24..27 (4 bytes)
    public readonly uint Flags;          // Offset 28..31 (4 bytes)

    /// <summary>
    /// Constructs a new ActivationHeader with raw uint flags.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ActivationHeader(
        ulong sequenceId,
        int tokenIndex,
        int layerIndex,
        int hiddenDim,
        int expertId,
        uint flags = 0,
        uint magic = ActivationMagic)
    {
        Magic = magic;
        SequenceId = sequenceId;
        TokenIndex = tokenIndex;
        LayerIndex = layerIndex;
        HiddenDim = hiddenDim;
        ExpertId = expertId;
        Flags = flags;
    }

    /// <summary>
    /// Constructs a new ActivationHeader with typed <see cref="ActivationFlags"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ActivationHeader(
        ulong sequenceId,
        int tokenIndex,
        int layerIndex,
        int hiddenDim,
        int expertId,
        ActivationFlags flags)
        : this(sequenceId, tokenIndex, layerIndex, hiddenDim, expertId, (uint)flags, ActivationMagic)
    {
    }

    /// <summary>
    /// Gets the typed flags bitmask.
    /// </summary>
    public ActivationFlags TypedFlags => (ActivationFlags)Flags;

    /// <summary>
    /// Writes the header to a destination byte span in Little-Endian format.
    /// </summary>
    /// <param name="destination">Destination span of at least 32 bytes.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void WriteTo(Span<byte> destination)
    {
        if (destination.Length < HeaderSize)
        {
            ThrowDestinationTooSmall(destination.Length);
        }

        BinaryPrimitives.WriteUInt32LittleEndian(destination.Slice(0, 4), Magic);
        BinaryPrimitives.WriteUInt64LittleEndian(destination.Slice(4, 8), SequenceId);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(12, 4), TokenIndex);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(16, 4), LayerIndex);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(20, 4), HiddenDim);
        BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(24, 4), ExpertId);
        BinaryPrimitives.WriteUInt32LittleEndian(destination.Slice(28, 4), Flags);
    }

    /// <summary>
    /// Writes the header to an <see cref="IBufferWriter{T}"/> and advances by 32 bytes.
    /// </summary>
    /// <param name="writer">Target buffer writer.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void WriteTo(IBufferWriter<byte> writer)
    {
        ArgumentNullException.ThrowIfNull(writer);
        Span<byte> destination = writer.GetSpan(HeaderSize);
        WriteTo(destination);
        writer.Advance(HeaderSize);
    }

    /// <summary>
    /// Deserializes an ActivationHeader from a 32-byte source span.
    /// </summary>
    /// <param name="source">Input span containing at least 32 bytes.</param>
    /// <returns>Deserialized <see cref="ActivationHeader"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ActivationHeader ReadFrom(ReadOnlySpan<byte> source)
    {
        if (source.Length < HeaderSize)
        {
            ThrowSourceTooSmall(source.Length);
        }

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(0, 4));
        if (magic != ActivationMagic)
        {
            ThrowInvalidMagic(magic);
        }

        ulong seqId = BinaryPrimitives.ReadUInt64LittleEndian(source.Slice(4, 8));
        int tokenIdx = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(12, 4));
        int layerIdx = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(16, 4));
        int hiddenDim = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(20, 4));
        int expertId = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(24, 4));
        uint flags = BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(28, 4));

        return new ActivationHeader(seqId, tokenIdx, layerIdx, hiddenDim, expertId, flags, magic);
    }

    /// <summary>
    /// Attempts to read an ActivationHeader from a span without throwing on invalid magic or insufficient length.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool TryRead(ReadOnlySpan<byte> source, out ActivationHeader header)
    {
        if (source.Length < HeaderSize)
        {
            header = default;
            return false;
        }

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(0, 4));
        if (magic != ActivationMagic)
        {
            header = default;
            return false;
        }

        ulong seqId = BinaryPrimitives.ReadUInt64LittleEndian(source.Slice(4, 8));
        int tokenIdx = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(12, 4));
        int layerIdx = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(16, 4));
        int hiddenDim = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(20, 4));
        int expertId = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(24, 4));
        uint flags = BinaryPrimitives.ReadUInt32LittleEndian(source.Slice(28, 4));

        header = new ActivationHeader(seqId, tokenIdx, layerIdx, hiddenDim, expertId, flags, magic);
        return true;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowDestinationTooSmall(int actualLength) =>
        throw new ArgumentException($"Destination span length ({actualLength}) is smaller than required HeaderSize ({HeaderSize}).", nameof(actualLength));

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowSourceTooSmall(int actualLength) =>
        throw new ArgumentException($"Source span length ({actualLength}) is smaller than required HeaderSize ({HeaderSize}).", nameof(actualLength));

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ThrowInvalidMagic(uint actualMagic) =>
        throw new InvalidOperationException($"Invalid Glacier Activation frame magic: 0x{actualMagic:X8}. Expected: 0x{ActivationMagic:X8}");

    public bool Equals(ActivationHeader other) =>
        Magic == other.Magic &&
        SequenceId == other.SequenceId &&
        TokenIndex == other.TokenIndex &&
        LayerIndex == other.LayerIndex &&
        HiddenDim == other.HiddenDim &&
        ExpertId == other.ExpertId &&
        Flags == other.Flags;

    public override bool Equals(object? obj) => obj is ActivationHeader other && Equals(other);

    public override int GetHashCode() =>
        HashCode.Combine(Magic, SequenceId, TokenIndex, LayerIndex, HiddenDim, ExpertId, Flags);

    public static bool operator ==(ActivationHeader left, ActivationHeader right) => left.Equals(right);
    public static bool operator !=(ActivationHeader left, ActivationHeader right) => !left.Equals(right);

    public override string ToString() =>
        $"ActivationHeader [Seq={SequenceId}, Tok={TokenIndex}, Layer={LayerIndex}, Dim={HiddenDim}, Exp={ExpertId}, Flags=0x{Flags:X2}]";
}
