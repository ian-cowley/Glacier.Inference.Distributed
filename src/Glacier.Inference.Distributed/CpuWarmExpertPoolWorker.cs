namespace Glacier.Inference.Distributed;

using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Shuffle;
using Glacier.Inference.Gguf;
using Glacier.Inference.Model;
using Glacier.Inference.Quant;

/// <summary>
/// High-performance CPU Warm Expert Pool stage worker executing sparse MoE SwiGLU computation
/// directly over system RAM resident weights using AVX-512 and AVX2 SIMD instructions.
/// Eliminates PCIe host-to-device weight paging bottlenecks by computing activations in host memory.
/// </summary>
public sealed unsafe class CpuWarmExpertPoolWorker : IInferenceStageWorker
{
    private readonly LayerWeights[]? _layers;
    private readonly int _dim;
    private readonly int _expertFfnDim;
    private readonly int _numExperts;
    private readonly GgufType _weightType;

    private readonly byte* _standaloneGateWeight;
    private readonly byte* _standaloneUpWeight;
    private readonly byte* _standaloneDownWeight;
    private readonly bool _ownsStandaloneMemory;
    private readonly bool _multithreaded;

    private int _disposed;

    public int StageRank { get; }
    public string Name { get; }

    /// <summary>
    /// Gets the hidden dimension (D) of activation vectors processed by this worker.
    /// </summary>
    public int HiddenDim => _dim;

    /// <summary>
    /// Gets the intermediate feed-forward dimension (d_ffn) of expert layers.
    /// </summary>
    public int ExpertFfnDim => _expertFfnDim;

    /// <summary>
    /// Gets the number of resident experts in this pool.
    /// </summary>
    public int NumExperts => _numExperts;

    /// <summary>
    /// Constructs a CpuWarmExpertPoolWorker backed by real transformer layer weights from a loaded GGUF model.
    /// </summary>
    public CpuWarmExpertPoolWorker(
        int stageRank,
        LayerWeights[] layers,
        int dim,
        int expertFfnDim,
        string name = "CPU Warm Expert Pool (AVX-512 / Zen 5)",
        bool multithreaded = true)
    {
        ArgumentNullException.ThrowIfNull(layers);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(expertFfnDim);

        StageRank = stageRank;
        _layers = layers;
        _dim = dim;
        _expertFfnDim = expertFfnDim;
        _numExperts = layers.Length > 0 && layers[0].IsMoe ? 8 : 1;
        _weightType = layers.Length > 0 ? layers[0].FfnGateExpsType : GgufType.F32;
        _standaloneGateWeight = null;
        _standaloneUpWeight = null;
        _standaloneDownWeight = null;
        _ownsStandaloneMemory = false;
        _multithreaded = multithreaded;
        Name = name;
    }

    /// <summary>
    /// Constructs a CpuWarmExpertPoolWorker with standalone unmanaged expert weights.
    /// Used for benchmarking, testing, or distributed partitions where expert blocks are pinned in RAM.
    /// </summary>
    public CpuWarmExpertPoolWorker(
        int stageRank,
        int dim,
        int expertFfnDim,
        int numExperts,
        GgufType weightType,
        byte* gateWeight,
        byte* upWeight,
        byte* downWeight,
        bool ownsMemory = false,
        string name = "CPU Warm Expert Pool (Unmanaged)",
        bool multithreaded = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(expertFfnDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numExperts);

        StageRank = stageRank;
        _layers = null;
        _dim = dim;
        _expertFfnDim = expertFfnDim;
        _numExperts = numExperts;
        _weightType = weightType;
        _standaloneGateWeight = gateWeight;
        _standaloneUpWeight = upWeight;
        _standaloneDownWeight = downWeight;
        _ownsStandaloneMemory = ownsMemory;
        _multithreaded = multithreaded;
        Name = name;
    }

    /// <summary>
    /// Factory method creating a test/benchmark worker populated with deterministic synthetic weights.
    /// Fully zero-copy and Native AOT compatible.
    /// </summary>
    public static CpuWarmExpertPoolWorker CreateSynthetic(
        int stageRank,
        int dim,
        int expertFfnDim,
        int numExperts = 8,
        GgufType weightType = GgufType.F32,
        string name = "CPU Warm Expert Pool (Synthetic)",
        bool multithreaded = true)
    {
        long gateRowBytes = GgufTypes.GetRowBytes(weightType, dim);
        long gateBytes = (long)numExperts * expertFfnDim * gateRowBytes;

        long upRowBytes = GgufTypes.GetRowBytes(weightType, dim);
        long upBytes = (long)numExperts * expertFfnDim * upRowBytes;

        long downRowBytes = GgufTypes.GetRowBytes(weightType, expertFfnDim);
        long downBytes = (long)numExperts * dim * downRowBytes;

        byte* gateWeight = (byte*)NativeMemory.AlignedAlloc((nuint)gateBytes, 64);
        byte* upWeight = (byte*)NativeMemory.AlignedAlloc((nuint)upBytes, 64);
        byte* downWeight = (byte*)NativeMemory.AlignedAlloc((nuint)downBytes, 64);

        // Populate deterministic non-zero weights for realistic SIMD execution
        if (weightType == GgufType.F32)
        {
            float* fGate = (float*)gateWeight;
            float* fUp = (float*)upWeight;
            float* fDown = (float*)downWeight;

            long totalGateFloats = (long)numExperts * expertFfnDim * dim;
            long totalDownFloats = (long)numExperts * dim * expertFfnDim;

            float scaleIn = 1.0f / MathF.Sqrt(dim);
            float scaleOut = 1.0f / MathF.Sqrt(expertFfnDim);

            for (long i = 0; i < totalGateFloats; i++)
            {
                fGate[i] = ((i % 17) - 8) * scaleIn * 0.1f;
                fUp[i] = ((i % 13) - 6) * scaleIn * 0.1f;
            }

            for (long i = 0; i < totalDownFloats; i++)
            {
                fDown[i] = ((i % 19) - 9) * scaleOut * 0.1f;
            }
        }
        else
        {
            NativeMemory.Clear(gateWeight, (nuint)gateBytes);
            NativeMemory.Clear(upWeight, (nuint)upBytes);
            NativeMemory.Clear(downWeight, (nuint)downBytes);
        }

        return new CpuWarmExpertPoolWorker(
            stageRank: stageRank,
            dim: dim,
            expertFfnDim: expertFfnDim,
            numExperts: numExperts,
            weightType: weightType,
            gateWeight: gateWeight,
            upWeight: upWeight,
            downWeight: downWeight,
            ownsMemory: true,
            name: name,
            multithreaded: multithreaded);
    }

    /// <inheritdoc />
    public ValueTask<RefCountedPartitionBuffer> ProcessActivationAsync(
        ActivationHeader header,
        ReadOnlyMemory<byte> inputActivation,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var result = ProcessActivation(header, inputActivation.Span);
        return ValueTask.FromResult(result);
    }

    /// <inheritdoc />
    public RefCountedPartitionBuffer ProcessActivation(
        ActivationHeader header,
        ReadOnlySpan<byte> inputActivation)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);

        int expectedBytes = _dim * sizeof(float);
        if (inputActivation.Length < expectedBytes)
        {
            throw new ArgumentException(
                $"Input activation size ({inputActivation.Length} bytes) is smaller than required hidden dimension ({expectedBytes} bytes for D={_dim}).",
                nameof(inputActivation));
        }

        // Allocate 64-byte aligned unmanaged output buffer for activation return
        var outputBuffer = RefCountedPartitionBuffer.Allocate(expectedBytes);

        fixed (byte* inPtr = inputActivation)
        {
            ProcessActivation(header, (float*)inPtr, (float*)outputBuffer.UnmanagedPointer);
        }

        return outputBuffer;
    }

    /// <summary>
    /// Processes an activation vector directly into caller-provided unmanaged float buffers.
    /// Completely zero GC heap allocation hot path.
    /// </summary>
    public void ProcessActivation(
        ActivationHeader header,
        float* inputActivation,
        float* outputActivation)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        ArgumentNullException.ThrowIfNull(inputActivation);
        ArgumentNullException.ThrowIfNull(outputActivation);

        if (_layers != null)
        {
            int layerIdx = Math.Clamp(header.LayerIndex, 0, _layers.Length - 1);
            var layer = _layers[layerIdx];
            int expertIdx = Math.Max(0, header.ExpertId);

            long gateSlice = (long)_expertFfnDim * GgufTypes.GetRowBytes(layer.FfnGateExpsType, _dim);
            long upSlice = (long)_expertFfnDim * GgufTypes.GetRowBytes(layer.FfnUpExpsType, _dim);
            long downSlice = (long)_dim * GgufTypes.GetRowBytes(layer.FfnDownExpsType, _expertFfnDim);

            byte* expGate = layer.FfnGateExpsWeight + (long)expertIdx * gateSlice;
            byte* expUp = layer.FfnUpExpsWeight + (long)expertIdx * upSlice;
            byte* expDown = layer.FfnDownExpsWeight + (long)expertIdx * downSlice;

            ExecuteExpertSwiGlu(
                layer.FfnGateExpsType, expGate, layer.FfnGateExpsBias,
                layer.FfnUpExpsType, expUp, layer.FfnUpExpsBias,
                layer.FfnDownExpsType, expDown, layer.FfnDownExpsBias,
                expertIdx, inputActivation, outputActivation);
        }
        else
        {
            int expId = Math.Clamp(header.ExpertId, 0, _numExperts - 1);
            long gateSlice = (long)_expertFfnDim * GgufTypes.GetRowBytes(_weightType, _dim);
            long upSlice = (long)_expertFfnDim * GgufTypes.GetRowBytes(_weightType, _dim);
            long downSlice = (long)_dim * GgufTypes.GetRowBytes(_weightType, _expertFfnDim);

            byte* expGate = _standaloneGateWeight + (long)expId * gateSlice;
            byte* expUp = _standaloneUpWeight + (long)expId * upSlice;
            byte* expDown = _standaloneDownWeight + (long)expId * downSlice;

            ExecuteExpertSwiGlu(
                _weightType, expGate, null,
                _weightType, expUp, null,
                _weightType, expDown, null,
                expId, inputActivation, outputActivation);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void ExecuteExpertSwiGlu(
        GgufType gateType, byte* gateWeight, float* gateBias,
        GgufType upType, byte* upWeight, float* upBias,
        GgufType downType, byte* downWeight, float* downBias,
        int expertIdx, float* x, float* outX)
    {
        bool useStack = _expertFfnDim <= 16384;
        float* gate;
        float* up;
        float* act;

        if (useStack)
        {
            float* sGate = stackalloc float[_expertFfnDim];
            float* sUp = stackalloc float[_expertFfnDim];
            float* sAct = stackalloc float[_expertFfnDim];
            gate = sGate;
            up = sUp;
            act = sAct;
        }
        else
        {
            gate = (float*)NativeMemory.AlignedAlloc((nuint)(_expertFfnDim * sizeof(float)), 64);
            up = (float*)NativeMemory.AlignedAlloc((nuint)(_expertFfnDim * sizeof(float)), 64);
            act = (float*)NativeMemory.AlignedAlloc((nuint)(_expertFfnDim * sizeof(float)), 64);
        }

        try
        {
            // MatVecMul: gate = x * GateWeight
            MatVecMulVectorized(gateType, gateWeight, x, gate, _dim, _expertFfnDim);
            if (gateBias != null)
            {
                AddVector(gate, gateBias + (long)expertIdx * _expertFfnDim, _expertFfnDim);
            }

            // MatVecMul: up = x * UpWeight
            MatVecMulVectorized(upType, upWeight, x, up, _dim, _expertFfnDim);
            if (upBias != null)
            {
                AddVector(up, upBias + (long)expertIdx * _expertFfnDim, _expertFfnDim);
            }

            // SwiGLU: act = silu(gate) * up
            QuantKernels.SwiGLU(gate, up, act, _expertFfnDim);

            // MatVecMul: outX = act * DownWeight
            MatVecMulVectorized(downType, downWeight, act, outX, _expertFfnDim, _dim);
            if (downBias != null)
            {
                AddVector(outX, downBias + (long)expertIdx * _dim, _dim);
            }
        }
        finally
        {
            if (!useStack)
            {
                NativeMemory.AlignedFree(gate);
                NativeMemory.AlignedFree(up);
                NativeMemory.AlignedFree(act);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void MatVecMulVectorized(GgufType type, byte* weightData, float* x, float* y, int nCols, int nRows)
    {
        if (!_multithreaded && type == GgufType.F32)
        {
            float* wF32 = (float*)weightData;
            for (int r = 0; r < nRows; r++)
            {
                float sum = 0.0f;
                float* row = wF32 + (long)r * nCols;
                int c = 0;
                if (Vector256.IsHardwareAccelerated)
                {
                    int vLen = Vector256<float>.Count;
                    var vSum = Vector256<float>.Zero;
                    for (; c <= nCols - vLen; c += vLen)
                    {
                        var vx = Vector256.Load(x + c);
                        var vw = Vector256.Load(row + c);
                        vSum += vx * vw;
                    }
                    sum += Vector256.Sum(vSum);
                }
                for (; c < nCols; c++)
                {
                    sum += row[c] * x[c];
                }
                y[r] = sum;
            }
            return;
        }

        if (nCols % 32 == 0)
        {
            QuantKernels.MatVecMul(type, weightData, x, y, nCols, nRows);
        }
        else
        {
            // Fallback for non-multiple-of-32 test dimensions
            if (type == GgufType.F32)
            {
                float* wF32 = (float*)weightData;
                for (int r = 0; r < nRows; r++)
                {
                    float sum = 0.0f;
                    float* row = wF32 + (long)r * nCols;
                    for (int c = 0; c < nCols; c++)
                    {
                        sum += row[c] * x[c];
                    }
                    y[r] = sum;
                }
            }
            else
            {
                throw new NotSupportedException($"Dimension {nCols} is not a multiple of 32 for quantization {type}.");
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddVector(float* target, float* bias, int length)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            int vLen = Vector256<float>.Count;
            for (; i <= length - vLen; i += vLen)
            {
                var vt = Vector256.Load(target + i);
                var vb = Vector256.Load(bias + i);
                (vt + vb).Store(target + i);
            }
        }
        for (; i < length; i++)
        {
            target[i] += bias[i];
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;

        if (_ownsStandaloneMemory)
        {
            if (_standaloneGateWeight != null) NativeMemory.AlignedFree(_standaloneGateWeight);
            if (_standaloneUpWeight != null) NativeMemory.AlignedFree(_standaloneUpWeight);
            if (_standaloneDownWeight != null) NativeMemory.AlignedFree(_standaloneDownWeight);
        }
    }
}
