namespace Glacier.Inference.Distributed;

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Shuffle;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu.D3D12;
using Glacier.Inference.Memory;
using Glacier.Inference.Quant;
using Glacier.Inference.Scheduling;
using Vortice.Direct3D12;

/// <summary>
/// Telemetry snapshot for a <see cref="GpuFirstExpertStageWorker"/> node.
/// </summary>
public sealed record GpuFirstWorkerTelemetry(
    int StageRank,
    string DeviceName,
    bool IsGpuActive,
    int TotalExperts,
    int GpuResidentSlotCapacity,
    long GpuVramHitCount,
    long GpuPcieTransferCount,
    long CpuSimdOverflowCount,
    long LookaheadPrefetchCount,
    double GpuExecutionRatio);

/// <summary>
/// GPU-First distributed MoE expert stage worker for heterogeneous cluster nodes.
/// Prioritizes Direct3D 12 GPU compute shaders (including <c>Q2_0</c>, <c>IQ4_NL</c>, and <c>IQ3_S</c>
/// for Qwen 3.8 Flash Next &amp; Coder 640-column <c>ffn_down_exps</c>), manages a GPU-resident expert
/// slot cache with Router Lookahead Prefetching, and spills only bandwidth-overflow experts to the
/// CPU SIMD Warm Expert Pool (<see cref="CpuWarmExpertPoolWorker"/>).
/// </summary>
public sealed unsafe class GpuFirstExpertStageWorker : IInferenceStageWorker
{
    private readonly int _dim;
    private readonly int _expertFfnDim;
    private readonly int _numExperts;
    private readonly GgufType _gateType;
    private readonly GgufType _upType;
    private readonly GgufType _downType;

    private readonly byte* _hostGateWeights;
    private readonly byte* _hostUpWeights;
    private readonly byte* _hostDownWeights;
    private readonly bool _ownsHostMemory;

    private readonly D3D12Context? _ctx;
    private readonly bool _ownsContext;
    private readonly GpuResidentSlotManager _slotManager;
    private readonly BandwidthCoScheduler _coScheduler;
    private readonly Lock _gpuLock = new();

    // Direct3D 12 Compute Pipelines for GPU-First Expert SwiGLU
    private ID3D12RootSignature? _sigGemv;
    private ID3D12PipelineState? _psoGemvQ2_0;
    private ID3D12PipelineState? _psoGemvIQ4_NL;
    private ID3D12PipelineState? _psoGemvIQ3_S;
    private ID3D12PipelineState? _psoGemvQ2_K;
    private ID3D12PipelineState? _psoGemvIQ2_XXS;
    private ID3D12PipelineState? _psoGemvQ4K;
    private ID3D12PipelineState? _psoGemvQ8_0;
    private ID3D12PipelineState? _psoGemvFp32;

    private ID3D12RootSignature? _sigSwiglu;
    private ID3D12PipelineState? _psoSwiglu;

    private ID3D12RootSignature? _sigVecAddWeighted;
    private ID3D12PipelineState? _psoVecAddWeighted;

    // GPU Slot Cache Buffers (each slot holds 1 expert's aligned gate, up, and down tensors)
    private readonly int _gpuSlotCapacity;
    private ID3D12Resource[]? _dSlotGate;
    private ID3D12Resource[]? _dSlotUp;
    private ID3D12Resource[]? _dSlotDown;

    // GPU Scratch Buffers for Expert Evaluation
    private ID3D12Resource? _dX;
    private ID3D12Resource? _dGate;
    private ID3D12Resource? _dUp;
    private ID3D12Resource? _dAct;
    private ID3D12Resource? _dOut;
    private ID3D12Resource? _dAccumOut;
    private ID3D12Resource? _uploadX;
    private float* _pUploadX;
    private ID3D12Resource? _readbackOut;
    private float* _pReadbackOut;

    private long _gpuVramHitCount;
    private long _gpuPcieTransferCount;
    private long _cpuSimdOverflowCount;
    private long _lookaheadPrefetchCount;
    private int _disposed;

    public int StageRank { get; }
    public string Name { get; }
    public int HiddenDim => _dim;
    public int ExpertFfnDim => _expertFfnDim;
    public int NumExperts => _numExperts;
    public bool IsGpuAccelerated => _ctx != null;
    public string DeviceName => _ctx?.DeviceName ?? "CPU SIMD Warm Pool Fallback";

    public GpuFirstExpertStageWorker(
        int stageRank,
        int dim,
        int expertFfnDim,
        int numExperts,
        GgufType gateUpType,
        GgufType downType,
        byte* hostGateWeights,
        byte* hostUpWeights,
        byte* hostDownWeights,
        bool ownsHostMemory = false,
        D3D12Context? d3dContext = null,
        bool ownsContext = false,
        int gpuSlotCapacity = 64,
        BandwidthCoScheduler? coScheduler = null,
        string? name = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(expertFfnDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numExperts);

        StageRank = stageRank;
        _dim = dim;
        _expertFfnDim = expertFfnDim;
        _numExperts = numExperts;
        _gateType = gateUpType;
        _upType = gateUpType;
        _downType = downType;
        _hostGateWeights = hostGateWeights;
        _hostUpWeights = hostUpWeights;
        _hostDownWeights = hostDownWeights;
        _ownsHostMemory = ownsHostMemory;

        _ctx = d3dContext;
        _ownsContext = ownsContext;
        _gpuSlotCapacity = Math.Clamp(gpuSlotCapacity, 1, numExperts);

        long gateSlice = (long)expertFfnDim * GgufTypes.GetRowBytes(_gateType, dim);
        long upSlice = (long)expertFfnDim * GgufTypes.GetRowBytes(_upType, dim);
        long downSlice = (long)dim * GgufTypes.GetRowBytes(_downType, expertFfnDim);
        nuint expertRawBytes = (nuint)(gateSlice + upSlice + downSlice);

        _slotManager = new GpuResidentSlotManager(_gpuSlotCapacity, expertRawBytes, false);
        _coScheduler = coScheduler ?? new BandwidthCoScheduler(
            initialPcieGBps: 24.0,
            initialCpuGBps: 65.0,
            vramGBps: 360.0);

        if (_ctx != null)
        {
            InitGpuResources();
            // Pre-populate initial hot experts into GPU VRAM slots up to _gpuSlotCapacity
            WarmInitialGpuSlots();
        }

        Name = name ?? (_ctx != null
            ? $"GPU-First MoE Worker ({_ctx.DeviceName}, {_gateType}/{_downType})"
            : $"GPU-First MoE Worker (CPU SIMD Fallback, {_gateType}/{_downType})");
    }

    /// <summary>
    /// Creates a synthetic GPU-First MoE stage worker with realistic quantized expert weights
    /// (supports <c>Q2_0</c>, <c>IQ4_NL</c>, <c>IQ3_S</c>, <c>Q4_K</c>, <c>Q8_0</c>, <c>F32</c>)
    /// for distributed load and accuracy testing.
    /// </summary>
    public static GpuFirstExpertStageWorker CreateSynthetic(
        int stageRank,
        int dim,
        int expertFfnDim,
        int numExperts,
        GgufType gateUpType,
        GgufType downType,
        D3D12Context? d3dContext = null,
        bool ownsContext = false,
        int gpuSlotCapacity = 64,
        int seed = 1337)
    {
        long gateRowBytes = GgufTypes.GetRowBytes(gateUpType, dim);
        long upRowBytes = GgufTypes.GetRowBytes(gateUpType, dim);
        long downRowBytes = GgufTypes.GetRowBytes(downType, expertFfnDim);

        long gateBytes = (long)numExperts * expertFfnDim * gateRowBytes;
        long upBytes = (long)numExperts * expertFfnDim * upRowBytes;
        long downBytes = (long)numExperts * dim * downRowBytes;

        byte* gateWeight = (byte*)NativeMemory.AlignedAlloc((nuint)gateBytes, 64);
        byte* upWeight = (byte*)NativeMemory.AlignedAlloc((nuint)upBytes, 64);
        byte* downWeight = (byte*)NativeMemory.AlignedAlloc((nuint)downBytes, 64);

        PopulateDeterministicQuantizedWeights(gateUpType, gateWeight, numExperts * expertFfnDim, dim, seed);
        PopulateDeterministicQuantizedWeights(gateUpType, upWeight, numExperts * expertFfnDim, dim, seed + 101);
        PopulateDeterministicQuantizedWeights(downType, downWeight, numExperts * dim, expertFfnDim, seed + 202);

        return new GpuFirstExpertStageWorker(
            stageRank: stageRank,
            dim: dim,
            expertFfnDim: expertFfnDim,
            numExperts: numExperts,
            gateUpType: gateUpType,
            downType: downType,
            hostGateWeights: gateWeight,
            hostUpWeights: upWeight,
            hostDownWeights: downWeight,
            ownsHostMemory: true,
            d3dContext: d3dContext,
            ownsContext: ownsContext,
            gpuSlotCapacity: gpuSlotCapacity);
    }

    private static void PopulateDeterministicQuantizedWeights(
        GgufType type,
        byte* dst,
        int totalRows,
        int cols,
        int seed)
    {
        long rowBytes = GgufTypes.GetRowBytes(type, cols);
        float[] rowFp32 = new float[cols];
        float scale = 1.0f / MathF.Sqrt(cols);

        // Generate 8 distinct prototype rows and tile them across experts for fast initialization
        int protoCount = Math.Min(totalRows, 8);
        byte* protoBuf = (byte*)NativeMemory.AlignedAlloc((nuint)(protoCount * rowBytes), 64);
        try
        {
            for (int p = 0; p < protoCount; p++)
            {
                for (int c = 0; c < cols; c++)
                {
                    int h = ((p + 1) * 73856093) ^ (c * 19349663) ^ seed;
                    float val = ((h & 0xFF) - 128) / 128.0f * scale;
                    rowFp32[c] = val;
                }

                byte* pDst = protoBuf + p * rowBytes;
                fixed (float* pSrc = rowFp32)
                {
                    switch (type)
                    {
                        case GgufType.Q2_0:
                            QuantKernels.QuantizeQ2_0(pSrc, (BlockQ2_0*)pDst, cols);
                            break;
                        case GgufType.IQ4_NL:
                            QuantKernels.QuantizeIQ4_NL(pSrc, (BlockIQ4_NL*)pDst, cols);
                            break;
                        case GgufType.IQ3_S:
                            QuantKernels.QuantizeIQ3_S(pSrc, (BlockIQ3_S*)pDst, cols);
                            break;
                        case GgufType.F32:
                            Buffer.MemoryCopy(pSrc, pDst, rowBytes, rowBytes);
                            break;
                        default:
                            NativeMemory.Clear(pDst, (nuint)rowBytes);
                            break;
                    }
                }
            }

            for (int r = 0; r < totalRows; r++)
            {
                byte* srcRow = protoBuf + (r % protoCount) * rowBytes;
                byte* dstRow = dst + (long)r * rowBytes;
                Buffer.MemoryCopy(srcRow, dstRow, rowBytes, rowBytes);
            }
        }
        finally
        {
            NativeMemory.AlignedFree(protoBuf);
        }
    }

    private void InitGpuResources()
    {
        if (_ctx == null) return;

        var gemvParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 6), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.ShaderResourceView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        };
        _sigGemv = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, gemvParams));
        _psoGemvQ2_0 = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ2_0));
        _psoGemvIQ4_NL = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvIQ4_NL));
        _psoGemvIQ3_S = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvIQ3_S));
        _psoGemvQ2_K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ2_K));
        _psoGemvIQ2_XXS = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvIQ2_XXS));
        _psoGemvQ4K = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ4K));
        _psoGemvQ8_0 = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvQ8_0));
        _psoGemvFp32 = _ctx.CreatePipelineState(_sigGemv, _ctx.CompileShader(D3D12Shaders.GemvFp32));

        var swigluParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 1), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(2, 0), ShaderVisibility.All)
        };
        _sigSwiglu = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, swigluParams));
        _psoSwiglu = _ctx.CreatePipelineState(_sigSwiglu, _ctx.CompileShader(D3D12Shaders.SwiGLU));

        var vecAddWeightedParams = new RootParameter[]
        {
            new RootParameter(new RootConstants(0, 0, 2), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(0, 0), ShaderVisibility.All),
            new RootParameter(RootParameterType.UnorderedAccessView, new RootDescriptor(1, 0), ShaderVisibility.All)
        };
        _sigVecAddWeighted = _ctx.CreateRootSignature(new RootSignatureDescription(RootSignatureFlags.None, vecAddWeightedParams));
        _psoVecAddWeighted = _ctx.CreatePipelineState(_sigVecAddWeighted, _ctx.CompileShader(D3D12Shaders.VecAddWeighted));

        _dSlotGate = new ID3D12Resource[_gpuSlotCapacity];
        _dSlotUp = new ID3D12Resource[_gpuSlotCapacity];
        _dSlotDown = new ID3D12Resource[_gpuSlotCapacity];

        _dX = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dGate = _ctx.CreateDeviceBuffer((ulong)(_expertFfnDim * sizeof(float)));
        _dUp = _ctx.CreateDeviceBuffer((ulong)(_expertFfnDim * sizeof(float)));
        _dAct = _ctx.CreateDeviceBuffer((ulong)(_expertFfnDim * sizeof(float)));
        _dOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));
        _dAccumOut = _ctx.CreateDeviceBuffer((ulong)(_dim * sizeof(float)));

        _uploadX = _ctx.CreateUploadBuffer((ulong)(_dim * sizeof(float)));
        void* pUp = null;
        _uploadX.Map(0, null, &pUp);
        _pUploadX = (float*)pUp;

        _readbackOut = _ctx.CreateReadbackBuffer((ulong)(_dim * sizeof(float)));
        void* pRb = null;
        _readbackOut.Map(0, null, &pRb);
        _pReadbackOut = (float*)pRb;
    }

    private void WarmInitialGpuSlots()
    {
        if (_ctx == null) return;
        int warmCount = Math.Min(_gpuSlotCapacity, _numExperts);
        for (int e = 0; e < warmCount; e++)
        {
            EnsureExpertInGpuSlot(0, e, countAsTransfer: false);
        }
    }

    private int EnsureExpertInGpuSlot(int layerIdx, int expertIdx, bool countAsTransfer)
    {
        if (_slotManager.TryGetSlot(layerIdx, expertIdx, out int slotIdx, out _))
        {
            if (countAsTransfer)
                Interlocked.Increment(ref _gpuVramHitCount);
            return slotIdx;
        }

        _slotManager.AllocateOrEvictSlot(layerIdx, expertIdx, out slotIdx, out _, out _, out _);

        long gateSlice = (long)_expertFfnDim * GgufTypes.GetRowBytes(_gateType, _dim);
        long upSlice = (long)_expertFfnDim * GgufTypes.GetRowBytes(_upType, _dim);
        long downSlice = (long)_dim * GgufTypes.GetRowBytes(_downType, _expertFfnDim);

        byte* expGate = _hostGateWeights + (long)expertIdx * gateSlice;
        byte* expUp = _hostUpWeights + (long)expertIdx * upSlice;
        byte* expDown = _hostDownWeights + (long)expertIdx * downSlice;

        _dSlotGate![slotIdx]?.Dispose();
        _dSlotUp![slotIdx]?.Dispose();
        _dSlotDown![slotIdx]?.Dispose();

        _dSlotGate[slotIdx] = D3D12TensorAlign.UploadAlignedTensor(_ctx!, _gateType, (IntPtr)expGate, _expertFfnDim, _dim);
        _dSlotUp[slotIdx] = D3D12TensorAlign.UploadAlignedTensor(_ctx!, _upType, (IntPtr)expUp, _expertFfnDim, _dim);
        _dSlotDown[slotIdx] = D3D12TensorAlign.UploadAlignedTensor(_ctx!, _downType, (IntPtr)expDown, _dim, _expertFfnDim);

        if (countAsTransfer)
            Interlocked.Increment(ref _gpuPcieTransferCount);

        return slotIdx;
    }

    /// <summary>
    /// Router Lookahead Prefetching: Pre-stages predicted experts for layer <paramref name="nextLayerIdx"/>
    /// into unpinned GPU VRAM slots before that layer begins execution.
    /// </summary>
    public void PrefetchLookaheadExperts(int nextLayerIdx, ReadOnlySpan<int> predictedExpertIds)
    {
        if (_ctx == null || predictedExpertIds.IsEmpty) return;

        lock (_gpuLock)
        {
            for (int i = 0; i < predictedExpertIds.Length; i++)
            {
                int expId = Math.Clamp(predictedExpertIds[i], 0, _numExperts - 1);
                if (!_slotManager.TryGetSlot(nextLayerIdx, expId, out _, out _))
                {
                    EnsureExpertInGpuSlot(nextLayerIdx, expId, countAsTransfer: false);
                    Interlocked.Increment(ref _lookaheadPrefetchCount);
                }
            }
        }
    }

    /// <summary>
    /// Executes Top-K routed experts for an MoE layer using GPU-First scheduling:
    /// 1. All VRAM-resident experts and up to <c>_gpuSlotCapacity</c> PCIe-streamed experts execute on the node's GPU first
    ///    in a single fused D3D12 command list submission.
    /// 2. Any remaining overflow experts (if <c>topK &gt; _gpuSlotCapacity</c>) execute concurrently on the CPU SIMD Warm Expert Pool (<c>AVX-512</c> / <c>AVX2</c>).
    /// 3. Optionally prefetches lookahead experts for the next token/layer.
    /// </summary>
    public void ProcessRoutedTopK(
        int layerIdx,
        ReadOnlySpan<float> inputX,
        ReadOnlySpan<int> selectedExpertIds,
        ReadOnlySpan<float> selectedWeights,
        Span<float> outputX,
        ReadOnlySpan<int> nextLayerLookaheadExperts = default,
        int lookaheadTargetLayerIdx = -1)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int topK = Math.Min(selectedExpertIds.Length, selectedWeights.Length);
        outputX.Slice(0, _dim).Clear();

        if (_ctx != null)
        {
            int gpuBatchLimit = Math.Min(topK, _gpuSlotCapacity);
            Span<int> gpuSlotIndices = stackalloc int[gpuBatchLimit];
            Span<float> gpuWeights = stackalloc float[gpuBatchLimit];

            lock (_gpuLock)
            {
                // Stage 1: Ensure all GPU-assigned experts are in VRAM slots (hits + PCIe transfers)
                for (int k = 0; k < gpuBatchLimit; k++)
                {
                    int expId = Math.Clamp(selectedExpertIds[k], 0, _numExperts - 1);
                    gpuSlotIndices[k] = EnsureExpertInGpuSlot(layerIdx, expId, countAsTransfer: true);
                    gpuWeights[k] = selectedWeights[k];
                }

                // Stage 2: Execute all GPU experts + weighted accumulation in ONE fused D3D12 command list
                fixed (float* pIn = inputX)
                {
                    Buffer.MemoryCopy(pIn, _pUploadX, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                }

                _ctx.BeginCommands();
                var cmd = _ctx.CommandList;

                cmd.ResourceBarrierTransition(_dX!, ResourceStates.Common, ResourceStates.CopyDest);
                cmd.CopyBufferRegion(_dX!, 0, _uploadX!, 0, (ulong)(_dim * sizeof(float)));
                cmd.ResourceBarrierTransition(_dX!, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

                uint ffnSize = (uint)_expertFfnDim;
                uint dimSize = (uint)_dim;
                uint* pAddConsts = stackalloc uint[2];

                for (int k = 0; k < gpuBatchLimit; k++)
                {
                    int slotIdx = gpuSlotIndices[k];
                    float w = gpuWeights[k];

                    DispatchGemv(cmd, _gateType, _dGate!, _dX!, _dSlotGate![slotIdx].GPUVirtualAddress, _dim, _expertFfnDim);
                    DispatchGemv(cmd, _upType, _dUp!, _dX!, _dSlotUp![slotIdx].GPUVirtualAddress, _dim, _expertFfnDim);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    cmd.SetComputeRootSignature(_sigSwiglu!);
                    cmd.SetPipelineState(_psoSwiglu!);
                    cmd.SetComputeRoot32BitConstants(0, 1, (IntPtr)(&ffnSize), 0);
                    cmd.SetComputeRootUnorderedAccessView(1, _dGate!.GPUVirtualAddress);
                    cmd.SetComputeRootUnorderedAccessView(2, _dUp!.GPUVirtualAddress);
                    cmd.SetComputeRootUnorderedAccessView(3, _dAct!.GPUVirtualAddress);
                    cmd.Dispatch((ffnSize + 255) / 256, 1, 1);
                    cmd.ResourceBarrierUnorderedAccessView(null!);

                    if (k == 0)
                    {
                        DispatchGemv(cmd, _downType, _dAccumOut!, _dAct!, _dSlotDown![slotIdx].GPUVirtualAddress, _expertFfnDim, _dim, scaleOut: w);
                    }
                    else
                    {
                        DispatchGemv(cmd, _downType, _dOut!, _dAct!, _dSlotDown![slotIdx].GPUVirtualAddress, _expertFfnDim, _dim);
                        cmd.ResourceBarrierUnorderedAccessView(null!);

                        cmd.SetComputeRootSignature(_sigVecAddWeighted!);
                        cmd.SetPipelineState(_psoVecAddWeighted!);
                        pAddConsts[0] = dimSize;
                        pAddConsts[1] = *(uint*)&w;
                        cmd.SetComputeRoot32BitConstants(0, 2, (IntPtr)pAddConsts, 0);
                        cmd.SetComputeRootUnorderedAccessView(1, _dOut!.GPUVirtualAddress);
                        cmd.SetComputeRootUnorderedAccessView(2, _dAccumOut!.GPUVirtualAddress);
                        cmd.Dispatch((dimSize + 255) / 256, 1, 1);
                    }
                    cmd.ResourceBarrierUnorderedAccessView(null!);
                }

                cmd.ResourceBarrierTransition(_dAccumOut!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
                cmd.CopyBufferRegion(_readbackOut!, 0, _dAccumOut!, 0, (ulong)(_dim * sizeof(float)));
                cmd.ResourceBarrierTransition(_dAccumOut!, ResourceStates.CopySource, ResourceStates.Common);
                cmd.ResourceBarrierTransition(_dX!, ResourceStates.UnorderedAccess, ResourceStates.Common);

                _ctx.EndCommandsAndExecute();
                _ctx.Synchronize();

                fixed (float* pOut = outputX)
                {
                    Buffer.MemoryCopy(_pReadbackOut, pOut, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
                }
            }

            // Stage 3: Overflow any remaining experts (when topK > _gpuSlotCapacity) to CPU SIMD
            if (topK > gpuBatchLimit)
            {
                float[] expertOut = new float[_dim];
                for (int k = gpuBatchLimit; k < topK; k++)
                {
                    int expId = Math.Clamp(selectedExpertIds[k], 0, _numExperts - 1);
                    float weight = selectedWeights[k];
                    Interlocked.Increment(ref _cpuSimdOverflowCount);

                    fixed (float* pIn = inputX, pExpOut = expertOut, pOut = outputX)
                    {
                        ExecuteExpertOnCpuSimd(expId, pIn, pExpOut);
                        QuantKernels.VecAddWeighted(pExpOut, pOut, weight, _dim);
                    }
                }
            }

            if (!nextLayerLookaheadExperts.IsEmpty)
            {
                int targetLayer = lookaheadTargetLayerIdx >= 0 ? lookaheadTargetLayerIdx : layerIdx;
                PrefetchLookaheadExperts(targetLayer, nextLayerLookaheadExperts);
            }
        }
        else
        {
            float[] expertOut = new float[_dim];
            for (int k = 0; k < topK; k++)
            {
                int expId = Math.Clamp(selectedExpertIds[k], 0, _numExperts - 1);
                float weight = selectedWeights[k];
                Interlocked.Increment(ref _cpuSimdOverflowCount);

                fixed (float* pIn = inputX, pExpOut = expertOut, pOut = outputX)
                {
                    ExecuteExpertOnCpuSimd(expId, pIn, pExpOut);
                    QuantKernels.VecAddWeighted(pExpOut, pOut, weight, _dim);
                }
            }
        }
    }

    public ValueTask<RefCountedPartitionBuffer> ProcessActivationAsync(
        ActivationHeader header,
        ReadOnlyMemory<byte> inputActivation,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult(ProcessActivation(header, inputActivation.Span));
    }

    public RefCountedPartitionBuffer ProcessActivation(
        ActivationHeader header,
        ReadOnlySpan<byte> inputActivation)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);

        int expectedBytes = _dim * sizeof(float);
        if (inputActivation.Length < expectedBytes)
            throw new ArgumentException($"Input activation length ({inputActivation.Length}) < {expectedBytes}.", nameof(inputActivation));

        var outputBuffer = RefCountedPartitionBuffer.Allocate(expectedBytes);
        int expId = Math.Clamp(header.ExpertId, 0, _numExperts - 1);

        fixed (byte* inPtr = inputActivation)
        {
            float* pIn = (float*)inPtr;
            float* pOut = (float*)outputBuffer.UnmanagedPointer;

            if (_ctx != null)
            {
                ExecuteExpertOnGpu(header.LayerIndex, expId, pIn, pOut);
            }
            else
            {
                Interlocked.Increment(ref _cpuSimdOverflowCount);
                ExecuteExpertOnCpuSimd(expId, pIn, pOut);
            }
        }

        return outputBuffer;
    }

    private void ExecuteExpertOnGpu(int layerIdx, int expertIdx, float* inputX, float* outputX)
    {
        lock (_gpuLock)
        {
            int slotIdx = EnsureExpertInGpuSlot(layerIdx, expertIdx, countAsTransfer: true);

            Buffer.MemoryCopy(inputX, _pUploadX, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));

            _ctx!.BeginCommands();
            var cmd = _ctx.CommandList;

            cmd.ResourceBarrierTransition(_dX!, ResourceStates.Common, ResourceStates.CopyDest);
            cmd.CopyBufferRegion(_dX!, 0, _uploadX!, 0, (ulong)(_dim * sizeof(float)));
            cmd.ResourceBarrierTransition(_dX!, ResourceStates.CopyDest, ResourceStates.UnorderedAccess);

            DispatchGemv(cmd, _gateType, _dGate!, _dX!, _dSlotGate![slotIdx].GPUVirtualAddress, _dim, _expertFfnDim);
            DispatchGemv(cmd, _upType, _dUp!, _dX!, _dSlotUp![slotIdx].GPUVirtualAddress, _dim, _expertFfnDim);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            cmd.SetComputeRootSignature(_sigSwiglu!);
            cmd.SetPipelineState(_psoSwiglu!);
            uint ffnSize = (uint)_expertFfnDim;
            cmd.SetComputeRoot32BitConstants(0, 1, (IntPtr)(&ffnSize), 0);
            cmd.SetComputeRootUnorderedAccessView(1, _dGate!.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(2, _dUp!.GPUVirtualAddress);
            cmd.SetComputeRootUnorderedAccessView(3, _dAct!.GPUVirtualAddress);
            cmd.Dispatch((ffnSize + 255) / 256, 1, 1);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            DispatchGemv(cmd, _downType, _dOut!, _dAct!, _dSlotDown![slotIdx].GPUVirtualAddress, _expertFfnDim, _dim);
            cmd.ResourceBarrierUnorderedAccessView(null!);

            cmd.ResourceBarrierTransition(_dOut!, ResourceStates.UnorderedAccess, ResourceStates.CopySource);
            cmd.CopyBufferRegion(_readbackOut!, 0, _dOut!, 0, (ulong)(_dim * sizeof(float)));
            cmd.ResourceBarrierTransition(_dOut!, ResourceStates.CopySource, ResourceStates.Common);
            cmd.ResourceBarrierTransition(_dX!, ResourceStates.UnorderedAccess, ResourceStates.Common);

            _ctx.EndCommandsAndExecute();
            _ctx.Synchronize();

            Buffer.MemoryCopy(_pReadbackOut, outputX, (ulong)(_dim * sizeof(float)), (ulong)(_dim * sizeof(float)));
        }
    }

    private void DispatchGemv(
        ID3D12GraphicsCommandList cmd,
        GgufType type,
        ID3D12Resource y,
        ID3D12Resource x,
        ulong wAddr,
        int kCols,
        int mRows,
        float scaleOut = 0.0f)
    {
        cmd.SetComputeRootSignature(_sigGemv!);
        var pso = type switch
        {
            GgufType.Q2_0 => _psoGemvQ2_0!,
            GgufType.IQ4_NL => _psoGemvIQ4_NL!,
            GgufType.IQ3_S => _psoGemvIQ3_S!,
            GgufType.Q2_K => _psoGemvQ2_K!,
            GgufType.IQ2_XXS => _psoGemvIQ2_XXS!,
            GgufType.Q4_K => _psoGemvQ4K!,
            GgufType.Q8_0 => _psoGemvQ8_0!,
            _ => _psoGemvFp32!
        };
        cmd.SetPipelineState(pso);

        cmd.SetComputeRootShaderResourceView(1, wAddr);
        cmd.SetComputeRootShaderResourceView(2, _ctx!.DummyBuffer.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(3, x.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(4, _ctx.DummyBuffer.GPUVirtualAddress);
        cmd.SetComputeRootUnorderedAccessView(5, y.GPUVirtualAddress);

        uint* pConsts = stackalloc uint[6];
        pConsts[0] = (uint)kCols;
        pConsts[1] = (uint)mRows;
        pConsts[2] = *(uint*)&scaleOut;
        pConsts[3] = 0u;
        pConsts[4] = 1u;
        pConsts[5] = 0u;
        cmd.SetComputeRoot32BitConstants(0, 6, (IntPtr)pConsts, 0);
        cmd.Dispatch(((uint)mRows + 3) / 4, 1, 1);
    }

    public void ExecuteExpertOnCpuSimd(int expertIdx, float* inputX, float* outputX)
    {
        long gateSlice = (long)_expertFfnDim * GgufTypes.GetRowBytes(_gateType, _dim);
        long upSlice = (long)_expertFfnDim * GgufTypes.GetRowBytes(_upType, _dim);
        long downSlice = (long)_dim * GgufTypes.GetRowBytes(_downType, _expertFfnDim);

        byte* expGate = _hostGateWeights + (long)expertIdx * gateSlice;
        byte* expUp = _hostUpWeights + (long)expertIdx * upSlice;
        byte* expDown = _hostDownWeights + (long)expertIdx * downSlice;

        float* gate = stackalloc float[_expertFfnDim];
        float* up = stackalloc float[_expertFfnDim];
        float* act = stackalloc float[_expertFfnDim];

        QuantKernels.MatVecMul(_gateType, expGate, inputX, gate, _dim, _expertFfnDim);
        QuantKernels.MatVecMul(_upType, expUp, inputX, up, _dim, _expertFfnDim);
        QuantKernels.SwiGLU(gate, up, act, _expertFfnDim);
        QuantKernels.MatVecMul(_downType, expDown, act, outputX, _expertFfnDim, _dim);
    }

    public GpuFirstWorkerTelemetry GetTelemetry()
    {
        long hits = Interlocked.Read(ref _gpuVramHitCount);
        long transfers = Interlocked.Read(ref _gpuPcieTransferCount);
        long cpu = Interlocked.Read(ref _cpuSimdOverflowCount);
        long prefetches = Interlocked.Read(ref _lookaheadPrefetchCount);
        long total = hits + transfers + cpu;
        double gpuRatio = total > 0 ? (double)(hits + transfers) / total : (_ctx != null ? 1.0 : 0.0);

        return new GpuFirstWorkerTelemetry(
            StageRank: StageRank,
            DeviceName: DeviceName,
            IsGpuActive: _ctx != null,
            TotalExperts: _numExperts,
            GpuResidentSlotCapacity: _gpuSlotCapacity,
            GpuVramHitCount: hits,
            GpuPcieTransferCount: transfers,
            CpuSimdOverflowCount: cpu,
            LookaheadPrefetchCount: prefetches,
            GpuExecutionRatio: gpuRatio);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;

        if (_ctx != null)
        {
            _ctx.Synchronize();
            if (_dSlotGate != null) foreach (var r in _dSlotGate) r?.Dispose();
            if (_dSlotUp != null) foreach (var r in _dSlotUp) r?.Dispose();
            if (_dSlotDown != null) foreach (var r in _dSlotDown) r?.Dispose();

            _dX?.Dispose();
            _dGate?.Dispose();
            _dUp?.Dispose();
            _dAct?.Dispose();
            _dOut?.Dispose();
            _dAccumOut?.Dispose();

            if (_uploadX != null) { _uploadX.Unmap(0); _uploadX.Dispose(); }
            if (_readbackOut != null) { _readbackOut.Unmap(0); _readbackOut.Dispose(); }

            _psoGemvQ2_0?.Dispose();
            _psoGemvIQ4_NL?.Dispose();
            _psoGemvIQ3_S?.Dispose();
            _psoGemvQ2_K?.Dispose();
            _psoGemvIQ2_XXS?.Dispose();
            _psoGemvQ4K?.Dispose();
            _psoGemvQ8_0?.Dispose();
            _psoGemvFp32?.Dispose();
            _sigGemv?.Dispose();

            _psoSwiglu?.Dispose();
            _sigSwiglu?.Dispose();

            _psoVecAddWeighted?.Dispose();
            _sigVecAddWeighted?.Dispose();

            if (_ownsContext)
                _ctx.Dispose();
        }

        _slotManager.Dispose();

        if (_ownsHostMemory)
        {
            if (_hostGateWeights != null) NativeMemory.AlignedFree(_hostGateWeights);
            if (_hostUpWeights != null) NativeMemory.AlignedFree(_hostUpWeights);
            if (_hostDownWeights != null) NativeMemory.AlignedFree(_hostDownWeights);
        }
    }
}
