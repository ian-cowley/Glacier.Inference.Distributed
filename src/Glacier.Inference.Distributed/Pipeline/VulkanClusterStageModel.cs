namespace Glacier.Inference.Distributed.Pipeline;

using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Glacier.Inference.Gguf;
using Glacier.Inference.Gpu;
using Glacier.Inference.Memory;
using Glacier.Inference.Model;

/// <summary>
/// Experimental Vulkan compute stage scaffold for distributed LLM pipeline stages.
/// Weight upload and compute shader dispatch are not yet wired; callers should use D3D12, CUDA, or CPU SIMD stage runners.
/// </summary>
[System.Diagnostics.CodeAnalysis.Experimental("GLACIER_EXP001")]
public sealed unsafe class VulkanClusterStageModel : IClusterStageModel
{
    private readonly VulkanContext _ctx;
    private readonly bool _ownsContext;
    private readonly int _startLayer;
    private readonly int _layerCount;
    private readonly int _dim;
    private readonly int _vocabSize;
    private readonly int _headCount;
    private readonly int _headCountKv;
    private readonly int _headDim;
    private readonly int _maxSeqLen;
    private readonly bool _isFirstStage;
    private readonly bool _isLastStage;
    private readonly string _backendName;

    // SPIR-V Compute Pipelines & Layouts
    private IntPtr _pipelineGemmQ8;
    private IntPtr _layoutGemmQ8;
    private IntPtr _dslGemmQ8;

    private IntPtr _pipelineGemmF32;
    private IntPtr _layoutGemmF32;
    private IntPtr _dslGemmF32;

    private IntPtr _pipelineRmsNorm;
    private IntPtr _layoutRmsNorm;
    private IntPtr _dslRmsNorm;

    private IntPtr _pipelineRope;
    private IntPtr _layoutRope;
    private IntPtr _dslRope;

    private IntPtr _pipelineAttn;
    private IntPtr _layoutAttn;
    private IntPtr _dslAttn;

    private IntPtr _pipelineGeluMul;
    private IntPtr _layoutGeluMul;
    private IntPtr _dslGeluMul;

    private IntPtr _pipelineAddNorm;
    private IntPtr _layoutAddNorm;
    private IntPtr _dslAddNorm;

    // GPU Storage & Staging Buffers
    private IntPtr _bufIn;
    private IntPtr _memIn;
    private IntPtr _bufOut;
    private IntPtr _memOut;
    private IntPtr _bufResidual;
    private IntPtr _memResidual;

    // KV Cache & State Management
    private readonly KVCache _kvCache;
    private bool _disposed;

    public string BackendName => _backendName;
    public int StartLayer => _startLayer;
    public int LayerCount => _layerCount;
    public int EndLayer => _startLayer + _layerCount;
    public bool IsFirstStage => _isFirstStage;
    public bool IsLastStage => _isLastStage;
    public int Dim => _dim;
    public VulkanContext Context => _ctx;

    public VulkanClusterStageModel(
        VulkanContext ctx,
        int startLayer,
        int layerCount,
        int dim,
        int headCount,
        int headCountKv,
        int headDim,
        int maxSeqLen = 4096,
        bool isFirstStage = false,
        bool isLastStage = false,
        bool ownsContext = false,
        string? backendName = null,
        int vocabSize = 0)
    {
        ArgumentNullException.ThrowIfNull(ctx);
        ArgumentOutOfRangeException.ThrowIfNegative(startLayer);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dim);

        _ctx = ctx;
        _ownsContext = ownsContext;
        _startLayer = startLayer;
        _layerCount = layerCount;
        _dim = dim;
        _vocabSize = vocabSize > 0 ? vocabSize : dim;
        _headCount = headCount;
        _headCountKv = headCountKv;
        _headDim = headDim;
        _maxSeqLen = maxSeqLen;
        _isFirstStage = isFirstStage;
        _isLastStage = isLastStage;
        _backendName = backendName ?? $"Vulkan GPU [Device {_ctx.DeviceOrdinal}] ({_ctx.DeviceName})";

        _kvCache = new KVCache(_layerCount, _headCountKv, _headDim, _maxSeqLen);
        InitializeVulkanPipelinesAndBuffers();
    }

    public VulkanClusterStageModel(
        ModelWeights weights,
        int maxSeqLen,
        int startLayer,
        int layerCount,
        bool isLastStage,
        int deviceOrdinal = 0)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _ctx = new VulkanContext(deviceOrdinal);
        _ownsContext = true;
        _startLayer = startLayer;
        _layerCount = layerCount;
        _dim = weights.EmbeddingLength;
        _vocabSize = weights.VocabSize;
        _headCount = weights.HeadCount;
        _headCountKv = weights.HeadCountKv;
        _headDim = weights.HeadDim;
        _maxSeqLen = maxSeqLen;
        _isFirstStage = startLayer == 0;
        _isLastStage = isLastStage;
        _backendName = $"Vulkan GPU [Device {_ctx.DeviceOrdinal}] ({_ctx.DeviceName})";

        _kvCache = new KVCache(_layerCount, _headCountKv, _headDim, _maxSeqLen);
        InitializeVulkanPipelinesAndBuffers();
    }

    private void InitializeVulkanPipelinesAndBuffers()
    {
        if (_ctx.DeviceHandle == IntPtr.Zero) return;

        ulong actBytes = (ulong)(_dim * sizeof(float));
        ulong outFloats = (ulong)Math.Max(_dim, _isLastStage ? _vocabSize : _dim);
        ulong outBytes = outFloats * sizeof(float);
        uint storageUsage = VulkanDriver.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        uint hostVisible = VulkanDriver.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VulkanDriver.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        // Allocate activation and properly-sized output/logits buffers
        _ctx.CreateBuffer(actBytes, storageUsage, hostVisible, out _bufIn, out _memIn);
        _ctx.CreateBuffer(outBytes, storageUsage, hostVisible, out _bufOut, out _memOut);
        _ctx.CreateBuffer(actBytes, storageUsage, hostVisible, out _bufResidual, out _memResidual);

        // Precompile and link SPIR-V compute pipelines
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.GemmQ8Spirv, 4, 64, out _pipelineGemmQ8, out _layoutGemmQ8, out _dslGemmQ8);
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.GemmF32Spirv, 3, 64, out _pipelineGemmF32, out _layoutGemmF32, out _dslGemmF32);
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.RmsNormSpirv, 3, 64, out _pipelineRmsNorm, out _layoutRmsNorm, out _dslRmsNorm);
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.RopeSpirv, 2, 64, out _pipelineRope, out _layoutRope, out _dslRope);
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.AttentionSpirv, 4, 64, out _pipelineAttn, out _layoutAttn, out _dslAttn);
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.GeluMulSpirv, 2, 64, out _pipelineGeluMul, out _layoutGeluMul, out _dslGeluMul);
        VulkanComputePipeline.CreatePipeline(_ctx.DeviceHandle, VulkanShaders.AddNormSpirv, 3, 64, out _pipelineAddNorm, out _layoutAddNorm, out _dslAddNorm);
    }

    public void ForwardStage(
        int token,
        int pos,
        ReadOnlySpan<float> inputX,
        Span<float> outputX,
        Span<float> logits,
        bool computeLogits)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        throw new NotSupportedException(
            "VulkanClusterStageModel compute shader dispatch and weight upload are not yet implemented; use D3D12, CUDA, or CPU SIMD stage runners.");
    }

    public void ForwardBatchStage(
        ReadOnlySpan<int> tokens,
        int startPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        bool computeLogits)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        throw new NotSupportedException(
            "VulkanClusterStageModel compute shader dispatch and weight upload are not yet implemented; use D3D12, CUDA, or CPU SIMD stage runners.");
    }

    public void ResetKvCache()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _kvCache.Reset();
        _ctx.Synchronize();
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_ctx.DeviceHandle != IntPtr.Zero)
        {
            _ctx.Synchronize();

            if (_bufIn != IntPtr.Zero) _ctx.DestroyBuffer(_bufIn, _memIn);
            if (_bufOut != IntPtr.Zero) _ctx.DestroyBuffer(_bufOut, _memOut);
            if (_bufResidual != IntPtr.Zero) _ctx.DestroyBuffer(_bufResidual, _memResidual);

            DestroyPipelineResources(_pipelineGemmQ8, _layoutGemmQ8, _dslGemmQ8);
            DestroyPipelineResources(_pipelineGemmF32, _layoutGemmF32, _dslGemmF32);
            DestroyPipelineResources(_pipelineRmsNorm, _layoutRmsNorm, _dslRmsNorm);
            DestroyPipelineResources(_pipelineRope, _layoutRope, _dslRope);
            DestroyPipelineResources(_pipelineAttn, _layoutAttn, _dslAttn);
            DestroyPipelineResources(_pipelineGeluMul, _layoutGeluMul, _dslGeluMul);
            DestroyPipelineResources(_pipelineAddNorm, _layoutAddNorm, _dslAddNorm);
        }

        _kvCache.Dispose();

        if (_ownsContext)
        {
            _ctx.Dispose();
        }
    }

    private void DestroyPipelineResources(IntPtr pipeline, IntPtr layout, IntPtr dsl)
    {
        if (pipeline != IntPtr.Zero) VulkanDriver.DestroyPipeline(_ctx.DeviceHandle, pipeline, IntPtr.Zero);
        if (layout != IntPtr.Zero) VulkanDriver.DestroyPipelineLayout(_ctx.DeviceHandle, layout, IntPtr.Zero);
        if (dsl != IntPtr.Zero) VulkanDriver.DestroyDescriptorSetLayout(_ctx.DeviceHandle, dsl, IntPtr.Zero);
    }
}
