namespace Glacier.Inference.Distributed.Pipeline;

using System;

/// <summary>
/// Defines the standard execution interface for a distributed pipeline stage model across backends
/// (CPU SIMD, Direct3D 12, Vulkan).
/// </summary>
public interface IClusterStageModel : IDisposable
{
    /// <summary>
    /// Human-readable backend name (e.g. "Direct3D 12 GPU [Device 0]", "Vulkan GPU [Device 0]", "CPU SIMD").
    /// </summary>
    string BackendName { get; }

    /// <summary>
    /// Executes the forward pass for a single token through the layers of this pipeline stage.
    /// </summary>
    void ForwardStage(int token, int pos, ReadOnlySpan<float> inputX, Span<float> outputX, Span<float> logits, bool computeLogits);

    /// <summary>
    /// Executes batched forward pass for prompt prefill across this pipeline stage.
    /// </summary>
    void ForwardBatchStage(
        ReadOnlySpan<int> tokens,
        int startPos,
        ReadOnlySpan<float> inputXBatch,
        Span<float> outputXBatch,
        Span<float> logits,
        bool computeLogits);

    /// <summary>
    /// Resets the internal KV cache and sequence position state.
    /// </summary>
    void ResetKvCache();
}
