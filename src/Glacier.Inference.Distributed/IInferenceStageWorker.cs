namespace Glacier.Inference.Distributed;

using System;
using System.Buffers;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Shuffle;

/// <summary>
/// Represents a stage worker in the distributed pipeline (e.g. Stage 0 Attention/Root, Stage 1 Warm Expert Pool, Stage 2 Accumulation).
/// Processes incoming activation tensors and returns zero-allocation ref-counted partition buffers.
/// </summary>
public interface IInferenceStageWorker : IDisposable
{
    /// <summary>
    /// Pipeline stage index/rank (e.g., 0 = Attention/Root, 1 = Warm Expert Pool, 2 = Accumulator).
    /// </summary>
    int StageRank { get; }

    /// <summary>
    /// Descriptive name or hardware role of this stage worker (e.g. "GPU Root Attention", "AVX-512 Warm Expert Pool").
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Asynchronously processes an activation vector for a token/layer.
    /// </summary>
    /// <param name="header">Metadata describing the activation frame.</param>
    /// <param name="inputActivation">Contiguous memory holding the activation vector (e.g., FP32 floats).</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>A ref-counted partition buffer containing the output activation tensor.</returns>
    ValueTask<RefCountedPartitionBuffer> ProcessActivationAsync(
        ActivationHeader header,
        ReadOnlyMemory<byte> inputActivation,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Synchronously processes an activation vector on the current thread with zero task/promise allocations.
    /// </summary>
    /// <param name="header">Metadata describing the activation frame.</param>
    /// <param name="inputActivation">Span holding the activation vector.</param>
    /// <returns>A ref-counted partition buffer containing the output activation tensor.</returns>
    RefCountedPartitionBuffer ProcessActivation(
        ActivationHeader header,
        ReadOnlySpan<byte> inputActivation);
}
