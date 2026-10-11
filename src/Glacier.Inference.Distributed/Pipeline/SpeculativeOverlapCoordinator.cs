namespace Glacier.Inference.Distributed.Pipeline;

using System;
using System.Diagnostics;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Cluster.Protocol;
using Glacier.Inference.Distributed.Transport;
using Glacier.Inference.Engine;

/// <summary>
/// Abstraction for forward pass execution of a pipeline stage model.
/// </summary>
public interface IStageForwardModel
{
    /// <summary>
    /// Executes a forward step for a single token at sequence position pos.
    /// </summary>
    void ForwardStage(int token, int pos, ReadOnlySpan<float> inputX, Span<float> outputX, Span<float> logits, bool computeLogits);
}

/// <summary>
/// Coordinates asynchronous speculative two-stage pipeline overlap for distributed cluster inference.
/// Stage 0 evaluates candidate token N+1 concurrently on local GPU while Stage 1 is evaluating
/// subsequent layers and LM Head on token N across Gigabit LAN.
/// On speculation hit, achieves overlapped step latency max(T0, T1) + Tnet (target >19.5-22 tok/s).
/// On speculation miss, performs zero-allocation in-place KV cache overwrite at sequence position P+1.
/// </summary>
public sealed class SpeculativeOverlapCoordinator : IDisposable
{
    private readonly IDraftProvider _draftProvider;
    private readonly int _hiddenDim;
    private readonly int _maxSeqLen;
    private readonly int[] _tokenHistory;
    private int _historyCount;

    // Double-buffered intermediate activation scratchpads (zero heap allocation during hot loop)
    private float[] _curActivation;
    private float[] _specActivation;
    private readonly byte[] _sendBuffer;
    private readonly int[] _singleDraftBuf = new int[1];

    private readonly SpeculativePipelineMetrics _metrics = new();
    private bool _disposed;

    public SpeculativePipelineMetrics Metrics => _metrics;
    public IDraftProvider DraftProvider => _draftProvider;
    public int HiddenDim => _hiddenDim;
    public int MaxSeqLen => _maxSeqLen;
    public ReadOnlySpan<int> TokenHistory => _tokenHistory.AsSpan(0, _historyCount);
    public ReadOnlySpan<float> CurrentActivationSpan => _curActivation.AsSpan(0, _hiddenDim);
    public ReadOnlySpan<float> SpeculativeActivationSpan => _specActivation.AsSpan(0, _hiddenDim);
    public Memory<float> CurrentActivationMemory => _curActivation.AsMemory(0, _hiddenDim);
    public Memory<float> SpeculativeActivationMemory => _specActivation.AsMemory(0, _hiddenDim);

    public SpeculativeOverlapCoordinator(
        int hiddenDim,
        int maxSeqLen = 8192,
        IDraftProvider? draftProvider = null)
    {
        _hiddenDim = hiddenDim;
        _maxSeqLen = maxSeqLen;
        _draftProvider = draftProvider ?? new PromptLookupDraftProvider(minMatchLength: 2, maxMatchLength: 4);

        _tokenHistory = new int[maxSeqLen];
        _historyCount = 0;

        _curActivation = GC.AllocateArray<float>(hiddenDim, pinned: true);
        _specActivation = GC.AllocateArray<float>(hiddenDim, pinned: true);

        // Pre-allocate send staging buffer for combined header (64B) + payload (FP32/FP16/INT8)
        int maxPayloadBytes = Math.Max(hiddenDim * sizeof(float), 64 + hiddenDim * sizeof(float));
        _sendBuffer = GC.AllocateArray<byte>(FusedSocketTransport.CombinedHeaderSize + maxPayloadBytes, pinned: true);
    }

    /// <summary>
    /// Seeds the prompt token sequence into the draft history tracker.
    /// </summary>
    public void SeedPromptHistory(ReadOnlySpan<int> promptTokens)
    {
        int copyCount = Math.Min(promptTokens.Length, _maxSeqLen);
        promptTokens.Slice(0, copyCount).CopyTo(_tokenHistory.AsSpan(0, copyCount));
        _historyCount = copyCount;
    }

    /// <summary>
    /// Records a committed token into the history buffer.
    /// </summary>
    public void RecordCommittedToken(int token)
    {
        if (_historyCount < _maxSeqLen)
        {
            _tokenHistory[_historyCount++] = token;
        }
        else
        {
            // Circular shift left when context capacity reached
            Array.Copy(_tokenHistory, 1, _tokenHistory, 0, _maxSeqLen - 1);
            _tokenHistory[^1] = token;
        }
    }

    /// <summary>
    /// Queries the draft engine (Prompt Lookup N-gram matching) for a candidate next token.
    /// Zero heap allocation hot path (&lt;1 us).
    /// </summary>
    public int ProposeCandidateToken()
    {
        if (_historyCount <= 1) return -1;

        int drafted = _draftProvider.Draft(
            _tokenHistory.AsSpan(0, _historyCount),
            maxDraftTokens: 1,
            _singleDraftBuf.AsSpan());

        return drafted > 0 ? _singleDraftBuf[0] : -1;
    }

    /// <summary>
    /// Executes a speculative draft forward pass on Stage 0 at sequence position pos.
    /// Stores the candidate intermediate activation into _specActivation and writes candidate KV into slot pos.
    /// </summary>
    public void ExecuteSpeculativeDraft(IStageForwardModel stage0Model, int candidateToken, int pos)
    {
        ArgumentNullException.ThrowIfNull(stage0Model);
        _metrics.SpeculativeAttempts++;

        stage0Model.ForwardStage(
            candidateToken,
            pos,
            ReadOnlySpan<float>.Empty,
            _specActivation.AsSpan(0, _hiddenDim),
            Span<float>.Empty,
            computeLogits: false);
    }

    /// <summary>
    /// Resolves the outcome of a speculative evaluation against the ground-truth token from Stage 1.
    /// On Hit: Swaps activation scratchpads in-place with 0 allocations; slot pos KV cache is already valid.
    /// On Miss: Re-runs true token at pos on Stage 0, unconditionally overwriting candidate KV slot in-place.
    /// </summary>
    /// <returns>True if speculation hit; false if miss.</returns>
    public bool ResolveSpeculation(
        IStageForwardModel stage0Model,
        int candidateToken,
        int trueToken,
        int pos,
        float topLogit = 0f,
        double stage1ComputeMs = 0.0)
    {
        ArgumentNullException.ThrowIfNull(stage0Model);

        if (candidateToken >= 0 && candidateToken == trueToken)
        {
            // SPECULATION HIT!
            _metrics.SpeculativeHits++;

            // Swap scratchpad buffers with zero allocation
            (_curActivation, _specActivation) = (_specActivation, _curActivation);

            RecordCommittedToken(trueToken);
            return true;
        }
        else
        {
            // SPECULATION MISS (or no candidate was proposed)
            if (candidateToken >= 0)
            {
                _metrics.SpeculativeMisses++;
            }
            else
            {
                _metrics.FallbackSequentialSteps++;
            }

            // Zero-allocation in-place KV cache rollback:
            // Re-evaluating trueToken at sequence position pos on Stage 0 overwrites
            // slot pos in-place across all heads without cache eviction or buffer re-allocation.
            stage0Model.ForwardStage(
                trueToken,
                pos,
                ReadOnlySpan<float>.Empty,
                _curActivation.AsSpan(0, _hiddenDim),
                Span<float>.Empty,
                computeLogits: false);

            RecordCommittedToken(trueToken);
            return false;
        }
    }

    /// <summary>
    /// Orchestrates an asynchronous speculative overlapped token generation step over the socket transport bus.
    /// Concurrently evaluates candidate token N+1 on Stage 0 while Stage 1 is evaluating Layers + LM Head on token N.
    /// </summary>
    public async ValueTask<(int TrueTokenId, float TopLogit, bool SpecHit, double StepMs)> ExecuteStepAsync(
        IStageForwardModel stage0Model,
        NonBlockingDuplexSocketBus stage1Bus,
        int currentToken,
        int currentPos,
        bool isInitialStep,
        WirePrecisionMode precisionMode,
        ulong stepCorrelationId,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(stage0Model);
        ArgumentNullException.ThrowIfNull(stage1Bus);

        var swStep = Stopwatch.StartNew();

        // 1. Initial step (first step after prompt prefill): compute Stage 0 forward pass for currentToken
        if (isInitialStep)
        {
            stage0Model.ForwardStage(
                currentToken,
                currentPos,
                ReadOnlySpan<float>.Empty,
                _curActivation.AsSpan(0, _hiddenDim),
                Span<float>.Empty,
                computeLogits: false);

            RecordCommittedToken(currentToken);
        }

        // 2. Transmit activation X_{27}^(N) across the socket to Stage 1
        int actByteCount = QuantizationKernels.GetPackedTokenBytes(precisionMode, _hiddenDim);
        int payloadLen = FusedActivationHeader.HeaderSize + actByteCount;
        int totalFrameLen = ClusterFrameHeader.HeaderSize + payloadLen;

        var clusterHdr = new ClusterFrameHeader(
            messageType: (ClusterMessageType)0x70, // MsgActivation
            correlationId: stepCorrelationId,
            payloadLength: (ulong)payloadLen,
            flags: ClusterFrameFlags.None);

        var actHdr = new FusedActivationHeader(
            sequenceId: stepCorrelationId,
            tokenIndex: currentPos,
            layerIndex: 0,
            hiddenDim: _hiddenDim,
            precisionMode: precisionMode,
            flags: FusedActivationFlags.FinalLayer);

        clusterHdr.WriteTo(_sendBuffer.AsSpan(0, ClusterFrameHeader.HeaderSize));
        actHdr.WriteTo(_sendBuffer.AsSpan(ClusterFrameHeader.HeaderSize, FusedActivationHeader.HeaderSize));

        var payloadSpan = _sendBuffer.AsSpan(FusedSocketTransport.CombinedHeaderSize, actByteCount);
        QuantizationKernels.QuantizeToken(_curActivation.AsSpan(0, _hiddenDim), payloadSpan, precisionMode);

        int sent = 0;
        var toSend = _sendBuffer.AsMemory(0, totalFrameLen);
        while (sent < totalFrameLen)
        {
            int r = await stage1Bus.Socket.SendAsync(toSend.Slice(sent), SocketFlags.None, ct).ConfigureAwait(false);
            if (r <= 0) throw new SocketException((int)SocketError.ConnectionReset);
            sent += r;
        }

        // 3. Propose candidate next token via Prompt Lookup Draft Provider
        int candidateToken = -1;
        if (currentPos + 1 < _maxSeqLen)
        {
            candidateToken = ProposeCandidateToken();
        }

        // 4. Concurrently execute Stage 0 forward pass for candidateToken at position currentPos + 1
        var swStage0 = Stopwatch.StartNew();
        if (candidateToken >= 0)
        {
            ExecuteSpeculativeDraft(stage0Model, candidateToken, currentPos + 1);
        }
        swStage0.Stop();
        _metrics.TotalStage0ComputeMs += swStage0.Elapsed.TotalMilliseconds;

        // 5. Receive Stage 1 confirmation result token from the duplex socket bus (bounded by 5.0s watchdog)
        var tokenFrame = await stage1Bus.ReceiveTokenAsync(ct).ConfigureAwait(false);
        int trueTokenId = tokenFrame.TokenId;
        float topLogit = tokenFrame.TopLogit;
        _metrics.TotalStage1ComputeMs += tokenFrame.ComputeMs;

        // 6. Barrier Synchronization & In-Place Rollback Resolution
        bool isHit = ResolveSpeculation(stage0Model, candidateToken, trueTokenId, currentPos + 1, topLogit, tokenFrame.ComputeMs);

        swStep.Stop();
        double stepMs = swStep.Elapsed.TotalMilliseconds;
        _metrics.TotalElapsedMs += stepMs;

        return (trueTokenId, topLogit, isHit, stepMs);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _draftProvider.Dispose();
    }
}
