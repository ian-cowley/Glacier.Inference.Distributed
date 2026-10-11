namespace Glacier.Inference.Distributed;

using System;
using System.Runtime.CompilerServices;

/// <summary>
/// Dynamic precision switcher selecting optimal wire precision mode (FP32, FP16, INT8_Symmetric, FP8_E4M3)
/// based on batch size, context depth, and measured network roundtrip latency (RTT).
/// Implements automatic congestion backoff to INT8 to alleviate TCP queuing delays under network saturation.
/// </summary>
public sealed class DynamicPrecisionSwitcher
{
    private double _currentEmaLatencyMs = -1.0;
    private WirePrecisionMode _lastSelectedPrecision;
    private int _totalSwitches;

    /// <summary>
    /// Optional manual or CLI override forcing a specific precision mode.
    /// </summary>
    public WirePrecisionMode? ForcedPrecision { get; set; }

    /// <summary>
    /// Batch token threshold above which prompt prefill switches to high-compression INT8 (default: 32).
    /// </summary>
    public int PrefillBatchThreshold { get; set; } = 32;

    /// <summary>
    /// Roundtrip latency threshold in milliseconds triggering network congestion backoff to INT8 (default: 2.5 ms).
    /// </summary>
    public double CongestionLatencyThresholdMs { get; set; } = 2.5;

    /// <summary>
    /// Sequence position depth beyond which single-token decode strictly uses FP16 to protect attention logit fidelity (default: 16,384 tokens).
    /// </summary>
    public int DeepContextThreshold { get; set; } = 16384;

    /// <summary>
    /// Exponential moving average smoothing factor for network RTT updates (default: 0.2).
    /// </summary>
    public double EmaAlpha { get; set; } = 0.2;

    /// <summary>
    /// Current measured network roundtrip latency in milliseconds (smoothed via EMA).
    /// Returns 0.5 ms default if no measurements have been recorded.
    /// </summary>
    public double CurrentEmaLatencyMs => _currentEmaLatencyMs > 0.0 ? _currentEmaLatencyMs : 0.5;

    /// <summary>
    /// Most recently selected precision mode.
    /// </summary>
    public WirePrecisionMode LastSelectedPrecision => _lastSelectedPrecision;

    /// <summary>
    /// Total count of dynamic precision switching events.
    /// </summary>
    public int TotalSwitches => _totalSwitches;

    public DynamicPrecisionSwitcher(WirePrecisionMode? forcedPrecision = null)
    {
        ForcedPrecision = forcedPrecision;
        _lastSelectedPrecision = forcedPrecision ?? WirePrecisionMode.FP16;
        _currentEmaLatencyMs = -1.0; // Uninitialized: first measurement establishes baseline directly
    }

    /// <summary>
    /// Updates the measured network roundtrip latency using exponential moving average (EMA).
    /// The first recorded measurement establishes the initial baseline directly.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void UpdateMeasuredLatency(double rttMs)
    {
        if (rttMs <= 0.0) return;

        if (_currentEmaLatencyMs <= 0.0)
        {
            _currentEmaLatencyMs = rttMs;
        }
        else
        {
            _currentEmaLatencyMs = (1.0 - EmaAlpha) * _currentEmaLatencyMs + EmaAlpha * rttMs;
        }
    }

    /// <summary>
    /// Alias for <see cref="UpdateMeasuredLatency(double)"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void RecordRtt(double rttMs) => UpdateMeasuredLatency(rttMs);

    /// <summary>
    /// Dynamically selects wire precision based on batch size, sequence position, decode mode, and measured latency.
    /// </summary>
    public WirePrecisionMode SelectPrecision(
        int batchSize,
        int sequencePosition = 0,
        bool isDecode = false,
        double? overrideRttMs = null)
    {
        if (ForcedPrecision.HasValue)
        {
            _lastSelectedPrecision = ForcedPrecision.Value;
            return ForcedPrecision.Value;
        }

        double rtt = overrideRttMs ?? (_currentEmaLatencyMs > 0.0 ? _currentEmaLatencyMs : 0.5);
        WirePrecisionMode selected;

        if (isDecode)
        {
            // Autoregressive single-token decode phase (B = 1)
            if (sequencePosition > DeepContextThreshold)
            {
                // In deep context (>16k tokens), maintain FP16 precision to protect attention logits
                selected = WirePrecisionMode.FP16;
            }
            else if (rtt > CongestionLatencyThresholdMs)
            {
                // Congestion backoff: drop payload to 5.2 KB to alleviate queuing
                selected = WirePrecisionMode.INT8_Symmetric;
            }
            else
            {
                selected = WirePrecisionMode.FP16;
            }
        }
        else
        {
            // Prompt prefill batch phase
            if (batchSize >= PrefillBatchThreshold)
            {
                // Prefill batch >= 32: Bandwidth bound, use INT8 (4x throughput gain)
                selected = WirePrecisionMode.INT8_Symmetric;
            }
            else if (rtt > CongestionLatencyThresholdMs)
            {
                // Small batch under network congestion: back off to INT8
                selected = WirePrecisionMode.INT8_Symmetric;
            }
            else
            {
                selected = WirePrecisionMode.FP16;
            }
        }

        if (selected != _lastSelectedPrecision)
        {
            _totalSwitches++;
            _lastSelectedPrecision = selected;
        }

        return selected;
    }

    /// <summary>
    /// Static optimal wire precision selector matching contract heuristic.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static WirePrecisionMode SelectOptimalWirePrecision(
        int batchSize,
        bool isDecode,
        double lanLatencyMs,
        WirePrecisionMode? forcedOverride = null)
    {
        if (forcedOverride.HasValue) return forcedOverride.Value;
        if (isDecode) return WirePrecisionMode.FP16;
        if (lanLatencyMs > 2.5) return WirePrecisionMode.INT8_Symmetric;
        return batchSize >= 32 ? WirePrecisionMode.INT8_Symmetric : WirePrecisionMode.FP16;
    }

    /// <summary>
    /// Resets the switcher state and EMA latency tracker.
    /// </summary>
    public void Reset()
    {
        _currentEmaLatencyMs = -1.0;
        _totalSwitches = 0;
        _lastSelectedPrecision = ForcedPrecision ?? WirePrecisionMode.FP16;
    }
}
