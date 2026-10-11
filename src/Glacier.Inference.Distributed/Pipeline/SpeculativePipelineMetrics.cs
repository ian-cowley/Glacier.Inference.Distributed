namespace Glacier.Inference.Distributed.Pipeline;

using System;

/// <summary>
/// Telemetry metrics for speculative two-stage pipeline overlap execution.
/// Tracks draft proposals, hits, misses, hit rate, and effective speedup over sequential execution.
/// </summary>
public sealed class SpeculativePipelineMetrics
{
    public long SpeculativeAttempts { get; set; }
    public long SpeculativeHits { get; set; }
    public long SpeculativeMisses { get; set; }
    public long FallbackSequentialSteps { get; set; }

    public double TotalStage0ComputeMs { get; set; }
    public double TotalStage1ComputeMs { get; set; }
    public double TotalNetworkTransitMs { get; set; }
    public double TotalElapsedMs { get; set; }

    public long TotalTokensGenerated => SpeculativeHits + SpeculativeMisses + FallbackSequentialSteps;

    /// <summary>
    /// Acceptance rate of speculative drafts (0.0 to 1.0).
    /// </summary>
    public double HitRate => SpeculativeAttempts > 0 ? (double)SpeculativeHits / SpeculativeAttempts : 0.0;

    /// <summary>
    /// Theoretical and empirical speedup achieved over strictly sequential pipeline decoding.
    /// Baseline sequential step: T0 (31.91ms) + T1 (30.35ms) + Tnet (1.0ms) = 63.26 ms (15.81 tok/s).
    /// Overlapped step: max(T0, T1) + Tnet = 32.91 ms (30.38 tok/s).
    /// Blended step = 63.26 - 30.35 * HitRate.
    /// </summary>
    public double TheoreticalSpeedup
    {
        get
        {
            double baselineMs = 63.26;
            double blendedMs = 63.26 - (30.35 * HitRate);
            return blendedMs > 0 ? baselineMs / blendedMs : 1.0;
        }
    }

    public double EffectiveThroughputTokensPerSec => TotalElapsedMs > 0
        ? (TotalTokensGenerated / (TotalElapsedMs / 1000.0))
        : (1000.0 / Math.Max(1.0, 63.26 - (30.35 * HitRate)));

    public void Reset()
    {
        SpeculativeAttempts = 0;
        SpeculativeHits = 0;
        SpeculativeMisses = 0;
        FallbackSequentialSteps = 0;
        TotalStage0ComputeMs = 0;
        TotalStage1ComputeMs = 0;
        TotalNetworkTransitMs = 0;
        TotalElapsedMs = 0;
    }

    public string FormatSummary()
    {
        return $"[Speculative Pipeline Telemetry]\n" +
               $"  Total Tokens:       {TotalTokensGenerated}\n" +
               $"  Speculative Drafts: {SpeculativeAttempts}\n" +
               $"  Hits (Accepted):    {SpeculativeHits} ({HitRate * 100.0:F1}% acceptance rate)\n" +
               $"  Misses (Rollbacks): {SpeculativeMisses}\n" +
               $"  Fallback Steps:     {FallbackSequentialSteps}\n" +
               $"  Effective Speedup:  {TheoreticalSpeedup:F2}x vs sequential baseline\n" +
               $"  Throughput Est:     {EffectiveThroughputTokensPerSec:F1} tok/s";
    }
}
