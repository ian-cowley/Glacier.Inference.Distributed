namespace Glacier.Inference.Distributed.Cli;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading;
using System.Threading.Tasks;

public record ModelPreset(string Key, string FileName, string Url, long ExpectedSize, string Description);

public static class ModelDownloader
{
    public static readonly Dictionary<string, ModelPreset> Presets = new(StringComparer.OrdinalIgnoreCase)
    {
        ["qwen2-57b"] = new(
            "qwen2-57b",
            "qwen2-57b-a14b-instruct-q3_k_m.gguf",
            "https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct-GGUF/resolve/main/qwen2-57b-a14b-instruct-q3_k_m.gguf",
            27505173312L,
            "Qwen2-57B-A14B (57B total, 14B active, 28 layers, 64 experts) @ Q3_K_M (25.62 GB)"
        ),
        ["mixtral-8x22b"] = new(
            "mixtral-8x22b",
            "Mixtral-8x22B-Instruct-v0.1.i1-IQ2_XXS.gguf",
            "https://huggingface.co/mradermacher/Mixtral-8x22B-Instruct-v0.1-i1-GGUF/resolve/main/Mixtral-8x22B-Instruct-v0.1.i1-IQ2_XXS.gguf",
            37889916608L,
            "Mixtral-8x22B-Instruct (141B total, 39B active, 56 layers, 8 experts) @ IQ2_XXS (35.29 GB)"
        ),
        ["mixtral-8x22b-iq2"] = new(
            "mixtral-8x22b-iq2",
            "Mixtral-8x22B-Instruct-v0.1.i1-IQ2_XXS.gguf",
            "https://huggingface.co/mradermacher/Mixtral-8x22B-Instruct-v0.1-i1-GGUF/resolve/main/Mixtral-8x22B-Instruct-v0.1.i1-IQ2_XXS.gguf",
            37889916608L,
            "Mixtral-8x22B-Instruct (141B total, 39B active, 56 layers, 8 experts) @ IQ2_XXS (35.29 GB)"
        ),
        ["deepseek-coder-v2-lite"] = new(
            "deepseek-coder-v2-lite",
            "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
            "https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
            9568912384L,
            "DeepSeek-Coder-V2-Lite (16B total, 2.4B active, 27 layers, 64 experts) @ Q4_K_M (8.91 GB)"
        ),
        ["deepseek-r1-14b"] = new(
            "deepseek-r1-14b",
            "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
            "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
            8988110240L,
            "DeepSeek-R1-Distill-Qwen-14B (Dense reasoning model, 48 layers) @ Q4_K_M (8.37 GB)"
        ),
        ["qwen2.5-32b"] = new(
            "qwen2.5-32b",
            "Qwen2.5-32B-Instruct-Q4_K_M.gguf",
            "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf",
            19851336576L,
            "Qwen2.5-32B-Instruct (Dense flagship, 64 layers) @ Q4_K_M (18.49 GB)"
        ),
        ["qwen2.5-14b"] = new(
            "qwen2.5-14b",
            "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            8988110240L,
            "Qwen2.5-14B-Instruct (Dense high-speed, 48 layers) @ Q4_K_M (8.37 GB)"
        ),
        ["qwen3.8-flash-next-coder"] = new(
            "qwen3.8-flash-next-coder",
            "Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ3_S.gguf",
            "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/resolve/main/Qwen3.8-Flash-Next-GSQ-RCO-Coder-IQ3_S.gguf",
            19542212608L,
            "Qwen3.8-Flash-Next-GSQ-RCO-Coder (42B RCO-pruned 256 experts, Hybrid GDN+MoE, IQ3_S/IQ4_NL) (18.20 GB)"
        ),
        ["qwen3.8-flash-next-q2_0"] = new(
            "qwen3.8-flash-next-q2_0",
            "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.gguf",
            "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0.gguf",
            22871023616L,
            "Qwen3.8-Flash-Next-GSQ-RCO (80B total, 512 experts, Hybrid GDN+MoE, Fast Q2_0 lookup-free) (21.30 GB)"
        ),
        ["qwen3.8-flash-next-iq3_s"] = new(
            "qwen3.8-flash-next-iq3_s",
            "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf",
            "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S.gguf",
            31245905920L,
            "Qwen3.8-Flash-Next-GSQ-RCO (80B total, 512 experts, Hybrid GDN+MoE, IQ3_S/IQ4_NL) (29.10 GB)"
        )
    };

    private static readonly HttpClient SharedHttpClient = CreateSharedHttpClient();

    private static HttpClient CreateSharedHttpClient()
    {
        var handler = new SocketsHttpHandler
        {
            EnableMultipleHttp2Connections = true,
            PooledConnectionLifetime = TimeSpan.FromMinutes(15)
        };
        var client = new HttpClient(handler)
        {
            Timeout = Timeout.InfiniteTimeSpan
        };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("GlacierInferenceDistributed/1.2.1");
        return client;
    }

    public static Task<int> DownloadModelAsync(
        string modelOrUrl,
        string? destDir = null,
        int concurrency = 8,
        CancellationToken ct = default)
        => DownloadModelCoreAsync(SharedHttpClient, modelOrUrl, destDir, concurrency, ct);

    internal static async Task<int> DownloadModelCoreAsync(
        HttpClient httpClient,
        string modelOrUrl,
        string? destDir = null,
        int concurrency = 8,
        CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(httpClient);
        concurrency = Math.Clamp(concurrency, 1, 32);
        destDir ??= Path.Combine(Directory.GetCurrentDirectory(), "models");
        Directory.CreateDirectory(destDir);

        string url;
        string fileName;
        long expectedSize = -1;
        string description;

        if (Presets.TryGetValue(modelOrUrl, out var preset))
        {
            url = preset.Url;
            fileName = preset.FileName;
            expectedSize = preset.ExpectedSize;
            description = preset.Description;
        }
        else
        {
            url = modelOrUrl;
            var uri = new Uri(url);
            fileName = Path.GetFileName(uri.LocalPath);
            description = $"Custom GGUF Model ({fileName})";
        }

        string targetPath = Path.Combine(destDir, fileName);

        if (File.Exists(targetPath))
        {
            var info = new FileInfo(targetPath);
            if (expectedSize > 0 && info.Length == expectedSize)
            {
                Console.WriteLine($"[Download] Model '{fileName}' already exists and matches expected size ({info.Length:N0} bytes). Skipping download.");
                return 0;
            }
        }

        Console.WriteLine("================================================================================");
        Console.WriteLine("  GLACIER PURE C# MULTI-STREAM MODEL DOWNLOADER (.NET 10)");
        Console.WriteLine($"  Model:       {description}");
        Console.WriteLine($"  Target Path: {targetPath}");
        Console.WriteLine($"  Streams:     {concurrency} Concurrent Sockets");
        Console.WriteLine("================================================================================\n");

        Console.Write("[Download] Probing remote file size... ");
        long totalSize = expectedSize;
        if (totalSize <= 0)
        {
            using var headReq = new HttpRequestMessage(HttpMethod.Head, url);
            using var headResp = await httpClient.SendAsync(headReq, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
            if (headResp.Content.Headers.ContentLength.HasValue)
            {
                totalSize = headResp.Content.Headers.ContentLength.Value;
            }
            else
            {
                // Fallback to range request 0-0
                using var rangeReq = new HttpRequestMessage(HttpMethod.Get, url);
                rangeReq.Headers.Range = new RangeHeaderValue(0, 0);
                using var rangeResp = await httpClient.SendAsync(rangeReq, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
                if (rangeResp.Content.Headers.ContentRange?.Length.HasValue == true)
                {
                    totalSize = rangeResp.Content.Headers.ContentRange.Length.Value;
                }
            }
        }

        if (totalSize <= 0)
        {
            Console.WriteLine("Failed to determine remote file size.");
            return 1;
        }

        Console.WriteLine($"{totalSize / (1024.0 * 1024.0 * 1024.0):F2} GB ({totalSize:N0} bytes).");

        // Stage into an atomic .part file so an interrupted/failed sparse allocation never poses as a completed targetPath
        string tempTargetPath = targetPath + ".part";
        Console.Write("[Disk] Pre-allocating sparse file structure on disk... ");
        using (var preAlloc = new FileStream(tempTargetPath, FileMode.Create, FileAccess.ReadWrite, FileShare.ReadWrite, 4096, FileOptions.Asynchronous))
        {
            preAlloc.SetLength(totalSize);
        }
        Console.WriteLine("Done.");

        var fileStream = new FileStream(
            tempTargetPath,
            FileMode.Open,
            FileAccess.Write,
            FileShare.ReadWrite,
            4096,
            FileOptions.Asynchronous);

        int effectiveConcurrency = (int)Math.Min(concurrency, totalSize);
        long chunkSize = totalSize / effectiveConcurrency;
        long totalBytesDownloaded = 0;
        var stopwatch = Stopwatch.StartNew();
        using var progressCts = CancellationTokenSource.CreateLinkedTokenSource(ct);

        // Background progress reporter
        var progressTask = Task.Run(async () =>
        {
            long lastBytes = 0;
            double lastSeconds = 0;
            while (!progressCts.Token.IsCancellationRequested)
            {
                await Task.Delay(1000, progressCts.Token).ConfigureAwait(ConfigureAwaitOptions.SuppressThrowing);
                if (progressCts.Token.IsCancellationRequested) break;

                long curBytes = Interlocked.Read(ref totalBytesDownloaded);
                double curSecs = stopwatch.Elapsed.TotalSeconds;
                double speedMBs = ((curBytes - lastBytes) / (1024.0 * 1024.0)) / Math.Max(0.1, curSecs - lastSeconds);
                double pct = (curBytes / (double)totalSize) * 100.0;
                double curGB = curBytes / (1024.0 * 1024.0 * 1024.0);
                double totalGB = totalSize / (1024.0 * 1024.0 * 1024.0);

                Console.Write($"\r[Progress] {curGB:F2} / {totalGB:F2} GB ({pct:F1}%) @ {speedMBs:F1} MB/s | Elapsed: {stopwatch.Elapsed:mm\\:ss}   ");

                lastBytes = curBytes;
                lastSeconds = curSecs;
            }
        }, CancellationToken.None);

        Console.WriteLine($"[Download] Launching {effectiveConcurrency} parallel HTTP Range streams...");
        var fileHandle = fileStream.SafeFileHandle;
        var tasks = new Task[effectiveConcurrency];
        for (int i = 0; i < effectiveConcurrency; i++)
        {
            long start = i * chunkSize;
            long end = (i == effectiveConcurrency - 1) ? totalSize - 1 : (i + 1) * chunkSize - 1;
            long expectedChunkLen = end - start + 1;

            tasks[i] = Task.Run(async () =>
            {
                using var req = new HttpRequestMessage(HttpMethod.Get, url);
                req.Headers.Range = new RangeHeaderValue(start, end);

                using var resp = await httpClient.SendAsync(req, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
                if (resp.StatusCode != HttpStatusCode.PartialContent)
                {
                    throw new InvalidOperationException(
                        $"Server did not return HTTP 206 PartialContent for Range bytes={start}-{end} (received {(int)resp.StatusCode} {resp.StatusCode}). Refusing to corrupt target file.");
                }

                using var netStream = await resp.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
                byte[] buffer = new byte[2 * 1024 * 1024]; // 2 MB buffer
                long writeOffset = start;
                long bytesReadForChunk = 0;

                while (bytesReadForChunk < expectedChunkLen)
                {
                    ct.ThrowIfCancellationRequested();
                    int bytesToRead = (int)Math.Min(buffer.Length, expectedChunkLen - bytesReadForChunk);
                    int read = await netStream.ReadAsync(buffer.AsMemory(0, bytesToRead), ct).ConfigureAwait(false);
                    if (read == 0) break;

                    await RandomAccess.WriteAsync(fileHandle, buffer.AsMemory(0, read), writeOffset, ct).ConfigureAwait(false);
                    writeOffset += read;
                    bytesReadForChunk += read;
                    Interlocked.Add(ref totalBytesDownloaded, read);
                }

                if (bytesReadForChunk != expectedChunkLen)
                {
                    throw new EndOfStreamException(
                        $"Truncated HTTP Range response for bytes={start}-{end}: expected {expectedChunkLen:N0} bytes, read {bytesReadForChunk:N0} bytes.");
                }
            }, ct);
        }

        try
        {
            await Task.WhenAll(tasks).ConfigureAwait(false);
            await fileStream.FlushAsync(ct).ConfigureAwait(false);
        }
        catch
        {
            await fileStream.DisposeAsync().ConfigureAwait(false);
            try { if (File.Exists(tempTargetPath)) File.Delete(tempTargetPath); } catch { }
            throw;
        }
        finally
        {
            progressCts.Cancel();
            await progressTask.ConfigureAwait(false);
        }

        await fileStream.DisposeAsync().ConfigureAwait(false);

        stopwatch.Stop();
        long verifiedBytes = Interlocked.Read(ref totalBytesDownloaded);
        var tempInfo = new FileInfo(tempTargetPath);
        if (verifiedBytes != totalSize || tempInfo.Length != totalSize)
        {
            try { if (File.Exists(tempTargetPath)) File.Delete(tempTargetPath); } catch { }
            Console.WriteLine($"[Error] Size mismatch! Expected {totalSize:N0} bytes, downloaded {verifiedBytes:N0} bytes, disk size {tempInfo.Length:N0} bytes.\n");
            return 1;
        }

        File.Move(tempTargetPath, targetPath, overwrite: true);

        double avgSpeed = (totalSize / (1024.0 * 1024.0)) / Math.Max(0.1, stopwatch.Elapsed.TotalSeconds);
        Console.WriteLine($"\n\n[SUCCESS] Completed in {stopwatch.Elapsed:mm\\:ss} ({avgSpeed:F1} MB/s average)!");
        Console.WriteLine($"[Verified] 100% Exact Byte Match ({verifiedBytes:N0} bytes downloaded & verified on disk).");
        Console.WriteLine($"Model ready for cluster inference: {targetPath}\n");
        return 0;
    }
}
