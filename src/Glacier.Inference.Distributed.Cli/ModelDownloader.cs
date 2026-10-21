namespace Glacier.Inference.Distributed.Cli;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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
        )
    };

    public static async Task<int> DownloadModelAsync(string modelOrUrl, string? destDir = null, int concurrency = 8)
    {
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

        var handler = new SocketsHttpHandler
        {
            EnableMultipleHttp2Connections = true,
            PooledConnectionLifetime = TimeSpan.FromMinutes(15)
        };
        using var httpClient = new HttpClient(handler);
        httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("GlacierInferenceDistributed/1.0");

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
            using var headResp = await httpClient.SendAsync(headReq, HttpCompletionOption.ResponseHeadersRead);
            if (headResp.Content.Headers.ContentLength.HasValue)
            {
                totalSize = headResp.Content.Headers.ContentLength.Value;
            }
            else
            {
                // Fallback to range request 0-0
                using var rangeReq = new HttpRequestMessage(HttpMethod.Get, url);
                rangeReq.Headers.Range = new RangeHeaderValue(0, 0);
                using var rangeResp = await httpClient.SendAsync(rangeReq, HttpCompletionOption.ResponseHeadersRead);
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

        Console.Write("[Disk] Pre-allocating sparse file structure on disk... ");
        using (var preAlloc = new FileStream(targetPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite, 4096, FileOptions.Asynchronous))
        {
            if (preAlloc.Length != totalSize)
            {
                preAlloc.SetLength(totalSize);
            }
        }
        Console.WriteLine("Done.");

        using var fileStream = new FileStream(
            targetPath, 
            FileMode.Open, 
            FileAccess.Write, 
            FileShare.ReadWrite, 
            4096, 
            FileOptions.Asynchronous);
        var fileHandle = fileStream.SafeFileHandle;

        long chunkSize = totalSize / concurrency;
        long totalBytesDownloaded = 0;
        var stopwatch = Stopwatch.StartNew();
        var cts = new CancellationTokenSource();

        // Background progress reporter
        var progressTask = Task.Run(async () =>
        {
            long lastBytes = 0;
            double lastSeconds = 0;
            while (!cts.Token.IsCancellationRequested)
            {
                await Task.Delay(1000, cts.Token).ConfigureAwait(ConfigureAwaitOptions.SuppressThrowing);
                if (cts.Token.IsCancellationRequested) break;

                long curBytes = Volatile.Read(ref totalBytesDownloaded);
                double curSecs = stopwatch.Elapsed.TotalSeconds;
                double speedMBs = ((curBytes - lastBytes) / (1024.0 * 1024.0)) / Math.Max(0.1, curSecs - lastSeconds);
                double pct = (curBytes / (double)totalSize) * 100.0;
                double curGB = curBytes / (1024.0 * 1024.0 * 1024.0);
                double totalGB = totalSize / (1024.0 * 1024.0 * 1024.0);

                Console.Write($"\r[Progress] {curGB:F2} / {totalGB:F2} GB ({pct:F1}%) @ {speedMBs:F1} MB/s | Elapsed: {stopwatch.Elapsed:mm\\:ss}   ");

                lastBytes = curBytes;
                lastSeconds = curSecs;
            }
        });

        Console.WriteLine($"[Download] Launching {concurrency} parallel HTTP Range streams...");
        var tasks = new Task[concurrency];
        for (int i = 0; i < concurrency; i++)
        {
            long start = i * chunkSize;
            long end = (i == concurrency - 1) ? totalSize - 1 : (i + 1) * chunkSize - 1;
            tasks[i] = Task.Run(async () =>
            {
                using var req = new HttpRequestMessage(HttpMethod.Get, url);
                req.Headers.Range = new RangeHeaderValue(start, end);

                using var resp = await httpClient.SendAsync(req, HttpCompletionOption.ResponseHeadersRead);
                resp.EnsureSuccessStatusCode();

                using var netStream = await resp.Content.ReadAsStreamAsync();
                byte[] buffer = new byte[2 * 1024 * 1024]; // 2 MB buffer
                long writeOffset = start;

                while (writeOffset <= end)
                {
                    int bytesToRead = (int)Math.Min(buffer.Length, end - writeOffset + 1);
                    int read = await netStream.ReadAsync(buffer.AsMemory(0, bytesToRead));
                    if (read == 0) break;

                    await RandomAccess.WriteAsync(fileHandle, buffer.AsMemory(0, read), writeOffset);
                    writeOffset += read;
                    Interlocked.Add(ref totalBytesDownloaded, read);
                }
            });
        }

        try
        {
            await Task.WhenAll(tasks);
        }
        finally
        {
            cts.Cancel();
            await progressTask;
        }

        stopwatch.Stop();
        double avgSpeed = (totalSize / (1024.0 * 1024.0)) / Math.Max(0.1, stopwatch.Elapsed.TotalSeconds);
        Console.WriteLine($"\n\n[SUCCESS] Completed in {stopwatch.Elapsed:mm\\:ss} ({avgSpeed:F1} MB/s average)!");

        var finalInfo = new FileInfo(targetPath);
        if (finalInfo.Length == totalSize)
        {
            Console.WriteLine($"[Verified] 100% Exact Byte Match ({finalInfo.Length:N0} bytes).");
            Console.WriteLine($"Model ready for cluster inference: {targetPath}\n");
            return 0;
        }
        else
        {
            Console.WriteLine($"[Error] Size mismatch! Expected {totalSize:N0} bytes, got {finalInfo.Length:N0} bytes.\n");
            return 1;
        }
    }
}
