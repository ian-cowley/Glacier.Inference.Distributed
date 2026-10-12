namespace Glacier.Inference.Distributed.Cli;

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Glacier.Inference.Sampling;

public static partial class Program
{
    public static async Task<int> RunServeAsync(
        DistributedClusterSession session,
        string host,
        int port,
        string? apiKey)
    {
        if ((host == "0.0.0.0" || host == "::") && string.IsNullOrEmpty(apiKey))
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("╔═════════════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║ ⚠️  SECURITY NOTICE: CLUSTER IS BOUND TO 0.0.0.0 WITHOUT AUTHENTICATION! ║");
            Console.WriteLine("║     ANY DEVICE ON LAN CAN INVOKE DISTRIBUTED INFERENCE.                 ║");
            Console.WriteLine("║     TO PROTECT: Provide --api-key <key> or set GLACIER_API_KEY.         ║");
            Console.WriteLine("╚═════════════════════════════════════════════════════════════════════════╝");
            Console.ResetColor();
        }

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("=======================================================================");
        Console.WriteLine("     GLACIER DISTRIBUTED CLUSTER OPENAI & OLLAMA HTTP SERVICE         ");
        Console.WriteLine("     Heterogeneous Multi-Node Pipeline Parallelism | Pure C# .NET 10   ");
        Console.WriteLine("=======================================================================");
        Console.ResetColor();
        Console.WriteLine($"Model:        {session.ModelName} ({session.TotalLayers} Layers, Dim: {session.Dim})");
        Console.WriteLine($"Endpoint:     http://{host}:{port}");
        Console.WriteLine($"Cluster:      {session.Stages.Length} Stages ({session.LocalModels.Count} Local GPU/CPU + {session.Streams.Count} Remote LAN)");
        if (!string.IsNullOrEmpty(apiKey))
        {
            Console.WriteLine("API Auth:     Enabled (Bearer token / x-api-key)");
        }
        Console.WriteLine();

        var app = CreateServeApp(session, host, port, apiKey);

        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($">>> OpenAI API Base URL:  http://{host}:{port}/v1");
        Console.WriteLine($">>> Models Endpoint:      http://{host}:{port}/v1/models");
        Console.WriteLine($">>> Chat Endpoint:        http://{host}:{port}/v1/chat/completions");
        Console.WriteLine($">>> Cluster Status:       http://{host}:{port}/v1/cluster/status");
        Console.WriteLine($">>> Ollama Compatibility: http://{host}:{port}/api/generate\n");
        Console.ResetColor();

        await app.RunAsync();
        return 0;
    }

    public static WebApplication CreateServeApp(
        DistributedClusterSession session,
        string host = "0.0.0.0",
        int port = 8080,
        string? apiKey = null)
    {
        var builder = WebApplication.CreateBuilder();
        builder.Services.Configure<Microsoft.AspNetCore.Server.Kestrel.Core.KestrelServerOptions>(options =>
        {
            options.Limits.MaxRequestBodySize = 32 * 1024 * 1024; // 32MB request body
        });

        var app = builder.Build();
        if (port >= 0)
        {
            app.Urls.Add($"http://{host}:{port}");
        }

        ConfigureServeEndpoints(app, session, apiKey);
        return app;
    }

    public static void ConfigureServeEndpoints(
        WebApplication app,
        DistributedClusterSession session,
        string? apiKey = null)
    {
        string modelName = session.ModelName;

        // Security middleware for optional API key authentication
        if (!string.IsNullOrEmpty(apiKey))
        {
            app.Use(async (context, next) =>
            {
                // Health endpoint can be public or authenticated
                if (context.Request.Path == "/health")
                {
                    await next();
                    return;
                }

                string? authHeader = context.Request.Headers.Authorization.FirstOrDefault();
                string? xApiKey = context.Request.Headers["x-api-key"].FirstOrDefault();

                bool authorized = false;
                if (!string.IsNullOrEmpty(authHeader) && authHeader.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
                {
                    string token = authHeader.Substring(7).Trim();
                    authorized = CryptographicOperations.FixedTimeEquals(
                        Encoding.UTF8.GetBytes(token),
                        Encoding.UTF8.GetBytes(apiKey));
                }
                else if (!string.IsNullOrEmpty(xApiKey))
                {
                    authorized = CryptographicOperations.FixedTimeEquals(
                        Encoding.UTF8.GetBytes(xApiKey),
                        Encoding.UTF8.GetBytes(apiKey));
                }

                if (!authorized)
                {
                    context.Response.StatusCode = StatusCodes.Status401Unauthorized;
                    await context.Response.WriteAsync("Unauthorized: Invalid or missing API key.", context.RequestAborted);
                    return;
                }

                await next();
            });
        }

        // 1. GET /v1/models (OpenAI Standard)
        app.MapGet("/v1/models", () => Results.Json(new
        {
            @object = "list",
            data = new[]
            {
                new
                {
                    id = modelName,
                    @object = "model",
                    created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    owned_by = "glacier-cluster"
                }
            }
        }));

        // 2. GET /v1/cluster/status & /health
        var getClusterStatus = () => Results.Json(new
        {
            status = "healthy",
            cluster = "Glacier Distributed LLM Cluster",
            model = modelName,
            total_layers = session.TotalLayers,
            embedding_dim = session.Dim,
            vocab_size = session.VocabSize,
            stages_count = session.Stages.Length,
            local_stages_count = session.LocalModels.Count,
            remote_stages_count = session.Streams.Count,
            stages = session.Stages.Select((st, idx) => new
            {
                stage_index = idx,
                node_name = st.Name,
                host = st.Host,
                port = st.Port,
                start_layer = st.StartLayer,
                end_layer = st.EndLayer,
                layer_count = st.LayerCount,
                is_local = idx < session.LocalModels.Count,
                backend = idx < session.LocalModels.Count ? session.LocalModels[idx].BackendName : "Remote Gigabit TCP Worker"
            })
        });

        app.MapGet("/v1/cluster/status", getClusterStatus);
        app.MapGet("/health", getClusterStatus);

        // 3. POST /v1/chat/completions (OpenAI Standard Chat Completion)
        app.MapPost("/v1/chat/completions", async (HttpContext ctx) =>
        {
            JsonElement req;
            try
            {
                req = await ctx.Request.ReadFromJsonAsync<JsonElement>(ctx.RequestAborted);
            }
            catch
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Invalid JSON payload.", ctx.RequestAborted);
                return;
            }

            var messages = ParseMessages(req);
            if (messages.Count == 0)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Messages array required.", ctx.RequestAborted);
                return;
            }

            bool stream = req.TryGetProperty("stream", out var s) &&
                          (s.ValueKind == JsonValueKind.True || (s.ValueKind != JsonValueKind.False && s.GetBoolean()));

            var options = ParseSamplingOptions(req);
            string formattedPrompt = session.Tokenizer != null
                ? session.Tokenizer.FormatChat(messages)
                : string.Join("\n", messages.Select(m => $"{m.Role}: {m.Content}"));
            string completionId = $"chatcmpl-{Guid.NewGuid():N}";
            long createdTimestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            if (!stream)
            {
                var result = await session.GenerateAsync(formattedPrompt, options, ct: ctx.RequestAborted);
                var resp = new
                {
                    id = completionId,
                    @object = "chat.completion",
                    created = createdTimestamp,
                    model = modelName,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            message = new { role = "assistant", content = result.Text },
                            finish_reason = result.FinishReason
                        }
                    },
                    usage = new
                    {
                        prompt_tokens = result.PromptTokens,
                        completion_tokens = result.GeneratedTokens,
                        total_tokens = result.PromptTokens + result.GeneratedTokens
                    }
                };
                await ctx.Response.WriteAsJsonAsync(resp, ctx.RequestAborted);
                return;
            }

            // Streaming SSE response
            ctx.Response.ContentType = "text/event-stream; charset=utf-8";
            ctx.Response.Headers.CacheControl = "no-cache";

            var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(256)
            {
                SingleWriter = true,
                SingleReader = true,
                FullMode = BoundedChannelFullMode.Wait
            });

            var genTask = Task.Run(async () =>
            {
                try
                {
                    return await session.GenerateAsync(
                        formattedPrompt,
                        options,
                        onToken: token => channel.Writer.TryWrite(token),
                        ct: ctx.RequestAborted);
                }
                finally
                {
                    channel.Writer.Complete();
                }
            });

            // Initial chunk
            var initialChunk = new
            {
                id = completionId,
                @object = "chat.completion.chunk",
                created = createdTimestamp,
                model = modelName,
                choices = new[]
                {
                    new
                    {
                        index = 0,
                        delta = new { role = "assistant", content = "" },
                        finish_reason = (string?)null
                    }
                }
            };
            await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(initialChunk)}\n\n", ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);

            // Stream delta tokens
            await foreach (var token in channel.Reader.ReadAllAsync(ctx.RequestAborted))
            {
                var chunk = new
                {
                    id = completionId,
                    @object = "chat.completion.chunk",
                    created = createdTimestamp,
                    model = modelName,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            delta = new { content = token },
                            finish_reason = (string?)null
                        }
                    }
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(chunk)}\n\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }

            var genResult = await genTask;
            var finalChunk = new
            {
                id = completionId,
                @object = "chat.completion.chunk",
                created = createdTimestamp,
                model = modelName,
                choices = new[]
                {
                    new
                    {
                        index = 0,
                        delta = new { },
                        finish_reason = genResult.FinishReason
                    }
                }
            };
            await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(finalChunk)}\n\n", ctx.RequestAborted);
            await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
        });

        // 4. POST /v1/completions (OpenAI Standard Text Completion)
        app.MapPost("/v1/completions", async (HttpContext ctx) =>
        {
            JsonElement req;
            try
            {
                req = await ctx.Request.ReadFromJsonAsync<JsonElement>(ctx.RequestAborted);
            }
            catch
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Invalid JSON payload.", ctx.RequestAborted);
                return;
            }

            string prompt = "";
            if (req.TryGetProperty("prompt", out var p))
            {
                if (p.ValueKind == JsonValueKind.String) prompt = p.GetString() ?? "";
                else if (p.ValueKind == JsonValueKind.Array && p.GetArrayLength() > 0) prompt = p[0].GetString() ?? "";
            }

            if (string.IsNullOrEmpty(prompt))
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Prompt is required.", ctx.RequestAborted);
                return;
            }

            bool stream = req.TryGetProperty("stream", out var s) &&
                          (s.ValueKind == JsonValueKind.True || (s.ValueKind != JsonValueKind.False && s.GetBoolean()));

            var options = ParseSamplingOptions(req);
            string completionId = $"cmpl-{Guid.NewGuid():N}";
            long createdTimestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            if (!stream)
            {
                var result = await session.GenerateAsync(prompt, options, ct: ctx.RequestAborted);
                var resp = new
                {
                    id = completionId,
                    @object = "text_completion",
                    created = createdTimestamp,
                    model = modelName,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            text = result.Text,
                            finish_reason = result.FinishReason
                        }
                    },
                    usage = new
                    {
                        prompt_tokens = result.PromptTokens,
                        completion_tokens = result.GeneratedTokens,
                        total_tokens = result.PromptTokens + result.GeneratedTokens
                    }
                };
                await ctx.Response.WriteAsJsonAsync(resp, ctx.RequestAborted);
                return;
            }

            ctx.Response.ContentType = "text/event-stream; charset=utf-8";
            ctx.Response.Headers.CacheControl = "no-cache";

            var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(256)
            {
                SingleWriter = true,
                SingleReader = true,
                FullMode = BoundedChannelFullMode.Wait
            });

            var genTask = Task.Run(async () =>
            {
                try
                {
                    return await session.GenerateAsync(
                        prompt,
                        options,
                        onToken: token => channel.Writer.TryWrite(token),
                        ct: ctx.RequestAborted);
                }
                finally
                {
                    channel.Writer.Complete();
                }
            });

            await foreach (var token in channel.Reader.ReadAllAsync(ctx.RequestAborted))
            {
                var chunk = new
                {
                    id = completionId,
                    @object = "text_completion",
                    created = createdTimestamp,
                    model = modelName,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            text = token,
                            finish_reason = (string?)null
                        }
                    }
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(chunk)}\n\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }

            var genResult = await genTask;
            var finalChunk = new
            {
                id = completionId,
                @object = "text_completion",
                created = createdTimestamp,
                model = modelName,
                choices = new[]
                {
                    new
                    {
                        index = 0,
                        text = "",
                        finish_reason = genResult.FinishReason
                    }
                }
            };
            await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(finalChunk)}\n\n", ctx.RequestAborted);
            await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
        });

        // 5. GET /api/tags (Ollama compatible tags list)
        app.MapGet("/api/tags", () => Results.Json(new
        {
            models = new[]
            {
                new
                {
                    name = modelName,
                    model = modelName,
                    modified_at = DateTime.UtcNow.ToString("o"),
                    size = File.Exists(session.ModelPath) ? (long)new FileInfo(session.ModelPath).Length : 0L,
                    details = new
                    {
                        format = "gguf",
                        family = session.MasterGguf?.Architecture ?? session.Architecture,
                        parameter_size = $"{session.TotalLayers}L",
                        quantization_level = "ClusterDistributed"
                    }
                }
            }
        }));

        // 6. POST /api/chat (Ollama compatible chat)
        app.MapPost("/api/chat", async (HttpContext ctx) =>
        {
            JsonElement req;
            try
            {
                req = await ctx.Request.ReadFromJsonAsync<JsonElement>(ctx.RequestAborted);
            }
            catch
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Invalid JSON payload.", ctx.RequestAborted);
                return;
            }

            var messages = ParseMessages(req);
            if (messages.Count == 0)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Messages array required.", ctx.RequestAborted);
                return;
            }

            bool stream = !req.TryGetProperty("stream", out var s) || s.ValueKind != JsonValueKind.False;
            var options = ParseSamplingOptions(req);
            string formattedPrompt = session.Tokenizer != null
                ? session.Tokenizer.FormatChat(messages)
                : string.Join("\n", messages.Select(m => $"{m.Role}: {m.Content}"));

            if (!stream)
            {
                var result = await session.GenerateAsync(formattedPrompt, options, ct: ctx.RequestAborted);
                var respObj = new
                {
                    model = modelName,
                    created_at = DateTime.UtcNow.ToString("o"),
                    message = new { role = "assistant", content = result.Text },
                    done = true,
                    total_duration = (long)(result.TotalDuration.TotalSeconds * 1e9),
                    prompt_eval_count = result.PromptTokens,
                    prompt_eval_duration = (long)(result.PromptDuration.TotalSeconds * 1e9),
                    eval_count = result.GeneratedTokens,
                    eval_duration = (long)(result.GenerationDuration.TotalSeconds * 1e9)
                };
                await ctx.Response.WriteAsJsonAsync(respObj, ctx.RequestAborted);
                return;
            }

            ctx.Response.ContentType = "application/x-ndjson";
            var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(256)
            {
                SingleWriter = true,
                SingleReader = true,
                FullMode = BoundedChannelFullMode.Wait
            });

            var genTask = Task.Run(async () =>
            {
                try
                {
                    return await session.GenerateAsync(
                        formattedPrompt,
                        options,
                        onToken: token => channel.Writer.TryWrite(token),
                        ct: ctx.RequestAborted);
                }
                finally
                {
                    channel.Writer.Complete();
                }
            });

            await foreach (var token in channel.Reader.ReadAllAsync(ctx.RequestAborted))
            {
                var chunk = new
                {
                    model = modelName,
                    created_at = DateTime.UtcNow.ToString("o"),
                    message = new { role = "assistant", content = token },
                    done = false
                };
                await ctx.Response.WriteAsync(JsonSerializer.Serialize(chunk) + "\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }

            var genResult = await genTask;
            var finalChunk = new
            {
                model = modelName,
                created_at = DateTime.UtcNow.ToString("o"),
                message = new { role = "assistant", content = "" },
                done = true,
                total_duration = (long)(genResult.TotalDuration.TotalSeconds * 1e9),
                eval_count = genResult.GeneratedTokens,
                eval_duration = (long)(genResult.GenerationDuration.TotalSeconds * 1e9)
            };
            await ctx.Response.WriteAsync(JsonSerializer.Serialize(finalChunk) + "\n", ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
        });

        // 7. POST /api/generate (Ollama compatible generate)
        app.MapPost("/api/generate", async (HttpContext ctx) =>
        {
            JsonElement req;
            try
            {
                req = await ctx.Request.ReadFromJsonAsync<JsonElement>(ctx.RequestAborted);
            }
            catch
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Invalid JSON payload.", ctx.RequestAborted);
                return;
            }

            string prompt = req.TryGetProperty("prompt", out var p) && p.ValueKind == JsonValueKind.String ? p.GetString() ?? "" : "";
            if (string.IsNullOrEmpty(prompt))
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Prompt is required.", ctx.RequestAborted);
                return;
            }

            bool stream = !req.TryGetProperty("stream", out var s) || s.ValueKind != JsonValueKind.False;
            var options = ParseSamplingOptions(req);

            if (!stream)
            {
                var result = await session.GenerateAsync(prompt, options, ct: ctx.RequestAborted);
                var respObj = new
                {
                    model = modelName,
                    created_at = DateTime.UtcNow.ToString("o"),
                    response = result.Text,
                    done = true,
                    total_duration = (long)(result.TotalDuration.TotalSeconds * 1e9),
                    prompt_eval_count = result.PromptTokens,
                    prompt_eval_duration = (long)(result.PromptDuration.TotalSeconds * 1e9),
                    eval_count = result.GeneratedTokens,
                    eval_duration = (long)(result.GenerationDuration.TotalSeconds * 1e9)
                };
                await ctx.Response.WriteAsJsonAsync(respObj, ctx.RequestAborted);
                return;
            }

            ctx.Response.ContentType = "application/x-ndjson";
            var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(256)
            {
                SingleWriter = true,
                SingleReader = true,
                FullMode = BoundedChannelFullMode.Wait
            });

            var genTask = Task.Run(async () =>
            {
                try
                {
                    return await session.GenerateAsync(
                        prompt,
                        options,
                        onToken: token => channel.Writer.TryWrite(token),
                        ct: ctx.RequestAborted);
                }
                finally
                {
                    channel.Writer.Complete();
                }
            });

            await foreach (var token in channel.Reader.ReadAllAsync(ctx.RequestAborted))
            {
                var chunk = new
                {
                    model = modelName,
                    created_at = DateTime.UtcNow.ToString("o"),
                    response = token,
                    done = false
                };
                await ctx.Response.WriteAsync(JsonSerializer.Serialize(chunk) + "\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }

            var genResult = await genTask;
            var finalChunk = new
            {
                model = modelName,
                created_at = DateTime.UtcNow.ToString("o"),
                response = "",
                done = true,
                total_duration = (long)(genResult.TotalDuration.TotalSeconds * 1e9),
                eval_count = genResult.GeneratedTokens,
                eval_duration = (long)(genResult.GenerationDuration.TotalSeconds * 1e9)
            };
            await ctx.Response.WriteAsync(JsonSerializer.Serialize(finalChunk) + "\n", ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
        });
    }

    private static SamplingOptions ParseSamplingOptions(JsonElement req)
    {
        var options = new SamplingOptions();
        if (req.TryGetProperty("temperature", out var tempProp) && tempProp.TryGetSingle(out float temp))
            options.Temperature = temp;
        if (req.TryGetProperty("top_p", out var topPProp) && topPProp.TryGetSingle(out float topP))
            options.TopP = topP;
        if (req.TryGetProperty("top_k", out var topKProp) && topKProp.TryGetInt32(out int topK))
            options.TopK = topK;
        if (req.TryGetProperty("max_tokens", out var maxTokProp) && maxTokProp.TryGetInt32(out int maxTok))
            options.MaxTokens = maxTok;
        else if (req.TryGetProperty("max_completion_tokens", out var maxCompProp) && maxCompProp.TryGetInt32(out int maxComp))
            options.MaxTokens = maxComp;
        if (req.TryGetProperty("repetition_penalty", out var repProp) && repProp.TryGetSingle(out float rep))
            options.RepetitionPenalty = rep;

        // Nested Ollama "options"
        if (req.TryGetProperty("options", out var subOpt) && subOpt.ValueKind == JsonValueKind.Object)
        {
            if (subOpt.TryGetProperty("temperature", out var subTemp) && subTemp.TryGetSingle(out float st))
                options.Temperature = st;
            if (subOpt.TryGetProperty("top_p", out var subTopP) && subTopP.TryGetSingle(out float sp))
                options.TopP = sp;
            if (subOpt.TryGetProperty("top_k", out var subTopK) && subTopK.TryGetInt32(out int sk))
                options.TopK = sk;
            if (subOpt.TryGetProperty("num_predict", out var subNum) && subNum.TryGetInt32(out int np))
                options.MaxTokens = np;
            if (subOpt.TryGetProperty("repeat_penalty", out var subRep) && subRep.TryGetSingle(out float sr))
                options.RepetitionPenalty = sr;
        }
        return options;
    }

    private static List<(string Role, string Content)> ParseMessages(JsonElement req)
    {
        var list = new List<(string Role, string Content)>();
        if (!req.TryGetProperty("messages", out var messages) || messages.ValueKind != JsonValueKind.Array)
            return list;

        foreach (var m in messages.EnumerateArray())
        {
            string role = m.TryGetProperty("role", out var r) ? r.GetString() ?? "user" : "user";
            string content = "";
            if (m.TryGetProperty("content", out var c))
            {
                if (c.ValueKind == JsonValueKind.String)
                {
                    content = c.GetString() ?? "";
                }
                else if (c.ValueKind == JsonValueKind.Array)
                {
                    var sb = new StringBuilder();
                    foreach (var part in c.EnumerateArray())
                    {
                        if (part.TryGetProperty("type", out var typeProp) &&
                            typeProp.GetString() == "text" &&
                            part.TryGetProperty("text", out var textProp))
                        {
                            sb.Append(textProp.GetString());
                        }
                    }
                    content = sb.ToString();
                }
            }
            list.Add((role, content));
        }
        return list;
    }
}
