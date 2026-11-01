namespace Glacier.Inference.Distributed.Cli;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Channels;
using System.Threading.Tasks;
using Glacier.Inference.Sampling;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;

internal sealed record OpenAiModelItem(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("created")] long Created,
    [property: JsonPropertyName("owned_by")] string OwnedBy);

internal sealed record OpenAiModelsListResponse(
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("data")] OpenAiModelItem[] Data);

internal sealed record ClusterStageStatusItem(
    [property: JsonPropertyName("stage_index")] int StageIndex,
    [property: JsonPropertyName("node_name")] string NodeName,
    [property: JsonPropertyName("host")] string Host,
    [property: JsonPropertyName("port")] int Port,
    [property: JsonPropertyName("start_layer")] int StartLayer,
    [property: JsonPropertyName("end_layer")] int EndLayer,
    [property: JsonPropertyName("layer_count")] int LayerCount,
    [property: JsonPropertyName("is_local")] bool IsLocal,
    [property: JsonPropertyName("backend")] string Backend);

internal sealed record ClusterStatusResponse(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("cluster")] string Cluster,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("total_layers")] int TotalLayers,
    [property: JsonPropertyName("embedding_dim")] int EmbeddingDim,
    [property: JsonPropertyName("vocab_size")] int VocabSize,
    [property: JsonPropertyName("stages_count")] int StagesCount,
    [property: JsonPropertyName("local_stages_count")] int LocalStagesCount,
    [property: JsonPropertyName("remote_stages_count")] int RemoteStagesCount,
    [property: JsonPropertyName("stages")] ClusterStageStatusItem[] Stages);

internal sealed record OpenAiChatMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string Content);

internal sealed record OpenAiUsage(
    [property: JsonPropertyName("prompt_tokens")] int PromptTokens,
    [property: JsonPropertyName("completion_tokens")] int CompletionTokens,
    [property: JsonPropertyName("total_tokens")] int TotalTokens);

internal sealed record OpenAiChatChoice(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("message")] OpenAiChatMessage Message,
    [property: JsonPropertyName("finish_reason")] string? FinishReason);

internal sealed record OpenAiChatCompletionResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("created")] long Created,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("choices")] OpenAiChatChoice[] Choices,
    [property: JsonPropertyName("usage")] OpenAiUsage Usage);

internal sealed record OpenAiChatDelta(
    [property: JsonPropertyName("role"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? Role = null,
    [property: JsonPropertyName("content"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? Content = null);

internal sealed record OpenAiChatStreamChoice(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("delta")] OpenAiChatDelta Delta,
    [property: JsonPropertyName("finish_reason")] string? FinishReason);

internal sealed record OpenAiChatCompletionChunk(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("created")] long Created,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("choices")] OpenAiChatStreamChoice[] Choices);

internal sealed record OpenAiErrorDetail(
    [property: JsonPropertyName("message")] string Message,
    [property: JsonPropertyName("type")] string Type,
    [property: JsonPropertyName("param")] string? Param,
    [property: JsonPropertyName("code")] int Code);

internal sealed record OpenAiErrorEnvelope(
    [property: JsonPropertyName("error")] OpenAiErrorDetail Error);

internal sealed record OpenAiTextChoice(
    [property: JsonPropertyName("index")] int Index,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("finish_reason")] string? FinishReason);

internal sealed record OpenAiTextCompletionResponse(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("created")] long Created,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("choices")] OpenAiTextChoice[] Choices,
    [property: JsonPropertyName("usage")] OpenAiUsage Usage);

internal sealed record OpenAiTextCompletionChunk(
    [property: JsonPropertyName("id")] string Id,
    [property: JsonPropertyName("object")] string Object,
    [property: JsonPropertyName("created")] long Created,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("choices")] OpenAiTextChoice[] Choices);

internal sealed record OllamaModelDetails(
    [property: JsonPropertyName("format")] string Format,
    [property: JsonPropertyName("family")] string Family,
    [property: JsonPropertyName("parameter_size")] string ParameterSize,
    [property: JsonPropertyName("quantization_level")] string QuantizationLevel);

internal sealed record OllamaTagModelItem(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("modified_at")] string ModifiedAt,
    [property: JsonPropertyName("size")] long Size,
    [property: JsonPropertyName("details")] OllamaModelDetails Details);

internal sealed record OllamaTagsResponse(
    [property: JsonPropertyName("models")] OllamaTagModelItem[] Models);

internal sealed record OllamaChatResponse(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("created_at")] string CreatedAt,
    [property: JsonPropertyName("message")] OpenAiChatMessage Message,
    [property: JsonPropertyName("done")] bool Done,
    [property: JsonPropertyName("total_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? TotalDuration = null,
    [property: JsonPropertyName("prompt_eval_count"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? PromptEvalCount = null,
    [property: JsonPropertyName("prompt_eval_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? PromptEvalDuration = null,
    [property: JsonPropertyName("eval_count"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? EvalCount = null,
    [property: JsonPropertyName("eval_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? EvalDuration = null);

internal sealed record OllamaGenerateResponse(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("created_at")] string CreatedAt,
    [property: JsonPropertyName("response")] string Response,
    [property: JsonPropertyName("done")] bool Done,
    [property: JsonPropertyName("total_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? TotalDuration = null,
    [property: JsonPropertyName("prompt_eval_count"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? PromptEvalCount = null,
    [property: JsonPropertyName("prompt_eval_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? PromptEvalDuration = null,
    [property: JsonPropertyName("eval_count"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? EvalCount = null,
    [property: JsonPropertyName("eval_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? EvalDuration = null);

[JsonSourceGenerationOptions(GenerationMode = JsonSourceGenerationMode.Default)]
[JsonSerializable(typeof(OpenAiModelsListResponse))]
[JsonSerializable(typeof(ClusterStatusResponse))]
[JsonSerializable(typeof(OpenAiChatCompletionResponse))]
[JsonSerializable(typeof(OpenAiChatCompletionChunk))]
[JsonSerializable(typeof(OpenAiErrorEnvelope))]
[JsonSerializable(typeof(OpenAiTextCompletionResponse))]
[JsonSerializable(typeof(OpenAiTextCompletionChunk))]
[JsonSerializable(typeof(OllamaTagsResponse))]
[JsonSerializable(typeof(OllamaChatResponse))]
[JsonSerializable(typeof(OllamaGenerateResponse))]
internal partial class DistributedServeJsonContext : JsonSerializerContext
{
}

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
        builder.Services.ConfigureHttpJsonOptions(options =>
        {
            options.SerializerOptions.TypeInfoResolverChain.Insert(0, DistributedServeJsonContext.Default);
        });
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
        app.MapGet("/v1/models", () =>
        {
            var payload = new OpenAiModelsListResponse(
                Object: "list",
                Data:
                [
                    new OpenAiModelItem(
                        Id: modelName,
                        Object: "model",
                        Created: DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        OwnedBy: "glacier-cluster")
                ]);
            return Results.Json(payload, DistributedServeJsonContext.Default.OpenAiModelsListResponse);
        });

        // 2. GET /v1/cluster/status & /health
        Func<IResult> getClusterStatus = () =>
        {
            var stageItems = new ClusterStageStatusItem[session.Stages.Length];
            for (int idx = 0; idx < session.Stages.Length; idx++)
            {
                var st = session.Stages[idx];
                bool isLocal = idx < session.LocalModels.Count;
                stageItems[idx] = new ClusterStageStatusItem(
                    StageIndex: idx,
                    NodeName: st.Name,
                    Host: st.Host,
                    Port: st.Port,
                    StartLayer: st.StartLayer,
                    EndLayer: st.EndLayer,
                    LayerCount: st.LayerCount,
                    IsLocal: isLocal,
                    Backend: isLocal ? session.LocalModels[idx].BackendName : "Remote Gigabit TCP Worker");
            }

            var statusPayload = new ClusterStatusResponse(
                Status: "healthy",
                Cluster: "Glacier Distributed LLM Cluster",
                Model: modelName,
                TotalLayers: session.TotalLayers,
                EmbeddingDim: session.Dim,
                VocabSize: session.VocabSize,
                StagesCount: session.Stages.Length,
                LocalStagesCount: session.LocalModels.Count,
                RemoteStagesCount: session.Streams.Count,
                Stages: stageItems);

            return Results.Json(statusPayload, DistributedServeJsonContext.Default.ClusterStatusResponse);
        };

        app.MapGet("/v1/cluster/status", getClusterStatus);
        app.MapGet("/health", getClusterStatus);

        // 3. POST /v1/chat/completions (OpenAI Standard Chat Completion)
        app.MapPost("/v1/chat/completions", async (HttpContext ctx) =>
        {
            List<(string Role, string Content)> messages;
            bool stream;
            SamplingOptions options;

            try
            {
                using var doc = await JsonDocument.ParseAsync(ctx.Request.Body, cancellationToken: ctx.RequestAborted);
                var req = doc.RootElement;
                messages = ParseMessages(req);
                stream = req.TryGetProperty("stream", out var s) &&
                         (s.ValueKind == JsonValueKind.True || (s.ValueKind != JsonValueKind.False && s.GetBoolean()));
                options = ParseSamplingOptions(req);
            }
            catch
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Invalid JSON payload.", ctx.RequestAborted);
                return;
            }

            if (messages.Count == 0)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Messages array required.", ctx.RequestAborted);
                return;
            }

            string formattedPrompt = session.Tokenizer != null
                ? session.Tokenizer.FormatChat(messages)
                : string.Join("\n", messages.Select(m => $"{m.Role}: {m.Content}"));
            string completionId = $"chatcmpl-{Guid.NewGuid():N}";
            long createdTimestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            if (!stream)
            {
                try
                {
                    var result = await session.GenerateAsync(formattedPrompt, options, ct: ctx.RequestAborted);
                    var resp = new OpenAiChatCompletionResponse(
                        Id: completionId,
                        Object: "chat.completion",
                        Created: createdTimestamp,
                        Model: modelName,
                        Choices:
                        [
                            new OpenAiChatChoice(
                                Index: 0,
                                Message: new OpenAiChatMessage("assistant", result.Text),
                                FinishReason: result.FinishReason)
                        ],
                        Usage: new OpenAiUsage(
                            PromptTokens: result.PromptTokens,
                            CompletionTokens: result.GeneratedTokens,
                            TotalTokens: result.PromptTokens + result.GeneratedTokens));

                    await ctx.Response.WriteAsJsonAsync(
                        resp,
                        DistributedServeJsonContext.Default.OpenAiChatCompletionResponse,
                        cancellationToken: ctx.RequestAborted);
                    return;
                }
                catch (OperationCanceledException) when (ctx.RequestAborted.IsCancellationRequested)
                {
                    return;
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"[Serve:Error] Non-streaming GenerateAsync failed: {ex}");
                    Console.ResetColor();
                    if (!ctx.Response.HasStarted)
                    {
                        ctx.Response.StatusCode = 500;
                        var errPayload = new OpenAiErrorEnvelope(
                            new OpenAiErrorDetail(
                                Message: ex.Message,
                                Type: "cluster_generation_error",
                                Param: null,
                                Code: 500));
                        await ctx.Response.WriteAsJsonAsync(
                            errPayload,
                            DistributedServeJsonContext.Default.OpenAiErrorEnvelope,
                            cancellationToken: ctx.RequestAborted);
                    }
                    return;
                }
            }

            // Streaming SSE response
            ctx.Response.ContentType = "text/event-stream; charset=utf-8";
            ctx.Response.Headers.CacheControl = "no-cache";

            var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions
            {
                SingleWriter = true,
                SingleReader = true
            });

            var genTask = Task.Run(async () =>
            {
                try
                {
                    return await session.GenerateAsync(
                        formattedPrompt,
                        options,
                        onToken: token =>
                        {
                            if (!channel.Writer.TryWrite(token))
                                throw new InvalidOperationException("Failed to enqueue streamed token.");
                        },
                        ct: ctx.RequestAborted);
                }
                catch (OperationCanceledException) when (ctx.RequestAborted.IsCancellationRequested)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"[Serve:Error] Streaming GenerateAsync failed: {ex}");
                    Console.ResetColor();
                    throw;
                }
                finally
                {
                    channel.Writer.TryComplete();
                }
            });

            try
            {
                var initialChunk = new OpenAiChatCompletionChunk(
                    Id: completionId,
                    Object: "chat.completion.chunk",
                    Created: createdTimestamp,
                    Model: modelName,
                    Choices:
                    [
                        new OpenAiChatStreamChoice(
                            Index: 0,
                            Delta: new OpenAiChatDelta(Role: "assistant", Content: ""),
                            FinishReason: null)
                    ]);
                await ctx.Response.WriteAsync(
                    $"data: {JsonSerializer.Serialize(initialChunk, DistributedServeJsonContext.Default.OpenAiChatCompletionChunk)}\n\n",
                    ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);

                await foreach (var token in channel.Reader.ReadAllAsync(ctx.RequestAborted))
                {
                    var chunk = new OpenAiChatCompletionChunk(
                        Id: completionId,
                        Object: "chat.completion.chunk",
                        Created: createdTimestamp,
                        Model: modelName,
                        Choices:
                        [
                            new OpenAiChatStreamChoice(
                                Index: 0,
                                Delta: new OpenAiChatDelta(Content: token),
                                FinishReason: null)
                        ]);
                    await ctx.Response.WriteAsync(
                        $"data: {JsonSerializer.Serialize(chunk, DistributedServeJsonContext.Default.OpenAiChatCompletionChunk)}\n\n",
                        ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }

                var genResult = await genTask;

                var finalChunk = new OpenAiChatCompletionChunk(
                    Id: completionId,
                    Object: "chat.completion.chunk",
                    Created: createdTimestamp,
                    Model: modelName,
                    Choices:
                    [
                        new OpenAiChatStreamChoice(
                            Index: 0,
                            Delta: new OpenAiChatDelta(),
                            FinishReason: genResult.FinishReason)
                    ]);
                await ctx.Response.WriteAsync(
                    $"data: {JsonSerializer.Serialize(finalChunk, DistributedServeJsonContext.Default.OpenAiChatCompletionChunk)}\n\n",
                    ctx.RequestAborted);
                await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }
            catch (OperationCanceledException) when (ctx.RequestAborted.IsCancellationRequested)
            {
                try { await genTask; } catch { }
            }
            catch (Exception ex)
            {
                try { await genTask; } catch { }
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[Serve:Error] Generator task faulted: {ex}");
                Console.ResetColor();

                if (!ctx.RequestAborted.IsCancellationRequested)
                {
                    var errChunk = new OpenAiChatCompletionChunk(
                        Id: completionId,
                        Object: "chat.completion.chunk",
                        Created: createdTimestamp,
                        Model: modelName,
                        Choices:
                        [
                            new OpenAiChatStreamChoice(
                                Index: 0,
                                Delta: new OpenAiChatDelta(Content: $"\n\n[Cluster Generation Error: {ex.Message}]"),
                                FinishReason: "error")
                        ]);
                    await ctx.Response.WriteAsync(
                        $"data: {JsonSerializer.Serialize(errChunk, DistributedServeJsonContext.Default.OpenAiChatCompletionChunk)}\n\n",
                        ctx.RequestAborted);
                    await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }
            }
        });

        // 4. POST /v1/completions (OpenAI Standard Text Completion)
        app.MapPost("/v1/completions", async (HttpContext ctx) =>
        {
            string prompt = "";
            bool stream;
            SamplingOptions options;

            try
            {
                using var doc = await JsonDocument.ParseAsync(ctx.Request.Body, cancellationToken: ctx.RequestAborted);
                var req = doc.RootElement;
                if (req.TryGetProperty("prompt", out var p))
                {
                    if (p.ValueKind == JsonValueKind.String) prompt = p.GetString() ?? "";
                    else if (p.ValueKind == JsonValueKind.Array && p.GetArrayLength() > 0) prompt = p[0].GetString() ?? "";
                }
                stream = req.TryGetProperty("stream", out var s) &&
                         (s.ValueKind == JsonValueKind.True || (s.ValueKind != JsonValueKind.False && s.GetBoolean()));
                options = ParseSamplingOptions(req);
            }
            catch
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Invalid JSON payload.", ctx.RequestAborted);
                return;
            }

            if (string.IsNullOrEmpty(prompt))
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Prompt is required.", ctx.RequestAborted);
                return;
            }

            string completionId = $"cmpl-{Guid.NewGuid():N}";
            long createdTimestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

            if (!stream)
            {
                try
                {
                    var result = await session.GenerateAsync(prompt, options, ct: ctx.RequestAborted);
                    var resp = new OpenAiTextCompletionResponse(
                        Id: completionId,
                        Object: "text_completion",
                        Created: createdTimestamp,
                        Model: modelName,
                        Choices:
                        [
                            new OpenAiTextChoice(
                                Index: 0,
                                Text: result.Text,
                                FinishReason: result.FinishReason)
                        ],
                        Usage: new OpenAiUsage(
                            PromptTokens: result.PromptTokens,
                            CompletionTokens: result.GeneratedTokens,
                            TotalTokens: result.PromptTokens + result.GeneratedTokens));

                    await ctx.Response.WriteAsJsonAsync(
                        resp,
                        DistributedServeJsonContext.Default.OpenAiTextCompletionResponse,
                        cancellationToken: ctx.RequestAborted);
                    return;
                }
                catch (OperationCanceledException) when (ctx.RequestAborted.IsCancellationRequested)
                {
                    return;
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"[Serve:Error] /v1/completions failed: {ex}");
                    Console.ResetColor();
                    if (!ctx.Response.HasStarted)
                    {
                        ctx.Response.StatusCode = 500;
                        var errPayload = new OpenAiErrorEnvelope(
                            new OpenAiErrorDetail(ex.Message, "cluster_generation_error", null, 500));
                        await ctx.Response.WriteAsJsonAsync(
                            errPayload,
                            DistributedServeJsonContext.Default.OpenAiErrorEnvelope,
                            cancellationToken: ctx.RequestAborted);
                    }
                    return;
                }
            }

            ctx.Response.ContentType = "text/event-stream; charset=utf-8";
            ctx.Response.Headers.CacheControl = "no-cache";

            var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions
            {
                SingleWriter = true,
                SingleReader = true
            });

            var genTask = Task.Run(async () =>
            {
                try
                {
                    return await session.GenerateAsync(
                        prompt,
                        options,
                        onToken: token =>
                        {
                            if (!channel.Writer.TryWrite(token))
                                throw new InvalidOperationException("Failed to enqueue streamed token.");
                        },
                        ct: ctx.RequestAborted);
                }
                finally
                {
                    channel.Writer.TryComplete();
                }
            });

            try
            {
                await foreach (var token in channel.Reader.ReadAllAsync(ctx.RequestAborted))
                {
                    var chunk = new OpenAiTextCompletionChunk(
                        Id: completionId,
                        Object: "text_completion",
                        Created: createdTimestamp,
                        Model: modelName,
                        Choices:
                        [
                            new OpenAiTextChoice(Index: 0, Text: token, FinishReason: null)
                        ]);
                    await ctx.Response.WriteAsync(
                        $"data: {JsonSerializer.Serialize(chunk, DistributedServeJsonContext.Default.OpenAiTextCompletionChunk)}\n\n",
                        ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }

                var genResult = await genTask;
                var finalChunk = new OpenAiTextCompletionChunk(
                    Id: completionId,
                    Object: "text_completion",
                    Created: createdTimestamp,
                    Model: modelName,
                    Choices:
                    [
                        new OpenAiTextChoice(Index: 0, Text: "", FinishReason: genResult.FinishReason)
                    ]);
                await ctx.Response.WriteAsync(
                    $"data: {JsonSerializer.Serialize(finalChunk, DistributedServeJsonContext.Default.OpenAiTextCompletionChunk)}\n\n",
                    ctx.RequestAborted);
                await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }
            catch (OperationCanceledException) when (ctx.RequestAborted.IsCancellationRequested)
            {
                try { await genTask; } catch { }
            }
            catch (Exception ex)
            {
                try { await genTask; } catch { }
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[Serve:Error] Streaming /v1/completions faulted: {ex}");
                Console.ResetColor();
                if (!ctx.RequestAborted.IsCancellationRequested)
                {
                    var errChunk = new OpenAiTextCompletionChunk(
                        Id: completionId,
                        Object: "text_completion",
                        Created: createdTimestamp,
                        Model: modelName,
                        Choices:
                        [
                            new OpenAiTextChoice(Index: 0, Text: $"\n\n[Cluster Generation Error: {ex.Message}]", FinishReason: "error")
                        ]);
                    await ctx.Response.WriteAsync(
                        $"data: {JsonSerializer.Serialize(errChunk, DistributedServeJsonContext.Default.OpenAiTextCompletionChunk)}\n\n",
                        ctx.RequestAborted);
                    await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }
            }
        });

        // 5. GET /api/tags (Ollama compatible tags list)
        app.MapGet("/api/tags", () =>
        {
            var payload = new OllamaTagsResponse(
                Models:
                [
                    new OllamaTagModelItem(
                        Name: modelName,
                        Model: modelName,
                        ModifiedAt: DateTime.UtcNow.ToString("o"),
                        Size: File.Exists(session.ModelPath) ? new FileInfo(session.ModelPath).Length : 0L,
                        Details: new OllamaModelDetails(
                            Format: "gguf",
                            Family: session.MasterGguf?.Architecture ?? session.Architecture,
                            ParameterSize: $"{session.TotalLayers}L",
                            QuantizationLevel: "ClusterDistributed"))
                ]);
            return Results.Json(payload, DistributedServeJsonContext.Default.OllamaTagsResponse);
        });

        // 6. POST /api/chat (Ollama compatible chat)
        app.MapPost("/api/chat", async (HttpContext ctx) =>
        {
            List<(string Role, string Content)> messages;
            bool stream;
            SamplingOptions options;

            try
            {
                using var doc = await JsonDocument.ParseAsync(ctx.Request.Body, cancellationToken: ctx.RequestAborted);
                var req = doc.RootElement;
                messages = ParseMessages(req);
                stream = !req.TryGetProperty("stream", out var s) || s.ValueKind != JsonValueKind.False;
                options = ParseSamplingOptions(req);
            }
            catch
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Invalid JSON payload.", ctx.RequestAborted);
                return;
            }

            if (messages.Count == 0)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Messages array required.", ctx.RequestAborted);
                return;
            }

            string formattedPrompt = session.Tokenizer != null
                ? session.Tokenizer.FormatChat(messages)
                : string.Join("\n", messages.Select(m => $"{m.Role}: {m.Content}"));

            if (!stream)
            {
                try
                {
                    var result = await session.GenerateAsync(formattedPrompt, options, ct: ctx.RequestAborted);
                    var respObj = new OllamaChatResponse(
                        Model: modelName,
                        CreatedAt: DateTime.UtcNow.ToString("o"),
                        Message: new OpenAiChatMessage("assistant", result.Text),
                        Done: true,
                        TotalDuration: (long)(result.TotalDuration.TotalSeconds * 1e9),
                        PromptEvalCount: result.PromptTokens,
                        PromptEvalDuration: (long)(result.PromptDuration.TotalSeconds * 1e9),
                        EvalCount: result.GeneratedTokens,
                        EvalDuration: (long)(result.GenerationDuration.TotalSeconds * 1e9));

                    await ctx.Response.WriteAsJsonAsync(
                        respObj,
                        DistributedServeJsonContext.Default.OllamaChatResponse,
                        cancellationToken: ctx.RequestAborted);
                    return;
                }
                catch (OperationCanceledException) when (ctx.RequestAborted.IsCancellationRequested)
                {
                    return;
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"[Serve:Error] /api/chat failed: {ex}");
                    Console.ResetColor();
                    if (!ctx.Response.HasStarted)
                    {
                        ctx.Response.StatusCode = 500;
                        var errPayload = new OpenAiErrorEnvelope(
                            new OpenAiErrorDetail(ex.Message, "cluster_generation_error", null, 500));
                        await ctx.Response.WriteAsJsonAsync(
                            errPayload,
                            DistributedServeJsonContext.Default.OpenAiErrorEnvelope,
                            cancellationToken: ctx.RequestAborted);
                    }
                    return;
                }
            }

            ctx.Response.ContentType = "application/x-ndjson";
            var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions
            {
                SingleWriter = true,
                SingleReader = true
            });

            var genTask = Task.Run(async () =>
            {
                try
                {
                    return await session.GenerateAsync(
                        formattedPrompt,
                        options,
                        onToken: token =>
                        {
                            if (!channel.Writer.TryWrite(token))
                                throw new InvalidOperationException("Failed to enqueue streamed token.");
                        },
                        ct: ctx.RequestAborted);
                }
                finally
                {
                    channel.Writer.TryComplete();
                }
            });

            try
            {
                await foreach (var token in channel.Reader.ReadAllAsync(ctx.RequestAborted))
                {
                    var chunk = new OllamaChatResponse(
                        Model: modelName,
                        CreatedAt: DateTime.UtcNow.ToString("o"),
                        Message: new OpenAiChatMessage("assistant", token),
                        Done: false);
                    await ctx.Response.WriteAsync(
                        JsonSerializer.Serialize(chunk, DistributedServeJsonContext.Default.OllamaChatResponse) + "\n",
                        ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }

                var genResult = await genTask;
                var finalChunk = new OllamaChatResponse(
                    Model: modelName,
                    CreatedAt: DateTime.UtcNow.ToString("o"),
                    Message: new OpenAiChatMessage("assistant", ""),
                    Done: true,
                    TotalDuration: (long)(genResult.TotalDuration.TotalSeconds * 1e9),
                    EvalCount: genResult.GeneratedTokens,
                    EvalDuration: (long)(genResult.GenerationDuration.TotalSeconds * 1e9));
                await ctx.Response.WriteAsync(
                    JsonSerializer.Serialize(finalChunk, DistributedServeJsonContext.Default.OllamaChatResponse) + "\n",
                    ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }
            catch (OperationCanceledException) when (ctx.RequestAborted.IsCancellationRequested)
            {
                try { await genTask; } catch { }
            }
            catch (Exception ex)
            {
                try { await genTask; } catch { }
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[Serve:Error] Streaming /api/chat faulted: {ex}");
                Console.ResetColor();
                if (!ctx.RequestAborted.IsCancellationRequested)
                {
                    var errChunk = new OllamaChatResponse(
                        Model: modelName,
                        CreatedAt: DateTime.UtcNow.ToString("o"),
                        Message: new OpenAiChatMessage("assistant", $"\n\n[Cluster Generation Error: {ex.Message}]"),
                        Done: true);
                    await ctx.Response.WriteAsync(
                        JsonSerializer.Serialize(errChunk, DistributedServeJsonContext.Default.OllamaChatResponse) + "\n",
                        ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }
            }
        });

        // 7. POST /api/generate (Ollama compatible generate)
        app.MapPost("/api/generate", async (HttpContext ctx) =>
        {
            string prompt;
            bool stream;
            SamplingOptions options;

            try
            {
                using var doc = await JsonDocument.ParseAsync(ctx.Request.Body, cancellationToken: ctx.RequestAborted);
                var req = doc.RootElement;
                prompt = req.TryGetProperty("prompt", out var p) && p.ValueKind == JsonValueKind.String ? p.GetString() ?? "" : "";
                stream = !req.TryGetProperty("stream", out var s) || s.ValueKind != JsonValueKind.False;
                options = ParseSamplingOptions(req);
            }
            catch
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Invalid JSON payload.", ctx.RequestAborted);
                return;
            }

            if (string.IsNullOrEmpty(prompt))
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsync("Prompt is required.", ctx.RequestAborted);
                return;
            }

            if (!stream)
            {
                try
                {
                    var result = await session.GenerateAsync(prompt, options, ct: ctx.RequestAborted);
                    var respObj = new OllamaGenerateResponse(
                        Model: modelName,
                        CreatedAt: DateTime.UtcNow.ToString("o"),
                        Response: result.Text,
                        Done: true,
                        TotalDuration: (long)(result.TotalDuration.TotalSeconds * 1e9),
                        PromptEvalCount: result.PromptTokens,
                        PromptEvalDuration: (long)(result.PromptDuration.TotalSeconds * 1e9),
                        EvalCount: result.GeneratedTokens,
                        EvalDuration: (long)(result.GenerationDuration.TotalSeconds * 1e9));

                    await ctx.Response.WriteAsJsonAsync(
                        respObj,
                        DistributedServeJsonContext.Default.OllamaGenerateResponse,
                        cancellationToken: ctx.RequestAborted);
                    return;
                }
                catch (OperationCanceledException) when (ctx.RequestAborted.IsCancellationRequested)
                {
                    return;
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"[Serve:Error] /api/generate failed: {ex}");
                    Console.ResetColor();
                    if (!ctx.Response.HasStarted)
                    {
                        ctx.Response.StatusCode = 500;
                        var errPayload = new OpenAiErrorEnvelope(
                            new OpenAiErrorDetail(ex.Message, "cluster_generation_error", null, 500));
                        await ctx.Response.WriteAsJsonAsync(
                            errPayload,
                            DistributedServeJsonContext.Default.OpenAiErrorEnvelope,
                            cancellationToken: ctx.RequestAborted);
                    }
                    return;
                }
            }

            ctx.Response.ContentType = "application/x-ndjson";
            var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions
            {
                SingleWriter = true,
                SingleReader = true
            });

            var genTask = Task.Run(async () =>
            {
                try
                {
                    return await session.GenerateAsync(
                        prompt,
                        options,
                        onToken: token =>
                        {
                            if (!channel.Writer.TryWrite(token))
                                throw new InvalidOperationException("Failed to enqueue streamed token.");
                        },
                        ct: ctx.RequestAborted);
                }
                finally
                {
                    channel.Writer.TryComplete();
                }
            });

            try
            {
                await foreach (var token in channel.Reader.ReadAllAsync(ctx.RequestAborted))
                {
                    var chunk = new OllamaGenerateResponse(
                        Model: modelName,
                        CreatedAt: DateTime.UtcNow.ToString("o"),
                        Response: token,
                        Done: false);
                    await ctx.Response.WriteAsync(
                        JsonSerializer.Serialize(chunk, DistributedServeJsonContext.Default.OllamaGenerateResponse) + "\n",
                        ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }

                var genResult = await genTask;
                var finalChunk = new OllamaGenerateResponse(
                    Model: modelName,
                    CreatedAt: DateTime.UtcNow.ToString("o"),
                    Response: "",
                    Done: true,
                    TotalDuration: (long)(genResult.TotalDuration.TotalSeconds * 1e9),
                    EvalCount: genResult.GeneratedTokens,
                    EvalDuration: (long)(genResult.GenerationDuration.TotalSeconds * 1e9));
                await ctx.Response.WriteAsync(
                    JsonSerializer.Serialize(finalChunk, DistributedServeJsonContext.Default.OllamaGenerateResponse) + "\n",
                    ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }
            catch (OperationCanceledException) when (ctx.RequestAborted.IsCancellationRequested)
            {
                try { await genTask; } catch { }
            }
            catch (Exception ex)
            {
                try { await genTask; } catch { }
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[Serve:Error] Streaming /api/generate faulted: {ex}");
                Console.ResetColor();
                if (!ctx.RequestAborted.IsCancellationRequested)
                {
                    var errChunk = new OllamaGenerateResponse(
                        Model: modelName,
                        CreatedAt: DateTime.UtcNow.ToString("o"),
                        Response: $"\n\n[Cluster Generation Error: {ex.Message}]",
                        Done: true);
                    await ctx.Response.WriteAsync(
                        JsonSerializer.Serialize(errChunk, DistributedServeJsonContext.Default.OllamaGenerateResponse) + "\n",
                        ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }
            }
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
