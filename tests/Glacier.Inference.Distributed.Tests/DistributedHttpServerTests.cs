namespace Glacier.Inference.Distributed.Tests;

using System;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Glacier.Inference.Distributed.Cli;
using Xunit;

public class DistributedHttpServerTests
{
    private static int GetAvailablePort()
    {
        using var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    [Fact]
    public async Task V1Models_And_ClusterStatus_ReturnExpectedMetadata()
    {
        int port = GetAvailablePort();
        using var session = DistributedClusterSession.CreateMockSession("qwen2-57b-a14b-instruct", totalLayers: 28, dim: 3584, vocabSize: 151936);
        var app = Program.CreateServeApp(session, host: "127.0.0.1", port: port);

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            // 1. GET /v1/models
            var modelsResp = await client.GetAsync("/v1/models");
            Assert.Equal(HttpStatusCode.OK, modelsResp.StatusCode);
            var modelsJson = await modelsResp.Content.ReadFromJsonAsync<JsonElement>();
            Assert.Equal("list", modelsJson.GetProperty("object").GetString());
            var dataArr = modelsJson.GetProperty("data");
            Assert.True(dataArr.GetArrayLength() > 0);
            var first = dataArr[0];
            Assert.Equal("qwen2-57b-a14b-instruct", first.GetProperty("id").GetString());
            Assert.Equal("glacier-cluster", first.GetProperty("owned_by").GetString());

            // 2. GET /v1/cluster/status
            var statusResp = await client.GetAsync("/v1/cluster/status");
            Assert.Equal(HttpStatusCode.OK, statusResp.StatusCode);
            var statusJson = await statusResp.Content.ReadFromJsonAsync<JsonElement>();
            Assert.Equal("healthy", statusJson.GetProperty("status").GetString());
            Assert.Equal(28, statusJson.GetProperty("total_layers").GetInt32());
            Assert.Equal(3584, statusJson.GetProperty("embedding_dim").GetInt32());
            Assert.True(statusJson.GetProperty("stages").GetArrayLength() == 2);

            // 3. GET /health
            var healthResp = await client.GetAsync("/health");
            Assert.Equal(HttpStatusCode.OK, healthResp.StatusCode);

            // 4. GET /api/tags (Ollama compatible)
            var tagsResp = await client.GetAsync("/api/tags");
            Assert.Equal(HttpStatusCode.OK, tagsResp.StatusCode);
            var tagsJson = await tagsResp.Content.ReadFromJsonAsync<JsonElement>();
            Assert.True(tagsJson.GetProperty("models").GetArrayLength() > 0);
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }

    [Fact]
    public async Task ChatCompletions_NonStreaming_ReturnsOpenAiCompatibleResponse()
    {
        int port = GetAvailablePort();
        using var session = DistributedClusterSession.CreateMockSession("qwen2-57b-a14b-instruct");
        var app = Program.CreateServeApp(session, host: "127.0.0.1", port: port);

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            var requestBody = new
            {
                model = "qwen2-57b-a14b-instruct",
                messages = new[]
                {
                    new { role = "user", content = "Explain distributed tensor pipelining." }
                },
                stream = false,
                max_tokens = 50
            };

            var resp = await client.PostAsJsonAsync("/v1/chat/completions", requestBody);
            Assert.Equal(HttpStatusCode.OK, resp.StatusCode);

            var json = await resp.Content.ReadFromJsonAsync<JsonElement>();
            Assert.True(json.TryGetProperty("id", out var idProp) && idProp.GetString()!.StartsWith("chatcmpl-"));
            Assert.Equal("chat.completion", json.GetProperty("object").GetString());
            Assert.Equal("qwen2-57b-a14b-instruct", json.GetProperty("model").GetString());

            var choices = json.GetProperty("choices");
            Assert.True(choices.GetArrayLength() == 1);
            var choice = choices[0];
            Assert.Equal("assistant", choice.GetProperty("message").GetProperty("role").GetString());
            Assert.False(string.IsNullOrWhiteSpace(choice.GetProperty("message").GetProperty("content").GetString()));
            Assert.Equal("stop", choice.GetProperty("finish_reason").GetString());

            var usage = json.GetProperty("usage");
            Assert.True(usage.GetProperty("prompt_tokens").GetInt32() > 0);
            Assert.True(usage.GetProperty("completion_tokens").GetInt32() > 0);
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }

    [Fact]
    public async Task ChatCompletions_Streaming_EmitsServerSentEvents()
    {
        int port = GetAvailablePort();
        using var session = DistributedClusterSession.CreateMockSession("qwen2-57b-a14b-instruct");
        var app = Program.CreateServeApp(session, host: "127.0.0.1", port: port);

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            var requestBody = new
            {
                model = "qwen2-57b-a14b-instruct",
                messages = new[]
                {
                    new { role = "user", content = "Stream token output across cluster." }
                },
                stream = true,
                max_tokens = 20
            };

            var resp = await client.PostAsJsonAsync("/v1/chat/completions", requestBody);
            Assert.Equal(HttpStatusCode.OK, resp.StatusCode);
            Assert.Contains("text/event-stream", resp.Content.Headers.ContentType?.ToString());

            string sseContent = await resp.Content.ReadAsStringAsync();
            Assert.Contains("data: {\"id\":", sseContent);
            Assert.Contains("\"chat.completion.chunk\"", sseContent);
            Assert.Contains("data: [DONE]", sseContent);
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }

    [Fact]
    public async Task TextCompletions_ReturnsExpectedFormat()
    {
        int port = GetAvailablePort();
        using var session = DistributedClusterSession.CreateMockSession("qwen2-57b-a14b-instruct");
        var app = Program.CreateServeApp(session, host: "127.0.0.1", port: port);

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            var requestBody = new
            {
                model = "qwen2-57b-a14b-instruct",
                prompt = "Glacier is a distributed inference framework",
                stream = false,
                max_tokens = 30
            };

            var resp = await client.PostAsJsonAsync("/v1/completions", requestBody);
            Assert.Equal(HttpStatusCode.OK, resp.StatusCode);

            var json = await resp.Content.ReadFromJsonAsync<JsonElement>();
            Assert.Equal("text_completion", json.GetProperty("object").GetString());
            var choices = json.GetProperty("choices");
            Assert.True(choices.GetArrayLength() == 1);
            Assert.False(string.IsNullOrWhiteSpace(choices[0].GetProperty("text").GetString()));
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }

    [Fact]
    public async Task ApiKeyAuth_EnforcesSecurityChallenge_WhenConfigured()
    {
        const string testKey = "glacier-cluster-auth-key-secret-999";
        int port = GetAvailablePort();
        using var session = DistributedClusterSession.CreateMockSession("qwen2-57b-a14b-instruct");
        var app = Program.CreateServeApp(session, apiKey: testKey, host: "127.0.0.1", port: port);

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            // 1. Without credentials -> 401 Unauthorized
            var unauthResp = await client.GetAsync("/v1/models");
            Assert.Equal(HttpStatusCode.Unauthorized, unauthResp.StatusCode);

            // 2. With wrong Bearer token -> 401 Unauthorized
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", "wrong-password");
            var wrongResp = await client.GetAsync("/v1/models");
            Assert.Equal(HttpStatusCode.Unauthorized, wrongResp.StatusCode);

            // 3. With correct Bearer token -> 200 OK
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", testKey);
            var okResp = await client.GetAsync("/v1/models");
            Assert.Equal(HttpStatusCode.OK, okResp.StatusCode);

            // 4. With correct x-api-key header -> 200 OK
            using var client2 = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };
            client2.DefaultRequestHeaders.Add("x-api-key", testKey);
            var xApiKeyResp = await client2.GetAsync("/v1/models");
            Assert.Equal(HttpStatusCode.OK, xApiKeyResp.StatusCode);
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }

    [Fact]
    public async Task Endpoints_RejectInvalidPayloads()
    {
        int port = GetAvailablePort();
        using var session = DistributedClusterSession.CreateMockSession("qwen2-57b-a14b-instruct");
        var app = Program.CreateServeApp(session, host: "127.0.0.1", port: port);

        await app.StartAsync();
        try
        {
            using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") };

            // 1. POST /v1/chat/completions without messages -> 400 Bad Request
            var badChat = await client.PostAsJsonAsync("/v1/chat/completions", new { model = "test" });
            Assert.Equal(HttpStatusCode.BadRequest, badChat.StatusCode);

            // 2. POST /v1/completions without prompt -> 400 Bad Request
            var badCmpl = await client.PostAsJsonAsync("/v1/completions", new { model = "test" });
            Assert.Equal(HttpStatusCode.BadRequest, badCmpl.StatusCode);

            // 3. POST /api/generate without prompt -> 400 Bad Request
            var badGen = await client.PostAsJsonAsync("/api/generate", new { model = "test" });
            Assert.Equal(HttpStatusCode.BadRequest, badGen.StatusCode);
        }
        finally
        {
            await app.StopAsync();
            await app.DisposeAsync();
        }
    }
}
