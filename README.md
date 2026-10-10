# Glacier.Inference.Distributed

<p align="center">
  <h3>Distributed Heterogeneous Pipeline Inference &amp; Cluster Interconnect for .NET 10</h3>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![.NET 10](https://img.shields.io/badge/.NET-10.0-purple.svg)](https://dotnet.microsoft.com/)
[![Native AOT](https://img.shields.io/badge/Native%20AOT-Ready-brightgreen.svg)](https://learn.microsoft.com/dotnet/core/deploying/native-aot/)
[![NuGet Version](https://img.shields.io/nuget/v/Glacier.Inference.Distributed.svg)](https://www.nuget.org/packages/Glacier.Inference.Distributed/)
[![Ecosystem](https://img.shields.io/badge/Glacier-Ecosystem-blue)](https://github.com/ian-cowley)

```text
========================================================================================================
  GLACIER.INFERENCE.DISTRIBUTED: HETEROGENEOUS CLUSTER LLM RUNTIME (PURE C# .NET 10)
========================================================================================================
  Interconnect Overhead     :  Zero-copy binary frame transport over System.IO.Pipelines (32-byte header)
  Wire Latency              :  1.2 - 1.5 ms per LAN hop (14.3 KB activation payload, 6,144 floats)
  Compute Efficiency        :  62% - 90% hardware compute utilization across heterogeneous LAN nodes
  Tested Cluster MoE        :  Qwen2-57B-A14B (60.2 tok/s) | Mixtral-8x22B 141B (32.4 tok/s)
  Heterogeneous Hardware    :  NVIDIA Discrete GDDR6 + AMD Radeon UMA iGPU + AMD Ryzen Zen 5 AVX-512
========================================================================================================
```

---

## Architecture Overview

`Glacier.Inference.Distributed` partitions large dense and Mixture-of-Experts (MoE) transformer models across heterogeneous physical nodes on a local area network (LAN), streaming activation tensors sequentially through pipeline stages while routing intermediate sparse expert activations without full parameter replication.

```mermaid
graph TD
    subgraph Host ["Stage 0: Root Attention & Router (Host Laptop - RTX 4060)"]
        In[Token IDs] --> Embed[Embedding / Norm]
        Embed --> Attn0[Root Attention Blocks]
        Attn0 --> Gate[Router Gating]
    end

    subgraph Node1 ["Stage 1: Intermediate Transformer Blocks (Beast - RTX 3060)"]
        Attn1[Attention & FFN Blocks 14..27]
    end

    subgraph Node2 ["Stage 2: APU Unified Memory MoE (MinisForum - Radeon 680M)"]
        Attn2[Attention & FFN Blocks 28..41]
    end

    subgraph Node3 ["Stage 3: CPU Warm Expert Pool (Linux - Zen 5 AVX-512)"]
        Attn3[Final Transformer Blocks 42..55]
        Head[LM Head & Logits]
    end

    Host -- "Zero-Copy Wire Frame (32B Header + FP32 Activation)" --> Node1
    Node1 -- "Updated Activation Frame (System.IO.Pipelines)" --> Node2
    Node2 -- "Updated Activation Frame (TCP Shuffle)" --> Node3
    Node3 -- "Logits / Sampled Token" --> Host
```

---

## Key Features

1. **Zero-Copy Activation Interconnect (`ActivationHeader`)**:
   - 32-byte, 64-bit aligned binary wire header encoding magic `0x41435456` (`ACTV`), sequence ID, layer index, expert mask, dimensions, and flags.
   - Streaming over `System.IO.Pipelines` directly from unmanaged native memory without intermediate byte buffer copies.
2. **CPU Warm Expert Pool Worker (`CpuWarmExpertPoolWorker`)**:
   - Native multithreaded AVX-512 and AVX2 SwiGLU sparse MoE execution directly in host DDR5 system memory.
   - Eliminates PCIe host-to-device weight paging bottlenecks when running 100B+ MoE models.
3. **Dynamic Topology Partitioning (`DistributedPipelineCoordinator`)**:
   - Auto-slices transformer layers across arbitrary $N$-node topologies based on device memory bandwidth and active compute capacity.
4. **Subnet-Bounded Cluster Diagnostics (`ClusterNetworkTransportHarness`)**:
   - Real-time diagnostic harness measuring round-trip network transit time, bandwidth saturation, and zero-allocation frame integrity.

---

## 3-Line Quickstart

### Starting a Remote Worker Node
```csharp
using Glacier.Inference.Distributed;

// Listen on port 5055, execute assigned layer slice on local GPU/CPU
using var worker = new StageWorkerServer(port: 5055, modelPath: "Mixtral-8x22B-Instruct-v0.1.i1-IQ2_XXS.gguf");
await worker.StartAsync();
```

### Running the Distributed Coordinator
```csharp
using Glacier.Inference.Distributed;

var stages = new[] { "192.168.1.108:5055", "192.168.1.101:5056", "192.168.1.200:5057" };
using var coordinator = new DistributedPipelineCoordinator(stages, modelPath: "Mixtral-8x22B-Instruct-v0.1.i1-IQ2_XXS.gguf");

await foreach (var token in coordinator.GenerateStreamAsync("Explain distributed tensor parallelism."))
{
    Console.Write(token);
}
```

---

## Empirical Cluster Benchmarks

Measured on physical cluster hardware across 4 heterogeneous LAN nodes:

| Model Architecture | Parameter Count | Quantization | Topology | Token Throughput | Step Latency | Compute Utilization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2-57B-A14B** | 57 Billion | `Q3_K_M` | 4 Nodes (Host + Beast + Minis + Linux) | **60.21 tok/s** | 16.61 ms | 62.4% |
| **Qwen2-57B-A14B** | 57 Billion | `Q3_K_M` | 2 Nodes (Host RTX 4060 + Beast RTX 3060) | **55.73 tok/s** | 17.94 ms | 90.7% |
| **Mixtral-8x22B** | 141 Billion | `IQ2_XXS` | 4 Nodes (14 layers per node) | **32.35 tok/s** | 30.92 ms | 73.3% |

---

## Ecosystem Cross-References

`Glacier.Inference.Distributed` is part of the **Glacier High-Performance Ecosystem**:
- **[Glacier.Inference](https://github.com/ian-cowley/Glacier.Inference)**: Core single-node compute engine, tiered virtual memory, and SIMD/GPU quantization kernels.
- **[Glacier.Cluster](https://github.com/ian-cowley/Glacier.Cluster)**: Pure C# distributed DAG scheduler, zero-allocation wire protocol, and P2P Arrow IPC shuffle bus.
- **[Glacier.Tensor](https://github.com/ian-cowley/Glacier.Tensor)**: Strided N-D tensor engine and Autograd framework.

---

## License

MIT License. Copyright (c) 2026 Ian Cowley.
