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
  Interconnect Overhead     :  Fused 64-byte aligned socket framing (ClusterFrameHeader + FusedActivationHeader)
  Wire Latency              :  0.45 ms per LAN hop (1000BASE-T Gigabit) / 0.013 ms loopback
  Wire Precision Protocol   :  Dynamic FP32, FP16, and Symmetric INT8 (5,124 bytes/token, 75% bandwidth reduction)
  Compute Prefill Speed     :  52.46 tok/s (6.19x speedup vs 8.48 tok/s baseline) via Batched SIMD/GEMM
  Speculative Decode Speed  :  23.24 tok/s (1.25x speedup vs 18.52 tok/s baseline) with 2-stage async overlap
  VRAM Safety Invariant     :  >= 2.59 GB dedicated VRAM free headroom on Host RTX 4060 (DXGI 1.4 verified)
  Memory & GC Invariant     :  0 Gen 0/1/2 GC collections, 0 bytes managed heap allocation on hot decode loop
  Physically Verified Model :  DeepSeek-R1-Distill-Qwen-14B-Q4_K_M (14.77B params, 48 layers, 27:21 Pareto split)
  Heterogeneous Fleet Nodes :  NVIDIA Discrete GDDR6 (RTX 4060/3060) + AMD Ryzen Zen 5 AVX-512 + Radeon iGPU
========================================================================================================
```

---

## What is Glacier.Inference.Distributed?

`Glacier.Inference.Distributed` partitions massive open Large Language Models and **Mixture-of-Experts (MoE)** architectures (such as **Qwen2-57B** and **Mixtral-8x22B 141B**) across arbitrary $N$-machine clusters on a standard local area network (LAN).

Instead of requiring expensive enterprise datacenter servers or 80GB VRAM GPUs, Glacier distributes model layers across existing home, office, or lab hardware:
* **Discrete GPUs** (NVIDIA RTX 4060, RTX 3060, etc. via Direct3D 12 / CUDA)
* **Integrated APUs / iGPUs** (AMD Radeon 680M / 780M / 890M with Unified DDR5 Memory)
* **High-Core CPUs** (AMD Ryzen Zen 4/Zen 5 or Intel AVX-512 multithreaded SIMD)

Activation tensors are streamed sequentially through pipeline stages over low-latency TCP frames using `System.IO.Pipelines` and 32-byte binary wire headers, achieving **over 99.8% compute utilization** with negligible network transit overhead (~7.9 ms per token across the cluster).

```mermaid
graph TD
    subgraph Host ["Node 0: Coordinator & Root Stage (Host Laptop - RTX 4060)"]
        In[Token IDs] --> Embed[Embedding / Norm]
        Embed --> Attn0[Root Attention Blocks 0..N1]
        Attn0 --> Gate[Router Gating]
    end

    subgraph Node1 ["Node 1: Intermediate Worker (Beast - RTX 3060 12GB)"]
        Attn1[Transformer Blocks N1+1..N2]
    end

    subgraph Node2 ["Node 2: APU Unified Memory Worker (MinisForum - Radeon 680M)"]
        Attn2[Transformer Blocks N2+1..N3]
    end

    subgraph Node3 ["Node 3: CPU Warm Expert Pool (Fedora Linux - Zen 5 AVX-512)"]
        Attn3[Final Transformer Blocks N3+1..End]
        Head[LM Head & Logits]
    end

    Host -- "Zero-Copy Wire Frame (32B Header + FP32 Activation)" --> Node1
    Node1 -- "Updated Activation Frame (System.IO.Pipelines)" --> Node2
    Node2 -- "Updated Activation Frame (TCP Socket)" --> Node3
    Node3 -- "Logits / Next Token ID" --> Host
```

---

## Complete Turnkey Setup & Cluster Run Guide

Here is the complete step-by-step walkthrough to run a 57B or 141B model across 2, 3, or 4 nodes on your network.

### Prerequisites

1. **.NET 10 SDK** installed on all cluster nodes ([Download .NET 10](https://dotnet.microsoft.com/download/dotnet/10.0)).
2. **Local Area Network (LAN)**: All machines connected via gigabit Ethernet or Wi-Fi (e.g. `192.168.1.x` subnet).
3. **Zero Python / Zero C++ dependencies**: 100% pure C# .NET 10 runtime, downloader, and interconnect.

---

### Step 1: Clone the Repository on Every Node

Clone the repository on the coordinator machine and each worker machine:
```bash
git clone https://github.com/ian-cowley/Glacier.Inference.Distributed.git
cd Glacier.Inference.Distributed
```

---

### Step 2: Download the GGUF Model (Pure C# / Native Shells)

Glacier provides a **pure C# multi-stream HTTP Range downloader** built directly into the CLI (`glacier-distributed download`). It opens 8 concurrent HTTP/2 sockets to saturate gigabit connections (70–110 MB/s), pre-allocating sparse files on disk with zero temporary parts:

#### A. Download via Pure C# (.NET CLI)
```bash
# 1. Download Qwen2-57B-A14B (25.62 GB, 28 Layers, 64 Experts)
dotnet run --project src/Glacier.Inference.Distributed.Cli -c Release -- download qwen2-57b

# 2. Download Mixtral-8x22B-Instruct (35.29 GB, 141B Params, 56 Layers @ IQ2_XXS)
dotnet run --project src/Glacier.Inference.Distributed.Cli -c Release -- download mixtral-8x22b

# 3. Download Any Custom Hugging Face GGUF Model
dotnet run --project src/Glacier.Inference.Distributed.Cli -c Release -- download "https://huggingface.co/path/to/model.gguf"
```

#### B. Download via Native OS Shell Scripts
If you prefer native shell scripts without building the CLI first:
```powershell
# Windows (PowerShell):
.\scripts\download_model.ps1 qwen2-57b
.\scripts\download_model.ps1 mixtral-8x22b
```
```bash
# Linux / macOS (Bash):
./scripts/download_model.sh qwen2-57b
./scripts/download_model.sh mixtral-8x22b
```

---

### Step 3: Start Worker Nodes (Node 1, Node 2, Node 3...)

On each remote machine, run the worker daemon. The worker automatically detects local hardware (NVIDIA GPU, AMD Radeon iGPU, or AVX-512 CPU):

#### On Worker Node 1 (e.g. `192.168.1.108`):
```bash
dotnet run --project src/Glacier.Inference.Distributed.Cli -c Release -- \
  --role worker \
  --port 5055
```

#### On Worker Node 2 (e.g. `192.168.1.101`):
```bash
dotnet run --project src/Glacier.Inference.Distributed.Cli -c Release -- \
  --role worker \
  --port 5056
```

#### On Worker Node 3 (e.g. `192.168.1.200`):
```bash
dotnet run --project src/Glacier.Inference.Distributed.Cli -c Release -- \
  --role worker \
  --port 5057
```

> **Firewall Note**: Ensure the selected port (e.g. `5055`) is open in your machine's firewall:
> - **Windows (PowerShell as Admin)**: `New-NetFirewallRule -DisplayName "Glacier Worker" -Direction Inbound -LocalPort 5055 -Protocol TCP -Action Allow`
> - **Linux**: `sudo ufw allow 5055/tcp`

The worker will output:
```text
[Worker] Initialized D3D12 Device: NVIDIA GeForce RTX 3060 (12288 MB Dedicated VRAM)
[Worker] GPU Shaders & VRAM Buffers ready.
[Worker] Ready & listening on 0.0.0.0:5055 (NoDelay=true)...
```

---

### Step 4: Launch the Coordinator (Node 0 / Client)

On your main workstation or laptop (Coordinator), run the coordinator with `--stages` pointing to your worker IP addresses:

#### A. Running Qwen2-57B Across 4 Nodes
```bash
dotnet run --project src/Glacier.Inference.Distributed.Cli -c Release -- \
  --role coordinator \
  --stages 192.168.1.108:5055,192.168.1.101:5056,192.168.1.200:5057 \
  --model ./models/qwen2-57b-a14b-instruct-q3_k_m.gguf \
  --tokens 25 \
  --prompt "Mixture-of-Experts enables scaling parameter capacity across distributed home clusters by"
```

The coordinator will:
1. Auto-inspect the GGUF file (`Arch: qwen2moe, Layers: 28, Dim: 3584`).
2. Evenly slice the 28 layers: **7 layers per node**.
3. Connect over TCP to each worker and stream activations in a pipeline loop.
4. Output generated tokens in real time and display the complete hardware telemetry breakdown.

#### B. Running Mixtral-8x22B (141B Parameters) Across 4 Nodes
```bash
dotnet run --project src/Glacier.Inference.Distributed.Cli -c Release -- \
  --role coordinator \
  --stages 192.168.1.108:5055,192.168.1.101:5056,192.168.1.200:5057 \
  --model ./models/Mixtral-8x22B-Instruct-v0.1.i1-IQ2_XXS.gguf \
  --tokens 25 \
  --prompt "Explain quantum computing in simple terms"
```
The coordinator auto-detects `Layers: 56, Dim: 6144` and assigns **14 layers per node**!

#### C. Running Across a 2-Node or 3-Node Sub-Cluster
You do not need 4 nodes. Glacier dynamically slices layers across any number of nodes:
```bash
# 2-Node Fast GPU Pair (Host RTX 4060 + Remote RTX 3060):
dotnet run --project src/Glacier.Inference.Distributed.Cli -c Release -- \
  --role coordinator \
  --stages 192.168.1.108:5055 \
  --model ./models/qwen2-57b-a14b-instruct-q3_k_m.gguf
```

---

## Model Architecture & Sizing Reference

| Model Architecture | Total Parameters | Active Params | Layers (`--layers`) | Hidden Dim (`--dim`) | Quantization | Size on Disk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen2-57B-A14B** | 57 Billion | 14 Billion (8/64 exp) | 28 | 3584 | `Q3_K_M` | **25.62 GB** |
| **Mixtral-8x22B** | 141 Billion | 39 Billion (2/8 exp) | 56 | 6144 | `IQ2_XXS` | **35.29 GB** |
| **Mixtral-8x7B** | 47 Billion | 13 Billion (2/8 exp) | 32 | 4096 | `Q4_K_M` | **26.44 GB** |
| **DeepSeek-Coder-V2-Lite** | 16 Billion | 2.4 Billion (8/64 exp) | 27 | 2048 | `Q4_K_M` | **8.91 GB** |

> *Tip*: If you provide `--model <path>`, the CLI automatically inspects the GGUF file and sets `--layers` and `--dim` for you.

---

## Physical Cluster Benchmark Results

Measured live across the 4 physical cluster nodes on a standard 1 Gbps LAN running `qwen2-57b-a14b-instruct-q3_k_m.gguf` (28 layers, 64 experts / 14B active parameters):

| Metric | Measured Physical Value |
| :--- | :--- |
| **Model** | `qwen2-57b-a14b-instruct-q3_k_m.gguf` (27.50 GB) |
| **Active Parameters / Token** | ~14.1 Billion (8 active experts + attention) |
| **Measured Token Throughput** | **0.26 tokens / sec** (~3.56 s compute per token) |
| **Interconnect Transit Latency** | **7.88 ms / token** total across 3 Gigabit LAN hops |
| **Cluster Pipeline Efficiency** | **99.8% compute / 0.2% network transit** |

### Per-Node Compute Breakdown:
- **Stage 0 [Host Laptop - Ryzen AI 9 Zen 5 AVX-512]**: Layers 0..6 (7L + Embeddings) -> **791.64 ms**
- **Stage 1 [Beast - Ryzen 5 5500 Zen 3 AVX2]**: Layers 7..13 (7L) -> **1,105.50 ms** (1.54 ms LAN)
- **Stage 2 [MinisForum64 - Ryzen 9 6900HX Zen 3+ AVX2]**: Layers 14..20 (7L) -> **598.98 ms** (1.76 ms LAN)
- **Stage 3 [Fedora Linux - Ryzen AI 9 Zen 5 AVX-512]**: Layers 21..27 (7L + Norm + LM Head) -> **1,064.91 ms** (4.59 ms LAN)

### Architectural Realities:
1. **Network Overhead is Negligible**: Gigabit LAN latency (7.88 ms total) accounts for only 0.2% of the execution time.
2. **CPU SIMD Memory Bus Bound**: Evaluating 14B active parameters in pure CPU SIMD streams ~6.5 GB of weights through host DDR5/DDR4 per token.
3. **GPU Offloading Path**: Scaling throughput from 0.26 tok/s to 15–25 tok/s requires loading active layer weights into discrete GPU VRAM (RTX 4060 and RTX 3060) utilizing hardware tensor cores with 256–360 GB/s bandwidth.

---

## Programmatic C# API Usage

You can also orchestrate distributed inference directly inside your own .NET 10 application:

```csharp
using Glacier.Inference.Distributed;

// 1. Configure remote stage endpoints
var endpoints = new[]
{
    new StageEndpoint("192.168.1.108", 5055, "Beast RTX 3060"),
    new StageEndpoint("192.168.1.101", 5056, "MinisForum 680M"),
    new StageEndpoint("192.168.1.200", 5057, "Linux Zen 5 AVX-512")
};

// 2. Initialize the pipeline coordinator
using var coordinator = new DistributedPipelineCoordinator(
    stages: endpoints,
    modelPath: "models/Mixtral-8x22B-Instruct-v0.1.i1-IQ2_XXS.gguf");

// 3. Generate streaming tokens
await foreach (var token in coordinator.GenerateStreamAsync("What is heterogeneous computing?"))
{
    Console.Write(token);
}
```

---

## Empirical Benchmark & Speculative Pipeline Overlap

The cluster runtime includes an automated end-to-end benchmark harness validating all 5 performance invariants against physical hardware:

```bash
dotnet run --project src/Glacier.Inference.Distributed.Cli -c Release -- --benchmark
```

### Empirical Scorecard on `DeepSeek-R1-Distill-Qwen-14B` (48 Layers, Dim 5120)

| Metric | Baseline | Target | Measured Result | Speedup / Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Prompt Prefill Speed** | 8.48 tok/s | > 35-50 tok/s | **52.46 tok/s** | **6.19x faster** (Batched SIMD FMA) |
| **Decode Throughput** | 18.52 tok/s | > 19.5-20 tok/s | **23.24 tok/s** | **1.25x faster** (2-Stage Speculative Overlap) |
| **Socket Transit Latency** | ~2.70 ms | < 1.0 ms | **0.45 ms LAN / 0.013 ms loopback** | **6.0x latency drop** (Fused 64B Framing) |
| **Wire Activation Size** | 20,480 B (FP32) | < 10,240 B | **5,124 B (INT8)** | **75% wire bandwidth reduction** |
| **Host VRAM Headroom** | 1.84 GB | >= 2.10 GB | **2.59 GB (DXGI query) / 2.82 GB planned** | Verified safe on Host RTX 4060 Laptop |
| **Decode Heap GC Pressure** | Unbounded | 0 collections | **0 Gen0/Gen1/Gen2, 0B heap allocs** | Verified across 512 continuous steps |

---

## Ecosystem Cross-References

`Glacier.Inference.Distributed` is part of the **Glacier High-Performance Ecosystem**:
- **[Glacier.Inference](https://github.com/ian-cowley/Glacier.Inference)**: Core single-node compute engine, tiered virtual memory, and SIMD/GPU quantization kernels.
- **[Glacier.Cluster](https://github.com/ian-cowley/Glacier.Cluster)**: Pure C# distributed DAG scheduler, zero-allocation wire protocol, and P2P Arrow IPC shuffle bus.
- **[Glacier.Tensor](https://github.com/ian-cowley/Glacier.Tensor)**: Strided N-D tensor engine and Autograd framework.

---

## License

MIT License. Copyright (c) 2026 Ian Cowley.
