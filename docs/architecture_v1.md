# OpenGP — Dynamic Layer-Streaming Inference Engine
## Architecture Specification Document (v0.1-draft)
**Role:** Senior Systems & LLM Engineer  
**Reviewer:** Chief Technology Officer / Chief Architect  
**Status:** Pending Review & Approval  

---

## 1. Executive Summary & Design Principles

OpenGP is designed for asynchronous, high-throughput inference on resource-constrained, heterogeneous volunteer nodes. Unlike standard inference engines (vLLM, Ollama, TensorRT-LLM) which require the entire model weight set to be statically resident in GPU VRAM, OpenGP treats **one layer (or a slice of layers)** as the atomic unit of execution.

### Key Constraints & Trade-Offs
* **Target Workload:** Asynchronous batch tasks (e.g. document analysis, bulk processing, speculative verification). Time-To-First-Token (TTFT) is deprioritized; **total memory footprint guarantee and steady-state throughput are prioritized.**
* **Memory Hierarchy:** Explicitly orchestrates three tiers of physical storage:
  1. **Tier 1 (Disk / NVMe):** Full GGUF weight repository.
  2. **Tier 2 (Host RAM):** Pinned memory staging arena & KV Cache holding area.
  3. **Tier 3 (GPU VRAM):** Active compute buffer holding only the current layer $L_i$ (and optionally pre-buffered $L_{i+1}$).

---

## 2. 3-Tier Storage & Memory Flow

```
+-----------------------------------------------------------------------------------+
| TIER 1: Persistent Disk (GGUF File on SSD/NVMe)                                   |
|   [Token Embeddings] [Layer 0] [Layer 1] ... [Layer N-1] [Output Head & Norm]     |
+-----------------------------------------------------------------------------------+
                                   |
                                   | Asynchronous I/O (pread / io_uring / mmap)
                                   v
+-----------------------------------------------------------------------------------+
| TIER 2: Host System RAM (Staging Buffer)                                          |
|   - Pinned Host Ring Buffer: Slot A (Layer N) | Slot B (Layer N+1)                |
|   - Global Context: Tokenizer, Token Sequence, State Coordinator                  |
|   - Host KV-Cache Reservoir (Optional layer-spillover)                            |
+-----------------------------------------------------------------------------------+
                                   |
                                   | Async DMA Transfer (cudaMemcpyAsync / hipMemcpy)
                                   v
+-----------------------------------------------------------------------------------+
| TIER 3: Device GPU VRAM (Execution Core)                                          |
|   - Active Layer VRAM Slab (Layer N Weights)                                      |
|   - In-flight Activation Tensor [Batch, Seq, d_model]                             |
|   - Active Layer KV-Cache Slice [2, n_head_kv, Seq, head_dim]                     |
|   - Transient Scratch Buffer (ggml_gallocr for Attention & MLP intermediate ops)  |
+-----------------------------------------------------------------------------------+
```

---

## 3. Double-Buffered Overlapped Execution Pipeline

To prevent GPU compute units from idling during disk reads, OpenGP employs **double buffering** between Host RAM and GPU VRAM:

```
Timeline:
------------------------------------------------------------------------------------
Time (t)         | Slot 0 (Compute Core)         | Slot 1 (Prefetch Engine)
-----------------+-------------------------------+----------------------------------
t = 0            | GPU: Execute Layer 0          | Disk->Host: Read Layer 1 weights
                 | Input: h_0 -> Output: h_1     | Host->GPU: Async DMA Layer 1
-----------------+-------------------------------+----------------------------------
t = 1            | GPU: Execute Layer 1          | Disk->Host: Read Layer 2 weights
                 | Input: h_1 -> Output: h_2     | Host->GPU: Async DMA Layer 2
-----------------+-------------------------------+----------------------------------
t = 2            | GPU: Execute Layer 2          | Disk->Host: Read Layer 3 weights
                 | Input: h_2 -> Output: h_3     | Host->GPU: Async DMA Layer 3
------------------------------------------------------------------------------------
```

### Invariant:
The activation tensor $h_{i}$ (size: $\text{Batch} \times \text{Tokens} \times d_{\text{model}} \times \text{sizeof(float)}$) remains in GPU VRAM across the transition from $L_i$ to $L_{i+1}$. It is **never** round-tripped to disk.

---

## 4. State Management (KV-Cache vs. SSM State)

Inference is stateful:

1. **For Transformers (KV-Cache):**
   * Layer $i$ produces Key and Value tensors for all processed tokens.
   * **Memory Strategy:** Because a volunteer might only hold 1 layer at a time, each layer has its own KV-cache partition.
   * If VRAM is sufficient: All $N$ layer KV-caches reside in VRAM (KV-cache is much smaller than weights).
   * If VRAM is critically low (< 2 GB): Layer $i$'s KV-cache can be swapped to Host RAM when Layer $i$ is evicted and swapped back when Layer $i$ returns for the next token.

2. **For SSMs (e.g. Mamba / Recurrent states):**
   * State size is $O(1)$ constant regardless of sequence length.
   * Stored in a compact state buffer per layer ($\approx$ few KB to MB). Trivially retained in VRAM or swapped with near-zero latency.

---

## 5. Pre-Flight Model Profiler & Configuration Step

Before running inference, OpenGP executes an automated pre-flight check:
1. **GGUF Inspection:**
   * Reads metadata: `block_count` ($N$), $d_{\text{model}}$, `head_count`, `context_length`.
   * Maps out tensor offsets for each `blk.i.*`.
2. **Hardware Topology Probe:**
   * Probes available backends via `ggml_backend_dev_t`.
   * Probes free GPU VRAM, free Host RAM, and disk read bandwidth.
3. **Budget Partitioning:**
   * Computes:
     $$\text{Max Layer Size} = \max_{i} \left( \sum_{w \in \text{blk.i}} \text{size}(w) \right)$$
   * Computes KV-cache size per token.
   * Allocates static VRAM slabs: **Weight Slab (Double Buffered) + Activation Slab + KV Slab + Workspace Scratch**.
4. **Configuration Lock:** Outputs an execution plan detailing whether to stream 1 layer at a time, 2 layers at a time, or keep KV in host memory.

---

## 6. Proposed Implementation Plan for Milestone 1

* **Branch:** `feature/model-profiler-and-streaming-core`
* **Deliverables:**
  1. `opengp_profiler`: Command-line tool to inspect any GGUF model and output the exact VRAM/RAM budget for layer-by-layer streaming.
  2. `opengp_gguf_streamer`: C++ streaming reader that can seek directly to any block `blk.i` using `pread`/`mmap` and populate pre-allocated `ggml_tensor` buffers without full file reads.
  3. `opengp_layer_runner`: Single-layer forward evaluation test harness using `ggml_cgraph` and `ggml_gallocr`.

---

*Awaiting CTO review and directives.*
