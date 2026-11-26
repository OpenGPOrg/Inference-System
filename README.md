# OpenGP: Modular, Distributed-Ready C++ LLM Inference Engine

**OpenGP** is a self-contained C++17 LLM inference engine built from first principles on vendored GGML/GGUF (`vendor/ggml/`). It splits LLM execution into three independent, noun-named phases:

1. **Perception (`perception`)** — Prompt templating, dynamic regex pre-tokenization, $\mathcal{O}(M \log M)$ BPE merge tournament, zero-VRAM Host `mmap` `token_embd.weight` lookup, and $H_0 \in \mathbb{R}^{n \times d_{\text{model}}}$ assembly.
2. **Reasoner (`reasoner`)** — Model-agnostic Transformer block graph execution (`RMSNorm`, `Q/K/V`, optional `QK-Norm`, `RoPE` `NORM`/`NEOX`, unified `MHA/GQA/MQA`, `SwiGLU`), fixed upfront KV Cache with init-time auto-halving, NVIDIA CUDA + Apple Metal + Vulkan (AMD/Intel/iGPU) + multi-threaded CPU auto-fallback, hybrid GPU-resident / Host-`mmap` layer streaming, and modular `[layer_start .. layer_end)` slicing for distributed execution.
3. **Responder (`responder`)** — Unembedding `lm_head` (`output.weight` or tied fallback, auto GPU/CPU), Repetition Penalty + Temperature + Top-K + Min-P + Top-P Nucleus `Sampler`, dynamic EOT stop detection, streaming UTF-8 `Detokenizer`, and autoregressive Decode loop.

> 📖 **Architecture & Deep Dive:** See [`docs/architecture/OPENGP_DEVELOPER_GUIDE.md`](docs/architecture/OPENGP_DEVELOPER_GUIDE.md) for the complete inference flow diagrams, module API reference, and VRAM/RAM layout.

---

## Quick Start

### 1. Build
OpenGP requires only CMake ($\ge$ 3.18) and a C++17 compiler. Hardware backends are auto-detected non-obstructively:
- **NVIDIA CUDA (`ggml-cuda`):** auto-enabled on Linux/Windows when the CUDA Toolkit is installed.
- **Apple Metal (`ggml-metal`):** auto-enabled on macOS with embedded Metal shaders.
- **Vulkan (`ggml-vulkan`):** auto-enabled on Linux/Windows (AMD, Intel Arc, and Intel UHD/Iris iGPUs) when CUDA is absent and the Vulkan SDK/headers (`vulkan/vulkan.hpp`) + `glslc` are installed. Missing Vulkan tools never fail the build—CMake logs a status line and builds CPU-only. Use `-DOPENGP_VULKAN=OFF` to disable, or `-DOPENGP_VULKAN=ON` to build alongside CUDA.
- **CPU Fallback:** always compiled with native SIMD optimizations (`-march=native`).

#### Prerequisites for Non-NVIDIA GPUs (Intel Arc / UHD / Iris & AMD Radeon):
To run or build on machines with Intel or AMD graphics:
- **Ubuntu / Debian:** `sudo apt install -y libvulkan1 mesa-vulkan-drivers` (for building from source, also add `libvulkan-dev glslc`)
- **Arch / Manjaro:** `sudo pacman -S vulkan-icd-loader vulkan-intel` (or `vulkan-radeon` for AMD; add `shaderc` to build)
- **Fedora:** `sudo dnf install -y vulkan-loader mesa-vulkan-drivers` (add `vulkan-headers glslc` to build)
- **Windows:** Installed automatically with official Intel or AMD driver packages (`vulkan-1.dll`).
- **Quick verify:** Run `vulkaninfo --summary` to confirm your GPU is recognized.

```bash
git clone https://github.com/OpenGPOrg/Inference-System.git opengp
cd opengp
make
```

> **Developer Tip:**
> - `make clean` removes only OpenGP objects and binaries while preserving compiled `build/vendor/` GGML, CUDA & Vulkan kernels (subsequent builds finish in seconds).
> - `make distclean` wipes the entire `build/` directory.
> - Set `OPENGP_DISABLE_VULKAN=1` at runtime to bypass Vulkan and run on CPU without rebuilding.

---

### 2. Run End-to-End Inference (`opengp_infer`)
Place any supported Transformer GGUF model (e.g., LLaMA-3.1, Qwen3, Mistral, Stories) in `models/`:

```bash
# Single-prompt generation (default ctx=512, auto CUDA GPU or CPU fallback)
./build/opengp_infer --model models/your_model.gguf --prompt "What is the capital of France?"

# Custom context window & unlimited tokens until EOT (--max-tokens 0)
./build/opengp_infer --model models/your_model.gguf --ctx-size 4096 --max-tokens 0 -i

# Force multi-threaded CPU execution (simulating a GPU-less host)
./build/opengp_infer --model models/your_model.gguf --prompt "Once upon a time" --device cpu --lm-head cpu
```

### 3. Diagnostic Phase Tools
```bash
# Phase 1 only: inspect regex chunks, BPE token IDs, pos_ids, and H_0
./build/opengp_perception models/your_model.gguf "Hello world"

# Phase 2 only: test a contiguous layer slice or verify N-stage pipeline parity
./build/opengp_reasoner --model models/your_model.gguf --prompt "Once upon a time" --verify-pipeline 3
./build/opengp_reasoner --model models/your_model.gguf --prompt "Hello" --layer-start 0 --layer-end 16
```

---

## Key Architectural Highlights

- **Zero Hardcoding:** Reads `general.architecture`, `<arch>.*` hyperparameters, RoPE mode (`NORM` vs. `NEOX`), `rope_freqs.weight`, `attn_q/k_norm`, and stop tokens directly from GGUF headers.
- **KV-First Memory Budgeting & Auto-Halving:** Reserves/allocates the fixed KV Cache (`--ctx-size`, default `512`) first—halving cleanly at init if VRAM is constrained—then pins as many leading layers as fit permanently in VRAM while streaming remaining layers through a single reusable GPU staging buffer from zero-copy Host `mmap`.
- **Automatic Hardware Adaptability:** Runtime priority is **CUDA → Metal → Vulkan → CPU**. Uses NVIDIA CUDA (`OPENGP_HAS_CUDA=1`), Apple Metal (`OPENGP_HAS_METAL=1`), or Vulkan (`OPENGP_HAS_VULKAN=1`, with startup smoke-test, op-support preflight, and shared-memory budgeting for iGPUs) automatically when available and logs + falls back cleanly to multi-threaded CPU (`--device cpu` / `--threads <N>` / `OPENGP_DISABLE_VULKAN=1`) on driver/hardware failures or CPU-only machines.
- **Distributed Volunteer Ready:** `ReasoningEngine` accepts a `LayerRange` (`--layer-start`, `--layer-end`, `--no-final-norm`) so a node only maps weights and allocates KV Cache for its assigned layer slice, exchanging `ActivationPacket`s via `forward_packet()`.

---

## Project Layout

```text
opengp/
├── CMakeLists.txt
├── Makefile
├── docs/architecture/OPENGP_DEVELOPER_GUIDE.md
├── vendor/ggml/          # Self-contained GGML, GGUF, CPU SIMD, CUDA, Metal & Vulkan backends
├── include/
│   ├── perception/       # Phase 1 headers (TokenizerConfig, BpeEngine, EmbedTable, PerceptionPipeline)
│   ├── reasoner/         # Phase 2 headers (BackendSelector, ModelConfig, WeightStore, KVCache, LayerGraph, ReasoningEngine)
│   └── responder/        # Phase 3 headers (LmHead, Sampler, Detokenizer, RespondingEngine)
└── src/
    ├── perception/       # Phase 1 implementation + opengp_perception CLI
    ├── reasoner/         # Phase 2 implementation + opengp_reasoner CLI
    └── responder/        # Phase 3 implementation + opengp_infer CLI
```
