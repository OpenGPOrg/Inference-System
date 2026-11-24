# OpenGP: Modular, Distributed-Ready C++ LLM Inference Engine

**OpenGP** is a self-contained C++17 LLM inference engine built from first principles on vendored GGML/GGUF (`vendor/ggml/`). It splits LLM execution into three independent, noun-named phases:

1. **Perception (`perception`)** — Prompt templating, dynamic regex pre-tokenization, $\mathcal{O}(M \log M)$ BPE merge tournament, zero-VRAM Host `mmap` `token_embd.weight` lookup, and $H_0 \in \mathbb{R}^{n \times d_{\text{model}}}$ assembly.
2. **Reasoner (`reasoner`)** — Model-agnostic Transformer block graph execution (`RMSNorm`, `Q/K/V`, optional `QK-Norm`, `RoPE` `NORM`/`NEOX`, unified `MHA/GQA/MQA`, `SwiGLU`), fixed upfront KV Cache with init-time auto-halving, NVIDIA CUDA + Apple Metal GPU + multi-threaded CPU auto-fallback, hybrid GPU-resident / Host-`mmap` layer streaming, and modular `[layer_start .. layer_end)` slicing for distributed execution.
3. **Responder (`responder`)** — Unembedding `lm_head` (`output.weight` or tied fallback, auto GPU/CPU), Repetition Penalty + Temperature + Top-K + Min-P + Top-P Nucleus `Sampler`, dynamic EOT stop detection, streaming UTF-8 `Detokenizer`, and autoregressive Decode loop.

> 📖 **Architecture & Deep Dive:** See [`docs/architecture/OPENGP_DEVELOPER_GUIDE.md`](docs/architecture/OPENGP_DEVELOPER_GUIDE.md) for the complete inference flow diagrams, module API reference, and VRAM/RAM layout.

---

## Quick Start

### 1. Build
OpenGP requires only CMake ($\ge$ 3.18) and a C++17 compiler. On Linux/Windows with the NVIDIA CUDA Toolkit installed, `ggml-cuda` is enabled automatically; on macOS, `ggml-metal` is enabled automatically with embedded Metal shaders; otherwise it builds CPU-only with native SIMD optimizations.

```bash
git clone https://github.com/OpenGPOrg/Inference-System.git opengp
cd opengp
make
```

> **Developer Tip:**
> - `make clean` removes only OpenGP objects and binaries while preserving compiled `build/vendor/` GGML & CUDA kernels (subsequent builds finish in a few seconds).
> - `make distclean` wipes the entire `build/` directory.

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
- **Automatic Hardware Adaptability:** Uses NVIDIA CUDA GPU (`OPENGP_HAS_CUDA=1`) or Apple Metal GPU (`OPENGP_HAS_METAL=1`) automatically when available and seamlessly falls back to multi-threaded CPU (`--device cpu` / `--threads <N>`) on CPU-only machines.
- **Distributed Volunteer Ready:** `ReasoningEngine` accepts a `LayerRange` (`--layer-start`, `--layer-end`, `--no-final-norm`) so a node only maps weights and allocates KV Cache for its assigned layer slice, exchanging `ActivationPacket`s via `forward_packet()`.

---

## Project Layout

```text
opengp/
├── CMakeLists.txt
├── Makefile
├── docs/architecture/OPENGP_DEVELOPER_GUIDE.md
├── vendor/ggml/          # Self-contained GGML, GGUF, CPU SIMD & CUDA backends
├── include/
│   ├── perception/       # Phase 1 headers (TokenizerConfig, BpeEngine, EmbedTable, PerceptionPipeline)
│   ├── reasoner/         # Phase 2 headers (ModelConfig, WeightStore, KVCache, LayerGraph, ReasoningEngine)
│   └── responder/        # Phase 3 headers (LmHead, Sampler, Detokenizer, RespondingEngine)
└── src/
    ├── perception/       # Phase 1 implementation + opengp_perception CLI
    ├── reasoner/         # Phase 2 implementation + opengp_reasoner CLI
    └── responder/        # Phase 3 implementation + opengp_infer CLI
```
