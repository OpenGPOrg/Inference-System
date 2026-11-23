# OpenGP: Next-Generation Modular LLM Inference Engine

OpenGP is a flexible, modular LLM inference engine built from first principles in modern C++ (C++17). It separates inference into three independent, pluggable phases:
1. **Perception Phase** (Prompt Templating, Regex Pre-tokenization, ID-based BPE, Host-RAM $W_{\text{emb}}$ mmap, and $H_0$ Activation Assembly)
2. **Reasoning Phase** (Layer Processing, Transformer Attention Backbone, SwiGLU FFN, KV Cache)
3. **Expression Phase** (LM Head, Defuzzification, Sampling, Logits)

---

## Quick Start (Zero Dependencies Beyond CMake & C++ Compiler)

### 1. Build
```bash
git clone <this-repo> opengp
cd opengp
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 2. Run the Perception Engine CLI
Place any GGUF model in `models/` (e.g. LLaMA-3, Qwen-2.5/3, Mistral, Stories) and run:

```bash
./opengp_perception ../models/your_model.gguf "What is an inference engine?"
```

---

## Architectural Highlights of the Perception Module
- **100% Model-Agnostic:** Reads pre-tokenizer regex flavors, chat templates, and stop tokens dynamically from the GGUF header with zero hardcoded values.
- **$\mathcal{O}(N \log N)$ Integer BPE Tournament:** Operates directly on `uint32_t` Token IDs using a doubly linked list with $\mathcal{O}(1)$ pointer splicing and a min-heap priority queue.
- **Zero VRAM Footprint:** Statically maps $W_{\text{emb}}$ into Host RAM (`mmap`). Only the prompt token rows are gathered in $\mathcal{O}(1)$ time (`offset = tid * row_bytes`) and SIMD dequantized to FP32.
- **Contract Handover:** Assembles contiguous activation tensor $H_0 \in \mathbb{R}^{n \times d_{\text{model}}}$ and coordinate vector $\text{pos\_ids} = [0, 1, \dots, n-1]$ ready for Layer 0 RMSNorm.
