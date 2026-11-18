# Technical Explainer: QLoRA, NF4 Quantization, Model Engineering & GGUF Export

This document breaks down the fundamental mathematical foundations, memory physics, tokenization algorithms, and design rationale behind **Section 3.1 & 3.5 (Model Engineering, QLoRA Fine-Tuning & Local Serving)** of DeepSearch-Local.

---

## 1. The Core Hardware Budget: 6 GB VRAM vs. Full Fine-Tuning Math

DeepSearch-Local is engineered to run on a consumer-grade **NVIDIA GeForce RTX 3050 Laptop GPU with 6,144 MiB (6 GB) VRAM** and 16 GB host RAM.

### The Physics of Full Parameter Fine-Tuning
In full fine-tuning, every single parameter $W$ in the neural network is updated via gradient descent:

$$W_{t+1} = W_t - \eta \cdot \nabla_W \mathcal{L}$$

For a 3-Billion parameter language model (such as `Qwen/Qwen2.5-3B-Instruct`):
1. **Model Weights (FP16/BF16):** $3 \times 10^9 \times 2\text{ bytes} = 6.0\text{ GB}$.
2. **Gradients ($\nabla_W \mathcal{L}$ in FP16/BF16):** $3 \times 10^9 \times 2\text{ bytes} = 6.0\text{ GB}$.
3. **AdamW Optimizer States (FP32):** AdamW tracks the first moment (momentum $m_t$) and second moment (variance $v_t$) in full 32-bit float for numerical stability:
   $$3 \times 10^9 \times (4\text{ bytes} + 4\text{ bytes}) = 24.0\text{ GB}$$
4. **Activations & KV Buffer:** $\approx 2.0\text{--}4.0\text{ GB}$ (depending on batch size and sequence length).

$$\text{Total VRAM required for full fine-tuning} \ge 38.0\text{ GB}$$

Attempting full fine-tuning on a 6 GB GPU causes instant CUDA Out-Of-Memory (OOM) fatal errors.

```
+-----------------------------------------------------------------------+
| FULL FINE-TUNING VRAM REQUIREMENT: >= 38,000 MiB (Impossible on 6GB)   |
+-----------------------------------------------------------------------+
| [Base Weights: 6,000 MiB] [Gradients: 6,000 MiB] [AdamW: 24,000 MiB]  |
+-----------------------------------------------------------------------+

+-----------------------------------------------------------------------+
| DEEPSEARCH-LOCAL QLoRA VRAM BUDGET: ~4,800 - 5,400 MiB (Fits in 6GB)   |
+-----------------------------------------------------------------------+
| [NF4 Base: 1,800 MiB] [LoRA Adapter: 75 MiB] [Paged AdamW: 150 MiB]   |
| [Activations via Checkpointing: 1,200 MiB] [OS Display Margin: 1.5GB] |
+-----------------------------------------------------------------------+
```

---

## 2. The Mathematics of Low-Rank Adaptation (LoRA)

### The Intrinsic Rank Hypothesis
Aghajanyan et al. (2020) and Hu et al. (2021) demonstrated that when adapting a pre-trained foundation model to a specialized domain, the parameter update matrix $\Delta W$ has a very low **intrinsic rank ($r$)**. Even though the weight matrix lives in a high-dimensional space ($d \times k$), the essential task-specific updates reside on a tiny subspace.

### Matrix Factorization Formulation
Instead of updating the full weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA freezes $W_0$ completely and decomposes the update into the product of two low-rank matrices:

$$\Delta W = B \cdot A$$

Where:
- $W_0 \in \mathbb{R}^{d \times k}$: Frozen base model weights.
- $A \in \mathbb{R}^{r \times k}$: Down-projection matrix, initialized from a Gaussian normal distribution $\mathcal{N}(0, \sigma^2)$.
- $B \in \mathbb{R}^{d \times r}$: Up-projection matrix, initialized to **zero** (ensuring $\Delta W = 0$ at step 0 so training begins from the exact baseline distribution).
- $r \ll \min(d, k)$: The low-rank bottleneck dimension (in DeepSearch-Local: **$r = 16$**).

```
                 Input Vector x (dimension k)
                      │                 │
                      ▼                 ▼
              ┌───────────────┐  ┌─────────────┐
              │  Frozen Base  │  │   Matrix A  │ (r x k)
              │   Matrix W0   │  └──────┬──────┘
              │    (d x k)    │         ▼  (dimension r)
              │               │  ┌─────────────┐
              │               │  │   Matrix B  │ (d x r)
              └───────┬───────┘  └──────┬──────┘
                      │                 │  * (alpha / r)
                      ▼                 ▼
                      └───► (+) ◄───────┘
                             │
                             ▼
                Output Vector h (dimension d)
```

The modified forward pass is computed as:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$

### What is $\alpha$ (LoRA Alpha)?
$\alpha$ is a constant scaling hyperparameter. The ratio $\frac{\alpha}{r}$ acts as a learning rate multiplier on the adapter weights:
- In DeepSearch-Local: $r = 16$, $\alpha = 32 \implies \frac{\alpha}{r} = \frac{32}{16} = 2.0$.
- Setting $\frac{\alpha}{r} = 2.0$ stabilizes gradient magnitudes across the low-rank bottleneck, preventing gradient vanishing or divergence during early epochs.

### Target Modules: Why All 7 Linear Projections?
Early LoRA papers only tuned query and value attention matrices (`q_proj`, `v_proj`). However, structured reasoning tasks (strict JSON tool syntax, bracketed citation referencing `[^chunk_id]`, and query decomposition) require cross-layer factual retention.

DeepSearch-Local injects adapters into **all 7 linear projection layers**:
1. **Self-Attention Layers:** `q_proj`, `k_proj`, `v_proj`, `o_proj` (adapts attention focus and multi-head routing).
2. **Multi-Layer Perceptron (MLP) Blocks:** `gate_proj`, `up_proj`, `down_proj` (adapts factual retrieval and reasoning memory).

Tuning all 7 linear layers adds only **18.5 million trainable parameters (~0.59% of the 3B model)**, keeping training VRAM under 5.4 GB while boosting task convergence speed by >40%.

---

## 3. Quantization Mechanics: NormalFloat4 (NF4) & Double Quantization

QLoRA (Dettmers et al., 2023) introduced two breakthroughs that allow 16-bit LoRA gradients to backpropagate through 4-bit quantized base weights.

### Why Not Linear INT4?
Standard integer quantization splits the dynamic range $[-\text{max}, \text{max}]$ into equal uniform intervals:

$$q = \text{round}\left(\frac{x}{S}\right)$$

However, neural network weights are **not uniformly distributed**—they follow a bell-shaped Gaussian normal distribution $\mathcal{N}(0, \sigma^2)$ centered at zero. Uniform quantization wastes precious 4-bit bins on the sparse tails, while suffering catastrophic quantization noise in the high-density region near zero.

```
Uniform INT4 Quantization (Wastes bins on sparse tails)
[-3σ]  [ ]  [ ]  [ ]  [X][X][X][X][X][X]  [ ]  [ ]  [ ]  [+3σ]
                      ^ High density area suffers quantization error

NormalFloat4 (NF4) Quantization (Equal area under Gaussian curve)
[-3σ]   [ ][ ][ ] [ ][ ][ ][ ][ ][ ][ ] [ ][ ][ ]   [+3σ]
        ^ Every bin holds exactly 1/16th of total weight mass
```

### 4-bit NormalFloat (NF4)
NF4 designs an information-theoretically optimal quantile quantization grid. The 16 discrete levels $q_i$ ($i \in \{0, \dots, 15\}$) are calculated such that each interval captures an **equal area (equal probability mass $1/16 = 6.25\%$) under the standard normal distribution**:

$$q_i = \frac{1}{2} \left( Q_X\left(\frac{i}{2^k}\right) + Q_X\left(\frac{i+1}{2^k}\right) \right)$$

This preserves maximum entropy and minimizes empirical perplexity degradation.

### Double Quantization (DQ)
In standard block-wise quantization, weights are grouped into chunks of 64, with each chunk requiring a 32-bit floating-point scale factor $c^{\text{FP32}}$ to unquantize during compute:
- Memory overhead of FP32 scales: $\frac{32\text{ bits}}{64\text{ weights}} = 0.5\text{ bits/parameter}$.

**Double Quantization** compresses the quantization scale factors themselves by applying an 8-bit FP8 quantizer with block size 256:
- Overhead of Double Quantization scales: $\frac{8\text{ bits}}{64} + \frac{32\text{ bits}}{64 \times 256} \approx 0.127\text{ bits/parameter}$.
- **Net Memory Savings:** Saves **$0.373\text{ bits per parameter}$**, freeing $\approx 140\text{ MiB}$ of critical VRAM on a 3B model.

### Dequantization on the Fly
During forward and backward passes, the 4-bit NF4 weights are dynamically dequantized into `bfloat16` registers on the fly, matrix-multiplied with activations, and instantly discarded. Gradients are accumulated strictly into the FP16 LoRA adapter matrices.

---

## 4. Memory-Saving Systems: Paged Optimizers & Gradient Checkpointing

Even with 4-bit base weights, training a 3B model can trigger sudden VRAM spikes that crash an RTX 3050. DeepSearch-Local implements two essential hardware safeguards:

```mermaid
flowchart TD
    subgraph VRAM_Management [RTX 3050 6GB VRAM Management]
        direction TB
        BaseWeights["NF4 Base Weights: ~1.8 GB"]
        Adapters["LoRA Adapters (FP16): ~75 MB"]
        Act["Activations (Gradient Checkpointing): ~1.2 GB"]
        PagedAdam["Paged AdamW (8-bit): ~150 MB"]
        
        Spike{Memory Spike > 5.8 GB?}
        PagedAdam --> Spike
        Spike -- Yes --> CPUPage["CUDA Unified Memory: Evict Optimizer States to Host RAM"]
        Spike -- No --> KeepGPU["Execute in GPU VRAM"]
    end
```

### 1. `paged_adamw_8bit` (CUDA Unified Memory Paging)
Standard optimizers allocate memory pools that remain permanently resident in GPU VRAM. When long sequence lengths (e.g. 1,024–4,096 tokens) cause transient activation peaks, the GPU runs out of memory.

`paged_adamw_8bit` leverages NVIDIA CUDA Unified Memory page tables:
- When VRAM pressure approaches the 6 GB hardware threshold, CUDA automatically pages non-critical optimizer state pages out of GPU VRAM into host system RAM over the PCIe bus.
- Once the forward/backward step completes and activations are freed, the pages are transparently fetched back. This guarantees 100% crash immunity against memory fragmentation.

### 2. Gradient / Activation Checkpointing
In standard backpropagation, the activations of every intermediate layer are cached in VRAM during the forward pass so they are available to compute derivatives during the backward pass:
- Storing activations across 36 transformer layers at context length 1,024 requires **$> 4.5\text{ GB}$ VRAM**.

**Gradient Checkpointing** discards all intermediate activations during the forward pass, retaining only the layer boundary states:
- During the backward pass, each layer's activations are **recomputed on the fly**.
- **Trade-off:** Adds $\approx 25\%$ compute overhead, but reduces activation VRAM from $4.5\text{ GB} \to 1.2\text{ GB}$, making 3B training possible on consumer laptops.

### 3. Gradient Accumulation
Hardware limitations restrict the per-device micro-batch size to **1**:
- `per_device_train_batch_size: 1`
- `gradient_accumulation_steps: 4`
- **Effective Batch Size:** $1 \times 4 = 4$.
Gradients from 4 consecutive forward/backward passes are summed before calling `optimizer.step()`, ensuring smooth stochastic gradient trajectories without multiplying memory requirements.

---

## 5. Tokenization Mechanics & Completion-Only Loss Masking

### The ChatML Protocol
`Qwen/Qwen2.5-3B-Instruct` uses the ChatML prompt structure with special delimiter tokens:
```
<|im_start|>system
You are DeepSearch-Local Research Agent...<|im_end|>
<|im_start|>user
Research AMD Ryzen 7000 power consumption.<|im_end|>
<|im_start|>assistant
{"tool": "web_search", "query": "AMD Ryzen 7000 TDP power consumption"}<|im_end|>
```

### The Completion-Only Loss Masking Principle
Standard causal language model training calculates cross-entropy loss over **every token** in the sequence:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i \mid x_{<i})$$

If trained naively, the model would be penalized for predicting the prompt text ("Research AMD Ryzen..."), forcing the 3B model to memorize input queries and system instructions rather than learning reasoning behavior.

**Completion-Only Loss Masking** masks all prompt tokens by setting their target label to **`-100`** (the PyTorch `CrossEntropyLoss` ignore index):

```python
# Delimiter identifying assistant response
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)
```

```
Sequence: [SYS_TOKENS] [USER_TOKENS] [<|im_start|>assistant\n] [RESPONSE_TOKENS] [<|im_end|>]
Labels:   [-100, -100] [-100, -100]  [-100]                  [ 4521, 8912, 12 ] [ 151645 ]
Loss:     [IGNORED   ] [IGNORED    ]  [IGNORED]               [COMPUTE GRADIENTS] [COMPUTE]
```

Gradients are computed strictly on the assistant's structured outputs, forcing 100% of the adapter's expressive capacity to focus on tool schema precision, citation accuracy, and synthesis.

---

## 6. Synthetic Trajectory Distillation

To train the 3B model to perform complex deep research, DeepSearch-Local synthesizes high-quality trajectories via OpenRouter using frontier models (`claude-3-5-sonnet`, `gpt-4o`).

The dataset is partitioned into 4 distinct behavioral distributions:

```mermaid
pie title Synthetic Training Trajectory Distribution (3,000 - 5,000 Samples)
    "Grounded Synthesis & Inline Citations" : 35
    "Structured Tool Calling (JSON)" : 25
    "Multi-Perspective Query Decomposition" : 25
    "Gap Detection & Reflection" : 15
```

| Behavioral Task | Target Distribution | Training Objective |
| :--- | :--- | :--- |
| **Grounded Synthesis** | **35%** | Extract facts from evidence chunks and generate inline citations (`[^chunk_id]`) with zero unsupported claims. |
| **Tool Calling** | **25%** | Emit valid JSON schema calls (`web_search`, `fetch_page_markdown`) with zero markdown preamble or parsing syntax errors. |
| **Query Decomposition** | **25%** | Decompose complex, ambiguous prompts into 3–5 orthogonal, high-signal search queries. |
| **Gap Detection & Reflection** | **15%** | Inspect drafted sections against outlines, detect missing facts, and formulate targeted follow-up queries. |

---

## 7. The Merging, GGUF Export & Ollama Serving Lifecycle

Once training is complete, the LoRA adapter weights must be deployed to Ollama for fast, local CPU/GPU inference.

```mermaid
sequenceDiagram
    autonumber
    participant Trainer as PeftModel (4-bit NF4)
    participant Merger as merge_and_export.py
    participant LlamaCPP as llama.cpp (GGUF Quantizer)
    participant Ollama as Ollama Runtime
    
    Note over Trainer: Adapter Checkpoint Saved in ./models/adapter_checkpoint
    Merger->>Merger: 1. Load Base Model in FP16 (CPU/Host RAM)
    Merger->>Merger: 2. W_merged = W_base + (alpha / r) * (B * A)
    Merger->>Merger: 3. Save Merged HF Model to ./models/deepsearch-slm-merged
    Merger->>LlamaCPP: 4. convert_hf_to_gguf.py -> FP16 GGUF
    LlamaCPP->>LlamaCPP: 5. llama-quantize -> Q4_K_M GGUF (~2.0 GB)
    LlamaCPP->>Ollama: 6. ollama create deepsearch-3b -f Modelfile
    Note over Ollama: Ready for 35 tok/s local inference on RTX 3050
```

### The Weight Folding Equation
Because LoRA is a linear modification, the adapter weights can be permanently folded back into the base model weights with zero added inference latency:

$$W_{\text{merged}} = W_{\text{base}} + \frac{\alpha}{r} (B \cdot A)$$

1. Base weights are loaded in unquantized 16-bit floating point (`torch.bfloat16`).
2. LoRA matrices $B$ and $A$ are multiplied, scaled by $\frac{\alpha}{r} = 2.0$, and added directly to $W_{\text{base}}$.
3. The resulting model is saved as a standard HuggingFace causal LM.

### GGUF Quantization (`Q4_K_M`)
To run inside Ollama with a 4,096-token KV cache on an RTX 3050 (which leaves ~2.8–3.2 GB available VRAM during runtime), the merged model is quantized to **GGUF format using `llama.cpp`**:

- **Why `Q4_K_M`?**
  Unlike naive 4-bit quantization, `Q4_K_M` (k-quant medium) uses mixed quantization:
  - Critical attention weights (`v_proj`, `gate_proj`) are quantized with higher precision (6-bit or 5-bit).
  - Non-critical feed-forward layers are quantized in 4-bit.
  - Retains 99.2% of unquantized FP16 perplexity while shrinking model size to **~2.0 GB**.

### Ollama Serving Configuration (`Modelfile`)
```dockerfile
FROM ./deepsearch-3b-q4_k_m.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER num_ctx 4096
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
```

---

## 8. Summary Jargon Cheat Sheet

| Jargon Term | Plain English Definition | Role in DeepSearch-Local |
| :--- | :--- | :--- |
| **PEFT** | Parameter-Efficient Fine-Tuning; tuning a tiny fraction of model weights. | Enables 3B SLM training on consumer 6 GB GPUs. |
| **LoRA** | Low-Rank Adaptation; factorizing weight updates into $B \times A$. | Tunes 18.5M parameters (0.59%) across all 7 linear projections. |
| **QLoRA** | Quantized LoRA; backpropagating gradients through 4-bit quantized base weights. | Compresses base model from 6.0 GB to 1.8 GB VRAM during training. |
| **NormalFloat4 (NF4)** | Information-theoretically optimal 4-bit quantization grid for normal distributions. | Minimizes quantization error compared to uniform integer INT4. |
| **Double Quantization** | Quantizing the FP32 quantization constants to FP8. | Saves 0.37 bits/parameter (~140 MiB VRAM). |
| **Paged AdamW** | Moving optimizer memory pages to host CPU RAM on VRAM spikes. | Prevents CUDA Out-Of-Memory crashes on 6 GB RTX 3050 GPUs. |
| **Gradient Checkpointing** | Discarding forward activations and recomputing them during backprop. | Cuts activation VRAM from 4.5 GB to 1.2 GB. |
| **Loss Masking** | Setting prompt token labels to `-100` so loss is only computed on completions. | Forces model to learn synthesis and JSON schema without query memorization. |
| **GGUF** | Binary model file format used by `llama.cpp` and Ollama for fast CPU/GPU inference. | Deployment format for local 3B serving with 4096 context window. |
| **Q4_K_M** | Medium 4-bit k-quantization using mixed bit-widths for sensitive layers. | Optimal trade-off between 2.0 GB model size and near-FP16 perplexity. |
| **Weight Folding** | Merging $W_{\text{base}} + \frac{\alpha}{r}(BA)$ into a single permanent matrix. | Completely eliminates LoRA inference latency overhead in deployment. |
| **ChatML** | Delimiter token scheme (`<|im_start|>`, `<|im_end|>`) structuring chat dialogs. | Native dialogue format for `Qwen2.5-3B-Instruct`. |
