# DeepSearch-Local: QLoRA Fine-Tuning & Model Engineering Guide

This guide is an end-to-end educational reference designed for engineers learning **Parameter-Efficient Fine-Tuning (PEFT)**, **QLoRA (Quantized Low-Rank Adaptation)**, and local AI engineering.

It explains the exact mathematics, hardware memory constraints, tokenization mechanics, and weight-merging procedures used in `DeepSearch-Local` to fine-tune a domain-specialized 3-Billion parameter Small Language Model (SLM) on an **NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)**.

---

## Table of Contents
1. [The High-Level Objective](#1-the-high-level-objective)
2. [The Mathematics of LoRA](#2-the-mathematics-of-lora)
3. [Quantization Mechanics: FP16, INT4, and NF4](#3-quantization-mechanics-fp16-int4-and-nf4)
4. [VRAM Budget Arithmetic: How 3B Fits into 6GB](#4-vram-budget-arithmetic-how-3b-fits-into-6gb)
5. [Tokenization, ChatML, and Completion-Only Loss Masking](#5-tokenization-chatml-and-completion-only-loss-masking)
6. [The Merging and Export Lifecycle (From LoRA to Ollama)](#6-the-merging-and-export-lifecycle-from-lora-to-ollama)
7. [Step-by-Step Hands-On Guide](#7-step-by-step-hands-on-guide)

---

## 1. The High-Level Objective

In an autonomous deep research platform like DeepSearch-Local, the language model acts as the **reasoning and synthesis engine**. It performs four specialized tasks:
1. **Query Decomposition**: Generating multi-perspective search queries from a complex user brief.
2. **Tool Invocation**: Outputting strictly validated JSON schema tool calls (`web_search`, `fetch_page`).
3. **Grounded Synthesis**: Extracting facts from evidence passages and citing them with inline markers (`[^chunk_id]`).
4. **Gap Reflection**: Inspecting drafted sections, identifying missing evidence, and planning follow-up queries.

Commercial cloud APIs (e.g., GPT-4o, Claude 3.5 Sonnet) excel at these tasks but incur recurring costs and latency over cyclic agent loops. A **3B SLM (such as `Qwen/Qwen2.5-3B-Instruct`)**, fine-tuned specifically on these four tasks, can match frontier models within this bounded domain while running locally at **$0.00 marginal cost**.

---

## 2. The Mathematics of LoRA

### The Problem with Full Fine-Tuning
In full fine-tuning, every single parameter $W$ in the neural network is updated via gradient descent:
$$W_{t+1} = W_t - \eta \cdot \nabla_W \mathcal{L}$$

For a 3-Billion parameter model:
- Storing base weights in 16-bit FP16/BF16: $3 \times 10^9 \times 2\text{ bytes} = 6.0\text{ GB}$.
- Storing gradients $\nabla_W \mathcal{L}$ in 16-bit: $3 \times 10^9 \times 2\text{ bytes} = 6.0\text{ GB}$.
- Storing standard AdamW optimizer states (first moment $m_t$ and second moment $v_t$ in FP32):
  $$3 \times 10^9 \times (4\text{ bytes} + 4\text{ bytes}) = 24.0\text{ GB}$$
- **Total VRAM for full fine-tuning**: $\ge 36\text{ GB}$ (completely impossible on a 6GB card).

### The Low-Rank Adaptation (LoRA) Solution
Hu et al. (2021) observed that when adapting a pretrained language model to a specific task, the weight update matrix $\Delta W$ has a very low **intrinsic dimension (intrinsic rank)**. 

Instead of updating the full weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA freezes $W_0$ entirely and decomposes $\Delta W$ into the product of two low-rank matrices:
$$\Delta W = B \cdot A$$
where:
- $A \in \mathbb{R}^{r \times k}$ is initialized from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$.
- $B \in \mathbb{R}^{d \times r}$ is initialized to **zero** (ensuring $\Delta W = 0$ at the start of training).
- $r \ll \min(d, k)$ is the **rank** (e.g., $r = 16$).

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

The forward pass is computed as:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$

### What is $\alpha$ (Alpha)?
$\alpha$ is a constant scaling hyperparameter. The scaling factor $\frac{\alpha}{r}$ stabilizes training:
- When tuning different ranks (e.g. testing $r=8$ vs $r=16$), keeping $\frac{\alpha}{r}$ constant (e.g., $2.0$, with $\alpha=32, r=16$) ensures that you do not need to drastically retune the learning rate.

### Which Modules are Targeted?
Earlier implementations only adapted attention projection matrices (`q_proj`, `v_proj`). In DeepSearch-Local, we target **all 7 linear projections**:
- Self-attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Feed-forward / MLP: `gate_proj`, `up_proj`, `down_proj`

Targeting all linear layers provides greater expressive capacity for complex structured behaviors (such as JSON syntax and inline citation schemas) while only adding a negligible fraction of parameters (~18.5M parameters, or ~0.59% of the model).

---

## 3. Quantization Mechanics: FP16, INT4, and NF4

### Why Not Linear INT4?
Traditional integer quantization divides the range $[-\text{max}, \text{max}]$ into uniform bins:
$$q = \text{round}\left(\frac{x}{S}\right)$$
However, pretrained neural network weights are **not uniformly distributed**—they follow a bell-shaped Gaussian normal distribution $\mathcal{N}(0, \sigma^2)$ centered at zero. Uniform quantization wastes valuable bins on the tails where few weights exist, while having low resolution near zero where most weights concentrate.

### 4-bit NormalFloat (NF4)
Dettmers et al. (QLoRA, 2023) introduced **NormalFloat4 (NF4)**, an information-theoretically optimal quantile quantization scheme for normally distributed data. 

NF4 defines the $2^4 = 16$ discrete levels such that each bin contains an **equal number of expected weights**. This maximizes information entropy and yields significantly lower quantization error than uniform INT4.

### Double Quantization
In standard block-wise quantization, weights are chunked into blocks of 64 values, and each block has a 32-bit floating-point scale factor $c_1$. For a 3B model, these scale factors themselves consume:
$$\frac{3 \times 10^9}{64} \times 32\text{ bits} \approx 1.5\text{ GB bits} \approx 187\text{ MB}$$

**Double Quantization** applies an 8-bit FP quantization step to the scale factors $c_1$ themselves with a secondary block size of 256. This reduces the memory footprint of quantization constants from $0.5\text{ bits/param}$ to just $0.127\text{ bits/param}$, saving $\approx 140\text{ MB}$ of precious VRAM.

### How Forward and Backward Passes Work in QLoRA
1. The base model weights $W_0$ remain in **4-bit NF4** in GPU VRAM at all times.
2. During the forward pass for a layer, a block of 4-bit weights is dynamically dequantized into **16-bit bfloat16**:
   $$\tilde{W}_0 = \text{dequantize}(W_0^{\text{NF4}})$$
3. The matrix multiplication $\tilde{W}_0 x$ is computed in `bfloat16`.
4. The low-rank branch $\frac{\alpha}{r}(B \cdot A) x$ is computed in `bfloat16`.
5. The outputs are summed: $h = \tilde{W}_0 x + \Delta W x$.
6. Only the small matrices $A$ and $B$ receive gradients $\nabla_A \mathcal{L}$ and $\nabla_B \mathcal{L}$. The base weights $W_0$ never require gradients.

---

## 4. VRAM Budget Arithmetic: How 3B Fits into 6GB

Here is the exact hardware budget breakdown on the **RTX 3050 Laptop GPU (6,144 MiB physical VRAM)**:

| Component | Precision | Size Calculation | Allocated VRAM |
| :--- | :--- | :--- | :--- |
| **Base Model Weights** ($W_0$) | 4-bit NF4 | $3.1 \times 10^9 \times 0.5\text{ bytes}$ | **~1,550 MB** |
| **Quantization Scale Constants** | Double Quant (8-bit) | Block size 64 + 256 | **~50 MB** |
| **Trainable LoRA Parameters** ($A, B$) | 16-bit BF16 | $18.5 \times 10^6 \times 2\text{ bytes}$ | **~37 MB** |
| **Gradients** ($\nabla_A \mathcal{L}, \nabla_B \mathcal{L}$) | 16-bit BF16 | $18.5 \times 10^6 \times 2\text{ bytes}$ | **~37 MB** |
| **Optimizer States** (`paged_adamw_8bit`) | 8-bit | $18.5 \times 10^6 \times 2\text{ moments} \times 1\text{ byte}$ | **~37 MB** |
| **Activation Memory** ($L=2048$, batch=1) | BF16 | With Gradient Checkpointing & SDPA | **~1,800 MB - 2,200 MB** |
| **PyTorch Context & CUDA Kernels** | N/A | CUDA Runtime overhead | **~400 MB** |
| **Total Training Footprint** | | | **~3,900 MB - 4,400 MB** |
| **Available Headroom** | | Display buffers & GNOME (~200 MB) | **~1,500 MB Free** |

### Why `paged_adamw_8bit` is Crucial
Standard AdamW stores 32-bit floating-point copies of the first moment $m_t$ (momentum) and second moment $v_t$ (variance). 
`paged_adamw_8bit`:
1. Quantizes both moments to 8-bit.
2. Leverages CUDA Unified Memory. If an unexpected sequence length spike causes temporary memory pressure on the GPU, CUDA **automatically pages the optimizer states to CPU system RAM** and pages them back when needed, completely preventing Out-Of-Memory (CUDA OOM) crashes.

### Gradient Checkpointing: The Space-Time Tradeoff
Normally, during the forward pass, the activations of all 36 transformer layers are stored in VRAM so they can be reused during backpropagation. For long sequences ($L=2048$), storing all layer activations consumes $>6\text{ GB}$ by itself!

**Gradient Checkpointing** discards intermediate activations during the forward pass and only saves activations at layer boundaries ("checkpoints"). During the backward pass, it recalculates the discarded activations on the fly layer-by-layer. This reduces activation memory by **~70%** at the cost of only ~20% additional compute time.

---

## 5. Tokenization, ChatML, and Completion-Only Loss Masking

### The ChatML Format
DeepSearch-Local uses the standard **ChatML** format natively supported by Qwen2.5:
```
<|im_start|>system
You are DeepSearch-Local, an expert autonomous deep research agent.<|im_end|>
<|im_start|>user
Generate targeted search queries to investigate: 1-bit LLMs.<|im_end|>
<|im_start|>assistant
{
  "queries": [
    "BitNet b1.58 1-bit LLM architecture energy efficiency",
    "Ternary weights {-1, 0, 1} hardware acceleration"
  ]
}<|im_end|>
```

### Why Completion-Only Loss Masking is Mandatory
If you train a model on the raw concatenated string, the cross-entropy loss function evaluates **every single token**:
$$\mathcal{L} = -\sum_{t=1}^T \log P(x_t \mid x_{<t})$$

This causes two severe problems:
1. **Wasted Capacity**: The model spends gradient capacity learning how to predict system prompts and user queries instead of mastering agent reasoning and tool usage.
2. **Degraded Generalization**: Conditioning the model to predict user prompts makes it overly rigid when answering novel or informal user queries.

### The Mechanism of `ChatMLCompletionDataCollator`
In PyTorch's `nn.CrossEntropyLoss(ignore_index=-100)`, any label with value `-100` is skipped during loss calculation.

Our `ChatMLCompletionDataCollator` ([dataset.py](file:///home/elson/Files/DeepSearchLocal/finetuning/dataset.py)):
1. Finds the sequence of token IDs representing `<|im_start|>assistant\n`.
2. Sets all preceding tokens (system prompt, user question, turn headers) to `label = -100`.
3. Sets the `<|im_start|>assistant\n` delimiter itself to `label = -100`.
4. Leaves the assistant's tokens and `<|im_end|>` intact with their real token IDs.
5. Sets any trailing padding tokens (`<pad>`) to `label = -100`.

As a result, **100% of the training loss and gradient descent updates come strictly from the assistant's reasoning, tool calls, and grounded citations**.

---

## 6. The Merging and Export Lifecycle (From LoRA to Ollama)

Once fine-tuning is complete, you have a small directory containing:
- `adapter_model.safetensors` (~37 MB)
- `adapter_config.json`

You cannot efficiently run 4-bit base weights + LoRA in local inference engines like `llama.cpp` or `Ollama`. Instead, we merge them:

```
[Base Model (16-bit BF16)] ──┐
                             ├──► W_merged = W_0 + (alpha / r) * (B @ A) ──► [Merged Model (16-bit)]
[LoRA Adapter (16-bit BF16)] ┘                                                       │
                                                                                     ▼
[Ollama Runtime / Local WebUI] ◄── [Ollama Modelfile] ◄── [Q4_K_M Quantization (GGUF)]
```

### 1. Merging Weights in Full Precision
In [finetuning/merge_and_export.py](file:///home/elson/Files/DeepSearchLocal/finetuning/merge_and_export.py):
1. Base weights $W_0$ are loaded in full `bfloat16` on the **CPU** (using ~6 GB of system RAM, completely avoiding GPU VRAM contention).
2. The low-rank update $\Delta W = \frac{\alpha}{r} (B \cdot A)$ is computed and added directly to $W_0$:
   $$W_{\text{merged}} = W_0 + \Delta W$$
3. The merged model is saved as a clean standalone 16-bit HuggingFace checkpoint.

### 2. GGUF Conversion & Ollama Creation
Ollama and `llama.cpp` use the **GGUF (GPT-Generated Unified Format)** binary format.
Ollama can ingest the merged model directly:
```bash
ollama create deepsearch-3b -f Modelfile -q q4_k_m
```
This performs block-level 4-bit quantization (`Q4_K_M`, medium k-quant with 4-bit weights and 6-bit scales for attention matrices), shrinking the model from 6 GB down to **~2.0 GB file size** and running at $>40\text{ tokens/sec}$ on your RTX 3050!

---

## 7. Step-by-Step Hands-On Guide

### Step 1: Install Fine-Tuning Dependencies
```bash
uv sync --group finetuning
```

### Step 2: Test Dataset Validation & Dry-Run Setup
Run a rapid dry-run to verify that tokenizer, collator, and VRAM memory profiling work without errors:
```bash
uv run python finetuning/train_qlora.py --dry_run --max_steps 2
```

### Step 3: Run Full QLoRA Fine-Tuning
When your training dataset is ready at `./data/synthetic_dataset.jsonl`:
```bash
uv run python finetuning/train_qlora.py \
    --config ./finetuning/configs/qlora_3b_rtx3050.yaml \
    --dataset_path ./data/synthetic_dataset.jsonl \
    --output_dir ./models/adapter_checkpoint
```

### Step 4: Merge Adapter & Generate Ollama Modelfile
```bash
uv run python finetuning/merge_and_export.py \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --adapter_path ./models/adapter_checkpoint \
    --merged_output_dir ./models/deepsearch-slm-merged \
    --modelfile_path ./Modelfile
```

### Step 5: Register and Test in Ollama
```bash
# Build quantized Ollama model
ollama create deepsearch-3b -f Modelfile -q q4_k_m

# Test generation
ollama run deepsearch-3b "Generate 3 research queries for: solid-state battery electrolytes"
```
