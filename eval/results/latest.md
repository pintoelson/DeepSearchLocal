# DeepSearch-Local: Base vs Fine-Tuned Benchmark Report
**Date:** 2026-10-27 10:56:27  
**Evaluated Samples:** 4  
**Judge Head-to-Head:** Fine-Tuned Wins: **4** | Ties: **0** | Base Wins: **0**

## 1. Aggregate Scorecard

| Task Capability | Samples | Base Score | Fine-Tuned Score | Delta (%) | Base Judge (1-5) | FT Judge (1-5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Query Decomposition** | 1 | 0.67 | **0.97** | **+45.6%** | 2.5 | **4.0** |
| **Tool Calling** | 1 | 0.00 | **1.00** | **+100000.0%** | 1.0 | **4.5** |
| **Grounded Synthesis** | 1 | 0.00 | **1.00** | **+100000.0%** | 4.0 | **5.0** |
| **Gap Reflection** | 1 | 1.00 | **1.00** | **0.0%** | 3.5 | **4.5** |

## 2. Inference Performance & Speed

- **Fine-Tuned Throughput:** 13.8 tokens/sec (Avg Latency: 8205.3 ms)
- **Base Model Throughput:** 27.1 tokens/sec (Avg Latency: 9583.5 ms)

---

## 3. Side-by-Side Prompt & Response Inspection

### Sample 1 [QUERY_DECOMPOSITION]: Current state and breakthroughs in 1-bit LLM architectures (BitNet b1.58)
**Item ID:** `decomp-01`  
**Judge Verdict:** `FINETUNED` (Base: 2.5/5.0 | FT: 4.0/5.0)  
**Judge Rationale:** *Model B provides five concrete, mostly orthogonal search queries spanning architecture, accuracy, training, scalability, and inference efficiency. Model A offers broad topic headings rather than search-ready queries, with overlap between its coverage of the field’s overview and current state; B could better target recent breakthroughs specific to BitNet b1.58.*  

<details>
<summary><b>View Input Prompt</b></summary>

```text
Decompose this research topic into 3 to 5 targeted, orthogonal web search queries: 'Current state and breakthroughs in 1-bit LLM architectures (BitNet b1.58)'
```

</details>

| Model A: Base SLM | Model B: Fine-Tuned SLM |
| :--- | :--- |
| **Score:** 0.67 (316 tokens, 11481.6ms) | **Score:** 0.97 (99 tokens, 7793.79ms) |

**Base Response:**
```text
To decompose the research topic "Current state and breakthroughs in 1-bit LLM architectures (BitNet b1.58)" into targeted, orthogonal web search queries, we can break it down into specific aspects that will help in gathering comprehensive information. Here are five targeted queries:

1. **Definition and Overview of 1-Bit LLM Architectures**
   - This query will provide an overview of what 1-bit LLM architectures are and how they differ from traditional large language model architectures.

2. **BitNet b1.58 Architecture Details**
   - This query focuses on understanding the specifics of the BitNet b1.58 architecture, including its design and implementation details.

3. **Applications and Use Cases of 1-Bit LLMs**
   - This query will help identify real-world applications and use cases where 1-bit LLMs have been successfully applied, which is crucial for understanding their practical impact.

4. **Advantages and Disadvantages of 1-Bit LLMs**
   - This query will highlight the benefits and limitations of using 1-bit LLM architectures, which is important for evaluating their suitability in different scenarios.

5. **Current State and Future Prospects of 1-Bit LLMs**
   - This query will provide insights into the current status of research and development in 1-bit LLMs, as well as future prospects and potential advancements in the field.

These queries are designed to cover various facets of the research topic, ensuring a thorough exploration of the subject matter.
```

**Fine-Tuned Response:**
```text
{
  "queries": [
    "BitNet b1.58 architecture details and hardware efficiency",
    "1‑bit LLMs vs traditional transformer models: accuracy trade‑offs",
    "Training strategies for 1‑bit language models: data‑augmentation techniques",
    "Scalability analysis of 1‑bit LLMs on large datasets (e.g., CommonCrawl)",
    "Impact of quantization on inference latency and memory bandwidth"
  ]
}
```

---

### Sample 2 [TOOL_CALLING]: Initial web search for BitNet 1-bit LLM efficiency metrics
**Item ID:** `tool-01`  
**Judge Verdict:** `FINETUNED` (Base: 1.0/5.0 | FT: 4.5/5.0)  
**Judge Rationale:** *Model A does not call the search tool and instead provides explanatory chatter. Model B makes a direct, well-formed web_search call with relevant query terms and a result limit, though peer-reviewed sources are not explicitly specified.*  

<details>
<summary><b>View Input Prompt</b></summary>

```text
Call the web search tool to find peer-reviewed papers on BitNet b1.58 energy efficiency and memory consumption compared to FP16 baselines.
```

</details>

| Model A: Base SLM | Model B: Fine-Tuned SLM |
| :--- | :--- |
| **Score:** 0.00 (142 tokens, 5831.72ms) | **Score:** 1.00 (48 tokens, 3794.22ms) |

**Base Response:**
```text
To execute this search effectively, I will use a web search tool to find relevant peer-reviewed papers. However, since I don't have direct access to web search capabilities here, I'll formulate a query that can be used in a web search engine.

Here's the refined query:

"Peer-reviewed papers on BitNet b1.58 energy efficiency and memory consumption compared to FP16 baselines"

This query should help locate relevant academic papers discussing the performance of BitNet b1.58 in comparison to FP16 (Fixed Point 16-bit) in terms of energy efficiency and memory consumption. Once I get the results from the web search, I can analyze them further if needed.
```

**Fine-Tuned Response:**
```text
{
  "tool": "web_search",
  "arguments": {
    "query": "BitNet b1.58 energy efficiency memory consumption FP16 baseline comparison",
    "max_results": 5
  }
}
```

---

### Sample 3 [GROUNDED_SYNTHESIS]: BitNet b1.58 Architecture and Energy Gains
**Item ID:** `synth-01`  
**Judge Verdict:** `FINETUNED` (Base: 4.0/5.0 | FT: 5.0/5.0)  
**Judge Rationale:** *Both responses accurately cover the ternary-weight mechanism, 71.4% energy reduction, and 2.4× DRAM savings, with no material unsupported claims. Model B is stronger because it supplies correct inline citations for each claim, while Model A provides none.*  

<details>
<summary><b>View Input Prompt</b></summary>

```text
Context:
[Chunk chunk_101]: BitNet b1.58 constrains network weights to ternary values {-1, 0, 1} using per-tensor or per-channel absolute maximum (absmax) quantization. Matrix multiplications are reduced to pure additions and subtractions.
[Chunk chunk_102]: Benchmark tests demonstrate that BitNet b1.58 achieves a 71.4% reduction in energy consumption for matrix multiplications compared to FP16 LLaMA baselines.
[Chunk chunk_103]: In terms of memory footprint, model weights require 2.4x less DRAM at 3B scale, and latency improves by 1.8x on custom FPGA accelerators.

Summarize the architectural mechanism, energy reduction percentage, and DRAM memory savings of BitNet b1.58.
```

</details>

| Model A: Base SLM | Model B: Fine-Tuned SLM |
| :--- | :--- |
| **Score:** 0.00 (183 tokens, 7557.31ms) | **Score:** 1.00 (145 tokens, 11755.64ms) |

**Base Response:**
```text
BitNet b1.58 employs a ternary weight quantization technique where network weights are constrained to the values {-1, 0, 1} through per-tensor or per-channel absolute maximum (absmax) quantization. This approach simplifies matrix multiplications to pure additions and subtractions, significantly reducing computational complexity and energy consumption.

The energy reduction achieved by BitNet b1.58 is notable, with benchmark tests showing a 71.4% decrease in energy consumption for matrix multiplications compared to floating-point precision 16 (FP16) LLaMA baselines.

In terms of memory usage, BitNet b1.58 reduces the model's memory footprint by 2.4 times at a 3 billion parameter scale, leading to improved DRAM efficiency. Additionally, it enhances latency performance by 1.8 times on custom FPGA accelerators.
```

**Fine-Tuned Response:**
```text
BitNet b1.58 employs ternary weight quantization constrained to the set {-1, 0, 1} via per-tensor or per-channel absolute maximum (absmax) scaling, which transforms matrix multiplications into sequences of pure addition and subtraction operations [^chunk_101]. This design yields a 71.4% energy reduction for matrix multiplications relative to floating-point16 LLaMA baselines [^chunk_102]. Regarding memory footprint, model weights occupy 2.4 times less DRAM at a 3 billion parameter scale, while achieving a 1.8x improvement in latency on custom FPGA accelerators [^chunk_103].
```

---

### Sample 4 [GAP_REFLECTION]: Reflecting on 1-bit LLM report draft gaps
**Item ID:** `refl-01`  
**Judge Verdict:** `FINETUNED` (Base: 3.5/5.0 | FT: 4.5/5.0)  
**Judge Rationale:** *Model B identifies all three major uncovered target areas and provides focused follow-up queries, including a direct query about accelerator support and deployment barriers. Model A recognizes the gaps but omits a hardware-support follow-up query, adds less central topics, and includes unnecessary explanatory chatter.*  

<details>
<summary><b>View Input Prompt</b></summary>

```text
Research Outline Target: Comprehensive technical analysis of BitNet b1.58 covering architecture, training stability, scaling laws, and hardware accelerator support.

Current Draft Section:
BitNet b1.58 uses ternary weights {-1, 0, 1} with absmax quantization, reducing matrix multiplication energy by 71.4%. Memory footprint is decreased by 2.4x.

Evaluate the draft against the target outline: identify missing critical topics and formulate 2 to 3 targeted follow-up queries.
```

</details>

| Model A: Base SLM | Model B: Fine-Tuned SLM |
| :--- | :--- |
| **Score:** 1.00 (434 tokens, 13463.56ms) | **Score:** 1.00 (166 tokens, 9477.59ms) |

**Base Response:**
```text
The current draft section provides some information about BitNet b1.58, but it lacks several key aspects that are crucial for a comprehensive technical analysis. Here are the identified gaps and corresponding follow-up queries:

### Missing Topics:
1. **Architecture Details**: The draft does not provide any details about the network's structure or how its ternary weights and absmax quantization affect its design.
2. **Training Stability Analysis**: There is no discussion on the stability during training, which is essential for understanding the reliability and performance of the model.
3. **Scaling Laws**: The scaling laws (e.g., how the performance scales with the size of the network) are not covered.
4. **Hardware Accelerator Support**: While mentioning hardware accelerator support, there is no specific information about which accelerators are used and how they enhance performance.
5. **Energy Efficiency Comparison**: The energy efficiency improvement is mentioned, but there should be a comparison with other similar models to validate the claim.
6. **Quantization Effects**: Further explanation on how the absmax quantization affects the model's accuracy and generalization capabilities would be beneficial.

### Follow-Up Queries:
1. **Can you provide more details about the BitNet b1.58 architecture? Specifically, how do the ternary weights and absmax quantization impact the network's structure and performance?**
   - This query aims to understand the architectural choices and their implications.

2. **Could you elaborate on the training process and stability of BitNet b1.58? What measures were taken to ensure stable training, and what were the results?**
   - This query seeks information on the robustness of the training process, which is crucial for real-world applications.

3. **How do the scaling laws of BitNet b1.58 compare to other similar models? Please include a brief comparison table or graph to illustrate the performance trends.**
   - This query helps in validating the claims made about the scaling laws and provides context for the model's performance across different sizes.

By addressing these gaps and following up with these queries, the draft can be significantly improved to meet the outlined research goals.
```

**Fine-Tuned Response:**
```text
{
  "has_gaps": true,
  "sections_to_revise": [
    "Architecture and Training Stability",
    "Scaling Laws and Performance",
    "Hardware Accelerator Support"
  ],
  "follow_up_queries": [
    "What are the specific architectural components and training challenges of BitNet b1.58, including the impact of ternary weight sparsity and absmax quantization on inference latency?",
    "What empirical scaling laws demonstrate how BitNet b1.58 scales with batch size, model depth, and hardware acceleration, and what are the corresponding accuracy and power efficiency results?",
    "Which hardware accelerators support BitNet b1.58, what are their supported tensor shapes and precision, and what are any reported deployment barriers such as memory bandwidth or kernel overhead?"
  ]
}
```

---
