"""Dual-engine model runner interface supporting Ollama API and HuggingFace/PEFT."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class GenerationResult:
    """Standardized output container for model generation."""

    text: str
    latency_ms: float
    output_tokens: int
    tokens_per_second: float
    model_name: str


class BaseModelRunner(ABC):
    """Abstract interface for local model execution."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> GenerationResult:
        """Generate assistant response from input ChatML messages."""
        pass


class OllamaRunner(BaseModelRunner):
    """Inference runner querying local Ollama instance via REST API."""

    def __init__(
        self,
        model_name: str = "deepsearch-3b:latest",
        base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> GenerationResult:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "stop": ["<|im_end|>", "<|endoftext|>"],
            },
        }

        start_time = time.perf_counter()
        res = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=120,
        )
        elapsed_s = time.perf_counter() - start_time

        if res.status_code != 200:
            raise RuntimeError(
                f"Ollama error (HTTP {res.status_code}) for model '{self.model_name}': {res.text[:300]}"
            )

        data = res.json()
        text = data.get("message", {}).get("content", "").strip()

        # Extract tokens from Ollama stats or calculate fallback
        eval_count = data.get("eval_count")
        if eval_count is None:
            eval_count = max(len(text.split()), 1)

        eval_duration_ns = data.get("eval_duration")
        if eval_duration_ns and eval_duration_ns > 0:
            tps = eval_count / (eval_duration_ns / 1e9)
        else:
            tps = eval_count / max(elapsed_s, 0.001)

        return GenerationResult(
            text=text,
            latency_ms=round(elapsed_s * 1000, 2),
            output_tokens=eval_count,
            tokens_per_second=round(tps, 2),
            model_name=self.model_name,
        )


class HuggingFaceRunner(BaseModelRunner):
    """Direct in-process inference runner using PyTorch and Transformers / PEFT."""

    def __init__(
        self,
        model_path_or_name: str = "Qwen/Qwen2.5-3B-Instruct",
        adapter_path: str | None = None,
        load_in_4bit: bool = True,
        device: str = "auto",
        enable_adapter: bool = True,
        shared_model: Any = None,
        shared_tokenizer: Any = None,
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_name = (adapter_path if enable_adapter else model_path_or_name) or model_path_or_name
        self.device = device
        self.enable_adapter = enable_adapter

        if shared_model is not None and shared_tokenizer is not None:
            self.model = shared_model
            self.tokenizer = shared_tokenizer
            return

        print(f"📦 Loading tokenizer for '{model_path_or_name}'...")
        tok_source = adapter_path if (adapter_path and os.path.isdir(adapter_path)) else model_path_or_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure 4-bit quantization to prevent OOM on RTX 3050 (6GB VRAM)
        quant_config = None
        torch_dtype = torch.bfloat16
        if load_in_4bit and torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        print(f"📦 Loading model weights for '{model_path_or_name}' (4-bit: {bool(quant_config)})...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path_or_name,
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if adapter_path and os.path.exists(adapter_path):
            print(f"🧩 Attaching LoRA adapter from '{adapter_path}'...")
            self.model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=torch_dtype,
            )
        else:
            self.model = base_model

        self.model.eval()

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> GenerationResult:
        import torch

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        start_time = time.perf_counter()
        with torch.no_grad():
            # If this runner represents the Base model on an attached PEFT model, disable adapter
            if hasattr(self.model, "disable_adapter") and not self.enable_adapter:
                with self.model.disable_adapter():
                    output_tokens = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature if temperature > 0 else 0.01,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            else:
                output_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 0.01,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        elapsed_s = time.perf_counter() - start_time

        gen_tokens = output_tokens[0][input_len:]
        num_new_tokens = len(gen_tokens)
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        tps = num_new_tokens / max(elapsed_s, 0.001)

        return GenerationResult(
            text=text,
            latency_ms=round(elapsed_s * 1000, 2),
            output_tokens=num_new_tokens,
            tokens_per_second=round(tps, 2),
            model_name=self.model_name,
        )


class MockRunner(BaseModelRunner):
    """Deterministic mock runner for unit testing and offline dry runs."""

    def __init__(self, model_name: str = "mock-model", is_finetuned: bool = False):
        self.model_name = model_name
        self.is_finetuned = is_finetuned

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> GenerationResult:
        last_msg = messages[-1]["content"] if messages else ""

        if "Context:" in last_msg and "Summarize" in last_msg:
            # Grounded synthesis
            if self.is_finetuned:
                text = (
                    "BitNet b1.58 constrains network weights to {-1, 0, 1} [^chunk_101]. "
                    "It achieves 71.4% energy reduction [^chunk_102] and 2.4x DRAM savings [^chunk_103]."
                )
            else:
                # Base model might use brackets or forget citation formatting
                text = (
                    "BitNet b1.58 uses ternary weights. It provides energy reduction [1] and memory savings."
                )
        elif "Call the web search tool" in last_msg or "Execute search" in str(messages):
            # Tool calling
            if self.is_finetuned:
                text = '```json\n{\n  "tool": "web_search",\n  "arguments": {\n    "query": "BitNet b1.58 energy efficiency FP16 benchmark"\n  }\n}\n```'
            else:
                text = 'I can help search for that! You should query "BitNet b1.58 energy efficiency".'
        elif "Decompose this research topic" in last_msg:
            # Query decomposition
            if self.is_finetuned:
                text = (
                    '1. "BitNet b1.58 ternary weight quantization architecture"\n'
                    '2. "BitNet b1.58 energy efficiency vs FP16 baselines"\n'
                    '3. "Hardware accelerators and FPGA deployment for 1-bit LLMs"'
                )
            else:
                text = "Here are some search queries: BitNet overview, 1-bit models."
        else:
            # Reflection
            if self.is_finetuned:
                text = (
                    "Identified Gaps: The draft lacks coverage of training stability and scaling laws.\n"
                    "Follow-up Queries:\n"
                    '1. "BitNet b1.58 training loss curves and stability optimizations"\n'
                    '2. "1-bit LLM scaling laws vs Chinchilla scaling"'
                )
            else:
                text = "The draft looks good but you could add more information about training."

        return GenerationResult(
            text=text,
            latency_ms=50.0,
            output_tokens=len(text.split()),
            tokens_per_second=120.0,
            model_name=self.model_name,
        )
