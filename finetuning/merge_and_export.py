"""Merge QLoRA adapter with base model and export to GGUF format for Ollama / llama.cpp.

Workflow:
1. Merge LoRA weights (Delta W) into base model weights (W_0):
   W = W_0 + (alpha / r) * (B @ A)
   Executed in full 16-bit precision (bfloat16) on CPU/GPU.
2. Save merged 16-bit HuggingFace checkpoint and tokenizer.
3. Convert HuggingFace checkpoint to GGUF format for llama.cpp / Ollama runtime.
4. Generate an Ollama Modelfile with 4,096 context window, stop tokens, and system prompt.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SYSTEM_PROMPT = (
    "You are DeepSearch-Local, an expert autonomous deep research agent. "
    "You decompose queries, invoke tools with valid JSON arguments, ground facts "
    "in retrieved evidence with inline citations [^chunk_id], and identify research gaps."
)


def merge_lora_to_hf(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    device: str = "cpu",
) -> Path:
    """Load base model and LoRA adapter, merge weights, and save full 16-bit checkpoint.

    Args:
        base_model_name: HuggingFace model identifier or local directory.
        adapter_path: Directory containing the trained PEFT adapter checkpoint.
        output_dir: Directory to save the merged HuggingFace model.
        device: Device to use for merging ('cpu' recommended to avoid 6GB VRAM limits).

    Returns:
        Path to output directory containing the merged model.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔄 Phase 1: Merging LoRA Adapter into Base Model")
    print("=" * 80)
    print(f"📦 Base Model:    {base_model_name}")
    print(f"🧩 Adapter Path:  {adapter_path}")
    print(f"💾 Output Target: {out_path}")
    print(f"💻 Device:        {device.upper()} (Safe memory isolation)")

    # 1. Load Tokenizer
    print("\n📦 Loading and saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path if (Path(adapter_path) / "tokenizer_config.json").is_file() else base_model_name,
        trust_remote_code=True,
    )
    tokenizer.save_pretrained(out_path)

    # Ensure vocab.json and merges.txt are copied if present in base model
    import shutil
    try:
        from huggingface_hub import hf_hub_download
        for fname in ["vocab.json", "merges.txt"]:
            dest = out_path / fname
            if not dest.is_file():
                try:
                    src = hf_hub_download(repo_id=base_model_name, filename=fname)
                    shutil.copyfile(src, dest)
                    print(f"   • Copied {fname} to {out_path} for GGUF compatibility")
                except Exception:
                    pass
    except Exception:
        pass

    # 2. Load Base Model in 16-bit precision
    # Note: CPU merge uses system RAM (~6GB for 3B SLM) preserving GPU VRAM completely.
    print(f"\n📦 Loading base model in 16-bit bfloat16 on {device.upper()}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # 3. Attach LoRA Adapter
    print(f"\n🧩 Attaching LoRA adapter from '{adapter_path}'...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16,
    )

    # 4. Merge Weights: W_merged = W_base + (alpha / r) * (B @ A)
    print("🧮 Merging adapter weights into base model layers (merge_and_unload)...")
    merged_model = model.merge_and_unload()

    # 5. Save Merged Model
    print(f"💾 Saving merged 16-bit model to '{out_path}'...")
    merged_model.save_pretrained(
        out_path,
        safe_serialization=True,
        max_shard_size="4GB",
    )

    # Ensure top-level rope_theta exists in config.json for GGUF/Ollama compatibility
    cfg_file = out_path / "config.json"
    if cfg_file.is_file():
        import json
        with cfg_file.open("r", encoding="utf-8") as f:
            cfg_data = json.load(f)
        if "rope_theta" not in cfg_data and "rope_parameters" in cfg_data:
            cfg_data["rope_theta"] = cfg_data["rope_parameters"].get("rope_theta", 1000000.0)
            with cfg_file.open("w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=2)

    print(f"✅ LoRA adapter successfully merged and saved to: {out_path}")
    return out_path


def generate_ollama_modelfile(
    model_path: str,
    modelfile_path: str = "./Modelfile",
    system_prompt: str | None = None,
    context_length: int = 4096,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> Path:
    """Generate a production-ready Ollama Modelfile configured for DeepSearch-Local.

    Supports either a direct .gguf file or the merged 16-bit HuggingFace safetensors directory.

    Args:
        model_path: Path to the converted GGUF file OR merged HF model directory.
        modelfile_path: Destination path for the Modelfile.
        system_prompt: Default agent system prompt.
        context_length: Maximum context window tokens (default: 4096).
        temperature: Generation temperature.
        top_p: Nucleus sampling probability.

    Returns:
        Path to the written Modelfile.
    """
    out_file = Path(modelfile_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    abs_model_path = os.path.abspath(model_path)

    # Generate Modelfile content with ChatML template and stop tokens
    content = f"""# DeepSearch-Local Ollama Modelfile
# Auto-generated by merge_and_export.py

FROM {abs_model_path}

# Qwen ChatML Template
TEMPLATE \"\"\"{{{{- if .Messages }}}}
{{{{- range .Messages }}}}<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ else if .Prompt }}}}{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ end -}}}}
\"\"\"

# Runtime Sampling & Context Parameters
PARAMETER num_ctx {context_length}
PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# Default Autonomous Deep Research System Prompt
SYSTEM \"\"\"{prompt}\"\"\"
"""

    with out_file.open("w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Generated Ollama Modelfile at: {out_file}")
    return out_file


def convert_to_gguf(
    hf_model_dir: str,
    output_gguf_path: str,
    out_type: str = "bf16",
) -> bool:
    """Convert a HuggingFace model directory to GGUF format using llama.cpp conversion tools.

    Args:
        hf_model_dir: Directory containing the merged HuggingFace model.
        output_gguf_path: Target path for the .gguf file.
        out_type: Output precision ('bf16', 'f16', 'q8_0').

    Returns:
        True if conversion succeeded, False otherwise.
    """
    print("\n" + "=" * 80)
    print("⚙️ Phase 2: Converting HuggingFace Model to GGUF Format")
    print("=" * 80)

    out_path = Path(output_gguf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Search for convert_hf_to_gguf.py in local tools or PATH
    candidate_scripts = [
        Path("./finetuning/tools/convert_hf_to_gguf.py"),
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
    ]

    script_path = None
    for cand in candidate_scripts:
        if cand.is_file():
            script_path = cand
            break

    if script_path is None:
        print("ℹ️ Note: 'convert_hf_to_gguf.py' from llama.cpp was not found locally.")
        print(f"👉 To convert '{hf_model_dir}' to GGUF, run:")
        print("   git clone https://github.com/ggerganov/llama.cpp.git")
        print(f"   python llama.cpp/convert_hf_to_gguf.py {hf_model_dir} --outfile {output_gguf_path} --outtype {out_type}")
        print(f"   ./llama.cpp/llama-quantize {output_gguf_path} {out_path.with_name(out_path.stem + '-q4_k_m.gguf')} Q4_K_M")
        return False

    cmd = [
        sys.executable,
        str(script_path),
        str(hf_model_dir),
        "--outfile",
        str(output_gguf_path),
        "--outtype",
        out_type,
    ]
    print(f"🚀 Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ GGUF file generated at: {output_gguf_path}")
        return True
    except subprocess.CalledProcessError as err:
        print(f"❌ GGUF conversion failed: {err}")
        return False


def export_pipeline(
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: str = "./models/adapter_checkpoint",
    merged_output_dir: str = "./models/deepsearch-slm-merged",
    output_gguf_path: str = "./models/deepsearch-slm-4096.gguf",
    modelfile_path: str = "./Modelfile",
    device: str = "cpu",
    skip_merge: bool = False,
    skip_gguf: bool = False,
):
    """Execute complete merge, conversion, and Ollama setup pipeline."""
    # 1. Merge weights
    if not skip_merge:
        merge_lora_to_hf(
            base_model_name=base_model_name,
            adapter_path=adapter_path,
            output_dir=merged_output_dir,
            device=device,
        )
    else:
        print(f"⏭️ Skipping merge. Using existing merged checkpoint at: {merged_output_dir}")

    # 2. GGUF Conversion
    if not skip_gguf:
        convert_to_gguf(
            hf_model_dir=merged_output_dir,
            output_gguf_path=output_gguf_path,
        )
    else:
        print("⏭️ Skipping GGUF conversion.")

    # 3. Generate Ollama Modelfile
    # If GGUF file exists, point to it; otherwise point directly to merged HF directory
    target_model_path = output_gguf_path if (not skip_gguf and Path(output_gguf_path).is_file()) else merged_output_dir
    generate_ollama_modelfile(
        model_path=target_model_path,
        modelfile_path=modelfile_path,
    )

    print("\n" + "=" * 80)
    print("🎉 Export Pipeline Complete!")
    print("To register and run in Ollama, execute:")
    print(f"   ollama create deepsearch-3b -f {modelfile_path} -q q4_k_m")
    print("   ollama run deepsearch-3b")
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSearch-Local Model Export & GGUF Conversion")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="./models/adapter_checkpoint")
    parser.add_argument("--merged_output_dir", type=str, default="./models/deepsearch-slm-merged")
    parser.add_argument("--gguf_path", type=str, default="./models/deepsearch-slm-4096.gguf")
    parser.add_argument("--modelfile_path", type=str, default="./Modelfile")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--skip_gguf", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_pipeline(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        merged_output_dir=args.merged_output_dir,
        output_gguf_path=args.gguf_path,
        modelfile_path=args.modelfile_path,
        device=args.device,
        skip_merge=args.skip_merge,
        skip_gguf=args.skip_gguf,
    )
