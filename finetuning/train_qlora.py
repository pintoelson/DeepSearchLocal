"""PEFT / QLoRA 4-bit fine-tuning script optimized for NVIDIA RTX 3050 6GB VRAM.

Key Architectural & Memory Components:
1. 4-bit NormalFloat (NF4) base weight quantization:
   Loads 3B parameters into ~1.5 GB VRAM instead of ~6.0 GB (FP16).
2. Double Quantization:
   Quantizes the first-stage quantization constants to 8-bit, saving ~0.37 bits/param (~140 MB).
3. Paged 8-bit AdamW (paged_adamw_8bit):
   Allocates 8-bit optimizer states; automatically pages memory to CPU system RAM during spikes.
4. Gradient Checkpointing:
   Discards intermediate forward-pass activations, recomputing them during backprop to save ~2 GB VRAM.
5. SDPA (Scaled Dot-Product Attention):
   Utilizes PyTorch's native FlashAttention/MemoryEfficient kernels.
6. Completion-Only Loss Collator:
   Masks prompt/system tokens with -100 so cross-entropy loss is computed exclusively on assistant tokens.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

# Prevent CUDA memory fragmentation on 6GB VRAM GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from finetuning.dataset import (
    ChatMLCompletionDataCollator,
    create_mock_dataset,
    load_and_prepare_dataset,
    validate_dataset_schema,
)


def print_gpu_memory_summary(stage: str = ""):
    """Log current CUDA VRAM allocation and reserved memory."""
    if not torch.cuda.is_available():
        print(f"[{stage}] CUDA is not available. Running on CPU.")
        return

    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    max_allocated = torch.cuda.max_memory_allocated() / (1024**2)
    device_name = torch.cuda.get_device_name(0)

    print(
        f"[{stage}] GPU ({device_name}): "
        f"Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB | Max Peak: {max_allocated:.1f} MB"
    )


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML training configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(
    config_path: str = "./finetuning/configs/qlora_3b_rtx3050.yaml",
    dataset_path_override: str | None = None,
    output_dir_override: str | None = None,
    max_seq_length_override: int | None = None,
    max_steps_override: int | None = None,
    dry_run: bool = False,
):
    """Execute the end-to-end QLoRA fine-tuning loop."""
    print("=" * 80)
    print("🚀 DeepSearch-Local: QLoRA Fine-Tuning Pipeline Starting")
    print("=" * 80)
    print_gpu_memory_summary("Initialization")

    # 1. Load Configuration
    config = load_config(config_path)
    model_cfg = config.get("model", {})
    quant_cfg = config.get("quantization", {})
    lora_cfg = config.get("lora", {})
    train_cfg = config.get("training", {})
    ds_cfg = config.get("dataset", {})

    base_model_name = model_cfg.get("base_model_name", "Qwen/Qwen2.5-3B-Instruct")
    output_dir = output_dir_override or train_cfg.get("output_dir", "./models/adapter_checkpoint")
    max_seq_length = max_seq_length_override or int(train_cfg.get("max_seq_length", 2048))
    response_template = ds_cfg.get("response_template", "<|im_start|>assistant\n")

    dataset_path = dataset_path_override or ds_cfg.get("dataset_path", "./data/synthetic_dataset.jsonl")
    config.setdefault("dataset", {})["dataset_path"] = dataset_path

    # Handle dry run or missing dataset
    if dry_run or not Path(dataset_path).is_file():
        if dry_run:
            print("🧪 --dry_run flag detected. Generating temporary mock dataset fixture...")
        else:
            print(f"⚠️ Dataset '{dataset_path}' not found. Generating temporary mock fixture for validation...")
        dataset_path = "./data/mock_dataset.jsonl"
        create_mock_dataset(dataset_path, count=8)
        config["dataset"]["dataset_path"] = dataset_path

    is_valid, err_msg, count = validate_dataset_schema(dataset_path)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {err_msg}")
    print(f"✅ Verified training dataset '{dataset_path}' containing {count} trajectories.")

    # 2. Tokenizer Setup
    print(f"\n📦 Loading tokenizer for base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding is required for causal LM batch training
    tokenizer.padding_side = "right"
    print(f"✅ Tokenizer initialized (vocab size: {len(tokenizer)}, pad_token: '{tokenizer.pad_token}')")

    # 3. Quantization Configuration (NF4 + Double Quant)
    print("\n⚙️ Configuring 4-bit NormalFloat (NF4) Quantization with Double Quantization...")
    compute_dtype = torch.bfloat16 if quant_cfg.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # 4. Base Model Loading
    print(f"📦 Loading base model '{base_model_name}' in 4-bit precision...")
    device_map = {"": 0} if torch.cuda.is_available() else None
    attn_impl = model_cfg.get("attn_implementation", "sdpa")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=compute_dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    print_gpu_memory_summary("Post Base Model Load")

    # 5. Prepare Model for k-bit Training & Enable Gradient Checkpointing
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=model_cfg.get("use_gradient_checkpointing", True),
    )
    if model_cfg.get("use_gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        print("✅ Gradient Checkpointing enabled (saving activation memory).")

    # 6. LoRA Adapter Injection
    print("\n🧩 Injecting LoRA adapter layers...")
    target_modules = lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=target_modules,
    )

    model = get_peft_model(model, peft_config)
    print_gpu_memory_summary("Post LoRA Injection")

    # Print Trainable Parameter Stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(
        f"📊 Parameter Efficiency: "
        f"{trainable_params:,} trainable / {all_params:,} total "
        f"({100 * trainable_params / all_params:.2f}% parameters updated)"
    )

    # 7. Dataset Preparation & Completion Masking Collator
    print("\n📚 Preparing dataset and applying ChatML template...")
    dataset_dict = load_and_prepare_dataset(config, tokenizer)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["validation"]
    print(f"✅ Dataset ready: {len(train_dataset)} train samples, {len(eval_dataset)} validation samples.")

    print(f"🎯 Configuring Completion-Only Collator (masking tokens before '{response_template}')...")
    data_collator = ChatMLCompletionDataCollator(
        tokenizer=tokenizer,
        response_template=response_template,
        max_length=max_seq_length,
    )

    # 8. Training Arguments
    num_epochs = int(train_cfg.get("num_train_epochs", 3))
    batch_size = int(train_cfg.get("per_device_train_batch_size", 1))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 4))
    lr = float(train_cfg.get("learning_rate", 2.0e-4))
    max_steps = max_steps_override if max_steps_override is not None else -1

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=int(train_cfg.get("warmup_steps", 10)),
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=2,
        fp16=bool(train_cfg.get("fp16", False)),
        bf16=bool(train_cfg.get("bf16", True)),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=data_collator,
    )

    # 9. Execute Training
    print("\n" + "=" * 80)
    print("🏋️ Beginning Model Fine-Tuning Loop")
    print(f"   Effective batch size: {batch_size * grad_accum} (batch={batch_size}, accum={grad_accum})")
    print(f"   Sequence Length: {max_seq_length} tokens")
    print(f"   Optimizer: {train_cfg.get('optim', 'paged_adamw_8bit')}")
    print("=" * 80)

    train_result = trainer.train()
    print_gpu_memory_summary("Training Complete")

    # 10. Save Adapter Checkpoint & Tokenizer
    print(f"\n💾 Saving LoRA adapter checkpoint to: {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save copy of training configuration for provenance
    with open(os.path.join(output_dir, "training_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    print(f"🎉 Fine-tuning finished! Adapter successfully saved to '{output_dir}'.")
    return train_result


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSearch-Local QLoRA Fine-Tuning Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="./finetuning/configs/qlora_3b_rtx3050.yaml",
        help="Path to YAML training configuration",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override path to JSONL training dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override directory to save adapter checkpoint",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="Override sequence length (e.g. 2048 or 4096)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override maximum training steps (for testing/dry runs)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run a rapid validation loop with generated mock dataset fixture",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        config_path=args.config,
        dataset_path_override=args.dataset_path,
        output_dir_override=args.output_dir,
        max_seq_length_override=args.max_seq_length,
        max_steps_override=args.max_steps,
        dry_run=args.dry_run,
    )
