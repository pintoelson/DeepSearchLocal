"""Local QLoRA fine-tuning and export package optimized for NVIDIA RTX 3050 (6GB VRAM)."""

from finetuning.dataset import (
    ChatMLCompletionDataCollator,
    create_mock_dataset,
    format_chatml_for_sft,
    load_and_prepare_dataset,
    validate_dataset_schema,
)
from finetuning.merge_and_export import (
    export_pipeline,
    generate_ollama_modelfile,
    merge_lora_to_hf,
)
from finetuning.train_qlora import train

__all__ = [
    "ChatMLCompletionDataCollator",
    "create_mock_dataset",
    "export_pipeline",
    "format_chatml_for_sft",
    "generate_ollama_modelfile",
    "load_and_prepare_dataset",
    "merge_lora_to_hf",
    "train",
    "validate_dataset_schema",
]
