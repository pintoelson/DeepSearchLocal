"""Unit and integration tests for the QLoRA Fine-Tuning and Export Pipeline."""

import json
import tempfile
from pathlib import Path

import torch
import yaml

from finetuning.dataset import (
    ChatMLCompletionDataCollator,
    create_mock_dataset,
    validate_dataset_schema,
)
from finetuning.merge_and_export import generate_ollama_modelfile


def test_qlora_config_loading():
    """Verify that qlora_3b_rtx3050.yaml contains valid hyperparameters for RTX 3050."""
    config_path = Path("finetuning/configs/qlora_3b_rtx3050.yaml")
    assert config_path.is_file(), "Config file does not exist"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Model checks
    assert config["model"]["base_model_name"] == "Qwen/Qwen2.5-3B-Instruct"
    assert config["model"]["torch_dtype"] == "bfloat16"
    assert config["model"]["use_gradient_checkpointing"] is True

    # Quantization checks
    assert config["quantization"]["load_in_4bit"] is True
    assert config["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert config["quantization"]["bnb_4bit_use_double_quant"] is True

    # LoRA checks
    assert config["lora"]["r"] == 16
    assert config["lora"]["lora_alpha"] == 32
    assert config["lora"]["task_type"] == "CAUSAL_LM"
    assert len(config["lora"]["target_modules"]) == 7

    # Training checks (VRAM budgeting on 6GB card)
    assert config["training"]["per_device_train_batch_size"] == 1
    assert config["training"]["gradient_accumulation_steps"] == 4
    assert config["training"]["max_seq_length"] in [1024, 2048, 4096]
    assert config["training"]["optim"] == "paged_adamw_8bit"


def test_mock_dataset_generation_and_validation():
    """Verify mock dataset creates syntactically valid ChatML records covering the 4 agent tasks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_path = Path(tmpdir) / "test_dataset.jsonl"
        create_mock_dataset(mock_path, count=4)

        is_valid, err, count = validate_dataset_schema(mock_path)
        assert is_valid is True, f"Validation failed with error: {err}"
        assert count == 4

        # Read records and verify messages structure
        with mock_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                assert "messages" in rec
                roles = [m["role"] for m in rec["messages"]]
                assert "user" in roles
                assert "assistant" in roles


def test_dataset_validator_catches_invalid_formats():
    """Verify that validate_dataset_schema rejects malformed or invalid records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. Non-existent file
        is_valid, err, _ = validate_dataset_schema(tmp / "nonexistent.jsonl")
        assert is_valid is False
        assert "does not exist" in err

        # 2. Invalid JSON syntax
        bad_json = tmp / "bad_syntax.jsonl"
        bad_json.write_text("{ unquoted_key: 123 }\n")
        is_valid, err, _ = validate_dataset_schema(bad_json)
        assert is_valid is False
        assert "Invalid JSON syntax" in err

        # 3. Missing messages key
        no_messages = tmp / "no_messages.jsonl"
        no_messages.write_text(json.dumps({"prompt": "hi", "response": "hello"}) + "\n")
        is_valid, err, _ = validate_dataset_schema(no_messages)
        assert is_valid is False
        assert "Missing top-level 'messages' key" in err

        # 4. Unknown role
        bad_role = tmp / "bad_role.jsonl"
        bad_role.write_text(
            json.dumps({"messages": [{"role": "admin", "content": "hello"}, {"role": "assistant", "content": "hi"}]})
            + "\n"
        )
        is_valid, err, _ = validate_dataset_schema(bad_role)
        assert is_valid is False
        assert "not in" in err

        # 5. Missing assistant message
        no_assistant = tmp / "no_assistant.jsonl"
        no_assistant.write_text(
            json.dumps({"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]})
            + "\n"
        )
        is_valid, err, _ = validate_dataset_schema(no_assistant)
        assert is_valid is False
        assert "must contain at least one 'assistant' response" in err


class DummyTokenizer:
    """Mock tokenizer simulating Qwen2.5 ChatML encoding for testing label masking."""

    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 2
        # Mock token dictionary
        self.vocab = {
            "<pad>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "system": 3,
            "user": 4,
            "assistant": 5,
            "\n": 6,
            "hello": 7,
            "world": 8,
            "agent": 9,
        }

    def encode(self, text, add_special_tokens=False):
        if text == "<|im_start|>assistant\n":
            return [1, 5, 6]
        if text == "<|im_end|>":
            return [2]
        return [self.vocab.get(w, 9) for w in text.split()]

    def __call__(self, texts, padding=True, truncation=True, max_length=2048, return_tensors="pt"):
        # Synthesize a fixed batch with response tokens [1, 5, 6]
        # Format: [system(1,3,6), user(1,4,6, 7), assistant(1,5,6, 8, 2), pad(0)]
        input_ids = torch.tensor(
            [
                [1, 3, 6, 9, 2, 1, 4, 6, 7, 2, 1, 5, 6, 8, 2, 0],
            ]
        )
        return {"input_ids": input_ids, "attention_mask": (input_ids != 0).long()}


def test_completion_collator_label_masking():
    """Verify that ChatMLCompletionDataCollator correctly masks prompt tokens with -100."""
    tokenizer = DummyTokenizer()
    collator = ChatMLCompletionDataCollator(
        tokenizer=tokenizer,
        response_template="<|im_start|>assistant\n",
        max_length=64,
    )

    batch_input = [{"text": "dummy input string"}]
    collated = collator(batch_input)

    input_ids = collated["input_ids"][0]
    labels = collated["labels"][0]

    # Verify input_ids and labels have identical shape
    assert input_ids.shape == labels.shape

    # Find where the assistant tokens are: [1, 5, 6, 8, 2, 0]
    # The assistant response tokens are: 8 and 2 (<|im_end|>)
    # Everything before that (system prompt, user prompt, and the delimiter 1,5,6) must be -100
    # The padding token (0) at the end must also be -100
    expected_labels = [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 8, 2, -100]
    assert labels.tolist() == expected_labels, f"Expected {expected_labels}, got {labels.tolist()}"


def test_ollama_modelfile_generation():
    """Verify that generate_ollama_modelfile produces valid Modelfile content."""
    with tempfile.TemporaryDirectory() as tmpdir:
        modelfile_path = Path(tmpdir) / "Modelfile"
        model_path = Path(tmpdir) / "test-model.gguf"
        model_path.touch()

        out_path = generate_ollama_modelfile(
            model_path=str(model_path),
            modelfile_path=str(modelfile_path),
            context_length=4096,
            temperature=0.2,
        )

        assert out_path.is_file()
        content = out_path.read_text(encoding="utf-8")

        assert f"FROM {model_path.resolve()}" in content
        assert "PARAMETER num_ctx 4096" in content
        assert "PARAMETER temperature 0.2" in content
        assert 'PARAMETER stop "<|im_end|>"' in content
        assert 'PARAMETER stop "<|endoftext|>"' in content
        assert "SYSTEM" in content
        assert "DeepSearch-Local" in content
