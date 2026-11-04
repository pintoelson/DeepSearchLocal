"""Dataset schema validation, loading, and ChatML formatting with completion-only loss masking.

DeepSearch-Local uses standard ChatML conversational JSONL format:
{
    "messages": [
        {"role": "system", "content": "You are DeepSearch-Local..."},
        {"role": "user", "content": "Research quantum error correction..."},
        {"role": "assistant", "content": "..."}
    ]
}

Completion-only masking ensures the LLM's cross-entropy loss is calculated
ONLY on assistant responses, preventing the model from wasting gradient capacity
memorizing user queries or system instructions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

VALID_ROLES = {"system", "user", "assistant"}


def validate_dataset_schema(dataset_path: str | Path) -> tuple[bool, str | None, int]:
    """Validate that a JSONL file strictly matches the expected ChatML dataset format.

    Args:
        dataset_path: Path to the .jsonl file to validate.

    Returns:
        Tuple of (is_valid, error_message, valid_sample_count).
    """
    path = Path(dataset_path)
    if not path.is_file():
        return False, f"Dataset file does not exist: {path}", 0

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as err:
                return False, f"Line {idx}: Invalid JSON syntax - {err}", count

            if not isinstance(record, dict) or "messages" not in record:
                return False, f"Line {idx}: Missing top-level 'messages' key", count

            messages = record["messages"]
            if not isinstance(messages, list) or len(messages) < 2:
                return (
                    False,
                    f"Line {idx}: 'messages' must be a list containing at least 2 messages (user + assistant)",
                    count,
                )

            has_assistant = False
            for m_idx, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    return False, f"Line {idx}, Message {m_idx}: Entry is not a dictionary", count
                role = msg.get("role")
                content = msg.get("content")

                if role not in VALID_ROLES:
                    return (
                        False,
                        f"Line {idx}, Message {m_idx}: Role '{role}' not in {VALID_ROLES}",
                        count,
                    )
                if not isinstance(content, str) or not content.strip():
                    return (
                        False,
                        f"Line {idx}, Message {m_idx}: Content must be a non-empty string",
                        count,
                    )
                if role == "assistant":
                    has_assistant = True

            if not has_assistant:
                return (
                    False,
                    f"Line {idx}: Trajectory must contain at least one 'assistant' response",
                    count,
                )

            count += 1

    if count == 0:
        return False, "Dataset file is empty or contains only blank lines", 0

    return True, None, count


def create_mock_dataset(output_path: str | Path, count: int = 4) -> Path:
    """Generate a small mock dataset covering the 4 core DeepSearch agent tasks for dry runs & tests.

    Tasks represented:
    1. Query Decomposition
    2. Tool Calling (DDG search / scrape)
    3. Grounded Synthesis with [^citations]
    4. Gap Detection & Reflection
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    samples = [
        # 1. Query Decomposition
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are DeepSearch-Local, an expert autonomous deep research agent.",
                },
                {
                    "role": "user",
                    "content": "Generate targeted search queries to investigate: Current state of 1-bit LLMs (BitNet b1.58).",
                },
                {
                    "role": "assistant",
                    "content": (
                        "{\n"
                        '  "queries": [\n'
                        '    "BitNet b1.58 1-bit LLM architecture energy efficiency",\n'
                        '    "Ternary weights {-1, 0, 1} matrix multiplication hardware acceleration",\n'
                        '    "BitNet b1.58 scaling laws vs FP16 Llama benchmark results"\n'
                        "  ]\n"
                        "}"
                    ),
                },
            ]
        },
        # 2. Tool Calling
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are DeepSearch-Local. Call tools strictly using JSON arguments.",
                },
                {
                    "role": "user",
                    "content": "Search for latest developments in solid-state battery electrolytes.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "{\n"
                        '  "tool": "web_search",\n'
                        '  "arguments": {\n'
                        '    "query": "solid state battery electrolyte breakthrough sulfide oxide polymer 2025 2026",\n'
                        '    "max_results": 5\n'
                        "  }\n"
                        "}"
                    ),
                },
            ]
        },
        # 3. Grounded Synthesis
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are DeepSearch-Local. Synthesize facts and cite sources using [^chunk_id].",
                },
                {
                    "role": "user",
                    "content": "Summarize the mechanism of BitNet b1.58 ternary quantization based on retrieved context.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "BitNet b1.58 constrains weights to the ternary set {-1, 0, 1} through absmax quantization [^chunk_a1]. "
                        "By eliminating floating-point matrix multiplications in favor of integer addition, it achieves up to "
                        "an 8.9x reduction in energy consumption compared to FP16 baselines [^chunk_b2]."
                    ),
                },
            ]
        },
        # 4. Gap Detection & Reflection
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are DeepSearch-Local. Reflect on current draft sections to identify missing evidence.",
                },
                {
                    "role": "user",
                    "content": "Evaluate current sections on solid-state batteries. Section 2 lacks commercial pilot timeline.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "{\n"
                        '  "has_gaps": true,\n'
                        '  "sections_to_revise": ["commercialization_timeline"],\n'
                        '  "follow_up_queries": [\n'
                        '    "Toyota CATL solid-state battery automotive commercial production timeline 2027 2030"\n'
                        "  ]\n"
                        "}"
                    ),
                },
            ]
        },
    ]

    selected = (samples * ((count // len(samples)) + 1))[:count]
    with out_file.open("w", encoding="utf-8") as f:
        for s in selected:
            f.write(json.dumps(s) + "\n")

    return out_file


def format_chatml_for_sft(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, str]:
    """Format a single conversation trajectory into a plain string using the tokenizer's chat template.

    Args:
        example: Dictionary containing a 'messages' list of role/content pairs.
        tokenizer: HuggingFace tokenizer with `apply_chat_template`.

    Returns:
        Dictionary with key 'text' containing the formatted ChatML string.
    """
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def load_and_prepare_dataset(
    config: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
) -> DatasetDict:
    """Load JSONL dataset, validate format, apply chat template, and split train/val.

    Args:
        config: Loaded YAML configuration dictionary.
        tokenizer: HuggingFace tokenizer for Qwen or target SLM.

    Returns:
        DatasetDict with 'train' and 'validation' splits.
    """
    ds_conf = config.get("dataset", {})
    ds_path = ds_conf.get("dataset_path", "./data/synthetic_dataset.jsonl")
    train_split_ratio = float(ds_conf.get("train_split_ratio", 0.95))

    is_valid, err, count = validate_dataset_schema(ds_path)
    if not is_valid:
        raise ValueError(f"Invalid dataset at {ds_path}: {err}")

    raw_dataset = Dataset.from_json(ds_path)

    # Format conversations to text via tokenizer's chat template
    formatted_dataset = raw_dataset.map(
        lambda ex: format_chatml_for_sft(ex, tokenizer),
        desc="Applying ChatML template",
        remove_columns=raw_dataset.column_names,
    )

    if count > 1 and train_split_ratio < 1.0:
        split = formatted_dataset.train_test_split(
            test_size=1.0 - train_split_ratio,
            seed=42,
        )
        return DatasetDict({"train": split["train"], "validation": split["test"]})

    return DatasetDict({"train": formatted_dataset, "validation": formatted_dataset})


class ChatMLCompletionDataCollator:
    """Collates and tokenizes ChatML text sequences while masking non-assistant tokens with -100.

    This ensures that PyTorch's CrossEntropyLoss only computes gradients for the assistant's
    reasoning, tool calls, and text generation tokens, completely ignoring the user input
    and system prompts.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        response_template: str = "<|im_start|>assistant\n",
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.max_length = max_length

        # Encode response template into token IDs without special BOS/EOS
        self.response_token_ids = tokenizer.encode(
            self.response_template,
            add_special_tokens=False,
        )

        # Qwen chat template uses <|im_end|> to terminate turns
        end_token_candidates = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        self.end_token_id = end_token_candidates[0] if end_token_candidates else tokenizer.eos_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]

        # Tokenize batch with dynamic padding up to max_length
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        pad_token_id = self.tokenizer.pad_token_id

        # Mask labels row-by-row
        for row_idx in range(input_ids.size(0)):
            seq_len = input_ids.size(1)
            row_input = input_ids[row_idx]
            row_labels = labels[row_idx]

            in_assistant = False
            resp_len = len(self.response_token_ids)
            i = 0

            while i < seq_len:
                if not in_assistant:
                    # Look ahead for response_template token match
                    if (
                        i + resp_len <= seq_len
                        and row_input[i : i + resp_len].tolist() == self.response_token_ids
                    ):
                        in_assistant = True
                        # Mask the delimiter itself with -100
                        row_labels[i : i + resp_len] = -100
                        i += resp_len
                        continue
                    else:
                        row_labels[i] = -100
                        i += 1
                else:
                    if self.end_token_id is not None and row_input[i].item() == self.end_token_id:
                        # Include the end-of-turn token in loss calculation so the model learns to stop
                        in_assistant = False
                        i += 1
                    else:
                        i += 1

            # Always mask padding tokens with -100
            if pad_token_id is not None:
                row_labels[row_input == pad_token_id] = -100

        batch["labels"] = labels
        return batch
