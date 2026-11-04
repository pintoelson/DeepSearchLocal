"""Interactive terminal chat interface for DeepSearch Base vs Fine-Tuned models."""

import argparse
import sys
from rich.console import Console
from rich.markdown import Markdown

console = Console()


def chat_ollama(model_name: str):
    """Chat interactively using Ollama streaming API."""
    import requests

    console.print(f"[bold green]💬 Starting interactive chat with Ollama model: [yellow]{model_name}[/yellow][/bold green]")
    console.print("[dim]Type your message and press Enter. Type 'exit', 'quit', or Ctrl+C to quit.\n[/dim]")

    messages = []
    base_url = "http://localhost:11434"

    while True:
        try:
            user_input = console.input("[bold cyan]You > [/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                break

            messages.append({"role": "user", "content": user_input})

            console.print(f"\n[bold magenta]{model_name} > [/bold magenta]", end="")
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": True,
            }

            res = requests.post(f"{base_url}/api/chat", json=payload, stream=True, timeout=60)
            if res.status_code != 200:
                console.print(f"\n[red]Ollama error (HTTP {res.status_code}): {res.text}[/red]")
                continue

            full_reply = []
            for line in res.iter_lines():
                if not line:
                    continue
                import json
                chunk = json.loads(line.decode("utf-8"))
                content = chunk.get("message", {}).get("content", "")
                console.print(content, end="", highlight=False)
                full_reply.append(content)

            console.print("\n")
            messages.append({"role": "assistant", "content": "".join(full_reply)})

        except KeyboardInterrupt:
            console.print("\n[yellow]Session ended.[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def chat_huggingface(adapter_path: str | None = None, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Chat interactively using PyTorch / Transformers / PEFT with 4-bit VRAM optimization."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
    from peft import PeftModel

    display_name = "Fine-Tuned SLM (Adapter)" if adapter_path else "Base SLM"
    console.print(f"[bold green]💬 Loading {display_name} in 4-bit precision...[/bold green]")

    tok_source = adapter_path or model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if torch.cuda.is_available() else None

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if adapter_path:
        console.print(f"🧩 Attaching LoRA adapter from: [bold]{adapter_path}[/bold]")
        model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch.bfloat16)
    else:
        model = base_model

    model.eval()
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    console.print(f"\n[bold green]✅ Model loaded! Starting interactive chat with [yellow]{display_name}[/yellow][/bold green]")
    console.print("[dim]Type your message and press Enter. Type 'exit', 'quit', or Ctrl+C to quit.\n[/dim]")

    messages = [
        {"role": "system", "content": "You are DeepSearch-Local, an expert autonomous deep research agent."}
    ]

    while True:
        try:
            user_input = console.input("[bold cyan]You > [/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                break

            messages.append({"role": "user", "content": user_input})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            console.print(f"\n[bold magenta]{display_name} > [/bold magenta]", end="")
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.2,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )

            gen_tokens = out[0][inputs["input_ids"].shape[1]:]
            reply = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            console.print("")
            messages.append({"role": "assistant", "content": reply})

        except KeyboardInterrupt:
            console.print("\n[yellow]Session ended.[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="Interactive Chatbot with Base or Fine-Tuned Model")
    parser.add_argument("--model", choices=["ft", "base"], default="ft", help="'ft' for Fine-Tuned, 'base' for Base model")
    parser.add_argument("--backend", choices=["ollama", "hf"], default="ollama", help="'ollama' (fast) or 'hf' (direct PyTorch)")
    args = parser.parse_args()

    if args.backend == "ollama":
        target = "deepsearch-3b:latest" if args.model == "ft" else "qwen2.5:3b"
        chat_ollama(target)
    else:
        if args.model == "ft":
            chat_huggingface(adapter_path="./models/adapter_checkpoint")
        else:
            chat_huggingface(adapter_path=None)


if __name__ == "__main__":
    main()
