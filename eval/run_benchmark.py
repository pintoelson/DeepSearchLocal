"""Comparative evaluation runner comparing Base SLM vs Fine-Tuned SLM.

Evaluates performance across the 4 core agent capabilities:
1. Structured Tool Calling (valid JSON schema & arguments)
2. Grounded Synthesis & Citations (exact [^chunk_id] syntax & zero hallucination)
3. Query Decomposition (orthogonal, targeted search queries)
4. Gap Detection & Reflection (missing fact identification & follow-up queries)

Supports:
- Dual-engine inference: Ollama REST API and direct HuggingFace/PEFT (4-bit/16-bit)
- Deterministic syntactic & citation metrics
- Cloud LLM-as-a-Judge (RWTH-GPT primary with Gemini & OpenRouter fallback)
- Rich terminal tables, JSON metric dumps, and side-by-side Markdown inspection reports
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from eval.llm_judge import FallbackJudge, MockJudge
from eval.metrics import (
    evaluate_gap_reflection,
    evaluate_grounded_synthesis,
    evaluate_query_decomposition,
    evaluate_tool_calling,
)
from eval.model_runner import (
    BaseModelRunner,
    GenerationResult,
    HuggingFaceRunner,
    MockRunner,
    OllamaRunner,
)

console = Console()


def load_benchmark_dataset(
    dataset_path: str = "./eval/test_dataset_tasks.jsonl",
    tasks: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load and filter benchmark prompts from JSONL file."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation dataset not found at {path}")

    records: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            item = json.loads(line_str)
            item_task = item.get("task", "")

            if tasks and "all" not in tasks and item_task not in tasks:
                continue

            current_count = task_counts.get(item_task, 0)
            if limit and current_count >= limit:
                continue

            records.append(item)
            task_counts[item_task] = current_count + 1

    return records


def evaluate_task_response(
    task: str,
    response_text: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    """Run task-specific deterministic metric evaluators."""
    if task == "tool_calling":
        allowed = item.get("allowed_tools", ["web_search", "fetch_page_markdown"])
        required = item.get("required_tool")
        return evaluate_tool_calling(response_text, allowed_tools=allowed, required_tool=required)

    elif task == "grounded_synthesis":
        chunks = item.get("context_chunks", [])
        return evaluate_grounded_synthesis(response_text, context_chunks=chunks)

    elif task == "query_decomposition":
        return evaluate_query_decomposition(response_text)

    elif task == "gap_reflection":
        expected_gaps = item.get("expected_gaps", [])
        return evaluate_gap_reflection(response_text, expected_gaps=expected_gaps)

    return {"score": 0.0}


def run_benchmarks(
    backend: str = "hf",
    base_model_name: str | None = None,
    ft_model_name: str | None = None,
    dataset_path: str = "./eval/test_dataset_tasks.jsonl",
    tasks: list[str] | None = None,
    limit: int | None = None,
    use_judge: bool = True,
    output_dir: str = "./eval/results",
) -> dict[str, Any]:
    """Run full automated benchmark suite comparing Base SLM vs Fine-Tuned SLM."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]🚀 DeepSearch-Local: Base vs Fine-Tuned Model Evaluation Suite[/bold cyan]")
    console.print(f"🔧 Backend Engine:  [bold yellow]{backend.upper()}[/bold yellow]")
    console.print(f"📂 Dataset Path:    [blue]{dataset_path}[/blue]")
    console.print(f"🎯 Tasks:           [green]{tasks or ['all']}[/green] (limit: {limit or 'none'})")
    console.print(f"⚖️ LLM Judge:       {'[green]Enabled (RWTH-GPT / Gemini)[/green]' if use_judge else '[yellow]Disabled (Heuristic only)[/yellow]'}\n")

    # 1. Initialize Runners
    base_runner: BaseModelRunner
    ft_runner: BaseModelRunner

    if backend == "ollama":
        base_name = base_model_name or "qwen2.5:3b"
        ft_name = ft_model_name or "deepsearch-3b:latest"
        console.print(f"🔌 Connecting to Ollama: Base='{base_name}', Fine-Tuned='{ft_name}'")
        base_runner = OllamaRunner(model_name=base_name)
        ft_runner = OllamaRunner(model_name=ft_name)

    elif backend == "mock":
        console.print("🧪 Using Deterministic Mock Runner (Offline Mode)")
        base_runner = MockRunner(model_name="mock-base-3b", is_finetuned=False)
        ft_runner = MockRunner(model_name="mock-deepsearch-3b", is_finetuned=True)

    else:  # 'hf'
        base_name = base_model_name or "Qwen/Qwen2.5-3B-Instruct"
        ft_adapter = ft_model_name or "./models/adapter_checkpoint"
        merged_path = "./models/deepsearch-slm-merged"

        # Check if we should use LoRA adapter sharing (most VRAM efficient)
        if os.path.isdir(ft_adapter) and os.path.isfile(os.path.join(ft_adapter, "adapter_config.json")):
            console.print(f"📦 HuggingFace Base Model: [bold]{base_name}[/bold]")
            console.print(f"🧩 Loading Model with LoRA Adapter: [bold]{ft_adapter}[/bold]")
            # Primary runner loads model + adapter into 4-bit VRAM
            ft_runner = HuggingFaceRunner(
                model_path_or_name=base_name,
                adapter_path=ft_adapter,
                load_in_4bit=True,
                enable_adapter=True,
            )
            # Base runner shares identical model in memory but disables adapter during inference
            base_runner = HuggingFaceRunner(
                model_path_or_name=base_name,
                enable_adapter=False,
                shared_model=ft_runner.model,
                shared_tokenizer=ft_runner.tokenizer,
            )
            console.print("✨ [green]VRAM Optimization:[/green] Base and Fine-Tuned models sharing weights in memory via PEFT disable_adapter!\n")
        elif os.path.isdir(merged_path):
            console.print(f"📦 Loading Merged Model: [bold]{merged_path}[/bold]")
            ft_runner = HuggingFaceRunner(model_path_or_name=merged_path, load_in_4bit=True)
            console.print(f"📦 Loading Base Model: [bold]{base_name}[/bold]")
            base_runner = HuggingFaceRunner(model_path_or_name=base_name, load_in_4bit=True)
        else:
            base_runner = HuggingFaceRunner(model_path_or_name=base_name, load_in_4bit=True)
            ft_runner = HuggingFaceRunner(model_path_or_name=base_name, adapter_path=ft_adapter, load_in_4bit=True)

    # 2. Initialize Judge
    judge = FallbackJudge(force_mock=(not use_judge))

    # 3. Load Dataset
    items = load_benchmark_dataset(dataset_path=dataset_path, tasks=tasks, limit=limit)
    console.print(f"📋 Loaded [bold green]{len(items)}[/bold green] benchmark test items across tasks.\n")

    results: list[dict[str, Any]] = []

    # 4. Execute Benchmark Loop
    for idx, item in enumerate(items, start=1):
        item_id = item.get("id", f"item-{idx}")
        task = item.get("task", "general")
        topic = item.get("topic", "")
        system_prompt = item.get("system_prompt", "You are DeepSearch-Local, an expert autonomous deep research agent.")
        user_prompt = item.get("user_prompt", "")

        console.print(f"[{idx}/{len(items)}] Evaluating [bold]{item_id}[/bold] ([cyan]{task}[/cyan]): {topic[:60]}...")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Generate from Base Model
        try:
            base_gen: GenerationResult = base_runner.generate(messages, max_tokens=512)
        except Exception as e:
            console.print(f"  ❌ Base model error: {e}")
            base_gen = GenerationResult(text=f"ERROR: {e}", latency_ms=0, output_tokens=0, tokens_per_second=0, model_name="base")

        # Generate from Fine-Tuned Model
        try:
            ft_gen: GenerationResult = ft_runner.generate(messages, max_tokens=512)
        except Exception as e:
            console.print(f"  ❌ Fine-tuned model error: {e}")
            ft_gen = GenerationResult(text=f"ERROR: {e}", latency_ms=0, output_tokens=0, tokens_per_second=0, model_name="ft")

        # Deterministic Task Metrics
        base_metrics = evaluate_task_response(task, base_gen.text, item)
        ft_metrics = evaluate_task_response(task, ft_gen.text, item)

        # LLM Judge Evaluation
        judge_eval = judge.evaluate_pair(
            task=task,
            topic=topic,
            user_prompt=user_prompt,
            response_base=base_gen.text,
            response_ft=ft_gen.text,
            context_data=item,
        )

        results.append({
            "id": item_id,
            "task": task,
            "topic": topic,
            "user_prompt": user_prompt,
            "base": {
                "text": base_gen.text,
                "latency_ms": base_gen.latency_ms,
                "tokens": base_gen.output_tokens,
                "tps": base_gen.tokens_per_second,
                "metrics": base_metrics,
                "judge_score": judge_eval.get("base_score", 3.0),
            },
            "finetuned": {
                "text": ft_gen.text,
                "latency_ms": ft_gen.latency_ms,
                "tokens": ft_gen.output_tokens,
                "tps": ft_gen.tokens_per_second,
                "metrics": ft_metrics,
                "judge_score": judge_eval.get("ft_score", 3.0),
            },
            "judge": judge_eval,
        })

    # 5. Aggregate Analysis
    summary = aggregate_metrics(results)

    # 6. Print Terminal Results
    print_terminal_summary(summary)

    # 7. Save Reports
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_path / f"benchmark_results_{ts}.json"
    md_path = out_path / f"benchmark_report_{ts}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    markdown_report = generate_markdown_report(summary, results)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(markdown_report)

    # Update latest symlinks
    latest_md = out_path / "latest.md"
    latest_json = out_path / "latest.json"
    try:
        latest_md.write_text(markdown_report, encoding="utf-8")
        latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    console.print(f"\n✅ [bold green]Benchmark Complete![/bold green]")
    console.print(f"📊 Detailed JSON saved to: [blue]{json_path}[/blue]")
    console.print(f"📝 Markdown Report saved to: [blue]{md_path}[/blue]")

    return {"summary": summary, "results": results, "json_path": str(json_path), "md_path": str(md_path)}


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate aggregated scores per task and overall."""
    summary: dict[str, Any] = {
        "total_samples": len(results),
        "tasks": {},
        "judge_summary": {"base_wins": 0, "ft_wins": 0, "ties": 0},
        "performance": {
            "base_avg_latency_ms": 0.0,
            "ft_avg_latency_ms": 0.0,
            "base_avg_tps": 0.0,
            "ft_avg_tps": 0.0,
        },
    }

    if not results:
        return summary

    base_latencies, ft_latencies = [], []
    base_tps_list, ft_tps_list = [], []

    for r in results:
        task = r["task"]
        if task not in summary["tasks"]:
            summary["tasks"][task] = {
                "count": 0,
                "base_scores": [],
                "ft_scores": [],
                "base_judge_scores": [],
                "ft_judge_scores": [],
            }

        t_data = summary["tasks"][task]
        t_data["count"] += 1
        t_data["base_scores"].append(r["base"]["metrics"].get("score", 0.0))
        t_data["ft_scores"].append(r["finetuned"]["metrics"].get("score", 0.0))
        t_data["base_judge_scores"].append(r["base"]["judge_score"])
        t_data["ft_judge_scores"].append(r["finetuned"]["judge_score"])

        base_latencies.append(r["base"]["latency_ms"])
        ft_latencies.append(r["finetuned"]["latency_ms"])
        base_tps_list.append(r["base"]["tps"])
        ft_tps_list.append(r["finetuned"]["tps"])

        winner = r["judge"].get("winner", "tie")
        if winner == "finetuned":
            summary["judge_summary"]["ft_wins"] += 1
        elif winner == "base":
            summary["judge_summary"]["base_wins"] += 1
        else:
            summary["judge_summary"]["ties"] += 1

    # Compute task averages
    for task, t_data in summary["tasks"].items():
        count = t_data["count"]
        t_data["base_avg_score"] = round(sum(t_data["base_scores"]) / count, 4)
        t_data["ft_avg_score"] = round(sum(t_data["ft_scores"]) / count, 4)
        t_data["score_improvement_pct"] = round(
            ((t_data["ft_avg_score"] - t_data["base_avg_score"]) / max(t_data["base_avg_score"], 0.001)) * 100, 1
        )
        t_data["base_avg_judge"] = round(sum(t_data["base_judge_scores"]) / count, 2)
        t_data["ft_avg_judge"] = round(sum(t_data["ft_judge_scores"]) / count, 2)

    # Compute overall performance
    total = len(results)
    summary["performance"]["base_avg_latency_ms"] = round(sum(base_latencies) / total, 1)
    summary["performance"]["ft_avg_latency_ms"] = round(sum(ft_latencies) / total, 1)
    summary["performance"]["base_avg_tps"] = round(sum(base_tps_list) / total, 1)
    summary["performance"]["ft_avg_tps"] = round(sum(ft_tps_list) / total, 1)

    return summary


def print_terminal_summary(summary: dict[str, Any]) -> None:
    """Print formatted comparative summary table in terminal."""
    table = Table(title="🏆 DeepSearch-Local Benchmark: Base vs Fine-Tuned SLM", header_style="bold magenta")
    table.add_column("Task Capability", style="cyan")
    table.add_column("Samples", justify="center")
    table.add_column("Base Score", justify="right")
    table.add_column("Fine-Tuned Score", justify="right")
    table.add_column("Δ Delta (%)", justify="right")
    table.add_column("Base Judge (1-5)", justify="right")
    table.add_column("FT Judge (1-5)", justify="right")

    for task, data in summary.get("tasks", {}).items():
        delta = data["score_improvement_pct"]
        delta_str = f"+{delta}%" if delta > 0 else f"{delta}%"
        delta_style = "bold green" if delta > 0 else ("yellow" if delta == 0 else "red")

        table.add_row(
            task.replace("_", " ").title(),
            str(data["count"]),
            f"{data['base_avg_score']:.2f}",
            f"{data['ft_avg_score']:.2f}",
            f"[{delta_style}]{delta_str}[/{delta_style}]",
            f"{data['base_avg_judge']:.1f}",
            f"{data['ft_avg_judge']:.1f}",
        )

    console.print(table)

    # Judge wins summary
    jw = summary.get("judge_summary", {})
    console.print(
        f"\n⚖️ [bold]Pairwise Judge Win Rates:[/bold] "
        f"[green]Fine-Tuned Wins: {jw.get('ft_wins', 0)}[/green] | "
        f"[yellow]Ties: {jw.get('ties', 0)}[/yellow] | "
        f"[red]Base Wins: {jw.get('base_wins', 0)}[/red]"
    )

    perf = summary.get("performance", {})
    console.print(
        f"⚡ [bold]Throughput:[/bold] Fine-Tuned: {perf.get('ft_avg_tps')} tok/s "
        f"({perf.get('ft_avg_latency_ms')} ms) vs Base: {perf.get('base_avg_tps')} tok/s "
        f"({perf.get('base_avg_latency_ms')} ms)\n"
    )


def generate_markdown_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    """Generate comprehensive Markdown report with side-by-side diff inspector."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = [
        f"# DeepSearch-Local: Base vs Fine-Tuned Benchmark Report",
        f"**Date:** {now_str}  ",
        f"**Evaluated Samples:** {summary.get('total_samples', 0)}  ",
        f"**Judge Head-to-Head:** Fine-Tuned Wins: **{summary['judge_summary'].get('ft_wins', 0)}** | Ties: **{summary['judge_summary'].get('ties', 0)}** | Base Wins: **{summary['judge_summary'].get('base_wins', 0)}**\n",
        "## 1. Aggregate Scorecard\n",
        "| Task Capability | Samples | Base Score | Fine-Tuned Score | Delta (%) | Base Judge (1-5) | FT Judge (1-5) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for task, data in summary.get("tasks", {}).items():
        delta = data["score_improvement_pct"]
        delta_str = f"+{delta}%" if delta > 0 else f"{delta}%"
        md.append(
            f"| **{task.replace('_', ' ').title()}** | {data['count']} | {data['base_avg_score']:.2f} | "
            f"**{data['ft_avg_score']:.2f}** | **{delta_str}** | {data['base_avg_judge']:.1f} | **{data['ft_avg_judge']:.1f}** |"
        )

    perf = summary.get("performance", {})
    md.extend([
        "",
        "## 2. Inference Performance & Speed",
        "",
        f"- **Fine-Tuned Throughput:** {perf.get('ft_avg_tps')} tokens/sec (Avg Latency: {perf.get('ft_avg_latency_ms')} ms)",
        f"- **Base Model Throughput:** {perf.get('base_avg_tps')} tokens/sec (Avg Latency: {perf.get('base_avg_latency_ms')} ms)",
        "",
        "---",
        "",
        "## 3. Side-by-Side Prompt & Response Inspection",
        "",
    ])

    for idx, r in enumerate(results, start=1):
        winner = r["judge"].get("winner", "tie").upper()
        md.extend([
            f"### Sample {idx} [{r['task'].upper()}]: {r['topic']}",
            f"**Item ID:** `{r['id']}`  ",
            f"**Judge Verdict:** `{winner}` (Base: {r['base']['judge_score']}/5.0 | FT: {r['finetuned']['judge_score']}/5.0)  ",
            f"**Judge Rationale:** *{r['judge'].get('reasoning', '')}*  ",
            "",
            "<details>",
            f"<summary><b>View Input Prompt</b></summary>",
            "",
            "```text",
            r["user_prompt"],
            "```",
            "",
            "</details>",
            "",
            "| Model A: Base SLM | Model B: Fine-Tuned SLM |",
            "| :--- | :--- |",
            f"| **Score:** {r['base']['metrics'].get('score', 0.0):.2f} ({r['base']['tokens']} tokens, {r['base']['latency_ms']}ms) | "
            f"**Score:** {r['finetuned']['metrics'].get('score', 0.0):.2f} ({r['finetuned']['tokens']} tokens, {r['finetuned']['latency_ms']}ms) |",
            "",
            "**Base Response:**",
            "```text",
            r["base"]["text"] if r["base"]["text"] else "(Empty)",
            "```",
            "",
            "**Fine-Tuned Response:**",
            "```text",
            r["finetuned"]["text"] if r["finetuned"]["text"] else "(Empty)",
            "```",
            "",
            "---",
            "",
        ])

    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser(description="DeepSearch-Local Comparative Benchmark Runner")
    parser.add_argument("--backend", choices=["hf", "ollama", "mock"], default="hf", help="Inference backend engine")
    parser.add_argument("--base_model", type=str, default=None, help="Base model identifier or path")
    parser.add_argument("--ft_model", type=str, default=None, help="Fine-tuned model / adapter identifier or path")
    parser.add_argument("--dataset_path", type=str, default="./eval/test_dataset_tasks.jsonl", help="Dataset path")
    parser.add_argument("--tasks", type=str, default="all", help="Comma-separated tasks to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Max test items per task")
    parser.add_argument("--no-judge", action="store_true", help="Disable LLM judge and run deterministic metrics only")
    parser.add_argument("--output_dir", type=str, default="./eval/results", help="Directory for output reports")

    args = parser.parse_args()
    tasks_list = [t.strip() for t in args.tasks.split(",")] if args.tasks else ["all"]

    run_benchmarks(
        backend=args.backend,
        base_model_name=args.base_model,
        ft_model_name=args.ft_model,
        dataset_path=args.dataset_path,
        tasks=tasks_list,
        limit=args.limit,
        use_judge=not args.no_judge,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
