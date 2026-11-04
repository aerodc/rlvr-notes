#!/usr/bin/env python3
"""
The same loop as toy_grpo.py, against a real model, using TRL.

Optional stage. Runs on a GPU; a 0.5B model on a rented single card is enough.
The point is not to learn TRL — it is to watch `reward_std` per group and see
the same four failure modes from toy_grpo.py on real training curves:

    reward_std -> 0            saturated or unsolvable, training has stopped
    reward flat, loss moving   you are optimising something other than the task
    length climbing, acc flat  length bias
    entropy -> 0               mode collapse

Usage:
    uv sync --extra train
    python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --max-steps 100
    python train_grpo.py --smoke          # 5 steps, just checks the wiring
"""

from __future__ import annotations

import argparse
import random
import re

# --------------------------------------------------------------------------- #
# Task: two-digit arithmetic, answer must be boxed
# --------------------------------------------------------------------------- #

BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def as_text(completion) -> str:
    """TRL hands back either a string or a chat-style list of message dicts."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for m in completion:
            if isinstance(m, dict):
                parts.append(str(m.get("content", "")))
            else:
                parts.append(str(m))
        return "\n".join(parts)
    return str(completion)


def extract_boxed(text: str) -> str | None:
    """Take the LAST boxed group; models often restate the answer."""
    matches = BOXED.findall(as_text(text))
    return matches[-1].strip() if matches else None


def normalise(value: str) -> str:
    v = value.strip().replace(",", "").replace(" ", "")
    try:
        return str(int(v))
    except ValueError:
        return v.lower()


def make_dataset(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        rows.append(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            f"Compute {a} * {b}. "
                            "Show your reasoning, then give the final answer as "
                            "\\boxed{number}."
                        ),
                    }
                ],
                "answer": str(a * b),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Rewards: format and correctness, kept separate on purpose
# --------------------------------------------------------------------------- #


def format_reward(completions, **kwargs):
    """Small shaping reward: did it follow the output contract?"""
    return [0.2 if extract_boxed(c) is not None else 0.0 for c in completions]


def accuracy_reward(completions, answer=None, **kwargs):
    """The verifiable part. Deterministic, exact match."""
    answers = answer if answer is not None else [None] * len(completions)
    out = []
    for c, expected in zip(completions, answers):
        pred = extract_boxed(c)
        if pred is None or expected is None:
            out.append(0.0)
        else:
            out.append(1.0 if normalise(pred) == normalise(expected) else 0.0)
    return out


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #


def build_config(args):
    """Build a GRPOConfig, ignoring options this TRL version does not know.

    TRL renames things between minor versions. Filtering against the dataclass
    fields keeps this script working across versions instead of dying on an
    unexpected keyword.
    """
    from dataclasses import fields

    from trl import GRPOConfig

    wanted = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "num_generations": args.num_generations,
        "max_prompt_length": 256,
        "max_completion_length": args.max_completion_length,
        "learning_rate": args.lr,
        "logging_steps": 1,
        "max_steps": args.max_steps,
        "save_steps": 10_000,
        "report_to": "none",
        "temperature": 1.0,
        "beta": 0.0,  # start with no KL; turn it on and watch what changes
        "use_vllm": False,
        "bf16": True,
    }
    valid = {f.name for f in fields(GRPOConfig)}
    dropped = sorted(set(wanted) - valid)
    if dropped:
        print(f"[config] ignoring options this TRL version does not have: {dropped}")
    return GRPOConfig(**{k: v for k, v in wanted.items() if k in valid})


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--steps", type=int, default=2000, help="size of the prompt set")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--num-generations", type=int, default=8, help="GRPO group size G")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--output-dir", default="runs/grpo-arith")
    p.add_argument("--smoke", action="store_true", help="5 steps, wiring check only")
    args = p.parse_args()

    if args.smoke:
        args.max_steps = 5
        args.steps = 32

    from datasets import Dataset
    from trl import GRPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset = Dataset.from_list(make_dataset(args.steps))
    print(f"[data] {len(dataset)} prompts; example: {dataset[0]['prompt'][0]['content'][:60]}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="bfloat16", device_map="auto"
    )

    config = build_config(args)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[accuracy_reward, format_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("[run] watching reward_std, not the loss. When it hits 0, learning has stopped.")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[done] {args.output_dir}")


if __name__ == "__main__":
    main()
