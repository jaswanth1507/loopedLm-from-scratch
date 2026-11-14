"""Compare recurrent depth with fixed depth on latent pointer chasing.

The looped model executes exactly the requested 1--4 loops during training.
At extrapolation time it receives 5--8 loops. The normal baseline has four
physical blocks, so it remains at four blocks for every test example.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.nn import functional as F

from looped_lm import LoopedLMConfig, LoopedLanguageModel, count_parameters
from normal_transformer import NormalTransformer


NODES, VOCAB_SIZE, PROMPT_LENGTH = 8, 16, 18
STEPS, BATCH_SIZE, EVAL_EVERY = 1_200, 96, 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR, OUTPUT_DIR = Path("data"), Path("results")


def load_split(name: str) -> tuple[Tensor, Tensor, Tensor]:
    prompts, answers, hops = [], [], []
    with (DATA_DIR / f"pointer_chase_{name}.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prompts.append([int(token) for token in row["prompt_tokens"].split()])
            answers.append(int(row["answer"]))
            hops.append(int(row["hops"]))
    return torch.tensor(prompts), torch.tensor(answers), torch.tensor(hops)


def sample(split: tuple[Tensor, Tensor, Tensor], batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
    prompts, answers, hops = split
    indices = torch.randint(len(prompts), (batch_size,))
    return prompts[indices].to(DEVICE), answers[indices].to(DEVICE), hops[indices].to(DEVICE)


def final_token_logits(model: torch.nn.Module, prompts: Tensor, hops: Tensor, looped: bool) -> Tensor:
    if looped:
        # Group by requested iteration count because each row has its own loop budget.
        output = torch.empty((len(prompts), VOCAB_SIZE), device=DEVICE)
        for loops in hops.unique().tolist():
            mask = hops == loops
            output[mask] = model(prompts[mask], loops=int(loops))[0][:, -1]
        return output
    return model(prompts)[0][:, -1]


@torch.no_grad()
def accuracy(model: torch.nn.Module, split: tuple[Tensor, Tensor, Tensor], looped: bool) -> float:
    model.eval()
    prompts, answers, hops = (value.to(DEVICE) for value in split)
    predictions = final_token_logits(model, prompts, hops, looped).argmax(dim=-1)
    return (predictions == answers).float().mean().item()


def train(name: str, model: torch.nn.Module, train_data: tuple[Tensor, Tensor, Tensor], validation: tuple[Tensor, Tensor, Tensor], extrapolation: tuple[Tensor, Tensor, Tensor], looped: bool) -> dict:
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    history = {"step": [], "in_range_accuracy": [], "extra_hop_accuracy": []}
    for step in range(1, STEPS + 1):
        model.train()
        prompts, answers, hops = sample(train_data, BATCH_SIZE)
        logits = final_token_logits(model, prompts, hops, looped)
        loss = F.cross_entropy(logits, answers)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % EVAL_EVERY == 0 or step == 1:
            in_range = accuracy(model, validation, looped)
            extra_hop = accuracy(model, extrapolation, looped)
            history["step"].append(step)
            history["in_range_accuracy"].append(in_range)
            history["extra_hop_accuracy"].append(extra_hop)
            print(f"{name:18} step {step:4d} | 1-4 hops {in_range:.1%} | 5-8 hops {extra_hop:.1%}")
    return history


def main() -> None:
    torch.manual_seed(9)
    torch.set_num_threads(1)
    if not (DATA_DIR / "pointer_chase_train.csv").exists():
        raise FileNotFoundError("Run: python make_reasoning_data.py")
    OUTPUT_DIR.mkdir(exist_ok=True)
    train_data = load_split("train")
    validation, extrapolation = load_split("validation"), load_split("extrapolation")

    # Both have about 30k parameters and four effective passes in the train range.
    looped = LoopedLanguageModel(LoopedLMConfig(VOCAB_SIZE, PROMPT_LENGTH, d_model=48, n_heads=4, dropout=0.0, num_loops=4))
    normal = NormalTransformer(LoopedLMConfig(VOCAB_SIZE, PROMPT_LENGTH, d_model=24, n_heads=4, dropout=0.0), layers=4)
    results = {
        "task": "latent pointer chasing through unordered node-to-node facts",
        "seed": 9,
        "looped": {"parameters": count_parameters(looped), **train("looped, variable loops", looped, train_data, validation, extrapolation, True)},
        "normal": {"parameters": count_parameters(normal), **train("normal, fixed depth", normal, train_data, validation, extrapolation, False)},
    }
    (OUTPUT_DIR / "reasoning_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(6.8, 3.8), layout="constrained")
    for key, color in (("looped", "tab:blue"), ("normal", "tab:orange")):
        label = "Looped" if key == "looped" else "Normal"
        axis.plot(results[key]["step"], [100 * x for x in results[key]["in_range_accuracy"]], color=color, label=f"{label}: trained 1–4 hops")
        axis.plot(results[key]["step"], [100 * x for x in results[key]["extra_hop_accuracy"]], color=color, linestyle="--", label=f"{label}: unseen 5–8 hops")
    axis.set(title="Latent pointer-chasing accuracy", xlabel="Training step", ylabel="Exact final-node accuracy (%)", ylim=(0, 100))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.savefig(OUTPUT_DIR / "reasoning_comparison.png", dpi=160)
    print(f"Saved {OUTPUT_DIR / 'reasoning_comparison.png'}")
    print(f"Parameters — looped: {results['looped']['parameters']:,}; normal: {results['normal']['parameters']:,}")


if __name__ == "__main__":
    main()
