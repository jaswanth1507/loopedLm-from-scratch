"""Train a looped LM and parameter-matched normal Transformer on prefix sums.

The first half of each sequence holds random numbers.  After a separator, the
model must emit their running sums modulo 16.  Loss and accuracy are measured
only on that answer half; random prompt tokens are not part of the score.
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


VOCAB_SIZE, SEPARATOR, NUMBERS = 17, 16, 8
STEPS, BATCH_SIZE, EVAL_BATCH_SIZE, EVAL_EVERY = 400, 64, 1_024, 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("results")
DATA_DIR = Path("data")


def load_split(name: str) -> Tensor:
    """Load human-readable prompt/answer rows into complete token sequences."""
    rows = []
    with (DATA_DIR / f"prefix_sums_{name}.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            prompt = [int(value) for value in row["prompt_numbers"].split()]
            answer = [int(value) for value in row["prefix_sums_mod_8"].split()]
            rows.append(prompt + [SEPARATOR] + answer)
    return torch.tensor(rows, dtype=torch.long)


def make_batch(sequences: Tensor, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """Sample fixed rows and return causal inputs, labels, and an answer mask."""
    selected = sequences[torch.randint(len(sequences), (batch_size,))].to(DEVICE)
    inputs, targets = selected[:, :-1], selected[:, 1:]
    # targets[:, NUMBERS:] are exactly the eight prefix-sum answer tokens.
    answer_mask = torch.zeros_like(targets, dtype=torch.bool)
    answer_mask[:, NUMBERS:] = True
    return inputs, targets, answer_mask


def answer_loss_and_accuracy(logits: Tensor, targets: Tensor, answer_mask: Tensor) -> tuple[Tensor, Tensor]:
    selected_logits = logits[answer_mask]
    selected_targets = targets[answer_mask]
    loss = F.cross_entropy(selected_logits, selected_targets)
    accuracy = (selected_logits.argmax(dim=-1) == selected_targets).float().mean()
    return loss, accuracy


@torch.no_grad()
def evaluate(model: torch.nn.Module, validation_sequences: Tensor) -> tuple[float, float]:
    model.eval()
    x, y, mask = make_batch(validation_sequences, EVAL_BATCH_SIZE)
    logits, _ = model(x)
    loss, accuracy = answer_loss_and_accuracy(logits, y, mask)
    return loss.item(), accuracy.item()


def train_one(name: str, model: torch.nn.Module, train_sequences: Tensor, validation_sequences: Tensor) -> dict:
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    history = {"step": [], "loss": [], "accuracy": []}
    for step in range(1, STEPS + 1):
        model.train()
        x, y, mask = make_batch(train_sequences, BATCH_SIZE)
        logits, _ = model(x)
        loss, _ = answer_loss_and_accuracy(logits, y, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % EVAL_EVERY == 0 or step == 1:
            validation_loss, validation_accuracy = evaluate(model, validation_sequences)
            history["step"].append(step)
            history["loss"].append(validation_loss)
            history["accuracy"].append(validation_accuracy)
            print(f"{name:18} step {step:3d} | loss {validation_loss:.3f} | accuracy {validation_accuracy:.1%}")
    return history


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(1)  # Stable, quick CPU runs on small matrices.
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not (DATA_DIR / "prefix_sums_train.csv").exists():
        raise FileNotFoundError("Run: python make_synthetic_data.py")
    train_sequences = load_split("train")
    validation_sequences = load_split("validation")

    # Four 32-wide untied blocks have a similar parameter count to one 64-wide
    # shared block run four times. Both have effective depth four.
    looped = LoopedLanguageModel(LoopedLMConfig(VOCAB_SIZE, block_size=17, d_model=64, n_heads=4, dropout=0.0, num_loops=4))
    normal = NormalTransformer(LoopedLMConfig(VOCAB_SIZE, block_size=17, d_model=32, n_heads=4, dropout=0.0), layers=4)
    results = {
        "task": "8-token modular prefix sums (mod 16); score only answer tokens",
        "seed": 7,
        "looped": {"parameters": count_parameters(looped), **train_one("looped (4 passes)", looped, train_sequences, validation_sequences)},
        "normal": {"parameters": count_parameters(normal), **train_one("normal (4 layers)", normal, train_sequences, validation_sequences)},
    }
    (OUTPUT_DIR / "synthetic_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for key, label in (("looped", "Looped, shared block ×4"), ("normal", "Normal, 4 untied blocks")):
        axes[0].plot(results[key]["step"], results[key]["loss"], marker="o", label=label)
        axes[1].plot(results[key]["step"], [100 * x for x in results[key]["accuracy"]], marker="o", label=label)
    axes[0].set(title="Validation loss", xlabel="Training step", ylabel="Cross-entropy")
    axes[1].set(title="Answer-token accuracy", xlabel="Training step", ylabel="Accuracy (%)", ylim=(0, 100))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(OUTPUT_DIR / "synthetic_comparison.png", dpi=160)
    print(f"\nSaved {OUTPUT_DIR / 'synthetic_comparison.png'}")
    print(f"Looped parameters: {results['looped']['parameters']:,}; normal parameters: {results['normal']['parameters']:,}")


if __name__ == "__main__":
    main()
