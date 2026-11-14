"""Generate a fixed latent multi-hop reasoning dataset.

Each prompt is an unordered table of eight ``node -> next_node`` facts plus a
start node and a requested hop count.  The one-token label is the node reached
after following that table repeatedly.  Intermediate nodes are never labels.
"""

import csv
from pathlib import Path

import torch


NODES, MAX_HOPS = 8, 8
DATA_DIR = Path("data")


def make_row(hops: int, generator: torch.Generator) -> tuple[str, int]:
    mapping = torch.randint(NODES, (NODES,), generator=generator)
    order = torch.randperm(NODES, generator=generator)
    facts = []
    for node in order.tolist():
        facts.extend((node, mapping[node].item()))
    start = torch.randint(NODES, (), generator=generator).item()
    answer = start
    for _ in range(hops):
        answer = mapping[answer].item()
    # Hop tokens are 8--15, leaving node tokens 0--7 unambiguous.
    prompt = facts + [start, NODES + hops - 1]
    return " ".join(map(str, prompt)), answer


def write_split(name: str, count: int, hop_values: range, generator: torch.Generator) -> None:
    with (DATA_DIR / f"pointer_chase_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prompt_tokens", "answer", "hops"])
        for index in range(count):
            hops = hop_values[index % len(hop_values)]
            prompt, answer = make_row(hops, generator)
            writer.writerow([prompt, answer, hops])


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    generator = torch.Generator().manual_seed(2027)
    write_split("train", 8_000, range(1, 5), generator)
    write_split("validation", 2_000, range(1, 5), generator)
    write_split("extrapolation", 2_000, range(5, 9), generator)
    print("Wrote fixed pointer-chasing train, validation, and extrapolation CSVs.")


if __name__ == "__main__":
    main()
