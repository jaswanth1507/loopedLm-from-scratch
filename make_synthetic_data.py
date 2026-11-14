"""Create the fixed synthetic dataset used by the comparison experiment."""

import csv
from pathlib import Path

import torch


NUMBERS = 8
DATA_DIR = Path("data")


def write_split(name: str, size: int, generator: torch.Generator) -> None:
    values = torch.randint(NUMBERS, (size, NUMBERS), generator=generator)
    answers = values.cumsum(dim=1).remainder(NUMBERS)
    with (DATA_DIR / f"prefix_sums_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prompt_numbers", "prefix_sums_mod_8"])
        for prompt, answer in zip(values.tolist(), answers.tolist()):
            writer.writerow([" ".join(map(str, prompt)), " ".join(map(str, answer))])


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    generator = torch.Generator().manual_seed(2026)
    write_split("train", 2_048, generator)
    write_split("validation", 512, generator)
    print("Wrote data/prefix_sums_train.csv and data/prefix_sums_validation.csv")


if __name__ == "__main__":
    main()
