"""Train the tiny looped LM on a local text file or the built-in demo corpus."""

import argparse

import torch

from looped_lm import LoopedLMConfig, LoopedLanguageModel, count_parameters


DEFAULT_TEXT = "looped language models reuse one transformer block across depth.\n" * 80


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="UTF-8 training text file")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    text = open(args.text, encoding="utf-8").read() if args.text else DEFAULT_TEXT
    chars = sorted(set(text))
    stoi = {char: i for i, char in enumerate(chars)}
    itos = dict(enumerate(chars))
    data = torch.tensor([stoi[char] for char in text], dtype=torch.long)
    if len(data) < 33:
        raise ValueError("need at least 33 characters of training text")

    config = LoopedLMConfig(vocab_size=len(chars), block_size=32, d_model=96, n_heads=4, num_loops=args.loops)
    model = LoopedLanguageModel(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"{count_parameters(model):,} parameters; {args.loops} shared-core loops")
    for step in range(args.steps):
        starts = torch.randint(len(data) - config.block_size - 1, (16,))
        x = torch.stack([data[i : i + config.block_size] for i in starts]).to(args.device)
        # Target at each position is the next character, not a wrapped copy.
        y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in starts]).to(args.device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:4d} | loss {loss.item():.3f}")

    model.eval()
    seed = torch.tensor([[stoi[text[0]]]], device=args.device)
    sample = model.generate(seed, max_new_tokens=160, temperature=0.8, top_k=8)
    print("".join(itos[token] for token in sample[0].tolist()))


if __name__ == "__main__":
    main()
