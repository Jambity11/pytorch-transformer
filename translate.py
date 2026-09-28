"""使用实验目录中的最优权重翻译单句。"""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from train import greedy_decode, make_model


def main():
    parser = argparse.ArgumentParser(description="使用训练好的 Transformer 翻译")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--text", required=True, help="待翻译的英文句子")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    args = parser.parse_args()

    run_dir = args.run_dir
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer_src = Tokenizer.from_file(str(run_dir / "tokenizer_en.json"))
    tokenizer_tgt = Tokenizer.from_file(str(run_dir / "tokenizer_it.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config, tokenizer_src, tokenizer_tgt).to(device)
    state = torch.load(run_dir / f"{args.checkpoint}.pt", map_location=device,
                       weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    ids = ([tokenizer_src.token_to_id("[SOS]")]
           + tokenizer_src.encode(args.text).ids
           + [tokenizer_src.token_to_id("[EOS]")])
    if len(ids) > config["seq_len"]:
        parser.error(f"输入太长：{len(ids)} 个 token，最大 {config['seq_len']}")
    ids += [tokenizer_src.token_to_id("[PAD]")] * (config["seq_len"] - len(ids))
    source = torch.tensor([ids], dtype=torch.int64, device=device)
    source_mask = (source != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(1)
    with torch.no_grad():
        result = greedy_decode(model, source, source_mask, tokenizer_tgt,
                               config["seq_len"], device)
    print(tokenizer_tgt.decode(result.cpu().tolist()))


if __name__ == "__main__":
    main()
