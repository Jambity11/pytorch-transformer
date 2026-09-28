"""可复现的英意翻译实验：训练、验证、测试和产物保存。"""

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from dataset import BilingualDataset, causal_mask
from model import build_transformer


def parse_args():
    defaults = get_config()
    parser = argparse.ArgumentParser(description="训练并评估 Transformer 翻译模型")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/default"),
                        help="保存配置、权重、曲线和结果的目录，默认 runs/default")
    parser.add_argument("--dataset-file", type=Path, help="本地 OPUS Books en-it Parquet 文件；不填则从 Hugging Face 下载")
    parser.add_argument("--max-samples", type=int, help="抽样句对数；不填使用全部数据")
    parser.add_argument("--epochs", type=int, help="训练总轮数，续训时也是总轮数；新实验默认 20")
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--seq-len", type=int, default=defaults["seq_len"])
    parser.add_argument("--d-model", type=int, default=defaults["d_model"])
    parser.add_argument("--layers", type=int, default=defaults["layers"])
    parser.add_argument("--heads", type=int, default=defaults["heads"])
    parser.add_argument("--d-ff", type=int, default=defaults["d_ff"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--eval-samples", type=int,
                        help="最终贪心解码的测试句数；新实验默认 100，0 表示全部")
    parser.add_argument("--resume", action="store_true", help="从 run-dir/last.pt 续训")
    parser.add_argument("--smoke-test", action="store_true",
                        help="在 CPU 上用两条内置句子检查一次参数更新")
    args = parser.parse_args()
    if (args.epochs is not None and args.epochs < 1) or args.batch_size < 1 or args.seq_len < 3:
        parser.error("epochs、batch-size 必须为正数，seq-len 必须至少为 3")
    if args.heads < 1 or args.d_model < 1 or args.d_model % args.heads or args.layers < 1 or args.d_ff < 1:
        parser.error("d-model、heads、layers 和 d-ff 必须为正数，且 d-model 必须能被 heads 整除")
    if args.max_samples is not None and args.max_samples < 10:
        parser.error("max-samples 至少为 10，才能划分训练、验证、测试集")
    if args.eval_samples is not None and args.eval_samples < 0:
        parser.error("eval-samples 不能为负数")
    return args


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_id = tokenizer_tgt.token_to_id("[SOS]")
    eos_id = tokenizer_tgt.token_to_id("[EOS]")
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.full((1, 1), sos_id, dtype=source.dtype, device=device)
    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        decoded = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        next_id = model.project(decoded[:, -1]).argmax(dim=1).item()
        decoder_input = torch.cat((decoder_input, torch.full((1, 1), next_id,
                                                              dtype=source.dtype, device=device)), dim=1)
        if next_id == eos_id:
            break
    return decoder_input.squeeze(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_tokenizer(dataset, lang, path):
    if path.exists():
        return Tokenizer.from_file(str(path))
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
    )
    tokenizer.train_from_iterator(
        (row["translation"][lang] for row in dataset), trainer=trainer
    )
    tokenizer.save(str(path))
    return tokenizer


def prepare_data(config, run_dir, resume):
    cache_dir = str(Path(__file__).resolve().parent / ".hf_cache" / "datasets")
    dataset_file = config.get("dataset_file")
    if dataset_file:
        path = Path(dataset_file)
        if not path.is_file():
            raise FileNotFoundError(f"找不到数据文件：{path}")
        source = {"kind": "local_parquet", "filename": path.name,
                  "sha256": file_sha256(path)}
        raw = load_dataset("parquet", data_files=str(path), split="train", cache_dir=cache_dir)
    else:
        source = {"kind": "huggingface", "name": "Helsinki-NLP/opus_books", "config": "en-it"}
        raw = load_dataset("opus_books", "en-it", split="train", cache_dir=cache_dir)

    if not {"translation", "id"}.issubset(raw.column_names):
        raise ValueError(f"数据列不符合 OPUS Books en-it 格式：{raw.column_names}")
    total = len(raw)
    shuffled = raw.shuffle(seed=config["seed"])
    if config["max_samples"]:
        shuffled = shuffled.select(range(min(config["max_samples"], total)))
    selected = len(shuffled)
    if selected < 10:
        raise ValueError("至少需要 10 条句对")

    test_size = max(1, round(selected * 0.1))
    val_size = max(1, round(selected * 0.1))
    first = shuffled.train_test_split(test_size=test_size, seed=config["seed"])
    second = first["train"].train_test_split(test_size=val_size, seed=config["seed"])
    splits = {"train": second["train"], "val": second["test"], "test": first["test"]}

    tokenizer_src = build_tokenizer(splits["train"], "en", run_dir / "tokenizer_en.json")
    tokenizer_tgt = build_tokenizer(splits["train"], "it", run_dir / "tokenizer_it.json")

    raw_counts = {name: len(split) for name, split in splits.items()}
    seq_len = config["seq_len"]
    for name, split in splits.items():
        splits[name] = split.filter(
            lambda row: (
                len(tokenizer_src.encode(row["translation"]["en"]).ids) + 2 <= seq_len
                and len(tokenizer_tgt.encode(row["translation"]["it"]).ids) + 1 <= seq_len
            ), desc=f"过滤 {name} 中的过长句子"
        )
        if not len(splits[name]):
            raise ValueError(f"{name} 集过滤后为空，请增大 seq-len")

    manifest = {
        "source": source,
        "dataset_fingerprint": raw._fingerprint,
        "original_examples": total,
        "selected_examples": selected,
        "split_before_filter": raw_counts,
        "split_after_filter": {name: len(split) for name, split in splits.items()},
        "source_vocab_size": tokenizer_src.get_vocab_size(),
        "target_vocab_size": tokenizer_tgt.get_vocab_size(),
        "split_seed": config["seed"],
        "tokenizer_training_split": "train",
        "software": {name: version(name) for name in
                     ("torch", "datasets", "tokenizers", "numpy", "pyarrow")},
    }
    manifest_path = run_dir / "data_manifest.json"
    if resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        keys = ("source", "original_examples", "selected_examples",
                "split_before_filter", "split_after_filter", "source_vocab_size",
                "target_vocab_size", "split_seed", "tokenizer_training_split")
        if any(previous[key] != manifest[key] for key in keys):
            raise ValueError("数据、划分或词表与原实验不一致，不能安全续训")
        if previous["software"] != manifest["software"]:
            print("注意：当前软件版本与原实验不同，数值结果可能有细微差异")
    else:
        save_json(manifest_path, manifest)

    wrapped = {
        name: BilingualDataset(split, tokenizer_src, tokenizer_tgt, "en", "it", seq_len)
        for name, split in splits.items()
    }
    print("数据划分（过滤后）：", manifest["split_after_filter"])
    return wrapped, tokenizer_src, tokenizer_tgt, manifest


def make_model(config, tokenizer_src, tokenizer_tgt):
    return build_transformer(
        tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(),
        config["seq_len"], config["seq_len"],
        d_model=config["d_model"], N=config["layers"],
        h=config["heads"], d_ff=config["d_ff"]
    )


def batch_loss(model, batch, device, loss_fn, target_vocab_size, pad_id):
    encoder_input = batch["encoder_input"].to(device)
    decoder_input = batch["decoder_input"].to(device)
    encoder_mask = batch["encoder_mask"].to(device)
    decoder_mask = batch["decoder_mask"].to(device)
    label = batch["label"].to(device)
    encoded = model.encode(encoder_input, encoder_mask)
    decoded = model.decode(encoded, encoder_mask, decoder_input, decoder_mask)
    logits = model.project(decoded)
    loss = loss_fn(logits.reshape(-1, target_vocab_size), label.reshape(-1))
    return loss, int((label != pad_id).sum().item())


def epoch_loss(model, loader, device, loss_fn, vocab_size, pad_id, optimizer=None):
    model.train(optimizer is not None)
    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(loader, leave=False, desc="训练" if optimizer else "验证")
    with torch.set_grad_enabled(optimizer is not None):
        for batch in iterator:
            loss, tokens = batch_loss(model, batch, device, loss_fn, vocab_size, pad_id)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * tokens
            total_tokens += tokens
            iterator.set_postfix(loss=f"{total_loss / total_tokens:.3f}")
    return total_loss / total_tokens, len(loader)


def save_checkpoint(path, model, optimizer, epoch, global_step, best_val_loss):
    temporary = path.with_suffix(".tmp")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }, temporary)
    temporary.replace(path)


def evaluate_test(model, dataset, tokenizer_tgt, config, run_dir, device, loss_fn):
    from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate

    model.eval()
    pad_id = tokenizer_tgt.token_to_id("[PAD]")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    test_loss, _ = epoch_loss(model, loader, device, loss_fn,
                              tokenizer_tgt.get_vocab_size(), pad_id)
    limit = config["eval_samples"] or len(dataset)
    limit = min(limit, len(dataset))
    predictions = []
    with torch.no_grad():
        for index in tqdm(range(limit), desc="测试集解码"):
            item = dataset[index]
            source = item["encoder_input"].unsqueeze(0).to(device)
            source_mask = item["encoder_mask"].unsqueeze(0).to(device)
            token_ids = greedy_decode(model, source, source_mask, tokenizer_tgt,
                                      config["seq_len"], device)
            predictions.append({
                "source": item["src_text"],
                "reference": item["tgt_text"],
                "prediction": tokenizer_tgt.decode(token_ids.cpu().tolist()),
            })
    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["source", "reference", "prediction"])
        writer.writeheader()
        writer.writerows(predictions)

    predicted = [row["prediction"] for row in predictions]
    expected = [row["reference"] for row in predictions]
    results = {
        "test_loss": test_loss,
        "bleu": BLEUScore()(predicted, [[text] for text in expected]).item(),
        "cer": CharErrorRate()(predicted, expected).item(),
        "wer": WordErrorRate()(predicted, expected).item(),
        "decoded_test_examples": limit,
        "all_test_examples": len(dataset),
        "checkpoint": "best.pt",
    }
    save_json(run_dir / "test_results.json", results)
    return results, predictions


def smoke_test():
    from types import SimpleNamespace

    class TinyTokenizer:
        def __init__(self, words):
            self.vocab = {word: index for index, word in enumerate(
                ["[UNK]", "[PAD]", "[SOS]", "[EOS]"] + words)}

        def token_to_id(self, token):
            return self.vocab[token]

        def encode(self, value):
            return SimpleNamespace(ids=[self.vocab.get(word, 0) for word in value.split()])

        def get_vocab_size(self):
            return len(self.vocab)

    pairs = [
        {"translation": {"en": "i like cats", "it": "amo i gatti"}},
        {"translation": {"en": "you like dogs", "it": "ami i cani"}},
    ]
    source = TinyTokenizer(["i", "you", "like", "cats", "dogs"])
    target = TinyTokenizer(["amo", "ami", "i", "gatti", "cani"])
    dataset = BilingualDataset(pairs, source, target, "en", "it", 8)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    model = build_transformer(source.get_vocab_size(), target.get_vocab_size(), 8, 8,
                              d_model=32, N=1, h=4, d_ff=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = model.projection_layer.proj.weight.detach().clone()
    loss_fn = nn.CrossEntropyLoss(ignore_index=target.token_to_id("[PAD]"))
    loss, _ = batch_loss(model, batch, torch.device("cpu"), loss_fn,
                         target.get_vocab_size(), target.token_to_id("[PAD]"))
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(before, model.projection_layer.proj.weight)
    print(f"CPU 快速检查通过：loss={loss.item():.4f}，完成一次参数更新")


def main():
    args = parse_args()
    if args.smoke_test:
        smoke_test()
        return
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config.json"
    if args.resume:
        if not config_path.is_file() or not (run_dir / "last.pt").is_file():
            raise FileNotFoundError("续训需要已有的 config.json 和 last.pt")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if args.dataset_file:
            config["dataset_file"] = str(args.dataset_file.resolve())
        if args.epochs is not None:
            config["epochs"] = args.epochs
        if args.eval_samples is not None:
            config["eval_samples"] = args.eval_samples
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"实验目录已有内容：{run_dir}；续训请加 --resume")
        run_dir.mkdir(parents=True, exist_ok=True)
        config = get_config()
        config.update({
            "dataset_file": str(args.dataset_file) if args.dataset_file else None,
            "max_samples": args.max_samples,
            "epochs": config["epochs"] if args.epochs is None else args.epochs,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "d_ff": args.d_ff,
            "lr": args.lr,
            "seed": args.seed,
            "eval_samples": config["eval_samples"] if args.eval_samples is None else args.eval_samples,
        })
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}；实验目录：{run_dir}")
    splits, tokenizer_src, tokenizer_tgt, _ = prepare_data(config, run_dir, args.resume)
    save_json(config_path, config)

    model = make_model(config, tokenizer_src, tokenizer_tgt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    pad_id = tokenizer_tgt.token_to_id("[PAD]")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    first_epoch, global_step, best_val_loss = 0, 0, float("inf")
    if args.resume:
        checkpoint = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        first_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_val_loss = checkpoint["best_val_loss"]
        print(f"从第 {first_epoch + 1} 轮续训")

    history_path = run_dir / "history.csv"
    if not args.resume:
        with history_path.open("w", encoding="utf-8", newline="") as output:
            csv.writer(output).writerow(["epoch", "train_loss", "val_loss", "seconds", "global_step"])
    val_loader = DataLoader(splits["val"], batch_size=config["batch_size"], shuffle=False)
    for epoch in range(first_epoch, config["epochs"]):
        set_seed(config["seed"] + epoch)
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        train_loader = DataLoader(splits["train"], batch_size=config["batch_size"],
                                  shuffle=True, generator=generator)
        started = time.perf_counter()
        train_loss, batches = epoch_loss(model, train_loader, device, loss_fn,
                                         tokenizer_tgt.get_vocab_size(), pad_id, optimizer)
        val_loss, _ = epoch_loss(model, val_loader, device, loss_fn,
                                  tokenizer_tgt.get_vocab_size(), pad_id)
        global_step += batches
        elapsed = time.perf_counter() - started
        improved = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, global_step, best_val_loss)
        if improved:
            shutil.copyfile(run_dir / "last.pt", run_dir / "best.pt")
        with history_path.open("a", encoding="utf-8", newline="") as output:
            csv.writer(output).writerow([epoch + 1, train_loss, val_loss, round(elapsed, 2), global_step])
        print(f"第 {epoch + 1} 轮：train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"用时={elapsed:.1f}s")

    from plot_history import plot_history
    plot_history(run_dir)
    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    print(f"使用验证损失最低的第 {best['epoch'] + 1} 轮权重评估测试集")
    results, _ = evaluate_test(model, splits["test"], tokenizer_tgt,
                               config, run_dir, device, loss_fn)
    results["best_epoch"] = best["epoch"] + 1
    save_json(run_dir / "test_results.json", results)
    print(f"测试结果：{results}")


if __name__ == "__main__":
    main()
