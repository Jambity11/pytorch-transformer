"""下载并校验实验使用的固定版本 OPUS Books en-it 数据。"""

import argparse
import hashlib
from pathlib import Path

import requests


REVISION = "1f9f6191d0e91a3c539c2595e2fe48fc1420de9b"
URL = ("https://huggingface.co/datasets/Helsinki-NLP/opus_books/resolve/"
       f"{REVISION}/en-it/train-00000-of-00001.parquet")
SHA256 = "d08901362614143a6dafe23248dfeec63302d7e098b74a8c9565c71555b923cb"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="下载并校验 OPUS Books 英意数据")
    parser.add_argument("--output", type=Path, default=Path("opus_books_en_it.parquet"))
    args = parser.parse_args()
    output = args.output
    if output.exists():
        if sha256(output) != SHA256:
            raise ValueError(f"现有文件校验失败，未覆盖：{output}")
        print(f"数据已存在且 SHA256 正确：{output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with requests.get(URL, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        with temporary.open("wb") as target:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    target.write(chunk)
    actual = sha256(temporary)
    if actual != SHA256:
        temporary.unlink()
        raise ValueError(f"下载校验失败：预期 {SHA256}，实际 {actual}")
    temporary.replace(output)
    print(f"下载完成并校验通过：{output}")


if __name__ == "__main__":
    main()
