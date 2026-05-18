from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack dataset result CSV files into result.zip.")
    parser.add_argument("--input_dir", default="results")
    parser.add_argument("--output", default="result.zip")
    parser.add_argument("--dataset1", default="dataset1_result.csv")
    parser.add_argument("--dataset2", default="dataset2_result.csv")
    return parser


def find_file(input_dir: Path, filename: str) -> Path:
    direct = input_dir / filename
    if direct.exists():
        return direct

    matches = list(input_dir.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"cannot find {filename} under {input_dir}")


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output = Path(args.output)
    dataset1_file = find_file(input_dir, args.dataset1)
    dataset2_file = find_file(input_dir, args.dataset2)

    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(dataset1_file, arcname="dataset1_result.csv")
        archive.write(dataset2_file, arcname="dataset2_result.csv")

    print(f"saved submission zip: {output}")


if __name__ == "__main__":
    main()
