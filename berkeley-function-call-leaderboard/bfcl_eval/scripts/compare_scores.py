import argparse
import json
from pathlib import Path


def load_score_header(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        raise ValueError(f"Empty score file: {path}")
    return json.loads(first_line)


def find_score_files(score_dir: Path, model: str) -> dict:
    model_dir = score_dir / model
    if not model_dir.exists():
        return {}
    files = {}
    for path in model_dir.rglob("BFCL_v4_*_score.json"):
        category = path.stem.replace("BFCL_v4_", "").replace("_score", "")
        files[category] = path
    return files


def format_delta(a, b):
    if a is None or b is None:
        return "n/a"
    return f"{(b - a):+.2%}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID to compare.")
    parser.add_argument(
        "--category",
        nargs="*",
        default=None,
        help="Optional list of categories to compare (default: all available).",
    )
    parser.add_argument(
        "--score-dir-original",
        default="score",
        help="Score dir for original runs (relative to repo root).",
    )
    parser.add_argument(
        "--score-dir-augmented",
        default="score_augmented",
        help="Score dir for augmented runs (relative to repo root).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    score_dir_orig = (repo_root / args.score_dir_original).resolve()
    score_dir_aug = (repo_root / args.score_dir_augmented).resolve()

    orig_files = find_score_files(score_dir_orig, args.model)
    aug_files = find_score_files(score_dir_aug, args.model)

    categories = set(orig_files.keys()) | set(aug_files.keys())
    if args.category:
        categories = set(args.category)

    rows = []
    for category in sorted(categories):
        orig_path = orig_files.get(category)
        aug_path = aug_files.get(category)

        orig_header = load_score_header(orig_path) if orig_path else None
        aug_header = load_score_header(aug_path) if aug_path else None

        orig_acc = orig_header.get("accuracy") if orig_header else None
        aug_acc = aug_header.get("accuracy") if aug_header else None

        rows.append(
            {
                "category": category,
                "orig_acc": f"{orig_acc:.2%}" if orig_acc is not None else "missing",
                "aug_acc": f"{aug_acc:.2%}" if aug_acc is not None else "missing",
                "delta": format_delta(orig_acc, aug_acc),
            }
        )

    if not rows:
        print("No score files found for comparison.")
        return

    print("Model:", args.model)
    print("Original score dir:", score_dir_orig)
    print("Augmented score dir:", score_dir_aug)
    print()
    print(f"{'category':<28} {'original':>10} {'augmented':>10} {'delta':>10}")
    print("-" * 62)
    for row in rows:
        print(
            f"{row['category']:<28} {row['orig_acc']:>10} {row['aug_acc']:>10} {row['delta']:>10}"
        )


if __name__ == "__main__":
    main()
