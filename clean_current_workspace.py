import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


KEEP_EXPERIMENT_DIRS = {
    "cleanup_manifests",
    "current_best_12971305063123648",
    "current_best_12971552723358022",
    "current_best_12971683198949764",
    "final_model",
    "round51_transition_markov",
    "round65_d2_rich_ranker",
    "round68_d2_strict_candidate",
    "round68_recommended_1_p010_m180",
    "round68_recommended_2_p014_m180",
    "round68_recommended_3_p006_m180",
    "round68_recommended_4_p018_m180",
    "round68_recommended_5_p010_m140",
    "round68_recommended_6_n006_m180",
}


DELETE_ROOT_DIRS = {
    "__pycache__",
    "result",
}


DELETE_FILE_SUFFIXES = (
    ".pyc",
    ".pid",
    ".winpid",
    ":Zone.Identifier",
)


def du(path):
    try:
        out = subprocess.check_output(["du", "-sh", str(path)], text=True)
        return out.split()[0]
    except Exception:
        return "unknown"


def is_inside(child, parent):
    child = child.resolve()
    parent = parent.resolve()
    return child == parent or parent in child.parents


def remove_path(path, root, manifest, dry_run, label):
    if not path.exists() and not path.is_symlink():
        return
    if not is_inside(path, root):
        manifest.write(f"SKIP_UNSAFE {path}\n")
        return
    size = du(path) if path.exists() else "unknown"
    rel = path.relative_to(root)
    manifest.write(f"{label} {size} {rel}\n")
    print(f"{label} {size} {rel}")
    if dry_run:
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def iter_deletable_files(root):
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        name = path.name
        if name.endswith(DELETE_FILE_SUFFIXES):
            yield path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--yes", action="store_true", help="Actually delete files.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not (root / "data").exists() or not (root / "experiments").exists():
        raise SystemExit(f"Unsafe project root: {root}")

    experiments = root / "experiments"
    manifest_dir = experiments / "cleanup_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"workspace_cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    dry_run = not args.yes

    before = du(root)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        manifest.write(f"Root: {root}\n")
        manifest.write(f"Before: {before}\n")
        manifest.write(f"Dry run: {dry_run}\n")
        manifest.write("Keep experiment dirs:\n")
        for name in sorted(KEEP_EXPERIMENT_DIRS):
            manifest.write(f"  {name}\n")

        for name in sorted(DELETE_ROOT_DIRS):
            remove_path(root / name, root, manifest, dry_run, "DELETE_ROOT")

        for path in sorted(iter_deletable_files(root)):
            remove_path(path, root, manifest, dry_run, "DELETE_FILE")

        for path in sorted(experiments.iterdir()):
            if not path.is_dir():
                continue
            if path.name in KEEP_EXPERIMENT_DIRS:
                manifest.write(f"KEEP experiments/{path.name}\n")
                continue
            remove_path(path, root, manifest, dry_run, "DELETE_EXPERIMENT")

        after = du(root)
        manifest.write(f"After: {after}\n")

    print(f"Before: {before}")
    print(f"After: {du(root)}")
    print(f"Manifest: {manifest_path}")
    if dry_run:
        print("Dry run only. Re-run with --yes to delete.")


if __name__ == "__main__":
    main()
