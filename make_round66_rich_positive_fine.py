import json
import os
import os.path as osp
import shutil

import numpy as np

from lgbm_candidate_ranker import apply_residual, pack_submission, read_scores, write_scores


OLD_BASE_DIR = "./experiments/current_best_12971305063123648"
NEW_BEST_DIR = "./experiments/current_best_12971552723358022"
RICH_RAW = "./experiments/round65_d2_rich_ranker/dataset2/dataset2_rich_raw.npy"
OUT_SUMMARY = "./experiments/round66_rich_positive_fine_summary.json"


def tag(value, scale=1000):
    return f"p{int(round(value * scale)):03d}"


def copy_d1(source_dir, run_root):
    src = osp.join(source_dir, "dataset1", "dataset1_result.csv")
    dst = osp.join(run_root, "dataset1", "dataset1_result.csv")
    if not osp.exists(src):
        raise FileNotFoundError(src)
    os.makedirs(osp.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def make_variant(name, d1_dir, d2_base, raw, alpha, margin):
    run_root = osp.join("./experiments", name)
    copy_d1(d1_dir, run_root)
    d2_scores, modified = apply_residual(d2_base, raw, alpha, margin)
    write_scores(d2_scores, osp.join(run_root, "dataset2", "dataset2_result.csv"))
    zip_path = pack_submission(run_root)
    record = {
        "name": name,
        "zip": zip_path,
        "dataset2_alpha": float(alpha),
        "dataset2_margin": float(margin),
        "dataset2_modified_rows": int(modified),
        "dataset2_modified_ratio": float(modified / len(d2_base)),
    }
    print("Variant:", record)
    return record


def alias(source_name, alias_name):
    src = osp.join("./experiments", source_name, "result.zip")
    dst_dir = osp.join("./experiments", alias_name)
    dst = osp.join(dst_dir, "result.zip")
    if not osp.exists(src):
        raise FileNotFoundError(src)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copyfile(src, dst)
    return dst


def main():
    raw = np.load(RICH_RAW)
    old_base_d2 = read_scores(osp.join(OLD_BASE_DIR, "dataset2", "dataset2_result.csv"))
    new_best_d2 = read_scores(osp.join(NEW_BEST_DIR, "dataset2", "dataset2_result.csv"))

    records = []
    old_grid = [
        (0.012, 0.18),
        (0.015, 0.18),
        (0.018, 0.18),
        (0.022, 0.18),
        (0.025, 0.18),
        (0.030, 0.18),
        (0.020, 0.14),
        (0.020, 0.15),
        (0.020, 0.16),
        (0.020, 0.17),
        (0.020, 0.19),
        (0.020, 0.20),
        (0.020, 0.22),
    ]
    for alpha, margin in old_grid:
        name = f"round66_oldbase_rich_{tag(alpha)}_m{int(round(margin * 100)):03d}"
        record = make_variant(name, OLD_BASE_DIR, old_base_d2, raw, alpha, margin)
        record["base_mode"] = "old_base_direct_alpha"
        records.append(record)

    # Very small incremental variants on top of the new online-best p020_m018.
    # These test whether the rich residual wants slightly more positive weight.
    inc_grid = [
        (0.004, 0.18),
        (0.006, 0.18),
        (0.008, 0.18),
        (0.010, 0.18),
        (0.006, 0.16),
        (0.006, 0.20),
    ]
    for alpha, margin in inc_grid:
        name = f"round66_newbest_rich_inc_{tag(alpha)}_m{int(round(margin * 100)):03d}"
        record = make_variant(name, NEW_BEST_DIR, new_best_d2, raw, alpha, margin)
        record["base_mode"] = "new_best_incremental"
        records.append(record)

    recommended_names = [
        "round66_oldbase_rich_p025_m018",
        "round66_oldbase_rich_p015_m018",
        "round66_oldbase_rich_p020_m016",
        "round66_oldbase_rich_p020_m020",
        "round66_newbest_rich_inc_p006_m018",
        "round66_oldbase_rich_p030_m018",
    ]
    recommended = []
    for i, name in enumerate(recommended_names, start=1):
        recommended.append(alias(name, f"round66_recommended_{i}_{name.replace('round66_', '')}"))

    summary = {
        "method": "round66_rich_positive_fine",
        "description": "Online feedback says round65 rich p020/m018 improved and n020 hurt. Fine sweep only the positive rich residual around that point.",
        "online_context": {
            "old_best": 1.2971305063123648,
            "round65_p020_m018": 1.2971552723358022,
            "round65_n020_m018": 1.2970572016927342,
        },
        "old_base_dir": OLD_BASE_DIR,
        "new_best_dir": NEW_BEST_DIR,
        "rich_raw": RICH_RAW,
        "records": records,
        "recommended_order": recommended,
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Summary:", OUT_SUMMARY)
    print("Recommended:")
    for path in recommended:
        print(path)


if __name__ == "__main__":
    main()
