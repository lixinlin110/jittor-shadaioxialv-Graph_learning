import json
import os
import os.path as osp
import shutil

import numpy as np

from lgbm_candidate_ranker import apply_residual, pack_submission, read_scores, write_scores


BASE_DIR = "./experiments/current_best_12971305063123648"
RICH_RAW = "./experiments/round65_d2_rich_ranker/dataset2/dataset2_rich_raw.npy"
OUT_SUMMARY = "./experiments/round67_rich_peak_clamp_summary.json"


def tag_alpha(alpha):
    milli = alpha * 1000
    if abs(milli - round(milli)) < 1e-9:
        return f"p{int(round(milli)):03d}"
    return f"p{int(round(alpha * 10000)):04d}"


def tag_margin(margin):
    return f"m{int(round(margin * 1000)):03d}"


def copy_d1(run_root):
    src = osp.join(BASE_DIR, "dataset1", "dataset1_result.csv")
    dst = osp.join(run_root, "dataset1", "dataset1_result.csv")
    if not osp.exists(src):
        raise FileNotFoundError(src)
    os.makedirs(osp.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def write_variant(name, base_d2, raw, alpha, margin):
    run_root = osp.join("./experiments", name)
    copy_d1(run_root)
    d2_scores, modified = apply_residual(base_d2, raw, alpha, margin)
    write_scores(d2_scores, osp.join(run_root, "dataset2", "dataset2_result.csv"))
    zip_path = pack_submission(run_root)
    record = {
        "name": name,
        "zip": zip_path,
        "dataset2_alpha": float(alpha),
        "dataset2_margin": float(margin),
        "dataset2_modified_rows": int(modified),
        "dataset2_modified_ratio": float(modified / len(base_d2)),
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
    base_d2 = read_scores(osp.join(BASE_DIR, "dataset2", "dataset2_result.csv"))
    grid = [
        (0.026, 0.180),
        (0.027, 0.180),
        (0.028, 0.180),
        (0.029, 0.180),
        (0.0275, 0.180),
        (0.025, 0.175),
        (0.025, 0.185),
        (0.027, 0.175),
        (0.027, 0.185),
    ]
    records = []
    for alpha, margin in grid:
        name = f"round67_rich_{tag_alpha(alpha)}_{tag_margin(margin)}"
        records.append(write_variant(name, base_d2, raw, alpha, margin))

    recommended_names = [
        "round67_rich_p0275_m180",
        "round67_rich_p027_m180",
        "round67_rich_p028_m180",
        "round67_rich_p026_m180",
        "round67_rich_p027_m175",
        "round67_rich_p027_m185",
    ]
    recommended = []
    for idx, name in enumerate(recommended_names, start=1):
        recommended.append(alias(name, f"round67_recommended_{idx}_{name.replace('round67_', '')}"))

    summary = {
        "method": "round67_rich_peak_clamp",
        "description": "Clamp around the online-proven rich residual peak. Round66 says p025/m018 and p030/m018 are almost tied; p015 and margin changes are worse.",
        "base_dir": BASE_DIR,
        "rich_raw": RICH_RAW,
        "online_context": {
            "round65_p020_m018": 1.2971552723358022,
            "round66_p025_m018": 1.2971683198949764,
            "round66_p030_m018": 1.2971678208356767,
            "round66_p015_m018": 1.2971133139227897,
            "round66_p020_m016": 1.2971546075193097,
            "round66_p020_m020": 1.2971575697992943,
        },
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
