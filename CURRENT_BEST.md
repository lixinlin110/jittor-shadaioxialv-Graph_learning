# Current Best

This repository tracks the effective code changes for the Jittor-7 Track 1 dynamic recommendation experiments.

Current best online score recorded during the latest iteration:

```text
1.2971683198949764
```

Best local artifact path:

```text
experiments/final_model/result.zip
experiments/current_best_12971683198949764/result.zip
```

Important note: `experiments/`, `data/`, and model/result archives are not intended to be committed to GitHub. The repository should keep reproducible code and lightweight notes only.

Effective code chain currently worth syncing:

- `lgbm_candidate_ranker.py`
- `make_round65_dataset2_rich_ranker.py`
- `make_round66_rich_positive_fine.py`
- `make_round67_rich_peak_clamp.py`
- `clean_current_workspace.py`
- `make_round68_dataset2_strict_candidate_ranker.py`

Round feedback summary:

- `round65_recommended_1_p020_m018`: `1.2971552723358022`
- `round66_recommended_1_oldbase_rich_p025_m018`: `1.2971683198949764`

The useful online signal so far is a very small positive dataset2 rich residual around alpha `0.025`, margin `0.18`. Negative rich residuals and broad hard-recent retraining did not improve online results.

Round68 local experiment:

- Method: dataset2-only strict candidate LambdaRank.
- Training positives: only historical positives that already appear in same-src public test candidate rows.
- Dataset1: fixed at current best `current_best_12971683198949764`.
- Base dataset2: current best `current_best_12971683198949764`.
- Strict split0-tail queries: `53135`.
- Strict split1 queries: `39706`.
- Validation: strict MRR `0.4000334347231581`, HitRate@10 `0.7392333333333333` on `30000` strict validation queries.

Round68 recommended upload order:

1. `experiments/round68_recommended_1_p010_m180/result.zip`
2. `experiments/round68_recommended_2_p014_m180/result.zip`
3. `experiments/round68_recommended_3_p006_m180/result.zip`
4. `experiments/round68_recommended_4_p018_m180/result.zip`
5. `experiments/round68_recommended_5_p010_m140/result.zip`
6. `experiments/round68_recommended_6_n006_m180/result.zip`
