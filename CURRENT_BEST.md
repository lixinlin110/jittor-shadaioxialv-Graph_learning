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

Round feedback summary:

- `round65_recommended_1_p020_m018`: `1.2971552723358022`
- `round66_recommended_1_oldbase_rich_p025_m018`: `1.2971683198949764`

The useful signal so far is a very small positive dataset2 rich residual around alpha `0.025`, margin `0.18`. Negative rich residuals and broad hard-recent retraining did not improve online results.
