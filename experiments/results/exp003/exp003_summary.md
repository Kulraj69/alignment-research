# Experiment 003 Results

## Setup
- Source artifact: experiments/results/exp001/exp001_results.json
- Samples: 100 activations from 50 prompt pairs
- Combined probe layers: [4, 5, 6, 7]
- Seed sweep: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

## Representation Geometry

| Rank | Layer | Silhouette (PCA-2) | Fisher ratio | PCA-2 variance |
|---:|---:|---:|---:|---:|
| 1 | 7 | 0.3002 | 0.1514 | 20.9% |
| 2 | 8 | 0.2915 | 0.1505 | 21.7% |
| 3 | 10 | 0.2911 | 0.1630 | 22.7% |
| 4 | 5 | 0.2911 | 0.1405 | 20.2% |
| 5 | 6 | 0.2881 | 0.1408 | 20.4% |

## Control-Subset Probe Robustness

| Subset | Pairs | Accuracy mean +/- std | AUC mean +/- std |
|---|---:|---:|---:|
| primary_all | 42 | 92.7% +/- 4.3% | 97.9% +/- 1.8% |
| standard_monitoring | 30 | 100.0% +/- 0.0% | 100.0% +/- 0.0% |
| lexical_matched | 12 | 72.5% +/- 15.2% | 88.3% +/- 10.8% |

## Cross-Control Transfer

| Train -> Test | Accuracy | AUC | P(watched) gap |
|---|---:|---:|---:|
| standard_to_lexical | 54.2% | 59.7% | 0.0587 |
| lexical_to_standard | 73.3% | 87.7% | 0.4791 |

## Style-Only Negative Control
- Normal-side mean P(watched): 0.0001
- Style-variant mean P(watched): 0.0029
- Style-control probability gap: 0.0028
- Style-variant predicted-watched rate: 0.0%

## Patching Matrix

| Layer | Watched-to-normal shift | Normal-to-watched shift | Directional asymmetry |
|---:|---:|---:|---:|
| 0 | 0.2311 | 0.1834 | 0.0477 |
| 3 | 0.3534 | 0.2552 | 0.0983 |
| 6 | 0.4358 | 0.4374 | -0.0016 |
| 9 | 0.6301 | 0.6556 | -0.0256 |
| 11 | 0.6747 | 0.6590 | 0.0157 |

## Length-Confound Audit
- Overall mean char gap: 32.08
- Char gap vs max activation distance: 0.9159
- Word gap vs max activation distance: 0.9048
- Char gap vs logit diff: 0.3375

### By Control Type

| Control | Pairs | Mean char gap | Char gap vs activation | Char gap vs logit diff |
|---|---:|---:|---:|---:|
| lexical_matched | 12 | 1.25 | 0.8144 | 0.5667 |
| standard_monitoring | 30 | 48.90 | 0.3551 | 0.0824 |
| style_only | 8 | 15.25 | 0.6406 | 0.1647 |

## Interpretation
- Primary pair-grouped separability remains strong under a 20-seed stress audit.
- Lexical-matched controls remain the pressure point; expand this subset before making strong claims.
- Style-only variants do not strongly trigger the watched detector, which is the desired negative-control pattern.
- Prompt length remains a plausible confound; add stricter length-matched pairs.
