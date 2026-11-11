# Current five-sign class distribution — 2026-11-12

Verified against the actual cached target arrays and index rows. Counts are video clips, not frames or augmented variants.

| Sign | Train | Validation | Test | Total | Training share |
|---|---:|---:|---:|---:|---:|
| HELLO | 14 | 4 | 14 | 32 | 18.7% |
| NO | 16 | 3 | 11 | 30 | 21.3% |
| PLEASE | 14 | 4 | 13 | 31 | 18.7% |
| THANKYOU | 16 | 3 | 12 | 31 | 21.3% |
| YES | 15 | 5 | 10 | 30 | 20.0% |
| **Total** | **75** | **19** | **60** | **154** | **100%** |

The official signer-disjoint split is preserved: 24 training, 4 validation and 11 test signers. Split shares are 48.7%, 12.3% and 39.0%. This is not a random 80/20 split.

Training counts are nearly balanced (largest/smallest = 16/14 = 1.14). Strong clip-count imbalance does not explain the NO/YES failures on its own. The much clearer limitation is only 14–16 source training clips per class; balance does not guarantee tracking quality or coverage of signers, viewpoints and signing variations. Online augmentation does not add new source clips.

The five-sign GRU, mean-pooling GRU and ST-GCN experiments use this same subset. No data, splits, model weights or training settings were changed. No inference or new test-set scoring occurred.

[Graph](assets/five-sign-class-distribution-2026-11-12.png) · [Verified counts and signer details](assets/five-sign-class-distribution-2026-11-12.json)

Reproduce: `.\.venv\Scripts\python.exe scripts/report_five_sign_distribution.py`
