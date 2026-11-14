# Official pretrained ASL Citizen baseline — 2026-11-14

## Baseline identity and workflow

Use the authors' released **I3D**, independently of the project GRU/ST-GCN:

`camera/video pixels → official 64-frame preprocessing → pretrained I3D → official temporal readout → sign scores`

This is the appearance-based baseline in [Microsoft's ASL Citizen repository](https://github.com/microsoft/ASL-citizen-code), using the [checkpoints_v1 release](https://github.com/microsoft/ASL-citizen-code/releases/tag/checkpoints_v1). It classifies 2,731 signs. The authors also provide a pose-based ST-GCN; I3D was chosen here to obtain a reference independent of our MediaPipe landmarks. No training, model conversion or geometry ablation was performed. The existing GRU geometry path, weights, archives and defaults remain unchanged.

- Source commit: `17f0148b00ef87a5957ec103da8d10bd8eb1fa23`.
- Official checkpoint: `models/asl-citizen-official-i3d-v1/ASL_citizen_I3D_weights.pt`.
- Checkpoint SHA-256: `3538319620331d5ba731e0f9ac79161deeb37a895b7680d215864fc00a14477d`.
- Official release ZIP SHA-256: `42c5349f95595fac79af77a902584510da92290e802903f3c96c4689a4b19a48`.
- Source and weight bytes are pinned in [the install specification](assets/official-citizen-i3d-install-spec-2026-11-14.json). MIT source licensing is retained under the ignored upstream directory.

## Following the official implementation

Use the unmodified upstream `pytorch_i3d.py`, `aslcitizen_dataset.py`, `videotransforms.py` and their [testing procedure](https://github.com/microsoft/ASL-citizen-code/blob/17f0148b00ef87a5957ec103da8d10bd8eb1fa23/I3D/aslcitizen_testing.py). Instantiate the 400-output architecture, replace its head with 2,731 outputs, then strictly load the complete released state dictionary, exactly as the upstream test script does. No randomly initialized head remains.

The label mapping is the sorted unique stripped glosses from the official training CSV. Its CSV hash and all indices are verified. Source loading uses the authors' temporal frame skipping/centering, endpoint append padding and center crop to 224×224. Despite the function's RGB name, the actual source keeps OpenCV BGR order; the adapter preserves it. Pixel values are scaled to [-1,1]. The model runs float32, without AMP, in evaluation/inference mode. The authors' readout interpolates per-frame logits to 64 steps and takes the temporal maximum before softmax.

The upstream test loader randomly chooses a first/last padding frame for short clips. This run seeds NumPy with 0, uses fixed sample order and zero loader workers. The original script shuffles with two workers; order-dependent padding can differ, so this is not a bit-identical reproduction of the authors' entire paper benchmark. These choices are disclosed rather than silently correcting source behavior.

## Matched official-test comparison

Run on the exact **182 official test videos** used in our fifteen-sign benchmark. Their five-sign subset contains **60** videos. No landmark arrays are used by I3D; the saved test index is metadata for matching video identities. Video bytes, upstream files, official weights and existing experiment checkpoints/archives are hashed before/after.

| I3D output task | Correct | Top-1 | Top-5 | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Original 2,731 outputs, evaluated on these 182 videos | 117/182 | 64.29% | 87.91% | Not averaged over thousands of absent classes |
| Same logits restricted to fifteen pilot labels | 165/182 | 90.66% | 100% | 0.8889 |
| Same logits restricted to five pilot labels | 58/60 | 96.67% | 100% | 0.9654 |

Restriction only selects and renormalizes output scores. It does not retrain or replace the 2,731-output head. It is a separate closed-set comparison, not the original full-vocabulary result. Five-label top-five is necessarily 100%.

For reference, the historical full-vocabulary project regularized GRU and ST-GCN predict **97/182 (53.30%)** and **90/182 (49.45%)** correctly on these same 182 videos, with all 2,731 labels competing. These are filtered prior complete-test predictions with their original model hashes verified, not new training.

| Pilot model | Correct | Top-1 | Macro F1 |
| --- | ---: | ---: | ---: |
| Official I3D, five-label view | 58/60 | 96.67% | 0.9654 |
| Project five-sign geometry GRU, preselected epoch 8 | 51/60 | 85.00% | 0.8407 |
| Project five-sign geometry GRU, retrospective epoch 5 | 59/60 | 98.33% | 0.9808 |
| Official I3D, fifteen-label view | 165/182 | 90.66% | 0.8889 |
| Current project fifteen-sign readout C=0.1 | 147/182 | 80.77% | 0.8033 |

This is not an architecture-only ablation: the released I3D learned from the full vocabulary and image pixels, while small project models learned from selected-class landmark sequences. The one-clip difference between five-sign epoch 5 and I3D is insufficient to establish superiority; epoch 5 was also identified using test results retrospectively.

The official fifteen-label view recognizes ME 12/13, YOU 12/12 and WE 12/12, but THANKYOU only **1/12**, usually confusing it with GOOD. The five-label view recognizes THANKYOU 11/12 because GOOD is excluded. This is direct evidence of output-set dependence; it does not establish a dataset mislabel or a sign-language semantic equivalence.

See [raw baseline results](assets/official-citizen-i3d-baseline-2026-11-14.json), [matched comparison](assets/official-citizen-i3d-comparison-2026-11-14.json), and the [comparison chart](assets/official-citizen-i3d-comparison-2026-11-14.png). Full 2,731-way logits are saved under `data/evaluations/official-citizen-i3d-2026-11-14-v1/full-vocabulary-logits.npy` for reproducible reporting. The inference/output phase took approximately 67 seconds locally with batch size 1.

## Confirmed four-trial webcam check

The user confirmed HELLO, THANKYOU, YOU and ME were performed as prompted with normal capture. Each trial was automatic, at most three seconds, with a 30-FPS input limit. The camera is closed and the official checkpoint hash remains unchanged. No video, still, landmark coordinates or embeddings were saved.

| Intended sign | Original full-vocabulary first guess | Fifteen-label first guess |
| --- | --- | --- |
| HELLO | WRITEONWALL | HELLO |
| THANKYOU | JAWDROP | GOOD |
| YOU | LAMP1 | HELLO |
| ME | BLOUSE | ME |

Full vocabulary is 0/4; the pilot view is 2/4. This is a small diagnostic, not a general accuracy estimate or a matched comparison against earlier GRU webcam captures. It demonstrates that transferring the authors' pretrained pixel model to this capture protocol also has difficulties. MediaPipe or our GRU cannot be the sole necessary component behind every observed live failure, but this does not rule out tracking problems in the GRU path. Capture boundaries, framing, posture, viewpoint and training/live distribution remain unresolved hypotheses.

The webcam RAM preprocessor exactly matches the upstream loader/pad/crop on a real public validation video and on synthetic frame-skipping boundaries. It introduces no alternate preprocessing to force a result. HELLO/ME successes in the pilot view depend on the smaller output set and must not be presented as full-vocabulary successes. Expected labels are attached after scoring and confirmed by the user.

## Reproduction

1. `python scripts/setup_official_citizen_i3d.py` installs/verifies pinned official sources, release weights and training-derived label mapping. Existing differing files are preserved with an error rather than overwritten.
2. `python scripts/evaluate_official_citizen_i3d.py --output data/evaluations/NEW_EMPTY_RUN` evaluates the matched dataset metadata through official preprocessing and weights.
3. `python scripts/check_official_citizen_webcam.py --report .cache/NEW_SESSION.json` runs the four guided live trials.
4. `python scripts/report_official_citizen_i3d.py --live SESSION --confirm-all` is only used after explicit confirmation; the report separates full-vocabulary and restricted outputs. Omitting live arguments preserves an already confirmed live record.

The reporting notebook contains the baseline and live outcomes. The author-reported full-test result is a separate published benchmark; this 182-video matched subset does not reproduce the full 32,941-video test evaluation. No retraining, feature redesign, default-recognizer replacement or Git push is part of this step.

## Final verification

All **189 tests passed**. Saved full-vocabulary logits reproduce the recorded ranks; both restricted confusion matrices reconstruct from their per-video predictions. All 567 protected file hashes match after the full suite, including prior experiment weights, archives, cached data, official sources and release weights. The notebook rendered without errors and all 35 artifact-manifest hashes match. See [verification record](assets/official-citizen-i3d-verification-2026-11-14.json).
