# Fifteen-sign GRU with face features

The user requested training a 15-sign model using the extracted face features
alongside hands and pose. The separate model and actual training code are in
[notebook 07](../notebooks/07_15_sign_face_gru.ipynb). The baseline and webcam
default were not changed.

## Input and protocol

The input has **1,741 features per frame**: the existing 254 hand/pose channels,
478×3 facial landmark coordinates, 52 blendshape scores and one face-validity
mask. Facial coordinates are corrected for aspect ratio, centered at the nose
and scaled by each clip's median valid inter-eye distance; missing faces are
zero-filled with the mask off. Geometry is face-local; coarse face location
remains present in the body pose features. Face-normalization choices are
experimental and not proven optimal.

The original raw frame indices select the same up-to-128 frames for face and
hand/pose. Temporal augmentation uses the same indices for every modality;
spatial roll is applied consistently. A preflight checked alignment and shape.
Sign labels and the 219/60/182 official signer-disjoint splits match the
original pilot. The model is a new 2-layer, 256-unit GRU with temporal mean
pooling, the existing 23 hand/trajectory geometry summaries and a 15-output
linear readout. It trains from scratch; after selecting the backbone the readout
is refitted on training clips using L2 logistic regression C=0.1.

Training stopped after epoch 17 under the sustained-gap rule; epoch 17 was
selected by the validation-competitive/small-gap rule. The model has 1,934,184
parameters. Source caches and the previous checkpoint remain unchanged.

## Results

| Model | Validation top-1 | Test top-1 | Test macro F1 |
|---|---:|---:|---:|
| Previous hand/pose trajectory readout | 54/60 (90.00%) | 147/182 (80.77%) | 80.33% |
| New hand/pose/face model | 53/60 (88.33%) | 141/182 (77.47%) | 78.00% |

Both use the same test clips and 15 labels. The new model did not improve overall
in this run. For individual test clips, THANKYOU changed from 5/12 to 8/12,
whereas YOU changed from 11/12 to 8/12; these class samples are small. The
model initialization, input dimension, face normalization and random training
path differ, and the earlier checkpoint had a target-informed selection step.
Thus this is not a strict face-only ablation and the difference does not prove
that facial information caused the change. The validation and test sets had
already been used in earlier project development; test results are not blind.

The saved model was reloaded and reproduced **141/182**, with matching
cross-entropy, on the test data. The new model has not been tested through a
webcam and currently requires a separate face-aware inference adapter.

Artifacts: `models/asl-gru-fifteen-signs-face-v1/model.pt`, `config.json`,
`labels.json`, `history.json` and `report.json`. Model and face-cache data are
local and Git-ignored. No previous model weights were overwritten. The curated
archive predates this experiment and does not include the new model.
