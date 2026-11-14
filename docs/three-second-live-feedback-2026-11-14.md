# Three-second webcam feedback - 2026-11-14

The frozen 15-class trajectory-readout GRU completed 12 automatic captures with a three-second input limit. The user reports intermittent ME recognition, successful correctly performed WE, and YOU confused with PLEASE or YES. They cannot recall the intended sign for individual capture IDs.

These observations are qualitative. No capture is assigned a ground-truth label, and no live accuracy is calculated. The reported approximately 50% ME success is not a measured accuracy or calibrated confidence estimate. Raw predictions are preserved in the session report, with a portable aggregate snapshot in [the feedback record](assets/three-second-live-feedback-2026-11-14.json).

Next check: prompted ME, WE, YOU, repeated twice; each capture ends automatically after three seconds. User confirmation is required before prompted labels are treated as performed labels. All 15 labels compete. The model, preprocessing and weights remain unchanged. Video, landmark coordinates and embeddings are not saved.

Earlier WE gestures from the two nine-trial sessions remain excluded under the existing correction ledger. This feedback does not restore those invalid results.
