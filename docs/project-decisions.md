# Project decisions

Agreed on 2026-10-25:

- ASL first; RSL remains a secondary comparison accepted by the supervisor.
- WLASL is the initial isolated-sign dataset.
- Start with 10 everyday signs, expand to 25 after a reliable end-to-end demo.
- Desired expressions include hello, thank you, goodbye, you're welcome, and
  see you. Dataset labels and ASL expressions must be checked individually.
- The provisional config uses hello, thank you, goodbye, please, sorry, yes,
  no, help, fine, and good. Review usable clips before freezing this list.
- Avoid separate bye/goodbye classes until overlap has been reviewed. Do not
  equate WLASL's welcome label with you're welcome without checking the sign.
- Local hardware is an RTX 4060 with 8 GB VRAM.
- Ordinary Python files form the main pipeline. Jupyter is optional.
- Prioritize implementation and the university report; publication is not required.
- Use a small landmark-sequence model first; continuous translation is later work.

Repository setup uses uv with a committed lockfile rather than parallel manual
requirements files. Development, full CPU, and full CUDA profiles share one
package and configuration. Uv is installed once as a standalone user tool; the
project uses a single `.venv`, including pip for familiar package inspection.
No dataset, trained model, or live recognizer exists
at this foundation milestone.

Next milestone: audit available WLASL clips for the draft vocabulary, establish
signer-disjoint manifests, and implement consistent training/webcam extraction.
