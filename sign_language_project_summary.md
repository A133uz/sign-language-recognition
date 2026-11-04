# Real-Time Sign Language Recognition Using Computer Vision and Deep Learning

Project summary: scope, architecture for the minor and major parts, resources, and paper structure.

**Current decisions (updated 2026-11-01):** ASL first using WLASL, aiming for all
available WLASL glosses (WLASL2000). Use 10 everyday signs to test the initial
pipeline, then expand the vocabulary. Coverage and accuracy need evaluation.
The supervisor accepts ASL-primary/RSL-secondary scope.
Local training hardware is an RTX 4060 with 8 GB VRAM. Publication is optional;
the application and university report take priority. Use ordinary Python files.
The older vocabulary and resource estimates below are longer-term planning notes.
See [README.md](README.md) for repository setup and current implementation status,
and [docs/project-decisions.md](docs/project-decisions.md) for the decision record.

---

## 1. Project overview

**Goal:** recognize signs shown to a laptop camera, convert them into words, and (in the major part) combine them into sentences and paragraphs.

**Submitted synopsis (summary):** a system that recognizes signs from video and converts them into words or phrases, focused on ASL and RSL. Hand and body features are extracted with computer vision and classified with deep learning. The minor project covers individual signs displayed as text. The outcome is a software application.

**Split of work**

| Part | Scope | Output |
|---|---|---|
| Minor project (semester 1) | Isolated sign recognition from a webcam, shown as text | Working real-time application and paper 1 |
| Major project (semester 2) | Continuous signing converted to sentences and paragraphs | Extended application and paper 2 |

---

## 2. Language choice

- **ASL is the main language.** It has the most data and published baselines (WLASL, MS-ASL, How2Sign, the Google Kaggle ISLR data).
- **RSL is a secondary experiment.** Slovo covers isolated RSL signs only. Public continuous RSL data is scarce, so sentence-level RSL work is not realistic for a student project.
- **Why this matters:** the synopsis says "ASL and RSL" as the primary focus. Doing both fully doubles the work, so agree with the supervisor that ASL is the full pipeline and RSL is a comparison run on a subset of Slovo.

---

## 3. Architecture: full pipeline

```
MINOR PROJECT (isolated signs)
+---------+    +-----------+    +------------+    +-----------------+
| Webcam  | -> | MediaPipe | -> | Classifier | -> | Live output     |
| frames  |    | landmarks |    | GRU / Trf  |    | word+confidence |
+---------+    +-----------+    +------------+    +-----------------+
                                                          |
                                                          v
MAJOR PROJECT (sentences)
+--------------+    +----------------+    +---------------------+
| Segmentation | -> | Text generator | -> | Final output        |
| sign limits  |    | glosses->text  |    | sentences/paragraph |
+--------------+    +----------------+    +---------------------+
```

The minor project should output a clean list of glosses (word, confidence, timestamp). The major project reads exactly that list, so the two halves connect.

---

## 4. Minor project architecture (isolated signs)

### 4.1 Pipeline

1. **Capture:** OpenCV reads webcam frames at roughly 20 to 30 fps.
2. **Landmark extraction:** MediaPipe (Holistic or Hands plus Pose) converts each frame to keypoints. Two hands give about 126 values per frame. Pose points can be added.
3. **Preprocessing:** normalize landmarks relative to the shoulders or wrist and scale them, fix the sequence length (for example 30 to 60 frames), and handle frames where a hand is missing.
4. **Classifier:** a sequence model on landmark windows.
   - Baseline: LSTM or GRU (2 layers, around 128 hidden units, dropout).
   - Comparison model: a small Transformer encoder.
5. **Real-time inference:** sliding window, a confidence threshold, and a rule that suppresses repeated predictions of the same sign.
6. **Interface:** a desktop application showing the live camera, the current prediction, and the recognized word history.

### 4.2 Why landmarks instead of raw video

- Training is far lighter (minutes to hours, not days).
- Runs in real time on a laptop CPU.
- Less sensitive to background, clothing, and lighting.
- Matches the format of the Google Kaggle ISLR data.

### 4.3 Realistic target

- 50 to 100 signs from WLASL or the Kaggle ISLR data (ASL).
- A matching subset from Slovo for the RSL experiment.
- Evaluation on signers who are not in the training data.

---

## 5. Major project architecture (continuous signing to sentences)

### 5.1 Pipeline

1. **Segmentation / continuous recognition:** split a continuous stream into individual signs, or recognize glosses directly. Options:
   - A "no-sign / rest" class with a sliding window (simplest, builds on the minor project).
   - Motion-based boundary detection.
   - A CTC-based continuous recognition model trained on a continuous dataset.
2. **Text generation:** turn the gloss sequence into fluent English. ASL grammar differs from English, so word-by-word joining gives poor sentences. Options:
   - A small seq2seq model (T5-type) trained on gloss-to-text pairs.
   - An LLM with prompting.
   - A small LLM fine-tuned with LoRA.
3. **Context for paragraphs:** pass earlier sentences into the generator so it can keep the text coherent.
4. **Output:** text with optional text-to-speech.

### 5.2 Research direction to follow

Recent sentence-level work includes gloss-free translation such as GFSLT-VLP (ICCV 2023), evaluated on German and Chinese sign language benchmarks. It is a good reference for the paper's related work, but reproducing it on ASL is a larger undertaking than the glosses-plus-text-generator route above.

### 5.3 Realistic target

Sentences from a limited vocabulary. Full paragraph translation of open-domain signing is still an open research problem and should be described as future work.

---

## 6. Datasets and existing work (verified links)

### ASL

| Resource | Notes | Link |
|---|---|---|
| WLASL | Word-level, about 2000 glosses | https://github.com/dxli94/WLASL |
| MS-ASL | 1000 classes, isolated signs | see MS-ASL paper |
| How2Sign | 80+ hours continuous ASL with English transcripts | http://how2sign.github.io/ |
| Google ISLR (Kaggle) | Isolated signs as MediaPipe landmarks, 250 words | https://www.kaggle.com/competitions/asl-signs |
| Google Fingerspelling (Kaggle) | Fingerspelling recognition | https://www.kaggle.com/competitions/asl-fingerspelling |

### RSL

| Resource | Notes | Link |
|---|---|---|
| Slovo | 20,000 videos, 1,000 isolated RSL classes, 194 signers | https://arxiv.org/abs/2305.14527 |
| Slovo code | Official repository | https://github.com/hukenovs/slovo |
| Sber RSL report | RSL dataset and fine-tuning report | https://arxiv.org/abs/2302.07693 |

### Open-source starting points (student/hobby projects, not verified by running)

- https://github.com/SiD-array/SignLens-RealTime-ASL-Recognition
- https://github.com/Rahul28428/Sign-Language-Detection
- https://github.com/man0405/Sign-Language-Detection
- https://github.com/matt-user/sign_language_detection
- https://github.com/JosephZahar/Google-Isolated-Sign-Language-Recognition-Kaggle

### Sentence-level research

- GFSLT-VLP paper: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhou_Gloss-Free_Sign_Language_Translation_Improving_from_Visual-Language_Pretraining_ICCV_2023_paper.pdf
- GFSLT-VLP code: https://github.com/zhoubenjia/GFSLT-VLP

Note: accuracy figures in small GitHub demos usually come from 3 to 10 signs and say little about real performance.

---

## 7. Computing resources (estimates)

| Stage | Needs |
|---|---|
| Live demo | Any recent laptop with webcam and about 8 GB RAM. CPU is enough. |
| Minor: landmark extraction | CPU, can take hours for many videos. Run once and save the arrays. |
| Minor: training | Free Colab or Kaggle GPU (T4 class). Training takes minutes to a couple of hours. |
| Minor: storage | Some GB for landmarks. Check that WLASL video links still work, since some are reported dead. The Kaggle data is already landmarks. |
| Major: continuous recognition | GPU with 16 to 24 GB (RTX 3090/4090 or cloud A100) is preferable. How2Sign is large, so using pose landmarks instead of raw video reduces disk and compute needs sharply. |
| Major: text generation | Hosted LLM prompting needs no GPU. LoRA fine-tuning of a small model fits on one 16 to 24 GB GPU. |

These are rough estimates. Confirm what GPU access the university provides.

---

## 8. Paper structure

### 8.1 Paper 1: minor project (isolated signs)

1. **Abstract:** problem, approach, datasets, main result, real-time performance.
2. **Introduction:** why sign recognition matters; ASL and RSL are separate languages; objectives and contributions.
3. **Related work:** datasets (WLASL, MS-ASL, How2Sign, Slovo, Kaggle ISLR); landmark-based vs video-based methods; from isolated recognition to sentence translation.
4. **Datasets:** subset used, number of signs, samples and signers, train/validation/test split, license notes.
5. **Methodology:** capture, landmark extraction, normalization, model architecture, training setup, real-time inference logic.
6. **Experiments and results:** see section 9.
7. **Discussion and limitations:** error analysis, confusable signs, lighting and signer variation.
8. **Ethics note:** the tool does not replace interpreters; Deaf community perspective; feedback from signers if possible.
9. **Conclusion and future work:** sentence-level extension (leads to the major project).
10. **References.**

### 8.2 Paper 2: major project (sentences)

1. **Abstract.**
2. **Introduction:** isolated vs continuous signing, why grammar matters, contributions.
3. **Related work:** continuous recognition, gloss-based vs gloss-free translation, How2Sign, GFSLT-VLP.
4. **Data:** continuous dataset or constructed gloss-text pairs, preprocessing.
5. **System architecture:** the minor-project recognizer, segmentation method, text generator, context handling.
6. **Experiments:** recognition error rate (WER), translation quality (BLEU or similar), ablations (segmentation methods, generator options), latency of the full pipeline.
7. **Human evaluation:** readability and meaning preservation judged by people, ideally including signers.
8. **Discussion and limitations.**
9. **Conclusion and future work.**
10. **References.**

---

## 9. Evaluation checklist

- **Signer-independent split.** Test on signers not present in training. A random split from the same signers inflates accuracy.
- **Top-1 and top-5 accuracy**, per-class results, and a confusion matrix.
- **Model comparison:** GRU/LSTM vs small Transformer; with and without normalization; hands only vs hands plus pose.
- **Real-time numbers:** latency per prediction and FPS on a named laptop.
- **Live demo test** with new people under varied lighting and backgrounds. Report the drop honestly.
- **ASL vs RSL comparison** on the same pipeline.
- **Always state vocabulary size** next to every accuracy number.

---

## 10. Wording and scope cautions

- Use "recognition" for the minor project. Use "translation" only for the sentence stage.
- Do not oversell results on very small vocabularies.
- Cite exact dataset versions or subsets and check license terms.
- Agree the ASL-primary, RSL-secondary scope with the supervisor before the first review.

---

## 11. Suggested working habits

- Keep an experiment log from day one (model, settings, split, result).
- Write related work and datasets sections early.
- Save figures as you go: pipeline diagram, confusion matrix, training curves, demo screenshots.
- Ask the supervisor for the university's report template and page expectations.
- Decide early whether the paper is a university report only or also a conference/journal submission. A publishable paper needs a clearer contribution than "we built a system".

---

## 12. Open questions to settle

1. Which 10 signs have sufficient available WLASL clips and signer coverage?
2. Can a signer-disjoint split retain useful coverage for each selected class?
3. What landmark feature schema and rejection strategy work best for the live demo?
4. Is additional university compute needed after measuring local training requirements?
