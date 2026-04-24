# ZipVoice-CA

Catalan fine-tune of [ZipVoice](https://github.com/k2-fsa/ZipVoice) for zero-shot text-to-speech.

<p align="center">
  <a href="https://huggingface.co/ebellob/ZipVoice-CA">
    <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Model-blue" alt="Hugging Face model">
  </a>
  <a href="https://erikupv.github.io/zipvoice-samples/">
    <img src="https://img.shields.io/badge/🔊%20Audio-Samples-green" alt="Audio samples">
  </a>
  <a href="https://github.com/k2-fsa/ZipVoice">
    <img src="https://img.shields.io/badge/Base%20Model-ZipVoice-orange" alt="Base ZipVoice repository">
  </a>
  <a href="https://huggingface.co/k2-fsa/ZipVoice">
    <img src="https://img.shields.io/badge/🤗%20Base%20Checkpoint-k2--fsa%2FZipVoice-yellow" alt="Base ZipVoice checkpoint">
  </a>
</p>

This repository contains the tools used to fine-tune ZipVoice for Catalan text-to-speech. It includes dataset preparation, fine-tuning, inference, and evaluation scripts for the released ZipVoice-CA checkpoint.

The reported metrics are intended as indicative benchmarks under this repository's evaluation setup, not as definitive state-of-the-art claims.

---

## Repository Structure

```text
.
├── create_dataset.py              # Builds Catalan train/dev/test TSV files
├── egs/zipvoice/run_finetune.sh   # Fine-tuning and inference recipe
├── eval/run_eval.sh               # Evaluation pipeline
├── requirements_zipvoice.txt      # Main ZipVoice environment requirements
├── requirements_whisper.txt       # Whisper environment requirements
├── requirements_eval.txt          # Additional evaluation requirements
└── zipvoice/                      # ZipVoice source code used by the recipe
```

---

## Performance Metrics

Results for the released ZipVoice-CA checkpoint.

| Dataset         | WER (%) ↓ | CER (%) ↓ | SIM-o ↑ | UTMOS ↑ |
| --------------- | --------: | --------: | ------: | ------: |
| Common Voice 17 |     10.96 |      3.00 |    0.68 |    3.17 |
| FestCat         |      7.31 |      2.56 |    0.65 |    3.46 |
| LaFrescat       |      7.61 |      2.56 |    0.67 |    3.54 |

Evaluation uses generated samples from the ZipVoice-CA recipe with `guidance_scale=1.0` and `num_step=25`.

---

## Evaluation Protocol

The evaluation set is built from three Catalan sources:

* held-out Common Voice 17 Catalan samples,
* FestCat prompts,
* LaFrescat prompts.

Metrics:

* **WER / CER**: ASR-based intelligibility metrics.
* **SIM-o**: speaker similarity between prompt and generated speech.
* **UTMOS**: automatic MOS-style naturalness estimate.

---

## Experimental Caveats

Full bitwise reproducibility is not guaranteed. Minor variations may arise from random initialization, sampling, non-deterministic GPU operations, preprocessing choices, filtering decisions, and future changes in dataset processing.

The reported scores are meant to show relative behavior and qualitative usefulness under this setup.

---

## Installation

This project uses three conda environments, each dedicated to a different stage of the pipeline:

* **`ZipVoice`**: training, fine-tuning, inference, and most preprocessing.
* **`whisper-env`**: Whisper-based transcription and WER/CER evaluation.
* **`utmos-env`**: UTMOS-based MOS evaluation.

Make sure Anaconda or Miniconda is installed before continuing.

### 1. Main Environment: `ZipVoice`

```bash
conda create -n ZipVoice python=3.11
conda activate ZipVoice
pip install -r requirements_zipvoice.txt
```

### 2. Whisper Environment: `whisper-env`

```bash
conda create -n whisper-env python=3.11
conda activate whisper-env
pip install -r requirements_whisper.txt
```

### 3. UTMOS Environment: `utmos-env`

```bash
conda create -n utmos-env python=3.9
conda activate utmos-env

git clone https://huggingface.co/spaces/sarulab-speech/UTMOS-demo
cd UTMOS-demo
pip install -r requirements.txt
```

---

## Data Preparation

Before fine-tuning or evaluation, generate the training, validation, and test TSV files with `create_dataset.py`.

Activate the main environment:

```bash
conda activate ZipVoice
```

### 1. Generate Training and Development Data

This creates:

* `custom_train.tsv`
* `custom_dev.tsv`

using a deterministic split of the Common Voice 17 Catalan dataset.

```bash
python create_dataset.py \
  --mode train \
  --out_dir egs/zipvoice/data_cat/raw/
```

### 2. Generate Test Data

This creates:

* `test.tsv`

using held-out Common Voice 17 samples, FestCat prompts, and LaFrescat prompts.

```bash
python create_dataset.py \
  --mode test \
  --out_dir egs/zipvoice/data_cat/raw/
```

The script stores the Common Voice split metadata under `egs/zipvoice/data_cat/raw/cv17_splits/` so train/dev/test prompt pools remain consistent across runs.

---

## Fine-Tuning

Once `custom_train.tsv` and `custom_dev.tsv` are ready, launch the ZipVoice fine-tuning recipe:

```bash
conda activate ZipVoice
cd egs/zipvoice

chmod +x run_finetune.sh
./run_finetune.sh
```

The script prepares manifests, computes features, downloads the base ZipVoice checkpoint, fine-tunes the model, averages checkpoints, and generates samples.

By default, generated samples are written under:

```text
egs/zipvoice/results/<experiment_name>/
```

---

## Inference Configuration

The provided recipe uses the following inference settings:

```bash
--guidance-scale 1.0
--num-step 25
```

These settings are kept in `egs/zipvoice/run_finetune.sh`.

---

## Evaluation

After fine-tuning and sample generation, run the evaluation pipeline from the repository root:

```bash
export ZIPVOICE_CA_ROOT=$(pwd)
export UTMOS_DIR=/path/to/UTMOS-demo

./eval/run_eval.sh finetune_eden_5e-4
```

Replace `finetune_eden_5e-4` with the experiment directory you want to evaluate.

The evaluation script computes speaker similarity, WER/CER, and UTMOS scores for the generated samples.

---

## Quick Smoke Test

For a small setup check, build reduced splits before launching a full run:

```bash
conda activate ZipVoice

python create_dataset.py \
  --mode train \
  --out_dir egs/zipvoice/data_cat/raw/ \
  --max_cv17_items 100 \
  --n_dev 10 \
  --n_test 10

python create_dataset.py \
  --mode test \
  --out_dir egs/zipvoice/data_cat/raw/ \
  --max_cv17_items 100 \
  --n_dev 10 \
  --n_test 10 \
  --max_items 5
```

Then inspect the generated files:

```bash
head egs/zipvoice/data_cat/raw/custom_train.tsv
head egs/zipvoice/data_cat/raw/custom_dev.tsv
head egs/zipvoice/data_cat/raw/test.tsv
```
---

## Citation and Acknowledgements

This work is based on [ZipVoice](https://github.com/k2-fsa/ZipVoice) and the pretrained checkpoint released by `k2-fsa`.

If you use this repository or the released model, please cite the original ZipVoice work and acknowledge this Catalan fine-tuning repository.
