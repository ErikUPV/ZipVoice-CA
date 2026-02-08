# ZipVoice-CA

**The model is now publicly available at [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/ebellob/ZipVoice-CA)**

This repository contains the tools and instructions necessary to **fine-tune the ZipVoice model for Catalan Text-to-Speech (TTS)**. It includes environment setup, data preparation, fine-tuning, and evaluation pipelines. Furthermore, in the table below, we can see the models' results vs [Matxa-TTS](https://huggingface.co/projecte-aina/matxa-tts-cat-multiaccent) in our experimental setup.
They are intended as **indicative benchmarks**, not as claims of state-of-the-art performance.

You can listen to some samples by clicking this [link](https://erikupv.github.io/zipvoice-samples/)


## Common Voice 17

| Modelo              | Scheduler  | LR   | WER (%) ↓ | CER (%) ↓ | SIM-o ↑  | UTMOS ↑  |
| ------------------- | ---------- | ---- | --------- | --------- | -------- | -------- |
| ZipVoice            | ConstantLR | 1e-4 | 10.96     | 3.00      | 0.68     | **3.17** |
| ZipVoice            | Eden       | 1e-4 | 11.85     | 3.32      | 0.67     | **3.17** |
| ZipVoice            | Eden       | 5e-4 | **10.55** | **2.85**  | 0.68     | 3.06     |
| Matxa-TTS           | –          | –    | 15.84     | 5.21      | –        | 2.94     |

## Festcat

| Modelo              | Scheduler  | LR   | WER (%) ↓ | CER (%) ↓ | SIM-o ↑  | UTMOS ↑  |
| ------------------- | ---------- | ---- | --------- | --------- | -------- | -------- |
| ZipVoice            | ConstantLR | 1e-4 | 7.31      | 2.56      | 0.65     | **3.46** |
| ZipVoice            | Eden       | 1e-4 | 8.25      | 2.86      | **0.67** | 3.44     |
| ZipVoice            | Eden       | 5e-4 | **6.55**  | **2.30**  | 0.63     | 3.42     |
| Matxa-TTS           | –          | –    | 8.34      | 2.96      | –        | 3.17     |

## LaFrescat

| Modelo              | Scheduler  | LR   | WER (%) ↓ | CER (%) ↓ | SIM-o ↑  | UTMOS ↑  |
| ------------------- | ---------- | ---- | --------- | --------- | -------- | -------- |
| ZipVoice            | ConstantLR | 1e-4 | 7.61      | 2.56      | **0.67** | **3.54** |
| ZipVoice            | Eden       | 1e-4 | 8.93      | 2.70      | 0.65     | 3.53     |
| ZipVoice            | Eden       | 5e-4 | **7.20**  | **2.00**  | **0.67** | 3.47     |
| Matxa-TTS           | –          | –    | 8.18      | 2.65      | –        | 3.24     |


## Experimental Caveats

- While scripts and configurations are provided, **full bitwise reproducibility is not guaranteed**.
- Minor variations may arise from:
  - random initialization and sampling,
  - non-deterministic GPU operations,
  - preprocessing and data filtering choices.
  - unforeseen changes in the processing of the data, such the splits themselves not being shuffled in the same way
- Results are reported to provide **relative trends and qualitative insights**, not definitive rankings.

---

## Installation (Environment Setup)

This project requires **three distinct conda environments**, each dedicated to a specific stage of the pipeline:

* **`ZipVoice`** — main training and fine-tuning
* **`whisper-env`** — Whisper-based transcription during data preparation
* **`utmos-env`** — UTMOS-based MOS evaluation

Make sure you have **Anaconda or Miniconda** installed before proceeding.

---

### 1. Main Environment: `ZipVoice`

This is the default environment used for core ZipVoice training and fine-tuning.

```bash
# Create the environment
conda create -n ZipVoice python=3.11
conda activate ZipVoice

# Install dependencies
pip install -r requirements_zipvoice.txt
```

---

### 2. Whisper Environment: `whisper-env`

This environment is used to process audio using OpenAI's Whisper model during data preparation.

```bash
# Create the environment
conda create -n whisper-env python=3.11
conda activate whisper-env

# Install dependencies
pip install -r requirements_whisper.txt
```

---

### 3. UTMOS Environment: `utmos-env`

This environment is required **only for evaluation**, to run UTMOS (MOS prediction). Follow these steps **exactly**:

```bash
# Create the environment
conda create -n utmos-env python=3.9
conda activate utmos-env
```

Then git clone the Huggingface repository:
```bash
git clone https://huggingface.co/spaces/sarulab-speech/UTMOS-demo
cd UTMOS-demo
pip install -r requirements.txt
```

---

## Data Preparation

Before fine-tuning, you must generate the **training**, **validation**, and **test** datasets using the `create_dataset.py` script.

Activate the main environment:

```bash
conda activate ZipVoice
```

The script must be run in **two modes**.

---

### 1. Generate Training and Development Data

This step creates:

* `custom_train.tsv`
* `custom_dev.tsv`

by splitting the **Common Voice 17 (Catalan)** dataset.

```bash
python create_dataset.py --mode train --out_dir egs/zipvoice/data_cat/raw/
```

---

### 2. Generate Test Data

This step creates:

* `test.tsv`

using:

* held-out Common Voice 17 test data
* FestCat prompts
* LaFrescat prompts

```bash
python create_dataset.py --mode test --out_dir egs/zipvoice/data_cat/raw/
```

---

## Fine-Tuning

Once the dataset files (`custom_train.tsv` and `custom_dev.tsv`) are ready, you can start fine-tuning the model.

```bash
conda activate ZipVoice
cd egs/zipvoice

# Run fine-tuning
./run_finetune.sh
```

This script launches the training loop and generates audio samples during and after training.
In case you can't run it directly:
```bash
chmod +x run_finetune.sh
```

---

## Evaluation

After fine-tuning completes and samples are generated, run the evaluation pipeline. This step requires:

* `test.tsv` (generated during data preparation)

From the `egs/zipvoice` directory:

```bash
# Run evaluation
$ZIPVOICE_CA_ROOT/eval/run_eval.sh
```

This will compute objective metrics and UTMOS-based MOS predictions for the generated samples.

---

Happy synthesizing! 🎙️
![Demo](https://media1.tenor.com/m/WvTbUOzmb8kAAAAd/happy-synthesizer-gumi.gif)
