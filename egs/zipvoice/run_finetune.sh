#!/bin/bash

# This script is an example of fine-tuning ZipVoice on your custom datasets.

# Add project root to PYTHONPATH
export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=6
stop_stage=7

# Number of jobs for data preparation
nj=8

# Whether the language of training data is one of Chinese and English
is_zh_en=0

# Language identifier, used when language is not Chinese or English
# see https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md
# Example of French: lang=fr
lang=ca

if [ $is_zh_en -eq 1 ]; then
      tokenizer=emilia
else
      tokenizer=espeak
      [ "$lang" = "default" ] && { echo "Error: lang is not set!" >&2; exit 1; }
fi

# You can set `max_len` according to statistics from the command 
# `lhotse cut describe data/fbank/custom_cuts_train.jsonl.gz`.
# Set `max_len` to 99% duration.

# Maximum length (seconds) of the training utterance, will filter out longer utterances
max_len=25

# Download directory for pre-trained models
download_dir=download/

# We suppose you have two TSV files: "data/raw/custom_train.tsv" and 
# "data/raw/custom_dev.tsv", where "custom" is your dataset name, 
# "train"/"dev" are used for training and validation respectively.

# Each line of the TSV files should be in one of the following formats:
# (1) `{uniq_id}\t{text}\t{wav_path}` if the text corresponds to the full wav,
# (2) `{uniq_id}\t{text}\t{wav_path}\t{start_time}\t{end_time}` if text corresponds
#     to part of the wav. The start_time and end_time specify the start and end
#     times of the text within the wav, which should be in seconds.
# > Note: {uniq_id} must be unique for each line.
for subset in train dev;do
      file_path=data_cat/raw/custom_${subset}.tsv
      [ -f "$file_path" ] || { echo "Error: expect $file_path !" >&2; exit 1; }
done

### Prepare the training data (1 - 4)

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
      echo "Stage 1: Prepare manifests for custom dataset from tsv files"

      for subset in train dev;do
            python3 -m zipvoice.bin.prepare_dataset \
                  --tsv-path data_cat/raw/custom_${subset}.tsv \
                  --prefix custom-finetune \
                  --subset raw_${subset} \
                  --num-jobs ${nj} \
                  --output-dir data_cat/manifests
      done
      # The output manifest files are "data/manifests/custom-finetune_cuts_raw_train.jsonl.gz".
      # and "data/manifests/custom-finetune_cuts_raw_dev.jsonl.gz".
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
      echo "Stage 2: Add tokens to manifests"
      # For "emilia" and "espeak" tokenizers, it's better to prepare the tokens 
      # before training. Otherwise, the on-the-fly tokenization can significantly
      # slow down the training.
      for subset in train dev;do
            python3 -m zipvoice.bin.prepare_tokens \
                  --input-file data_cat/manifests/custom-finetune_cuts_raw_${subset}.jsonl.gz \
                  --output-file data_cat/manifests/custom-finetune_cuts_${subset}.jsonl.gz \
                  --tokenizer ${tokenizer} \
                  --lang ${lang}
      done
      # The output manifest files are "data/manifests/custom-finetune_cuts_train.jsonl.gz".
      # and "data/manifests/custom-finetune_cuts_dev.jsonl.gz".
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
      echo "Stage 3: Compute Fbank for custom dataset"
      # You can skip this step and use `--on-the-fly-feats 1` in training stage
      for subset in train dev; do
            python3 -m zipvoice.bin.compute_fbank \
                  --source-dir data_cat/manifests \
                  --dest-dir data_cat/fbank \
                  --dataset custom-finetune \
                  --subset ${subset} \
                  --num-jobs ${nj}
      done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
      echo "Stage 4: Download pre-trained model, tokens file, and model config"
      # Uncomment this line to use HF mirror
      # export HF_ENDPOINT=https://hf-mirror.com
      hf_repo=k2-fsa/ZipVoice
      mkdir -p ${download_dir}
      for file in model.pt tokens.txt model.json; do
            huggingface-cli download \
                  --local-dir ${download_dir} \
                  ${hf_repo} \
                  zipvoice/${file}
      done
fi

### Training ZipVoice (5 - 6)

for EXP_DIR in finetune_eden_1e-4 finetune_eden_5e-4 finetune_constant_1e-4; do
      mkdir -p exp_cat/${EXP_DIR}
      mkdir -p results/${EXP_DIR}

      if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
            echo "Stage 5: Fine-tune the ZipVoice model"

            [ -z "$max_len" ] && { echo "Error: max_len is not set!" >&2; exit 1; }

            python3 -m zipvoice.bin.train_zipvoice \
                  --world-size 1 \
                  --use-fp16 1 \
                  --finetune 0 \
                  --base-lr 0.0005 \
                  --lr-epochs 5 \
                  --num-iters 50000 \
                  --num-iters 30000 \
                  --save-every-n 2000 \
                  --max-duration 350 \
                  --max-len ${max_len} \
                  --model-config ${download_dir}/zipvoice/model.json \
                  --checkpoint ${download_dir}/zipvoice/model.pt \
                  --tokenizer ${tokenizer} \
                  --lang ${lang} \
                  --token-file ${download_dir}/zipvoice/tokens.txt \
                  --dataset custom \
                  --train-manifest data_cat/fbank/custom-finetune_cuts_train.jsonl.gz \
                  --dev-manifest data_cat/fbank/custom-finetune_cuts_dev.jsonl.gz \
                  --exp-dir ${EXP_DIR}

      fi

      if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
            echo "Stage 6: Average the checkpoints for ZipVoice"
            python3 -m zipvoice.bin.generate_averaged_model \
                  --iter 30000 \
                  --avg 2 \
                  --model-name zipvoice \
                  --exp-dir ${EXP_DIR}
            # The generated model is exp_cat/zipvoice_finetune/iter-10000-avg-2.pt
      fi

      ### Inference with PyTorch models (7)

      if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
            echo "Stage 7: Inference of the ZipVoice model"

            python3 -m zipvoice.bin.infer_zipvoice \
                  --model-name zipvoice \
                  --model-dir ${EXP_DIR}/ \
                  --checkpoint-name iter-30000-avg-2.pt \
                  --tokenizer ${tokenizer} \
                  --lang ${lang} \
                  --test-list data_cat/raw/test.tsv \
                  --res-dir results/${EXP_DIR} \
                  --guidance-scale 1.0 \
                  --num-step 25
      fi
done
