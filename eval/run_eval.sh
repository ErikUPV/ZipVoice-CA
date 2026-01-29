#!/bin/bash

# Suite of evaluation scripts for ZipVoice-generated audio files.
# Requires 3 environments: your ZipVoice environment,
# a Whisper environment for WER evaluation, and a UTMOS environment for MOS evaluation.
# Usage: ./run_eval.sh <EXP_DIR>

EXP_DIR=$1 # e.g., finetune_eden_1e-4

export PYTHONPATH=../../:$PYTHONPATH

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

conda activate ZipVoice

python -m zipvoice.eval.speaker_similarity.sim \
            --wav-path egs/zipvoice/results/$EXP_DIR \
            --test-list egs/zipvoice/data_cat/raw/test.tsv \
            --model-dir TTS_eval_models/

conda activate whisper-env

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(python3 -c 'import os; import nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__) + "/lib")')

python eval/wer.py \
            --action predict \
            --datasets cv17 frescat festcat \
            --tsv_path egs/zipvoice/data_cat/raw/test.tsv \
            --results_dir egs/zipvoice/results/$EXP_DIR \
            --verbose

unset LD_LIBRARY_PATH

conda activate utmos-env

python eval/utmos.py \
            --input_dir egs/zipvoice/results/$EXP_DIR \
            --utmos_dir # Your UTMOS-demo repository path

