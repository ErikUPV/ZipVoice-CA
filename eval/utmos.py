import subprocess
import argparse
import pandas as pd
import os
import shutil
import glob
import tempfile

parser = argparse.ArgumentParser(description="Run UT-MOS prediction on a directory of audio files.")
parser.add_argument("--input_dir", type=str, required=True, help="Absolute directory containing audio files to predict.")
parser.add_argument('--utmos_dir', type=str, required=True, help='Absolute path of the UTMOS-demo repository root.')

args = parser.parse_args()

os.chdir(args.utmos_dir)

for dataset in ["cv17", "festcat", "frescat"]:
    files_to_move = glob.glob(f"{args.input_dir}/{dataset}*")
    destination_dir = os.path.join(args.input_dir, dataset)

    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    for file_path in files_to_move:
        # Avoid moving the destination directory into itself
        if os.path.abspath(file_path) != os.path.abspath(destination_dir):
            shutil.move(file_path, destination_dir)

    with tempfile.TemporaryDirectory() as temp_output_dir:
        out_path = temp_output_dir

        CMD = f"python predict.py --mode predict_dir --bs 64 --inp_dir {args.input_dir}/{dataset} --out_path {out_path}/{dataset}_utmos.csv"

        subprocess.run(CMD, shell=True, check=True)

        print(f"Predictions saved to {out_path}/{dataset}_utmos.csv")
        df = pd.read_csv(f"{out_path}/{dataset}_utmos.csv", names=["utmos"])
        print(f"Dataset: {dataset}, Mean UT-MOS: {df['utmos'].mean():.4f}")

    
