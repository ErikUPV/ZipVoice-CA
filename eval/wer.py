import faster_whisper
import jiwer
import os
import pandas as pd
from tqdm import tqdm
from whisper.normalizers import BasicTextNormalizer
import argparse
from num2words import num2words
import regex as re
import glob
import random

parser = argparse.ArgumentParser()
parser.add_argument("--action", type=str, default="transcribe", choices=["transcribe", "predict"],
                    help="Action to perform: if you already have transcriptions, you can calculate WER/CER directly.")
parser.add_argument("--datasets", type=str, nargs='+', default=None)
parser.add_argument("--tsv_path", type=str, default="test.tsv")
parser.add_argument('--results_dir', type=str, required=True)
parser.add_argument('--samples', type=int, default=None)
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()

refs, hyps = [], []
number_re = re.compile(r'\b\d+\b')

random.seed(42)


def normalize(text, normalizer):
    text = normalizer(text)
    text = number_re.sub(
        lambda x: num2words(
            int(x.group()),
            lang='ca'
            ),
        text
        )
    return text

def format_numbers(filename):
    number = filename.split('_')[-1].replace(".wav", "")
    return f"{filename.split('_')[0]}_{int(number):03d}.wav"

if args.action == "transcribe":

    model = faster_whisper.WhisperModel(model_size_or_path="large-v3")

    dir = args.results_dir
    csv = pd.read_csv(args.tsv_path, sep="\t", header=None, names=["file", "prompt_text", "prompt_wav", "text"])
    print(csv)

    if args.datasets:
        t_dataset = {
            k: {}
            for k in args.datasets
        }
    else:
        datasets = ["cv17", "festcat", "frescat", "default"]
        t_dataset = {
            k: {}
            for k in datasets
        }

    files = glob.glob(f"{dir}/**/*.wav", recursive=True)
    if args.samples is not None:
        random.shuffle(files)
        files = files[:args.samples]
    
    print(len(files))

    for file in tqdm(files, desc="Transcribing"):
        if not file.endswith(".wav"):
            continue
        
        formated_file = format_numbers(os.path.basename(file))

        dataset_name = os.path.dirname(file).split(os.sep)[-1]

        if args.datasets is not None:
            dataset_name = formated_file.split('_')[0]
            if dataset_name not in args.datasets:
                continue
        print(dataset_name)
        


        # print(file.split(".")[0])
        # print(csv.loc[csv["file"] == file.split(".")[0], "text"].values[0])
        generator, info = model.transcribe(file, language="ca")
        transcription = "".join(segment.text for segment in generator)
        id = formated_file
        if args.verbose:
            print(id)
            
        
        t_dataset[dataset_name][file] = (transcription, csv.loc[csv["file"].str.split('/').str[-1] == id, "text"].values[0])
        
        if args.verbose:
            print(t_dataset[dataset_name][file])
            print(len(t_dataset[dataset_name]))

    normalizer = BasicTextNormalizer(remove_diacritics=False)


elif args.action == "predict":
    df = pd.read_csv("results/test_finetune/transcriptions.csv")
    refs = df["reference"].tolist()
    hyps = df["transcription"].tolist()

for dataset in t_dataset.keys():
    refs = [normalize(ref.strip(), normalizer).strip() for _, ref in t_dataset[dataset].values()]
    hyps = [normalize(hyp.strip(), normalizer).strip() for hyp, _ in t_dataset[dataset].values()]

    wer_metric = jiwer.wer(
        refs,
        hyps,
    )

    cer_metric = jiwer.cer(
        refs,
        hyps,
    )

    print(f"[{dataset}] WER: {wer_metric:.4f}")
    print(f"[{dataset}] CER: {cer_metric:.4f}")


if args.action == "transcribe":
    transcription_dict = {
        "transcription" : [hyp for hyp in hyps],
        "reference" : [ref for ref in refs],
    }

    df: pd.DataFrame = pd.DataFrame.from_dict(transcription_dict)
    df.to_csv(f"{args.results_dir}/transcriptions.csv", index=False)

