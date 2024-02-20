
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import get_outputs
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_bleu4_array(origin_outputs, outputs):
    assert len(origin_outputs) == len(outputs)
    bleu4_array = np.zeros(len(origin_outputs))
    for idx, (origin_output, sft_output) in enumerate(zip(origin_outputs, outputs)):
        bleu_score = sentence_bleu([list(origin_output)], list(sft_output), smoothing_function=SmoothingFunction().method3)
        bleu4_array[idx] = bleu_score
    return bleu4_array

def draw(sft_bleu4_array, distill_bleu4_array):  
    plt.hist(  
        sft_bleu4_array,  
        color="#A16058",  
        bins=30,  
        alpha=0.7,  
        label="FT",  
    )  
    plt.hist(  
        distill_bleu4_array,  
        color="#85BFAC",  
        bins=30,  
        alpha=0.7,  
        label="SDFT",  
    )  
      
    sft_mean = sft_bleu4_array.mean()  
    distill_mean = distill_bleu4_array.mean()  
      
    plt.scatter([sft_mean], [0], color="#A16058", s=100, label="FT Mean", zorder=2.5)  
    plt.scatter([distill_mean], [0], color="#616C6E", s=100, label="SDFT Mean", zorder=2.5)  
      
    plt.text(sft_mean, plt.ylim()[1]*0.05, f'{sft_mean:.4f}', color="#A16058", ha='center')  
    plt.text(distill_mean, plt.ylim()[1]*0.05, f'{distill_mean:.4f}', color="#616C6E", ha='center')  
      
    plt.legend()  
    plt.xlabel("bleu4 Distribution")  
    plt.ylabel("Count")  
    plt.tight_layout()  
      
    plt.savefig("bleu4.svg")

def main(dataset, instruction_dataset):
    origin_file = f"predictions/origin/{instruction_dataset}/generated_predictions.jsonl"
    sft_file = f"predictions/{dataset}/sft/{instruction_dataset}/generated_predictions.jsonl"
    distill_file = f"predictions/{dataset}/distill/{instruction_dataset}/generated_predictions.jsonl"
    origin_outputs = get_outputs(origin_file)
    sft_outputs = get_outputs(sft_file)
    distill_outputs = get_outputs(distill_file)
    sft_bleu4_array = get_bleu4_array(origin_outputs, sft_outputs)
    distill_bleu4_array = get_bleu4_array(origin_outputs, distill_outputs)
    draw(sft_bleu4_array, distill_bleu4_array)
    print(f"sft_bleu4: {sft_bleu4_array.mean():.4f}")
    print(f"distill_bleu4: {distill_bleu4_array.mean():.4f}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--instruction_dataset", type=str, default="advbench")
    args = parser.parse_args()
    main(args.dataset, args.instruction_dataset)