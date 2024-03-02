import numpy as np
import matplotlib.pyplot as plt
import argparse
from utils import get_outputs
import jieba
from rouge_chinese import Rouge

def get_rougel_array(seed_outputs, outputs):
    assert len(seed_outputs) == len(outputs)
    rougel_array = np.zeros(len(seed_outputs))
    for idx, (seed_output, sft_output) in enumerate(zip(seed_outputs, outputs)):
        
        reference = list(jieba.cut(seed_output))
        hypothesis = list(jieba.cut(sft_output))
        rouge = Rouge()
        scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
        result = scores[0]
        rougel = result["rouge-l"]["f"]
        rougel_array[idx] = rougel
    return rougel_array

def draw(sft_rougel_array, sdft_rougel_array):  
    plt.hist(  
        sft_rougel_array,  
        color="#A16058",  
        bins=30,  
        alpha=0.7,  
        label="FT",  
    )  
    plt.hist(  
        sdft_rougel_array,  
        color="#85BFAC",  
        bins=30,  
        alpha=0.7,  
        label="SDFT",  
    )  
      
    sft_mean = sft_rougel_array.mean()  
    sdft_mean = sdft_rougel_array.mean()  
      
    plt.scatter([sft_mean], [0], color="#A16058", s=100, label="FT Mean", zorder=2.5)  
    plt.scatter([sdft_mean], [0], color="#616C6E", s=100, label="SDFT Mean", zorder=2.5)  
      
    plt.text(sft_mean, plt.ylim()[1]*0.05, f'{sft_mean:.4f}', color="#A16058", ha='center')  
    plt.text(sdft_mean, plt.ylim()[1]*0.05, f'{sdft_mean:.4f}', color="#616C6E", ha='center')  
      
    plt.legend()  
    plt.xlabel("Rouge-l Distribution")  
    plt.ylabel("Count")  
    plt.tight_layout()  
      
    plt.savefig("rougel.svg")

def main(dataset, instruction_dataset):
    seed_file = f"predictions/seed/{instruction_dataset}/generated_predictions.jsonl"
    sft_file = f"predictions/{dataset}/sft/{instruction_dataset}/generated_predictions.jsonl"
    sdft_file = f"predictions/{dataset}/sdft/{instruction_dataset}/generated_predictions.jsonl"
    seed_outputs = get_outputs(seed_file)
    sft_outputs = get_outputs(sft_file)
    sdft_outputs = get_outputs(sdft_file)
    sft_rougel_array = get_rougel_array(seed_outputs, sft_outputs)
    sdft_rougel_array = get_rougel_array(seed_outputs, sdft_outputs)
    draw(sft_rougel_array, sdft_rougel_array)
    print(f"sft_rougel: {sft_rougel_array.mean():.4f}")
    print(f"sdft_rougel: {sdft_rougel_array.mean():.4f}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--instruction_dataset", type=str, default="advbench-raw")
    args = parser.parse_args()
    main(args.dataset, args.instruction_dataset)