import json
import argparse

def get_acc(input_file, dataset):
    with open(input_file) as file:
        data = json.load(file)
    results = data["results"]
    result = results[dataset]
    acc = result["acc,none"]
    print(f"Accuracy for {dataset}: {acc:.2%}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    print("Evaluation on OpenLLM Leaderboard:")
    for dataset in ["mmlu", "truthfulqa", "ai2_arc", "hellaswag", "winogrande"]:
        get_acc(args.input_file, dataset)
    print()