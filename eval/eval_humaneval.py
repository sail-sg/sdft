import json
import argparse

def get_acc(input_file):
    with open(input_file) as file:
        data = json.load(file)
    acc = data["humanevalsynthesize-python"]["pass@1"]
    print("Evaluation on HumanEval:")
    print(f"Accuracy for HumanEval: {acc:.2%}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    get_acc(args.input_file)
