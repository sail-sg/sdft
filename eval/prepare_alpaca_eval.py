import json
import argparse
from pathlib import Path

def process(input_file, output_file):
    output_data_list = []
    instruction_file = "data/alpaca_eval.json"
    with open(instruction_file) as f:
        instruction_list = json.load(f)
    
    with open(input_file) as f:
        for line, data in zip(f, instruction_list):
            instruction = data["instruction"]
            output = json.loads(line)["predict"]
            item = {"instruction": instruction, "output": output}
            output_data_list.append(item)
    if not Path(output_file).parent.exists():
        Path(output_file).parent.mkdir(parents=True)
    with open(output_file, "w") as f:
        json.dump(output_data_list, f, indent=4)
        
def main(input_file, output_file):
    process(input_file, output_file)
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args.input_file, args.output_file)
