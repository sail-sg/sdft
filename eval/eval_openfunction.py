import re
import argparse
import json

def standardize_function_call(function_call):  
    function_call = re.sub(r'\s+', ' ', function_call.strip())
    function_call = function_call.replace("'", '"').replace(", ", ",")
    return function_call  

def main(input_file):
    total_cnt = 0
    correct_cnt = 0
    with open(input_file, "r") as file:
        for line in file:
            predict_data = json.loads(line)
            total_cnt += 1
            if check_openfunction(predict_data):
                correct_cnt += 1

    print(f"Accuracy for openfunction: {correct_cnt} / {total_cnt} = {(correct_cnt / total_cnt):.2%}\n")


def check_openfunction(predict_data):
    reference_answer = predict_data['label']
    reference_answer = standardize_function_call(reference_answer)
    answer = predict_data['predict']
    answer = standardize_function_call(answer)
    match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\((.*)\)', reference_answer)
    if not match:
        raise ValueError(f"Target function call is not in the correct format.\n{reference_answer}")
    func_name, params = match.groups()
    pattern = r'\b' + re.escape(func_name) + r'\(' + re.escape(params) + r'\)'
    return re.search(pattern, standardize_function_call(answer)) is not None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the results of the model")
    parser.add_argument("--input_file", type=str, help="Input jsonl file")
    args = parser.parse_args()
    main(args.input_file)
