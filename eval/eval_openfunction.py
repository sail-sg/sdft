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
            correct_cnt += check_openfunction(predict_data)

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
    # Exact keyword argument match
    if re.search(pattern, answer) is not None:
        return 1
    # keyword match failed, try position argument match, and assign half score
    answer_param_part = answer[max(answer.find('(') + 1, 0):answer.find(')')]
    ref_param_list = params.split(',')
    ans_param_list = answer_param_part.split(',')
    
    if len(ref_param_list) != len(ans_param_list):
        return 0
    for ref_param, ans_param in zip(ref_param_list, ans_param_list):
        try:
            value = ref_param.split('=')[1]
            if value not in ans_param:
                return 0
        except:
            return 0
    return 0.5
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the results of the model")
    parser.add_argument("--input_file", type=str, help="Input jsonl file")
    args = parser.parse_args()
    main(args.input_file)