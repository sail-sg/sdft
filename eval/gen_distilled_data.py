import json
import argparse
import datasets
from eval_math import check_math
from eval_openfunction import check_openfunction
from eval_magicoder import check_magicoder
from utils import strip_dict

def main(dataset, predict_jsonl):
    with open(f"data/{dataset}/{dataset}_train.json") as f:
        origin_dataset = json.load(f)
    check_func = get_check_func(dataset)
    distilled_dataset_name = f"distilled_{dataset}"
    output_data_list = get_output_data_list(
        origin_dataset, predict_jsonl, check_func
    )
    output_file = f"data/{dataset}/{distilled_dataset_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data_list, f, ensure_ascii=False, indent=4)


def get_check_func(dataset):
    if "gsm8k" in dataset or "MultiArith" in dataset:
        return check_math
    if "openfunction" in dataset:
        return check_openfunction
    if "magicoder" in dataset:
        return check_magicoder
    return lambda x: True


def get_output_data_list(origin_dataset, predict_jsonl, check_func):
    output_data_list = []
    answer_key = None
    with open(predict_jsonl, "r", encoding="utf-8") as f:
        for origin_data, line in zip(origin_dataset, f):
            predict_data = json.loads(line)
            strip_dict(origin_data)
            strip_dict(predict_data)
            if not answer_key:
                answer_key = find_answer_key(origin_data, predict_data)
            if verify(predict_data) and check_func(predict_data):
                origin_data[answer_key] = predict_data["predict"]
            output_data_list.append(origin_data)
    return output_data_list


def find_answer_key(origin_data, predict_data):
    for key in origin_data.keys():
        if origin_data[key] == predict_data["label"]:
            return key
    raise ValueError("answer key not found!")


def verify(predict_data):
    ban_set = {
        "reference answer",
        "your response",
        "my response",
        "your own response",
        "now it's your turn"
    }
    for ban in ban_set:
        if ban.lower() in predict_data["predict"].lower():
            return False
    if predict_data["predict"].startswith("Your turn"):
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate distilled dataset")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--predict_jsonl",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args.dataset, args.predict_jsonl)