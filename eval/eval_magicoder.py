import re
import json
import argparse

def main(input_file):
    total_cnt = 0
    correct_cnt = 0
    with open(input_file, "r") as file:
        for line in file:
            predict_data = json.loads(line)
            total_cnt += 1
            if check_magicoder(predict_data):
                correct_cnt += 1

    print(f"Accuracy: {correct_cnt} / {total_cnt} = {(correct_cnt / total_cnt):.2%}")

def contains_keywords(answer):
    keywords = {
        "```python"
    }
    for key in keywords:
        if key in answer:
            return True
    return False

def starts_right(answer):
    start_words = {
        "I have",
        "Here is",
        "I implemented"
    }
    for start_word in start_words:
        if answer.startswith(start_word):
            return True
    return False

def check_magicoder(predict_data):
    reference_answer = predict_data["label"].strip()
    answer = predict_data["predict"].strip()
    if not contains_keywords(answer):
        return False
    if not starts_right(answer):
        return False
    return True
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="Input jsonl file")
    args = parser.parse_args()
    main(args.input_file)
