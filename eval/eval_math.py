import argparse
import json
import os
from utils import find_last_integer


def main(input_file, output_file):
    results = []
    total_cnt = 0
    correct_cnt = 0
    with open(input_file, "r") as file:
        for line in file:
            predict_data = json.loads(line)
            ground_truth = find_last_integer(predict_data["label"])
            prediction = predict_data["predict"].strip()
            answer = find_last_integer(prediction)
            # Add to results
            results.append(
                {"ground_truth": ground_truth, "predict": prediction, "answer": answer}
            )
            total_cnt += 1
            if check_math(predict_data):
                correct_cnt += 1

    print(f"Accuracy for math: {correct_cnt} / {total_cnt} = {(correct_cnt / total_cnt):.2%}\n")
    # Output to JSON file
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as outfile:
            json.dump(results, outfile, indent=4)


def check_math(predict_data):
    ground_truth = find_last_integer(predict_data["label"])
    prediction = predict_data["predict"].strip()
    answer = find_last_integer(prediction)
    try:
        int_ground_truth = int(ground_truth)
        int_answer = int(answer)
        return int_ground_truth == int_answer
    except ValueError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the results of the model")
    parser.add_argument("--input_file", type=str, help="Input jsonl file")
    parser.add_argument(
        "--output_file", type=str, default=None, help="Output json file"
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file)
