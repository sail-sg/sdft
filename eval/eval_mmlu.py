import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    print("Evaluation on MMLU:")
    with open(args.input_file) as f:
        content = f.read()
        print(content + "\n")