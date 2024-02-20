import re
import json

def find_last_integer(s):
    INVALID = "-999999999"
    # This regular expression matches numbers, including negative numbers and decimals,
    # -? matches zero or one negative sign. This allows the number to be negative.
    # \d+ matches one or more digits. This is the integer part of the number.
    # (?:[,\s]\d+)* is a non-capturing group that matches zero or more occurrences of a comma or space followed by one or more digits.
    # This allows the number to have commas or spaces as thousand separators.
    # (?:\.\d+)? is another non-capturing group that matches zero or one occurrence of a decimal point followed by one or more digits.
    # This allows the number to have a decimal part.
    pattern = r"-?\d+(?:[,\s]\d+)*(?:\.\d+)?"
    numbers = re.findall(pattern, s)
    number = numbers[-1].replace(",", "").replace(" ", "") if numbers else INVALID
    if len(number.split(".")) > 1 and int(number.split(".")[1]) != 0:
        return INVALID
    return number.split(".")[0]

def strip_dict(d):
    for k, v in d.items():
        if isinstance(v, str):
            d[k] = v.strip()
            
def get_outputs(input_file):
    output_data_list = []
    with open(input_file) as f:
        for line in f:
            data = json.loads(line)
            prediction = data["predict"]
            output_data_list.append(prediction)
    return output_data_list