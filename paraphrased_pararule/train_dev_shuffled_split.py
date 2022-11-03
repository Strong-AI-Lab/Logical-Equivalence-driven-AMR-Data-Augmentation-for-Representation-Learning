import jsonlines
import random
import json

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

def parse(response, input_name, flag):
    if flag == "train":
        with open(input_name+'-train.jsonl', 'w') as f:
            for index in response:
                json.dump(index, f, default=set_default)
                f.write('\n')
    elif flag == "dev":
        with open(input_name+'-dev.jsonl', 'w') as f:
            for index in response:
                json.dump(index, f, default=set_default)
                f.write('\n')

result_list = []
input_name = "NonNegationRule-depth-1"
with open(input_name+".jsonl", "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        result_list.append(item)

random.shuffle(result_list)
split_separator = int(len(result_list) * 0.8)
parse(result_list[:split_separator], input_name, "train")
parse(result_list[split_separator:], input_name, "dev")
