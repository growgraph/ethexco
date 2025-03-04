from datasets import load_dataset
from pprint import pprint

dataset = load_dataset("vicgalle/alpaca-gpt4", split="train")
print(dataset.column_names)


for i in range(1000):
    d = dataset[i]
    if d["input"]:
        pprint(dataset[i])
