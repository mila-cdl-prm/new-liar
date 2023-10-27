"""
Merge columns of one dataset into another.

Overview:
- Load base dataset into a dictionary mapping `key` to dataset entry.
- For each additional dataset, create a similar dictionary.
- For each item in the base dataset, look up that key in the additional dataset.
    If an item can be retrieved, create a new dataset entry, where data from
    the base dataset should have priority.

Save a HuggingFace dataset under the same name (without the jsonl extension name)
"""

from typing import List, Tuple, Dict
import argparse
import json
import os

import datasets
from tqdm.auto import tqdm


from dataset_merge_utils import load_jsonl_file, build_index


parser = argparse.ArgumentParser()
parser.add_argument("base_dataset_jsonl")
parser.add_argument("jsonl_output_path")
parser.add_argument("extra_datasets_jsonl", nargs="+")
parser.add_argument("--key", default="example_id")
args = parser.parse_args()

# Load base dataset into a dictionary mapping `key` to dataset entry.
base_dataset = load_jsonl_file(args.base_dataset_jsonl)
base_dataset_index = build_index(base_dataset, args.key)

assert len(base_dataset) > 0

# Record the name of columns added and source of each column.
columns_added: Dict[str, str] = {
    k: args.base_dataset_jsonl for k in base_dataset[0].keys()
}

# For each additional dataset, create a similar dictionary.
for other_dataset_path in tqdm(args.extra_datasets_jsonl, ncols=75):
    other_dataset = load_jsonl_file(other_dataset_path)
    if len(other_dataset) == 0:
        print("Skipping empty dataset: {}".format(other_dataset_path))
        continue

    other_dataset_index = build_index(other_dataset, args.key)

    # Record names of columns added from this dataset.
    # Subtract to find list of newly-added column names.
    column_differences = other_dataset[0].keys() - base_dataset[0].keys()
    for column_name in list(column_differences):
        if column_name not in columns_added.keys():
            columns_added[column_name] = other_dataset_path

    # Create buffer to store updated entries.
    # Doing so avoids updating the index while iterating through it.
    base_dataset_index_updated = {}

    # For each item key in the index of the base dataset, check
    # if the other dataset index includes the same key.
    for key, base_entry in base_dataset_index.items():
        other_dataset_entry = other_dataset_index.get(key)

        if other_dataset_entry is not None:
            # If there is a match, add columns from the item from
            # the other dataset to the item from the base dataset.
            # Items from the base dataset should have priority.
            updated_entry = {**other_dataset_entry, **base_entry}
        else:
            updated_entry = base_entry

        # Update the base dataset index with the new item.
        base_dataset_index_updated[key] = updated_entry

    # Replace base_dataset_index with buffer.
    assert base_dataset_index.keys() == base_dataset_index_updated.keys()
    base_dataset_index = base_dataset_index_updated


for column_name, source_path in columns_added.items():
    print("{}: {}".format(column_name, source_path))

# Iterate through base dataset to create a jsonline format.
# Initialize output buffer.
dataset_output: List[Dict] = []
jsonl_output_buffer: List[str] = []

# For each value in the base dataset index, generate
# JSON string representation and add to buffer.
for entry in base_dataset_index.values():
    entry_json = json.dumps(entry)
    dataset_output.append(entry)
    jsonl_output_buffer.append(entry_json)

# Write JSONLines to the requested file.
with open(args.jsonl_output_path, "w") as output_file:
    output_file.write("\n".join(jsonl_output_buffer))
    print("Written jsonl output to {}".format(args.jsonl_output_path))


output_folder, output_jsonl_filename = os.path.split(args.jsonl_output_path)
output_basename, extension_name = os.path.splitext(output_jsonl_filename)
if extension_name is not None:
    dataset_output_path = os.path.join(output_folder, output_basename)
    dataset = datasets.Dataset.from_list(dataset_output)
    dataset_dict = dataset.train_test_split(test_size=0.3, seed=325)
    print(dataset_dict)

    dataset_dict.save_to_disk(dataset_output_path)
    print("DatasetDict saved to {}".format(dataset_output_path))
