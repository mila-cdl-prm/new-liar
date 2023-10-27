"""
Utils for:
- Loading jsonl files into a dictionary mapping a given indexing column
    to entries in the dataset.
"""
from typing import Dict, List, Any
import json

def load_jsonl_file(jsonl_path: str) -> List[Any]:
    """
    Load and parse a given JSONLines file into a list of JSON items.

    Params:
        jsonl_path: str, path to JSONLines file.
    
    Returns:
        List of parsed json items.
    """
    # Load input file as a list of lines.
    parsed_items: List[str] = []
    with open(jsonl_path, "r") as jsonl_file:
        lines = jsonl_file.read()
        for line in lines.splitlines():
            line_formatted = line.lstrip().rstrip()
            if len(line_formatted) > 0: 
                item = json.loads(line_formatted)    
                parsed_items.append(item)

    return parsed_items


def build_index(data_entries: List[Dict[str, Any]], primary_key: str) -> Dict[str, Dict]:
    """
    Given a list of dictionary objects and the name of a dictionary key 
    to be used as the primary key, create a lookup dictionary
    mapping values of the primary key to entries in the 
    jsonline file.

    Entries that do not include a `primary_key` attribute would
    not be included.

    Params:
        jsonl_lines: List[Dict[str, Any]], one line for a dictionary.
        primary_key: str, name of the primary key column.

    Returns:
        Dict[str, Dict], mapping values of primary key to 
        the corresponding entry (a dictionary) in the data file.
    """
    # Create buffer of output lookup map
    output: Dict[str, Dict] = {}

    # For each entry in the input:
    for data_entry in data_entries:

        # Retrieve key of the entry
        key = data_entry.get(primary_key)
        if key is None:
            continue

        # Set value of key in the output dictionary to this entry.
        output[key] = data_entry

    return output