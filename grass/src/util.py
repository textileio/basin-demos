from json import dumps
from re import search


# Function to extract the numerical suffix from the file name
def get_numeric_suffix(file_name):
    match = search(r"(\d+)", file_name)
    return int(match.group(1)) if match else -1


# Utility method for flattening row lists to adhere to DataFrame requirements
def flatten_row_lists(row: dict, header: str) -> dict:
    # Convert `media_urls` to JSON strings ()
    if isinstance(row.get(header), list):
        row[header] = dumps(row[header])
    return row
