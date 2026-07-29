import json
from pprint import pprint

from service import extract_document

INPUT_FILE_NAME = "../database/input/rg_real.jpg"
OUTPUT_FILE_NAME = "../database/ocr_predictions/rg_prediction.json"

result = extract_document(INPUT_FILE_NAME)

pprint(result)

# Save the result to a JSON file
with open(OUTPUT_FILE_NAME, "w") as f:
    json.dump(result, f)