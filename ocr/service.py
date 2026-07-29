import json
import re
from pathlib import Path

from ollama import chat

from prompt import OCR_PROMPT


class OCRException(Exception):
    pass


def extract_document(image_path: str) -> dict:
    """
    Extracts all textual fields from a document image.

    Parameters
    ----------
    image_path : str
        Path to the image.

    Returns
    -------
    dict
        JSON returned by the model.
    """

    image = Path(image_path)

    if not image.exists():
        raise FileNotFoundError(image)

    response = chat(
        model="qwen2.5vl:7b",
        messages=[
            {
                "role": "user",
                "content": OCR_PROMPT,
                "images": [str(image)],
            }
        ],
    )

    content = response["message"]["content"].strip()

    # Remove ```json ... ```
    content = re.sub(r"^```json", "", content)
    content = re.sub(r"^```", "", content)
    content = re.sub(r"```$", "", content).strip()

    try:
        return json.loads(content)

    except json.JSONDecodeError as e:
        raise OCRException(
            f"Model returned an invalid JSON.\n\n{content}"
        ) from e

