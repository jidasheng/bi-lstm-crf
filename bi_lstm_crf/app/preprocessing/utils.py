import json

START_TAG = "<START>"
STOP_TAG = "<STOP>"

PAD = "<PAD>"
OOV = "<OOV>"


def save_json_file(obj, file_path):
    with open(file_path, "w", encoding="utf8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))


def load_json_file(file_path):
    with open(file_path, encoding="utf8") as f:
        return json.load(f)
