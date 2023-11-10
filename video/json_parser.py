import json
import re


def parse_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def link_constructor(data):
    return f"rtsp://{data['login']}:{data['password']}@{data['ip']}"


def extract_id(file_path):
    match = re.search(r'(video|stream)_(\d+)', file_path)
    if match:
        return str(match.group(2))
    else:
        return None

