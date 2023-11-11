import json
import re


def parse_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def link_constructor(data):
    print("Linking constructor", flush=True)
    credentials = ""
    if 'login' in data:
        credentials = data['login']
        if 'password' in data:
            credentials += f":{data['password']}"
        credentials += "@"
    return f"rtsp://{credentials}{data['url']}"


def extract_id(file_path):
    match = re.search(r'(video|stream)_([a-z0-9-]+)', file_path)
    if match:
        return str(match.group(2))
    else:
        return None

