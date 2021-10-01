import json
from PIL import Image

def save_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, ensure_ascii=False)
        
def load_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def load_image(path):
    img = Image.open(path)
    return img