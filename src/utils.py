import json
import yaml
import pickle

def load_config(path = 'params.yaml'):
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def write_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def save_pkl(path, model):
    with open(path,'wb') as file:
        pickle.dump(model,file)

def load_pkl(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model
