# config by Umar Jamil

from pathlib import Path

def get_config(): 
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 500, 
        "d_model": 512, 
        "lang_src": "de", 
        "lang_tgt": "en", 
        "datasource": 'opus_books',
        "model_folder": "weights", 
        "kaggle": False, 
        "kaggle_model_folder": "/kaggle/input/langauage-model",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "count_gpu": 1,
        "TPU": True
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"  
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    print("Searching for exisiting model...")
    model_folder = f"{config['model_folder']}"
    if config["kaggle"]: 
        model_folder = f"{config['kaggle_model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])