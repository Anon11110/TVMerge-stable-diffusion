import os
import pickle
from typing import Any, Dict

def save_to_file(obj: Any, file_path: str) -> None:
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def load_from_file(file_path: str) -> Any:
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def create_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_params(params: Dict[str, Any], artifact_path: str) -> None:
    file_path = os.path.join(artifact_path, "params.pkl")
    save_to_file(params, file_path)

def load_params(artifact_path: str) -> Dict[str, Any]:
    file_path = os.path.join(artifact_path, "params.pkl")
    return load_from_file(file_path)
