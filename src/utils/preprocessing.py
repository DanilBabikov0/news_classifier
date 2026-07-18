import re
import pandas as pd
import os
import config

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    
    text = text.replace('\u2028', ' ')
    text = text.replace('\u2029', ' ') 

    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:—«»"\'()]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text.strip()

def load_and_clean(data_path: str, save_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[config.X_NAME]).copy()
    df[config.X_NAME] = df[config.X_NAME].apply(clean_text)
    df = df[df[config.X_NAME].str.len() > 0].copy()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
    return df