
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from funasr import AutoModel
import config


os.makedirs(config.EMBED_DIR, exist_ok=True)

model = AutoModel(model=config.MODEL_ID, hub=config.HUB)


df = pd.read_csv(config.CSV_PATH)
# 테스트용 100개
# df = df.head(100)  
print("총 데이터 수:", len(df))

embeddings = []
labels = []


for idx, row in tqdm(df.iterrows(), total=len(df)):
    wav_path = row['wav_path']
    label = row['label']

    try:
        rec_result = model.generate(wav_path, granularity="utterance", extract_embedding=True)
        feat = rec_result[0]['feats']
        embeddings.append(feat)
        labels.append(label)
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")

embeddings = np.array(embeddings)
labels = np.array(labels)

print("임베딩 개수:", embeddings.shape[0])
print("라벨 개수:", labels.shape[0])
print("임베딩 shape:", embeddings.shape)   # (개수, 768)
print("라벨 shape:", labels.shape)         # (개수,)

np.save(os.path.join(config.EMBED_DIR, "embeddings.npy"), embeddings)
np.save(os.path.join(config.EMBED_DIR, "labels.npy"), labels)

print("저장 완료!")
print("임베딩:", embeddings.shape)
print("라벨:", labels.shape)
