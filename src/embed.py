# src/embed.py
# 목적: 모든 음성을 emotion2vec에 넣어 768차원 임베딩을 추출하고 저장
#       (freeze 방식이라 한 번만 뽑아두고, 학습에서 재사용)

# ─────────────────────────────────────────
# import
# ─────────────────────────────────────────
# 필요한 것들을 import 하세요:
# - pandas (csv 읽기)
# - numpy (임베딩 배열로 저장)
# - os (폴더 생성, 경로)
# - tqdm (진행률 표시 — 16000개라 오래 걸리니 필수. from tqdm import tqdm)
# - funasr의 AutoModel (모델 로드 — test_model.py에서 썼던 거)
# - config (우리가 만든 설정 — import config)
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from funasr import AutoModel
import config


# ─────────────────────────────────────────
# 1. 저장 폴더 준비
# ─────────────────────────────────────────
# config.EMBED_DIR 폴더가 없으면 만들기
# 힌트: os.makedirs(경로, exist_ok=True)  ← exist_ok=True면 이미 있어도 에러 안 남
os.makedirs(config.EMBED_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 2. emotion2vec 모델 로드
# ─────────────────────────────────────────
# test_model.py에서 했던 것처럼 AutoModel 불러오기
# 힌트: config.MODEL_ID, config.HUB 를 사용
# model = AutoModel(model=..., hub=...)
model = AutoModel(model=config.MODEL_ID, hub=config.HUB)

# ─────────────────────────────────────────
# 3. 전처리된 데이터 읽기
# ─────────────────────────────────────────
# config.CSV_PATH 를 pandas로 읽기 (이건 우리가 만든 거라 인코딩 신경 안 써도 됨, utf-8)
# 이 df엔 'wav_path'(음성 경로)와 'label'(감정 숫자)이 들어있음
# 몇 개인지 print로 확인
df = pd.read_csv(config.CSV_PATH)
print("총 데이터 수:", len(df))

# ─────────────────────────────────────────
# 4. 각 음성에서 임베딩 추출 (핵심)
# ─────────────────────────────────────────
# 결과를 담을 빈 리스트 두 개 준비: 임베딩용, 라벨용
embeddings = []
labels = []

# df의 각 행을 반복 (tqdm으로 감싸서 진행률 보기)
# 힌트: for idx, row in tqdm(df.iterrows(), total=len(df)):
#

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

# ─────────────────────────────────────────
# 5. 리스트를 numpy 배열로 변환
# ─────────────────────────────────────────
# 리스트 상태론 저장 비효율 → numpy 배열로 바꾸기
# 힌트: np.array(embeddings), np.array(labels)
# 변환 후 shape 확인해보기 (임베딩은 (개수, 768), 라벨은 (개수,) 여야 함)
embeddings = np.array(embeddings)
labels = np.array(labels)

print("임베딩 개수:", embeddings.shape[0])
print("라벨 개수:", labels.shape[0])
print("임베딩 shape:", embeddings.shape)   # (개수, 768)
print("라벨 shape:", labels.shape)         # (개수,)

# ─────────────────────────────────────────
# 6. 저장
# ─────────────────────────────────────────
# 임베딩과 라벨을 파일로 저장 (다음 단계인 학습에서 불러다 씀)
# 힌트: np.save(경로, 배열) 사용
#   - config.EMBED_DIR 안에 저장
#   - 예: np.save(os.path.join(config.EMBED_DIR, "embeddings.npy"), embeddings)
#         np.save(os.path.join(config.EMBED_DIR, "labels.npy"), labels)
# 저장 완료 메시지 print


# ─────────────────────────────────────────
# 실행 확인용 (선택)
# ─────────────────────────────────────────
# 저장된 임베딩 개수, shape 등을 마지막에 print해서 확인