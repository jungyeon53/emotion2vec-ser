# src/config.py
# 프로젝트 전체에서 쓰는 설정값 모음

# ─────────────────────────────
# 모델 설정
# ─────────────────────────────

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_ID = "iic/emotion2vec_plus_base"   # 사용할 emotion2vec 모델
HUB = "hf"                                # Hugging Face에서 다운로드
EMBED_DIM = 768                           # emotion2vec 임베딩 차원 (실측 확인함)

# ─────────────────────────────
# 감정 클래스
# ─────────────────────────────
# NUM_CLASSES = 7                           # 7감정 분류
NUM_CLASSES = 4                           # 4감정으로 변경

LABEL_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}

# 숫자 → 감정 이름 (결과 해석·시각화용)
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

# ─────────────────────────────
# 경로
# ─────────────────────────────
CSV_PATH = "data/processed_5th_2nd.csv"   # 전처리된 데이터 (음성경로 + 라벨)
EMBED_DIR = "outputs/embeddings"          # 추출한 임베딩 저장 위치
MODEL_DIR = "outputs/models"              # 학습한 분류기 저장 위치

# ─────────────────────────────
# 학습 하이퍼파라미터
# ─────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 0.005   # 0.001 → 0.005
EPOCHS = 200
HIDDEN_DIM = 256                          # 분류기 은닉층 크기 (논문 따라)

# train/validation/test 분할 비율
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

# ─────────────────────────────
# 기타
# ─────────────────────────────
DEVICE = "mps"                            # M4 GPU 가속
SEED = 42                                 # 재현성