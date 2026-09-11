# emotion2vec-SER

emotion2vec 사전학습 모델로 음성 임베딩을 추출하고, MLP 분류기로 감정을 인식하는 Speech Emotion Recognition(SER) 파이프라인입니다.

논문 **"emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation"** 을 참고하여 구현했습니다.

---

## 개요

emotion2vec은 자기지도학습(Self-Supervised Learning) 방식으로 사전학습된 음성 감정 표현 모델입니다.
본 프로젝트는 `iic/emotion2vec_plus_base` 모델로 발화 단위 임베딩(768차원)을 추출한 뒤, 간단한 MLP로 4가지 감정을 분류합니다.

### 감정 클래스 (4감정)

| Label | 감정 |
|-------|------|
| 0 | angry |
| 1 | happiness |
| 2 | neutral |
| 3 | sadness |

---

## 발표 자료

프로젝트 전체 내용을 정리한 발표 자료입니다.

- [📄 슬라이드 미리보기 (PDF)](./docs/emotion2vec.pdf)
- [📥 발표 원본 (PPTX)](./docs/emotion2vec.pptx)

---

## 프로젝트 구조

```
emotion2vec-ser/
├── src/
│   ├── config.py        # 모델·경로·하이퍼파라미터 설정
│   ├── preprocess.py    # 원본 CSV 전처리 및 wav 경로 매핑
│   ├── embed.py         # emotion2vec 임베딩 추출 및 저장
│   └── train.py         # MLP 분류기 학습 및 평가
├── data/
│   └── processed_5th_2nd.csv   # 전처리된 데이터 (wav_path, label)
├── outputs/
│   ├── embeddings/
│   │   ├── embeddings.npy      # 추출된 임베딩 (N, 768)
│   │   └── labels.npy          # 감정 레이블 (N,)
│   └── models/
│       └── classifier.pt       # 학습된 MLP 분류기
└── README.md
```

---

## 데이터셋

**감정 분류를 위한 대화 음성 데이터셋 (5차년도 2차)**

- 5명의 평가자가 각 발화에 감정 레이블을 투표
- 과반수 투표로 최종 감정 결정 (동률 시 제외)
- 원본 7감정 중 데이터가 충분한 4감정(angry, happiness, neutral, sadness)만 사용
- 전처리 후 총 **13,239개** 발화 사용

| 감정 | 샘플 수 |
|------|--------|
| angry | 1,545 |
| happiness | 3,479 |
| neutral | 4,875 |
| sadness | 3,340 |

---

## 파이프라인

### 1. 전처리

```bash
cd src && python preprocess.py
```

원본 CSV에서 과반수 투표로 최종 감정 레이블을 결정하고, wav 파일 경로와 함께 `data/processed_5th_2nd.csv`로 저장합니다.

### 2. 임베딩 추출

```bash
cd src && python embed.py
```

`emotion2vec_plus_base` 모델로 각 발화의 utterance-level 임베딩(768차원)을 추출하여 `outputs/embeddings/`에 저장합니다.

### 3. 학습

```bash
cd src && python train.py
```

추출된 임베딩으로 MLP 분류기를 학습합니다.

---

## 모델 구조

```
Input (768) -> Linear -> ReLU -> Linear -> Output (4)
               768->256          256->4
```

- 임베딩 차원: 768 (emotion2vec_plus_base 출력)
- 은닉층: 256
- 출력: 4감정 클래스
- Loss: CrossEntropyLoss (class weight balanced 적용)
- Optimizer: Adam

---

## 하이퍼파라미터

| 설정 | 값 |
|------|----|
| EMBED_DIM | 768 |
| HIDDEN_DIM | 256 |
| NUM_CLASSES | 4 |
| LEARNING_RATE | 0.005 |
| EPOCHS | 200 |
| SEED | 42 |
| TRAIN / VAL / TEST | 70% / 10% / 20% |

---

## 평가 지표

- **WA** (Weighted Accuracy): 전체 정확도
- **UA** (Unweighted Accuracy): 클래스별 Recall 평균 (클래스 불균형 보정)
- **WF1** (Weighted F1): 가중 F1 스코어

---

## 재현성

모든 실험은 아래 seed 고정으로 재현 가능합니다.

```python
np.random.seed(42)
torch.manual_seed(42)
train_test_split(..., random_state=42)
```

---

## 참고 논문

Ma, Z., Zheng, M., Zhao, Z., Li, J., & Shan, W. (2023).
**emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation.**
*arXiv preprint arXiv:2312.15185.*

- 모델: [iic/emotion2vec_plus_base](https://huggingface.co/iic/emotion2vec_plus_base)
- 구현: [FunASR](https://github.com/modelscope/FunASR)
