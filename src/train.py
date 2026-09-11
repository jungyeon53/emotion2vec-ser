import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import config
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(config.SEED)
torch.manual_seed(config.SEED)

# 임베딩 불러오기
embeddings = np.load(f"{config.EMBED_DIR}/embeddings.npy")
labels = np.load(f"{config.EMBED_DIR}/labels.npy")

print("임베딩:", embeddings.shape)
print("라벨:", labels.shape)

#--------------------------------------------------
# 4감정만 남기기 (angry, happy, neutral, sad)
keep_labels = [0, 3, 4, 5]   # 남길 감정 (angry, happy, neutral, sad)

mask = np.isin(labels, keep_labels)
embeddings = embeddings[mask]
labels = labels[mask]

# 라벨 0~3으로 재매핑 (0,3,4,5 → 0,1,2,3)
remap = {0: 0, 3: 1, 4: 2, 5: 3}
labels = np.array([remap[l] for l in labels])

print(f"4감정 필터 후: {len(labels)}개")
print(f"라벨 분포: {np.bincount(labels)}")
#-----------------------------------------------

# 2. train / val / test 나누기

# 1차: 전체 → train(70%) + 임시(30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    embeddings, labels,
    test_size=0.3,           # 30%를 임시로 떼기
    stratify=labels,         # 감정 비율 유지
    random_state=config.SEED
)

# 2차: 임시(30%) → val(10%) + test(20%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=2/3,           # 임시의 2/3 = 전체의 20%가 test
    stratify=y_temp,         # 감정 비율 유지
    random_state=config.SEED
)

print("train:", X_train.shape, "/ val:", X_val.shape, "/ test:", X_test.shape)


# 임베딩(X)은 실수 → FloatTensor
X_train = torch.FloatTensor(X_train)
X_val = torch.FloatTensor(X_val)
X_test = torch.FloatTensor(X_test)

# 라벨(y)은 정수 → LongTensor
y_train = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val)
y_test = torch.LongTensor(y_test)


# 4. 분류기 정의

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(config.EMBED_DIM, config.HIDDEN_DIM)   # 768 → 256
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES) # 256 → 7

    def forward(self, x):
        x = self.fc1(x)    # 768 → 256
        x = self.relu(x)   # 활성화
        x = self.fc2(x)    # 256 → 7
        return x


# 5. 학습 도구 준비
# 모델(분류기) 실제로 만들기
model = Classifier()

# loss 함수 — 얼마나 틀렸나 계산
weights = compute_class_weight('balanced', classes=np.unique(y_train.numpy()), y=y_train.numpy())
weights = torch.FloatTensor(weights)
criterion = nn.CrossEntropyLoss(weight=weights)

# optimizer — 가중치를 어떻게 조정할지
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


# 6. 학습 루프

for epoch in range(config.EPOCHS):
    # 학습 모드
    model.train()

    # (a) 기울기 초기화
    optimizer.zero_grad()

    # (b) 예측
    outputs = model(X_train)

    # (c) loss 계산 (얼마나 틀렸나)
    loss = criterion(outputs, y_train)

    # (d) 역전파 (틀린 정도를 거꾸로 계산)
    loss.backward()

    # (e) 가중치 갱신
    optimizer.step()

    # (f) 몇 epoch마다 상황 출력
    if (epoch + 1) % 5 == 0:
        # val 정확도 계산
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_pred = val_outputs.argmax(dim=1)
            val_acc = (val_pred == y_val).float().mean()
        print(f"Epoch {epoch+1}/{config.EPOCHS}  loss: {loss.item():.4f}  val_acc: {val_acc:.4f}")


# 7. 평가 (test 최종 정확도)

model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_pred = test_outputs.argmax(dim=1)

# torch 텐서 → numpy (sklearn이 numpy 먹음)
y_true = y_test.numpy()
y_pred = test_pred.numpy()

# 3가지 지표
wa = accuracy_score(y_true, y_pred)
ua = recall_score(y_true, y_pred, average='macro')
wf1 = f1_score(y_true, y_pred, average='weighted')

print(f"\n=== Test 결과 ===")
print(f"WA  (weighted accuracy): {wa:.4f}")
print(f"UA  (unweighted accuracy): {ua:.4f}")
print(f"WF1 (weighted F1): {wf1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))


# 8. 모델 저장
import os

os.makedirs(config.MODEL_DIR, exist_ok=True)   # 저장 폴더 없으면 만들기
save_path = f"{config.MODEL_DIR}/classifier.pt"
torch.save(model.state_dict(), save_path)
print(f"\n모델 저장 완료: {save_path}")