
import pandas as pd
from collections import Counter
import os


csv_path = "/Users/kimjungyeon/Desktop/대학원/감정 분류를 위한 대화 음성 데이터셋/5차년도_2차.csv"
df = pd.read_csv(csv_path, encoding="cp949")

print("행 수", len(df))
print("컬럼명", list(df.columns))

emo_cols = ['1번 감정', '2번 감정', '3번 감정', '4번 감정', '5번 감정']

def get_majority_emotion(row):

    votes = [row[col] for col in emo_cols if pd.notnull(row[col])]

    if not votes:
        return None

    counter = Counter(votes)
    most_common = counter.most_common()

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return None

    return most_common[0][0]

df['최종감정'] = df.apply(get_majority_emotion, axis=1)

before = len(df)
df = df.dropna(subset=['최종감정'])

label_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}

df['label'] = df['최종감정'].map(label_map)

unmapped = df['label'].isna().sum()
if unmapped > 0:
    print(f"매핑 안 된 행 {unmapped}개! 감정 이름 확인 필요")
    print("매핑 안 된 감정:", df[df['label'].isna()]['최종감정'].unique())

else:
    print("모든 감정이 매핑되었습니다.")

wav_dir = "/Users/kimjungyeon/Desktop/대학원/감정 분류를 위한 대화 음성 데이터셋/5차년도_2차"
df['wav_path'] = df['wav_id'].apply(lambda x: os.path.join(wav_dir, f"{x}.wav"))
before = len(df)
df['exists'] = df['wav_path'].apply(os.path.exists)
df = df[df['exists']]

print(f"\nwav 매칭: {before} → {len(df)} (파일 없는 {before - len(df)}개 제거)")

final_df = df[['wav_path', 'label']].copy()
save_path = "data/processed_5th_2nd.csv"
final_df.to_csv(save_path, index=False)
print(f"\n저장 완료: {save_path} ({len(final_df)}개)")