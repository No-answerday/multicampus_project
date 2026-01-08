"""
감성 분석 모델 학습 스크립트

수정 사항:
  - ZeroDivisionError 방지 로직 추가 (데이터 0개일 때 중단)
  - 모델 저장 파일명 변경: logistic_regression_sentiment.joblib
  - NaN 값 처리를 위한 pd.notna() 도입
"""

import os
import pandas as pd
import numpy as np
import joblib
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt

# 한글 폰트 설정
from matplotlib import font_manager, rc
import platform

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":  # macOS
    plt.rc("font", family="AppleGothic")
else:  # 리눅스 (예: Google Colab, Ubuntu)
    plt.rc("font", family="NanumGothic")

plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지


def load_review_data(partitioned_reviews_dir):
    print("======================================================================")
    print(f"전처리된 리뷰 데이터 로드 중: {partitioned_reviews_dir}")

    # Hive 파티셔닝: category=*/data.parquet 패턴
    parquet_files = glob.glob(
        os.path.join(partitioned_reviews_dir, "category=*", "data.parquet")
    )

    if not parquet_files:
        print("[오류] Parquet 파일을 찾을 수 없습니다.")
        return []

    all_reviews = []
    for file_path in parquet_files:
        try:
            df = pd.read_parquet(file_path)
            category = os.path.basename(os.path.dirname(file_path)).replace(
                "category=", ""
            )
            print(f"  - {category}: {len(df):,}개 리뷰")
            all_reviews.extend(df.to_dict("records"))
        except Exception as e:
            print(f"파일 로드 오류: {file_path} - {e}")

    print(f"✓ 총 {len(all_reviews):,}개 리뷰 로드 완료")
    return all_reviews


def prepare_training_data(reviews):
    print("\n학습 데이터 준비 중...")
    X = []
    y = []

    # 데이터 샘플로 구조 확인 (상세하게)
    if reviews:
        sample = reviews[0]
        print(f"\n[데이터 구조 확인]")
        print(f"  - 전체 키: {list(sample.keys())}")
        print(f"  - label 존재: {'label' in sample}, 값: {sample.get('label')}")
        print(f"  - word2vec 존재: {'word2vec' in sample}")

        w2v_val = sample.get("word2vec")
        if w2v_val is not None:
            print(
                f"  - word2vec 타입: {type(w2v_val)}, 길이: {len(w2v_val) if hasattr(w2v_val, '__len__') else 'N/A'}"
            )
        else:
            print(f"  - word2vec 값: None")

        # 처음 10개 샘플에서 label과 word2vec 통계
        label_count = sum(1 for r in reviews[:100] if pd.notna(r.get("label")))
        w2v_count = sum(1 for r in reviews[:100] if r.get("word2vec") is not None)
        print(f"\n[처음 100개 샘플 확인]")
        print(f"  - label 있는 리뷰: {label_count}개")
        print(f"  - word2vec 있는 리뷰: {w2v_count}개")

    for review in reviews:
        w2v = review.get("word2vec")
        label = review.get("label")

        # label이 NaN이 아니며 존재하고, word2vec 벡터가 유효한 경우만 수집
        if pd.notna(label) and w2v is not None:
            if isinstance(w2v, (list, np.ndarray)) and len(w2v) > 0:
                X.append(w2v)
                y.append(int(label))

    data_count = len(y)

    if data_count == 0:
        print(f"✓ 학습 가능 데이터: 0개")
        print("[오류] 유효한 label과 word2vec 벡터가 없습니다.")
        print("전처리 파이프라인(src/preprocessing/main.py)을 확인하세요.")
        return np.array([]), np.array([])

    pos_count = sum(y)
    neg_count = data_count - pos_count

    print(f"✓ 학습 가능 데이터: {data_count:,}개")
    print(f"  - 긍정(1): {pos_count:,}개 ({pos_count / data_count * 100:.1f}%)")
    print(f"  - 부정(0): {neg_count:,}개 ({neg_count / data_count * 100:.1f}%)")

    return np.array(X), np.array(y)


def train_model(X_train, y_train):
    print("\n모델 학습 중...")
    # 클래스 불균형 대응을 위해 class_weight='balanced' 설정
    model = LogisticRegression(
        max_iter=1000, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    print("✓ 모델 학습 완료")
    return model


def evaluate_model(model, X_test, y_test, output_dir):
    print("\n모델 평가 중...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n정확도: {accuracy:.4f}")

    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred, target_names=["부정(0)", "긍정(1)"]))

    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["부정", "긍정"],
        yticklabels=["부정", "긍정"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("실제")
    plt.xlabel("예측")
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ 혼동 행렬 저장 완료: {cm_path}")


def main():
    print("=" * 70)
    print("감성 분석 모델 학습 (Logistic Regression)")
    print("=" * 70)

    # 경로 설정
    PROCESSED_DATA_DIR = "./data/processed_data"
    PARTITIONED_REVIEWS_DIR = os.path.join(PROCESSED_DATA_DIR, "partitioned_reviews")
    MODEL_OUTPUT_DIR = "./models"
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # 1. 데이터 로드
    reviews = load_review_data(PARTITIONED_REVIEWS_DIR)
    if not reviews:
        print("[중단] 로드된 리뷰 데이터가 없습니다.")
        return

    # 2. 학습 데이터 준비
    X, y = prepare_training_data(reviews)
    if X.size == 0:
        print("[중단] 학습 가능한 데이터가 없습니다.")
        return

    # 3. Train/Test 분할
    print("\n데이터 분할 중...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ 훈련: {len(X_train):,}개 / 테스트: {len(X_test):,}개")

    # 4. 모델 학습
    model = train_model(X_train, y_train)

    # 5. 모델 평가
    evaluate_model(model, X_test, y_test, MODEL_OUTPUT_DIR)

    # 6. 모델 저장 (파일명 변경 적용)
    model_path = os.path.join(MODEL_OUTPUT_DIR, "logistic_regression_sentiment.joblib")
    joblib.dump(model, model_path)
    print(f"\n✓ 모델 저장 완료: {model_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
