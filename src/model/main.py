"""
로지스틱 회귀 모델 학습 및 평가
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import joblib
import os

# 데이터 경로
PARQUET_PATH = "./data/processed_data/integrated_reviews_detail.parquet"
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_sentiment.pkl")


def load_and_prepare_data():
    """
    Parquet 파일에서 데이터 로드 및 전처리
    - label이 0 또는 1인 데이터만 사용
    - word2vec 벡터를 feature로 사용
    """
    print("데이터 로딩 중...")
    df = pd.read_parquet(PARQUET_PATH)

    # label이 0(부정) 또는 1(긍정)인 데이터만 필터링
    df = df[df["label"].isin([0, 1])].copy()

    # word2vec 벡터가 있는 데이터만 사용
    df = df[df["word2vec"].notna()].copy()

    print(f"전체 데이터: {len(df):,}개")
    print(f"긍정 리뷰: {(df['label'] == 1).sum():,}개")
    print(f"부정 리뷰: {(df['label'] == 0).sum():,}개")

    # word2vec 벡터를 numpy array로 변환
    X = np.array(df["word2vec"].tolist())
    y = df["label"].values

    return X, y, df


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """
    로지스틱 회귀 모델 학습 및 평가
    """
    print("\n로지스틱 회귀 모델 학습 중...")

    # 모델 생성 및 학습
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",  # 클래스 불균형 처리
        solver="lbfgs",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("모델 학습 완료!")

    # 학습 데이터 평가
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"\n[Train 성능]")
    print(f"정확도: {train_acc:.4f}")

    # 테스트 데이터 평가
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average="binary")
    test_recall = recall_score(y_test, y_pred, average="binary")
    test_f1 = f1_score(y_test, y_pred, average="binary")

    print(f"\n[Test 성능]")
    print(f"정확도: {test_acc:.4f}")
    print(f"정밀도: {test_precision:.4f}")
    print(f"재현율: {test_recall:.4f}")
    print(f"F1 점수: {test_f1:.4f}")

    # 상세 분류 리포트
    print("\n[분류 리포트]")
    print(classification_report(y_test, y_pred, target_names=["부정(0)", "긍정(1)"]))

    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred)
    print("\n[혼동 행렬]")
    print(f"                예측 부정  예측 긍정")
    print(f"실제 부정        {cm[0][0]:6d}    {cm[0][1]:6d}")
    print(f"실제 긍정        {cm[1][0]:6d}    {cm[1][1]:6d}")

    return model


def save_model(model):
    """모델 저장"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n모델 저장 완료: {MODEL_PATH}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("로지스틱 회귀 감성 분석 모델 학습")
    print("=" * 60)

    # 1. 데이터 로드 및 전처리
    X, y, df = load_and_prepare_data()

    # 2. Train/Test 분할 (8:2, stratify로 클래스 비율 유지)
    print("\n데이터 분할 중 (Train:Test = 8:2)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train 데이터: {len(X_train):,}개")
    print(f"Test 데이터: {len(X_test):,}개")
    print(
        f"Train 긍정 비율: {(y_train == 1).sum() / len(y_train):.2%}, "
        f"부정 비율: {(y_train == 0).sum() / len(y_train):.2%}"
    )
    print(
        f"Test 긍정 비율: {(y_test == 1).sum() / len(y_test):.2%}, "
        f"부정 비율: {(y_test == 0).sum() / len(y_test):.2%}"
    )

    # 3. 모델 학습 및 평가
    model = train_logistic_regression(X_train, y_train, X_test, y_test)

    # 4. 모델 저장
    save_model(model)

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
