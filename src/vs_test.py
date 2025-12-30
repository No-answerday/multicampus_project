import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from itertools import chain
import seaborn as sns
import random
import matplotlib.gridspec as gridspec

from matplotlib import rc
import platform

# 운영체제별 한글 폰트 설정
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
    plt.rcParams["axes.unicode_minus"] = False
    FONT_PATH = r"C:\WINDOWS\FONTS\MALGUNSL.TTF"
elif platform.system() == "Darwin":  # macOS
    plt.rc("font", family="AppleGothic")
    plt.rcParams["axes.unicode_minus"] = False
    FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
else:  # Linux
    plt.rc("font", family="NanumGothic")
    plt.rcParams["axes.unicode_minus"] = False
    FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# 파일 경로
DATA_DIR = "data/processed_data/선케어_태닝/선케어"

json_files = [
    os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".json")
]

print(f"총 파일 개수: {len(json_files)}")


# JSON 파일 통합
rows = []

for path in json_files:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print(f"불러오는 파일: {os.path.basename(path)}")

    for product in raw.get("data", []):
        p_info = product.get("product_info", {})
        reviews = product.get("reviews", {}).get("data", [])

        for r in reviews:
            rows.append(
                {
                    "source_file": os.path.basename(path),
                    # 상품
                    "product_id": p_info.get("product_id"),
                    "product_name": p_info.get("product_name"),
                    "category": p_info.get("category_path"),
                    "price": pd.to_numeric(p_info.get("price"), errors="coerce"),
                    # 리뷰
                    "score": r.get("score"),
                    "review_title": r.get("title", ""),
                    "review": r.get("content", ""),
                    "full_text": r.get("full_text", ""),
                    "helpful_count": r.get("helpful_count", 0),
                    "review_date": r.get("date"),
                    "has_image": int(r.get("has_image", False)),
                    # NLP 컬럼
                    "tokens": [],
                    "label": None,
                }
            )

df = pd.DataFrame(rows)

print("\n===== 데이터프레임 =====")
print(df.head())
print(df.info())


df["has_image"] = df["has_image"].fillna(0).astype(int)
df["helpful_count"] = df["helpful_count"].fillna(0).astype(int)
df["review_len"] = df["full_text"].astype(str).apply(len)
df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

print("검색 키워드:", raw.get("search_name"))
print("총 리뷰 수:", raw.get("total_collected_reviews"))
print("텍스트 리뷰 수:", raw.get("total_text_reviews"))
print("총 상품 수:", raw.get("total_product"))


# 리뷰 많은 상품 TOP 5
top_5_products = (
    df.groupby(["product_id", "product_name"])
    .size()
    .reset_index(name="review_count")
    .sort_values("review_count", ascending=False)
    .head(5)
)

print("\n===== 리뷰 많은 상품 TOP 5 =====")
print(top_5_products)

# 텍스트 있는 리뷰
df_text = df[df["review_len"] > 0].copy()
df_all = df.copy()

print("전체 리뷰 수:", len(df_all))
print("텍스트 리뷰 수:", len(df_text))


# 평점 분포
print("\n===== 평점 분포 =====")
print(df["score"].value_counts().sort_index())

# 리뷰 길이
print("\n===== 리뷰 길이 통계 =====")
print(df["review_len"].describe())


# 상품별 평균 평점
product_score = (
    df.groupby("product_name")
    .agg(
        mean_score=("score", "mean"),
        mean_helpful=("helpful_count", "mean"),
        review_count=("score", "count"),
    )
    .reset_index()
)

print("\n===== 상품별 평균 평점 & 평균 helpful_count =====")
print(product_score.head())

# 리뷰 수 TOP 10 상품
top_products = df["product_name"].value_counts().head(10).index


# 평점별 helpful_count
print("\n===== 평점별 helpful_count 통계 =====")
print(df.groupby("score")["helpful_count"].describe())

# 평점별 평균 리뷰 길이
print("\n===== 평점별 평균 리뷰 길이 =====")
print(df.groupby("score")["review_len"].mean())

# 평점별 리뷰 수 비율
print("\n===== 평점별 리뷰 수 비율 =====")
print(df["score"].value_counts(normalize=True).sort_index())


# 상품별 리뷰 수 분포
print("\n===== 상품별 리뷰 수 통계 =====")
print(df["product_id"].value_counts().describe())


# 상관계수
print("\n===== 상관계수 =====")
print("score - helpful_count :", df["score"].corr(df["helpful_count"]))
print("score - has_image :", df["score"].corr(df["has_image"]))

corr_product = product_score["mean_score"].corr(product_score["mean_helpful"])
print("상품 평균 평점 - 상품 평균 helpful_count :", corr_product)


# 시각화 1
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

df["score"].value_counts().sort_index().plot(kind="bar", ax=axes[0, 0])
axes[0, 0].set_title("평점 분포")

axes[0, 1].hist(df["review_len"], bins=50)
axes[0, 1].set_title("리뷰 길이 분포")

axes[0, 2].scatter(df["review_len"], df["helpful_count"], alpha=0.3)
axes[0, 2].set_xscale("log")
axes[0, 2].set_yscale("log")
axes[0, 2].set_title("리뷰 길이 vs Helpful Count")

sns.violinplot(x="score", y="review_len", data=df, ax=axes[1, 0])
axes[1, 0].set_title("평점별 리뷰 길이")

sns.boxplot(x="score", y="helpful_count", data=df, ax=axes[1, 1])
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("평점별 Helpful Count")

axes[1, 2].scatter(product_score["mean_score"], product_score["mean_helpful"])
axes[1, 2].set_title("상품 평균 평점 vs 평균 Helpful")

plt.tight_layout()
plt.show()


# 시각화 2
fig = plt.figure(figsize=(16, 8))
gs = gridspec.GridSpec(2, 3)

ax1 = fig.add_subplot(gs[0, 0])
df[df["product_name"].isin(top_products)].groupby("product_name")[
    "score"
].mean().sort_values().plot(kind="barh", ax=ax1)
ax1.set_title("TOP 10 상품 평균 평점")

ax2 = fig.add_subplot(gs[0, 2])
pivot = df[df["product_name"].isin(top_products)].pivot_table(
    index="product_name", columns="score", values="review", aggfunc="count"
)
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax2)

ax3 = fig.add_subplot(gs[1, :])
time_score = (
    df.dropna(subset=["review_date"])
    .set_index("review_date")
    .resample("ME")["score"]
    .mean()
)
time_score.plot(ax=ax3, linewidth=2)
ax3.set_title("월별 평균 평점 추이")

plt.tight_layout()
plt.show()


# ===== 워드클라우드(수정중) =====
