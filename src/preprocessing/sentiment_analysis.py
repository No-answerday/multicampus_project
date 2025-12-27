"""
감성 분석 및 단어 빈도 분석 유틸리티
"""

import math
from collections import Counter, defaultdict

# 피부 타입 키워드
SKIN_TYPES = {
    "건성": ["건성"],
    "지성": ["지성", "지성인"],
    "복합성": ["복합", "복합성"],
    "민감성": ["민감", "민감성"],
    "여드름성": ["여드름", "여드름성"],
}


def detect_skin_types(tokens):
    """토큰 리스트에서 피부 타입 감지"""
    token_set = set(tokens)
    found = []
    for skin, keys in SKIN_TYPES.items():
        if any(k in token_set for k in keys):
            found.append(skin)
    return found


def normalize_tfidf(tfidf):
    """TF-IDF 데이터를 딕셔너리로 정규화"""
    if isinstance(tfidf, dict):
        return {str(k): float(v) for k, v in tfidf.items()}
    if isinstance(tfidf, list):
        out = {}
        for item in tfidf:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                w, s = item
                out[str(w)] = float(s)
        return out
    return {}


def analyze_skin_type_frequency(reviews, top_n=20):
    """
    피부 타입별 단어 빈도 분석

    Args:
        reviews: 리뷰 리스트 (각 리뷰는 'tokens' 필드 포함)
        top_n: 상위 N개 단어

    Returns:
        dict: {피부타입: [(단어, 빈도), ...]}
    """
    skin_type_tokens = defaultdict(list)

    for review in reviews:
        tokens = review.get("tokens", [])
        if not isinstance(tokens, list) or not tokens:
            continue

        skin_types = detect_skin_types(tokens)
        for skin in skin_types:
            skin_type_tokens[skin].extend(tokens)

    result = {}
    for skin, all_tokens in skin_type_tokens.items():
        result[skin] = Counter(all_tokens).most_common(top_n)

    return result


def sentiment_tfidf_diff(reviews, top_n=30, min_doc_freq=20):
    """
    감성 특화 키워드 추출 (토큰 빈도 기반)

    score = (긍정 평균 빈도 - 부정 평균 빈도) * log1p(pos_n + neg_n)

    Args:
        reviews: 리뷰 리스트 (label, tokens 필드 포함)
        top_n: 추출할 키워드 개수
        min_doc_freq: 최소 문서 빈도

    Returns:
        tuple: (긍정 특화 키워드 리스트, 부정 특화 키워드 리스트)
    """
    pos_sum = defaultdict(float)
    pos_cnt = defaultdict(int)
    neg_sum = defaultdict(float)
    neg_cnt = defaultdict(int)

    for review in reviews:
        label = review.get("label")
        if label not in [0, 1]:
            continue

        # tokens에서 직접 빈도 계산
        tokens = review.get("tokens", [])
        if not isinstance(tokens, list) or not tokens:
            continue

        # 토큰별 빈도 계산 (TF)
        token_freq = Counter(tokens)
        total_tokens = len(tokens)

        if label == 1:
            for word, freq in token_freq.items():
                tf = freq / total_tokens if total_tokens > 0 else 0
                pos_sum[word] += tf
                pos_cnt[word] += 1
        else:
            for word, freq in token_freq.items():
                tf = freq / total_tokens if total_tokens > 0 else 0
                neg_sum[word] += tf
                neg_cnt[word] += 1

    if not pos_sum and not neg_sum:
        return [], []

    rows = []
    for w in set(pos_sum.keys()) | set(neg_sum.keys()):
        pc = pos_cnt.get(w, 0)
        nc = neg_cnt.get(w, 0)

        if max(pc, nc) < min_doc_freq:
            continue

        pos_mean = (pos_sum[w] / pc) if pc else 0.0
        neg_mean = (neg_sum[w] / nc) if nc else 0.0
        diff = pos_mean - neg_mean

        support = pc + nc
        score = diff * math.log1p(support)

        rows.append(
            {
                "word": w,
                "diff": diff,
                "pos_tfidf_mean": pos_mean,
                "neg_tfidf_mean": neg_mean,
                "pos_doc_count": pc,
                "neg_doc_count": nc,
                "support": support,
                "score": score,
            }
        )

    if not rows:
        return [], []

    # 점수 기준 정렬
    rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)

    positive_special = [
        {
            "word": r["word"],
            "diff": float(r["diff"]),
            "pos": float(r["pos_tfidf_mean"]),
            "neg": float(r["neg_tfidf_mean"]),
            "pos_n": int(r["pos_doc_count"]),
            "neg_n": int(r["neg_doc_count"]),
            "support": int(r["support"]),
            "score": float(r["score"]),
        }
        for r in rows_sorted[:top_n]
    ]

    negative_special = [
        {
            "word": r["word"],
            "diff": float(r["diff"]),
            "pos": float(r["pos_tfidf_mean"]),
            "neg": float(r["neg_tfidf_mean"]),
            "pos_n": int(r["pos_doc_count"]),
            "neg_n": int(r["neg_doc_count"]),
            "support": int(r["support"]),
            "score": float(r["score"]),
        }
        for r in reversed(rows_sorted[-top_n:])
    ]

    return positive_special, negative_special


def analyze_category_sentiment(products, top_n=30, min_doc_freq=20):
    """
    카테고리별 감성 키워드 분석

    Args:
        products: 상품 리스트
        top_n: 추출할 키워드 개수
        min_doc_freq: 최소 문서 빈도

    Returns:
        dict: 카테고리별 감성 키워드
    """
    all_reviews = []
    for product in products:
        reviews = product.get("reviews", {}).get("data", [])
        all_reviews.extend(reviews)

    positive_special, negative_special = sentiment_tfidf_diff(
        all_reviews, top_n=top_n, min_doc_freq=min_doc_freq
    )

    pos_count = sum(1 for r in all_reviews if r.get("label") == 1)
    neg_count = sum(1 for r in all_reviews if r.get("label") == 0)
    tokens_count = sum(
        1 for r in all_reviews if r.get("tokens") and isinstance(r.get("tokens"), list)
    )

    return {
        "positive_special": positive_special,
        "negative_special": negative_special,
        "review_count": len(all_reviews),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "tfidf_notna_count": tokens_count,
    }


def analyze_product_sentiment(product, top_n=30, min_doc_freq=5):
    """
    개별 상품의 감성 키워드 분석

    Args:
        product: 상품 데이터
        top_n: 추출할 키워드 개수
        min_doc_freq: 최소 문서 빈도 (상품별은 리뷰 수가 적을 수 있으므로 낮게 설정)

    Returns:
        dict: 상품별 감성 키워드
    """
    reviews = product.get("reviews", {}).get("data", [])

    positive_special, negative_special = sentiment_tfidf_diff(
        reviews, top_n=top_n, min_doc_freq=min_doc_freq
    )

    pos_count = sum(1 for r in reviews if r.get("label") == 1)
    neg_count = sum(1 for r in reviews if r.get("label") == 0)
    tokens_count = sum(
        1 for r in reviews if r.get("tokens") and isinstance(r.get("tokens"), list)
    )

    return {
        "positive_special": positive_special,
        "negative_special": negative_special,
        "review_count": len(reviews),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "tfidf_notna_count": tokens_count,
    }
