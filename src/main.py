import json
import time
import random
import undetected_chromedriver as uc
from get_product_urls import get_product_urls
from get_product_reviews import get_product_reviews


def main():
    start_time = time.time()
    KEYWORDS = ["사과", "바나나", "오렌지", "배"]  # 검색할 키워드 리스트
    PRODUCT_LIMIT = 10  # 키워드 당 수집할 상품 수
    REVIEW_TARGET = 10000  # 상품 당 수집할 리뷰 수 (유효 리뷰 기준)

    print(">>> 브라우저를 실행")

    options = uc.ChromeOptions()
    options.add_argument("--no-first-run")
    options.add_argument("--no-service-autorun")
    options.add_argument("--password-store=basic")
    options.add_argument("--window-size=1920,1080")
    # options.add_argument("--blink-settings=imagesEnabled=false")

    driver = uc.Chrome(options=options, use_subprocess=False)

    try:
        for k_idx, keyword in enumerate(KEYWORDS):
            try:
                print(f"\n{'='*50}")
                print(f">>> [{k_idx+1}/{len(KEYWORDS)}] 키워드 검색 시작: {keyword}")
                print(f"{'='*50}")

                urls = get_product_urls(driver, keyword, max_products=PRODUCT_LIMIT)
                print(f">>> [{keyword}] 수집된 URL: {len(urls)}개")

                crawled_data_list = []
                top_category = ""

                keyword_total_collected = 0  # 전체 수집 개수 (빈 내용 포함)
                keyword_total_text = 0  # 글 있는 리뷰 개수

                for idx, url in enumerate(urls):
                    print(f"\n   [{idx+1}/{len(urls)}] 상품 크롤링 중 ({keyword})...")

                    try:
                        data = get_product_reviews(
                            driver, url, idx + 1, target_review_count=REVIEW_TARGET
                        )

                        if data["product_info"]:
                            current_category = data["product_info"].get("category_path")
                            if not top_category and current_category:
                                top_category = current_category

                            r_data = data.get("reviews", {})
                            keyword_total_collected += r_data.get("total_count", 0)
                            keyword_total_text += r_data.get("text_count", 0)

                            crawled_data_list.append(data)
                            print(
                                f"     -> [완료] 전체: {r_data.get('total_count')}개 / 글있음: {r_data.get('text_count')}개"
                            )
                        else:
                            print("     -> [실패] 데이터 없음")

                        sleep_time = random.uniform(3, 4)
                        print(f"     -> {sleep_time:.1f}초 대기 중...")
                        time.sleep(sleep_time)

                    except Exception as e:
                        print(f"     -> [에러] {e}")

                result_json = {
                    "search_name": keyword,
                    "category": top_category,
                    "total_collected_reviews": keyword_total_collected,
                    "total_text_reviews": keyword_total_text,
                    "data": crawled_data_list,
                }

                if crawled_data_list:
                    filename = f"result_{keyword}.json"
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(result_json, f, indent=2, ensure_ascii=False)
                    print(f"\n[{keyword}] 저장 완료: {filename}")
                else:
                    print(f"\n[{keyword}] 수집된 데이터가 없습니다.")

                long_sleep = random.uniform(10, 12)
                print(f">>> 키워드 변경 대기 중 ({long_sleep:.1f}초)...")
                time.sleep(long_sleep)

            except Exception as e:
                print(f"\n!!! [{keyword}] 루프 에러: {e}")
                continue

    finally:
        print("\n>>> 브라우저를 종료합니다.")
        driver.quit()
        end_time = time.time()
        elapsed_time = end_time - start_time

        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print(f"\n총 실행 시간: {hours}시간 {minutes}분 {seconds}초")


if __name__ == "__main__":
    main()
