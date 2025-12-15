import json
import time
import random
import undetected_chromedriver as uc
from get_product_urls import get_product_urls
from get_product_reviews import get_product_reviews


def main():
    KEYWORD = "사과"
    PRODUCT_LIMIT = 3
    REVIEW_TARGET = 30

    # 1. 브라우저 설정 및 실행 (여기서 한 번만!)
    print(">>> 브라우저를 실행합니다...")
    options = uc.ChromeOptions()
    options.add_argument("--no-first-run")
    options.add_argument("--no-service-autorun")
    options.add_argument("--password-store=basic")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--blink-settings=imagesEnabled=false")
    driver = uc.Chrome(options=options, use_subprocess=False)

    try:
        # 2. URL 수집 (driver 전달)
        print(f">>> [{KEYWORD}] 검색 시작...")
        urls = get_product_urls(driver, KEYWORD, max_products=PRODUCT_LIMIT)
        print(f">>> 수집된 URL: {len(urls)}개")

        all_results = []

        # 3. 리뷰 수집 (driver 전달)
        for idx, url in enumerate(urls):
            print(f"\n[{idx+1}/{len(urls)}] 상품 크롤링 중...")

            try:
                # 같은 브라우저 창을 계속 사용
                data = get_product_reviews(
                    driver, url, target_review_count=REVIEW_TARGET
                )

                if data["product_info"]:
                    all_results.append(data)
                    print(f"  -> 완료: 리뷰 {data['reviews']['count']}개")
                else:
                    print("  -> 실패")

                time.sleep(random.uniform(3, 5))

            except Exception as e:
                print(f"  -> 에러: {e}")

        # 저장
        if all_results:
            filename = f"result_{KEYWORD}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\n✅ 저장 완료: {filename}")

    finally:
        print(">>> 브라우저를 종료합니다.")
        driver.quit()


if __name__ == "__main__":
    main()
