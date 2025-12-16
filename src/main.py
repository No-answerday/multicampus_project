import json
import time
import random
import undetected_chromedriver as uc
from selenium.common.exceptions import (
    UnexpectedAlertPresentException,
    NoSuchWindowException,
    WebDriverException,
)
import gc

from get_product_urls import get_product_urls
from get_product_reviews import get_product_reviews


def main():
    start_time = time.time()
    KEYWORDS = ["사과"]
    PRODUCT_LIMIT = 2
    REVIEW_TARGET = 50

    print(">>> 전체 작업을 시작합니다...")

    try:
        for k_idx, keyword in enumerate(KEYWORDS):
            crawled_data_list = []
            top_category = ""
            keyword_total_collected = 0
            keyword_total_text = 0

            # ---------------------------------------------------------
            # [단계 1] URL 수집
            # ---------------------------------------------------------
            print(f"\n{'='*50}")
            print(f">>> [{k_idx+1}/{len(KEYWORDS)}] '{keyword}' URL 수집 시작")
            print(f"{'='*50}")

            options = uc.ChromeOptions()
            options.add_argument("--no-first-run")
            options.add_argument("--no-service-autorun")
            options.add_argument("--password-store=basic")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--blink-settings=imagesEnabled=false")

            driver = uc.Chrome(options=options, use_subprocess=False)
            try:
                urls = get_product_urls(driver, keyword, max_products=PRODUCT_LIMIT)
                print(f">>> [{keyword}] URL {len(urls)}개 확보 완료")
                time.sleep(2)
            except Exception as e:
                print(f">>> URL 수집 중 에러: {e}")
                urls = []
            finally:
                print(">>> URL 수집 브라우저 종료 및 메모리 정리 중...")
                try:
                    driver.quit()
                    print("driver 종료")
                    try:
                        del driver
                    except:
                        pass
                    gc.collect()
                    time.sleep(random.uniform(15, 16))  # URL 수집 후 충분한 대기
                except:
                    pass

            if not urls:
                print(f">>> [{keyword}] 수집된 URL이 없어 넘어갑니다.")
                continue

            # ---------------------------------------------------------
            # [단계 2] 개별 상품 리뷰 수집 (리뷰 수에 따라 드라이버 재사용/재시작)
            # ---------------------------------------------------------
            print(f">>> [{keyword}] 상세 리뷰 수집 시작")

            # 첫 상품을 위한 드라이버 생성
            driver = None

            for idx, url in enumerate(urls):
                print(f"\n   [{idx+1}/{len(urls)}] 상품 처리 시작... ({keyword})")

                MAX_RETRIES = 2
                success = False

                for attempt in range(MAX_RETRIES):
                    print(
                        f"     -> [시도 {attempt+1}/{MAX_RETRIES}] 브라우저 실행 중..."
                    )

                    try:
                        # 드라이버가 없으면 새로 생성
                        if driver is None:
                            options = uc.ChromeOptions()
                            options.add_argument("--no-first-run")
                            options.add_argument("--no-service-autorun")
                            options.add_argument("--password-store=basic")
                            options.add_argument("--window-size=1920,1080")
                            options.add_argument("--blink-settings=imagesEnabled=false")
                            driver = uc.Chrome(options=options, use_subprocess=False)

                        # 수집 함수 호출
                        data = get_product_reviews(
                            driver, url, idx + 1, target_review_count=REVIEW_TARGET
                        )

                        if data and data.get("product_info"):
                            # 성공 데이터 처리
                            current_category = data["product_info"].get("category_path")
                            if not top_category and current_category:
                                top_category = current_category

                            r_data = data.get("reviews", {})
                            keyword_total_collected += r_data.get("total_count", 0)
                            keyword_total_text += r_data.get("text_count", 0)

                            crawled_data_list.append(data)
                            print(
                                f"     -> [성공] 수집 완료 (전체: {r_data.get('total_count', 0)}개)"
                            )
                            print(
                                f"     -> [성공] 수집 완료 (글 포함: {r_data.get('text_count', 0)}개)"
                            )
                            # 실제 수집한 리뷰 개수 확인
                            print(
                                f"     -> 수집한 리뷰 개수: {keyword_total_collected}개"
                            )

                            # 4500개 이상이면 드라이버 재시작, 아니면 유지
                            if keyword_total_collected >= 4500:
                                print(
                                    f"     -> 총 수집 리뷰 {keyword_total_collected}개 ≥ 4500 → 드라이버 재시작"
                                )
                                try:
                                    driver.quit()
                                    print("driver 종료")
                                except:
                                    pass
                                driver = None
                            else:
                                print(
                                    f"     -> 총 수집 리뷰 {keyword_total_collected}개 < 4500 → 드라이버 유지"
                                )

                            success = True
                            break
                        else:
                            print("     -> [실패] 데이터가 비어있습니다. 재시도합니다.")
                            # 드라이버 재시작
                            try:
                                driver.quit()
                                print("driver 종료")
                            except:
                                pass
                            driver = None

                    except Exception as e:
                        print(f"     -> [에러 발생] {e}")
                        # 에러 발생 시 드라이버 재시작
                        if driver:
                            try:
                                driver.quit()
                                print("driver 종료")
                            except:
                                pass
                        driver = None

                        # 잠시 대기 후 재시도
                        print("     -> 20초 후 재시도합니다...")
                        time.sleep(20)
                        continue

                # 2번 다 실패했을 경우
                if not success:
                    print(
                        f"     -> [최종 실패] {MAX_RETRIES}번 시도했으나 수집 실패. 다음 상품으로 넘어갑니다."
                    )

                # 다음 상품 넘어가기 전 대기
                gc.collect()
                sleep_time = random.uniform(15, 16)
                print(f"     -> 다음 상품 대기 중... ({sleep_time:.1f}초)")
                time.sleep(sleep_time)

            # 키워드 처리 완료 후 드라이버가 남아있으면 종료
            if driver:
                print(f">>> [{keyword}] 모든 상품 처리 완료 - 드라이버 종료")
                try:
                    driver.quit()
                    print("driver 종료")
                except:
                    pass
                driver = None

            # ---------------------------------------------------------
            # [단계 3] 키워드 완료 후 저장
            # ---------------------------------------------------------
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
                print(f"\n [{keyword}] 저장 완료: {filename}")
            else:
                print(f"\n[{keyword}] 수집된 데이터가 없습니다.")

            long_sleep = random.uniform(10, 15)
            print(f">>> 다음 키워드 준비 중 ({long_sleep:.1f}초)...")
            time.sleep(long_sleep)

    except KeyboardInterrupt:
        print("\n>>> 사용자에 의해 작업이 중단되었습니다.")

    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        print(f"\n총 실행 시간: {hours}시간 {minutes}분 {seconds}초")


if __name__ == "__main__":
    main()
