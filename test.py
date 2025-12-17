import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random


def test_star_filter():
    # ---------------------------------------------------------
    # [설정] 테스트할 상품 URL (리뷰가 많은 상품으로 설정 권장)
    # ---------------------------------------------------------
    TARGET_URL = "https://www.coupang.com/vp/products/1298620713?itemId=2311599581&vendorItemId=70308370600&q=%EC%82%AC%EA%B3%BC&searchId=2bbbb4bd4088418&sourceType=search&itemsCount=36&searchRank=7&rank=7"
    # (위 URL은 예시입니다. 본인이 테스트하고 싶은 URL로 교체하세요.)

    print(f">>> [TEST] 브라우저 실행 중...")

    # 옵션 설정 (main.py와 동일하게 유지)
    options = uc.ChromeOptions()
    options.add_argument("--no-first-run")
    options.add_argument("--no-service-autorun")
    options.add_argument("--password-store=basic")
    options.add_argument("--window-size=1920,1080")
    # 테스트 시 눈으로 확인하기 위해 이미지 로딩 켜두는 게 좋을 수 있음 (선택사항)
    # options.add_argument("--blink-settings=imagesEnabled=false")

    driver = uc.Chrome(options=options, use_subprocess=False)
    time.sleep(2)
    try:
        print(f">>> [TEST] 페이지 접속: {TARGET_URL}")
        driver.get(TARGET_URL)
        print(">>> [TEST] 페이지 로딩 대기 중...")
        # 접속 직후 알림창 처리
        try:
            time.sleep(1)
            alert = driver.switch_to.alert
            print(f"   -> ⚠️ 경고창 감지: {alert.text}")
            alert.accept()
        except:
            pass

        time.sleep(3)

        # -------------------------------------------------------
        # 1. 리뷰 섹션으로 이동
        # -------------------------------------------------------
        print(">>> [TEST] 리뷰 섹션으로 스크롤 이동...")
        # 대략적인 위치로 스크롤
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight * 0.4);")
        time.sleep(1)

        try:
            review_section = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "sdpReview"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", review_section)
            # 헤더에 가려지지 않게 약간 위로 보정
            driver.execute_script("window.scrollBy(0, -200);")
            time.sleep(2)
            print("   -> 리뷰 섹션 발견 완료")
        except:
            print("   -> ❌ 리뷰 섹션을 찾을 수 없습니다. (테스트 종료)")
            return

        # -------------------------------------------------------
        # 2. 별점 필터 순회 테스트
        # -------------------------------------------------------
        STAR_RATINGS = [
            {"score": 5, "text": "최고"},
            {"score": 4, "text": "좋음"},
            {"score": 3, "text": "보통"},
            {"score": 2, "text": "별로"},
            {"score": 1, "text": "나쁨"},
        ]

        print("\n>>> [TEST] 별점 필터 클릭 테스트 시작")

        for star_info in STAR_RATINGS:
            target_text = star_info["text"]
            print(f"\n   ------------------------------------")
            print(f"   [목표] '{target_text}' 필터 클릭 시도")

            # A. 드롭다운 열기
            try:
                # 메인 코드와 동일한 XPath 사용
                print("드롭다운 열기 시도 시작")

                dropdown_trigger = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable(
                        (
                            By.XPATH,
                            "//div[contains(@class, 'twc-flex') and contains(@class, 'twc-items-center') and contains(@class, 'twc-cursor-pointer')]//div[contains(@class, 'twc-text-[14px]')]",
                        )
                    )
                )
                # 현재 선택된 텍스트 출력해보기
                print(f"     -> 현재 드롭다운 상태: {dropdown_trigger.text.strip()}")

                driver.execute_script("arguments[0].click();", dropdown_trigger)
                print("     -> 드롭다운 버튼 클릭 성공 (팝업 열림)")
                time.sleep(1)  # 애니메이션 대기

            except Exception as e:
                print(f"     -> ❌ 드롭다운 버튼 클릭 실패: {e}")
                continue

            # B. 팝업 내 옵션 선택
            try:
                # data-radix-popper-content-wrapper 내부에서 텍스트로 찾기
                # 더 넓은 범위로 검색
                option_xpath = f"//*[@data-radix-popper-content-wrapper]//*[text()='{target_text}']"

                print(f"     -> '{target_text}' 옵션 찾는 중...")

                star_option = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, option_xpath))
                )

                print(f"     -> '{target_text}' 옵션 발견!")

                # 클릭 전 스크롤 및 클릭
                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", star_option
                )
                time.sleep(0.5)  # 눈으로 확인하기 위한 짧은 대기
                driver.execute_script("arguments[0].click();", star_option)

                print(f"     -> 옵션 '{target_text}' 클릭 성공!")

                # C. 결과 확인을 위한 대기
                print("     -> (로딩 대기 중...)")
                time.sleep(3)

            except Exception as e:
                print(f"     -> ❌ 옵션 클릭 실패: {e}")
                # 팝업 닫기 시도
                try:
                    driver.execute_script("document.body.click();")
                except:
                    pass

        print("\n>>> [TEST] 모든 테스트 완료.")

    except Exception as e:
        print(f"\n>>> [TEST] 치명적 에러 발생: {e}")

    finally:
        print(">>> 5초 뒤 브라우저가 종료됩니다...")
        time.sleep(5)
        driver.quit()


if __name__ == "__main__":
    test_star_filter()
