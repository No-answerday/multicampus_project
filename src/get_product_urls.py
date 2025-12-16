from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import time
import random
import urllib.parse


def get_product_urls(driver, keyword, max_products=5):
    encoded_keyword = urllib.parse.quote(keyword)
    # 초기 접속 (1페이지)
    search_url = (
        f"https://www.coupang.com/np/search?component=&q={encoded_keyword}&channel=user"
    )

    product_urls = []
    current_page = 1  # 현재 페이지 추적

    try:
        print(f"[get_urls] '{keyword}' 검색 페이지 접속 중...")
        driver.get(search_url)
        time.sleep(random.uniform(2.5, 3.5))

        while len(product_urls) < max_products:
            # 1. 페이지 로딩 대기 (상품 리스트)
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "product-list"))
                )
            except TimeoutException:
                print("   -> 상품 리스트를 찾을 수 없습니다. (로딩 지연 또는 없음)")
                break

            # 스크롤 조금 내려서 상품 로딩 유도
            driver.execute_script("window.scrollBy(0, 1500);")
            time.sleep(1.5)

            # 2. 현재 페이지 파싱
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            links = soup.select("ul#product-list > li > a")
            print(f"   -> {current_page}페이지 발견 상품 수: {len(links)}개")

            # 3. URL 수집
            for link in links:
                if len(product_urls) >= max_products:
                    break

                href = link.get("href")
                if not href or "javascript" in href or href == "#":
                    continue

                if href.startswith("/"):
                    full_url = f"https://www.coupang.com{href}"
                else:
                    full_url = href

                # 중복 방지 (이미 담은 URL이면 건너뜀)
                if full_url not in product_urls:
                    product_urls.append(full_url)

            print(f"   -> 현재 확보된 URL: {len(product_urls)}/{max_products}개")

            # 4. 목표 달성 여부 확인 및 페이지 이동
            if len(product_urls) >= max_products:
                print("   -> 목표 수량을 채웠습니다.")
                break

            # 5. 다음 페이지 이동 로직
            next_page = current_page + 1
            print(f"   -> 수량이 부족하여 {next_page}페이지로 이동을 시도합니다.")

            try:
                next_btn_xpath = (
                    f"//div[contains(@class, 'Pagination')]//a[text()='{next_page}']"
                )

                next_btn = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.XPATH, next_btn_xpath))
                )

                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", next_btn
                )
                time.sleep(0.5)
                driver.execute_script("arguments[0].click();", next_btn)

                # 페이지 변경 대기
                current_page += 1
                time.sleep(random.uniform(3, 4))

            except (TimeoutException, NoSuchElementException):
                print(
                    "   -> 다음 페이지 버튼을 찾을 수 없거나 더 이상 페이지가 없습니다."
                )
                break
            except Exception as e:
                print(f"   -> 페이지 이동 중 에러: {e}")
                break

    except Exception as e:
        print(f"[get_urls] 에러 발생: {e}")

    return product_urls
