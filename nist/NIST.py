import pandas as pd
import requests
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
import time
import requests
import os


def smiles_to_cas():
    with open("smiles_with_cas.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["smiles", "cas"])

        for i, smiles in enumerate(df["smiles"]):
            try:
                url = f"https://cactus.nci.nih.gov/chemical/structure/{smiles}/cas"
                response = requests.get(url, timeout=10)
                if response.status_code == 200 and response.text.strip():
                    cas = response.text.strip().split('\n')[0]
                else:
                    cas = "NA"
            except Exception:
                cas = "NA"

            writer.writerow([smiles, cas])
            print(f'{i}: {cas}')
            time.sleep(1)



def search_ir_spectrum_by_cas(cas_number: str, chromedriver_path: str, headless: bool = False):
    options = Options()
    if headless:
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
    service = Service(executable_path=os.path.expanduser(chromedriver_path))
    driver = webdriver.Chrome(service=service, options=options)

    # 2. 開啟 CAS 搜尋頁面
    driver.get("https://webbook.nist.gov/chemistry/cas-ser/")
    time.sleep(1)

    # 3. 勾選 “Gas phase IR spectrum”
    #    （如果頁面結構改動，需再調整 XPath）
    ir_cb = driver.find_element(
        By.XPATH,
        "//label[contains(normalize-space(.), 'IR spectrum')]/preceding-sibling::input[@type='checkbox']"
    )
    if not ir_cb.is_selected():
        ir_cb.click()
    time.sleep(0.3)
    
    gas_cb = driver.find_element(
        By.XPATH,
        "//label[contains(normalize-space(.), 'Gas phase')]/preceding-sibling::input[@type='checkbox']"
    )
    if not gas_cb.is_selected():
        gas_cb.click()
    time.sleep(0.3)
    # 4. 輸入 CAS 號
    cas_input = driver.find_element(By.ID, "ID")
    cas_input.clear()
    cas_input.send_keys(cas_number)
    time.sleep(0.3)

    # 5. 點擊 “Search” 按鈕送出
    submit_btn = driver.find_element(
        By.XPATH,
        "//input[@type='submit' and (@value='Search' or @value='CAS Search')]"
    )
    submit_btn.click()
    time.sleep(2)
    
    try:
        # 找到標題 "Information on this page:"
        info_section = driver.find_element(
            By.XPATH,
            "//strong[contains(text(), 'Information on this page')]/following-sibling::ul"
        )
        ir_link = info_section.find_element(By.PARTIAL_LINK_TEXT, "IR Spectrum")
        ir_link.click()

        try:
            spectrum_a = driver.find_element(
            By.XPATH,
            "//a[contains(@href,'JCAMP') and normalize-space(text())='spectrum']"
            )
            parent_p = spectrum_a.find_element(By.XPATH, "ancestor::p[1]")
            text = parent_p.text.strip()
            if "Download spectrum in JCAMP-DX format." in text:
                href = spectrum_a.get_attribute("href")
                if href.startswith("/"):
                    href = "https://webbook.nist.gov" + href
                print(href)

            return href

        except NoSuchElementException:
            return None

    except Exception as e:
        return None


    finally:
        driver.quit()


if __name__ == "__main__":
    start_time = time.time()
    CHROME_DRIVER = "./chromedriver/chromedriver"
    df = pd.read_csv("./test_cas_index.csv")
    CAS_NO = "1504-63-8"
    cas_test = "1137-68-4"
    cas_test1 = "479-45-8"

    with open("./final.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "cas", "href"])

        for i, row in df.iterrows():
            index = row["index"]
            cas = row["cas"]
            href = search_ir_spectrum_by_cas(cas, CHROME_DRIVER, headless=True)

            if href:
                resp = requests.get(href, timeout=15)
                if resp.status_code == 200:
                    out_path = f"./jcamp/{index}.jdx"
                    with open(out_path, "wb") as f:
                        f.write(resp.content)
                    writer.writerow([index, cas, href])

    end_time = time.time()
    print('Time: ', end_time-start_time)

    #
    # df = pd.read_csv("./test_cas.csv", keep_default_na=False)
    #
    # results = []
    # for index, row in df.iterrows():
    #     if row['cas'] != 'NA' and row['cas'] != '<!DOCTYPE html>':
    #
    #         results.append([index, row['cas']])
    #
    # df1 = pd.DataFrame(results, columns=["index", "cas"])
    # df1.to_csv("test_cas_index.csv", index=False)
