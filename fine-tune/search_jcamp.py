from rdkit import Chem
import subprocess
from bs4 import BeautifulSoup
import sys
import time

log_path = "./fine-tune/log.txt"
sys.stdout = open(log_path, "a", encoding="utf-8")
allowed_elements = {"C", "H", "O", "N", "Si", "P", "S", "F", "Cl", "Br", "I"}


def inchi_filter(inchi):
    mol = Chem.inchi.MolFromInchi(inchi) 
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in allowed_elements:
            return None
    smiles = Chem.MolToSmiles(mol)
    return smiles


if __name__ == '__main__':
    with open("./fine-tune/data_result_ex.csv", "a", encoding="utf-8") as result_file:
        with open("./fine-tune/multi_1.txt", "r", encoding="utf-8") as f:
            urls = [line.strip() for line in f if line.strip()]


        startIndex = 0
        for i, url in enumerate(urls[startIndex:], start=startIndex):
            print(f"處理第 {i+1} 筆:")
            

            # 用 wget 抓下來並覆蓋 result.txt
            subprocess.run(["wget", "-O", "./fine-tune/result.txt", url])

            # 讀 result.txt
            with open("./fine-tune/result.txt", "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f, "html.parser")

            # 找出所有 class="inchi-text"
            inchi_tags = soup.find_all("span", class_="inchi-text")

            if len(inchi_tags) >= 1:
                inchi = inchi_tags[0].text.strip()

            else:
                continue

            if not inchi.startswith("InChI="):
                print("Error: Not an InChI, Skip")
                continue
            
            smiles = inchi_filter(inchi)
            if smiles == None:
                continue
            
            ir_header = soup.find("h2", id="IR-Spec")
            if not ir_header:
                ir_header = soup.find("h2", id="IR-SPEC")
                if not ir_header:
                    print(f"Error 無有效 IR-SPEC 區塊 {url}")
                    continue

            if  ir_header.text.strip() == "Infrared Spectrum" or "IR Spectrum":
                link_tag = soup.find("a", href=lambda h: h and "JCAMP" in h and "Type=IR" in h)
                if link_tag:
                    download_url = "https://webbook.nist.gov" + link_tag["href"]
                    result_file.write(f"{smiles},{download_url}\n")
                    continue

                else:
                    content_tags = []
                    current = ir_header.find_next_sibling()
                    while current and current.name != "hr":
                        content_tags.append(current)
                        current = current.find_next_sibling()

                    sub_soup = BeautifulSoup("".join(str(tag) for tag in content_tags), "html.parser")

                    hrefs = []
                    for link in sub_soup.find_all("a", href=True):
                        if "ID=" in link["href"]:
                            href = "https://webbook.nist.gov" + link["href"]
                            text = link.text.strip()
                            hrefs.append((href, text))

                    if hrefs is None:
                        print(f"Error: href出現異常, {url}")

                    with open("./fine-tune/multi_2.txt", "a", encoding="utf-8") as multi_file:
                        for h, t in hrefs:
                            if "gas" in t.lower():
                                print(f"href: {t} -> {h}")
                                multi_file.write(h + "\n")
                                break

            time.sleep(0.2)
