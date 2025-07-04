from bs4 import BeautifulSoup


with open("./fine-tune/result.txt", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

with open("./fine-tune/molecule_list.txt", "a", encoding="utf-8") as out:
    links = soup.select("ol > li > a")
    for i, link in enumerate(links):
        href = link["href"]
        full_url = "https://webbook.nist.gov" + href
        # print(f"{i}: {full_url}")
        out.write(full_url+ "\n")
