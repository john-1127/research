total = 0
count = 0

with open("./output/sis/similarity.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            score = float(line.split()[-1])
            total += score
            count += 1
        except ValueError:
            print(f"跳過無法解析的行: {line}")

if count > 0:
    avg = total / count
    print(f"SIS 平均值: {avg:.6f}")
else:
    print("沒有可計算的 SIS 分數")

