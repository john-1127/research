sis_scores = {}

l = []
with open("./output/sis/qnn_pretrained.txt") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            sis_scores[parts[0]] = float(parts[1])
            l.append(float(parts[1]))

count = 0

for i in l:
    if i < 0.7:
        count += 1

print(count)

