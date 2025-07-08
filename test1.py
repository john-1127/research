def extract_and_filter(file_path, output_path):
    results = []

    with open(file_path, "r") as f:
        for line in f:
            try:
                parts = line.strip().split(",")
                score = float(parts[-1])
                l1 = float(parts[2])
                if l1 < 0.5 and score > 0.9:
                    results.append(line.strip())  # 保留原始行
            except ValueError:
                continue

    # 將符合條件的寫入 output file
    with open(output_path, "w") as out_f:
        for line in results:
            out_f.write(line + "\n")

# 使用範例
input_file = "./log_3.txt"
output_file = "./log_4.txt"
extract_and_filter(input_file, output_file)

