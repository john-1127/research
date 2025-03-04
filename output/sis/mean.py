import re

def extract_numbers_from_file(filename):
    numbers = []
    
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts:
                last_part = parts[-1]
                if re.match(r"^\d+(\.\d+)?$", last_part):
                    numbers.append(float(last_part))
    
    if numbers:
        avg = sum(numbers) / len(numbers)
        print(f"數值總和: {sum(numbers):.6f}")
        print(f"數值平均值: {avg:.6f}")
    else:
        print("未找到有效數字")

extract_numbers_from_file('/project/output/sis/sis_hybrid.txt')

