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
        print(f"Sum: {sum(numbers):.6f}")
        print(f"Mean Value: {avg:.6f}")
    else:
        print("Error")

extract_numbers_from_file('/workspaces/project/output/sis/similarity.txt')

