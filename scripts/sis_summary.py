import re
import csv
import statistics
import sys
import os

def extract_summary(input_file, group_name, model_name, output_csv):
    numbers = []

    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts:
                last_part = parts[-1]
                if re.match(r"^\d+(\.\d+)?$", last_part):
                    numbers.append(round(float(last_part), 4))

    if numbers:
        mean_val = round(statistics.mean(numbers), 4)
        median_val = round(statistics.median(numbers), 4)
        q1 = round(statistics.quantiles(numbers, n=4)[0], 4)
        q3 = round(statistics.quantiles(numbers, n=4)[2], 4)

        print(f"Group: {group_name}, Model: {model_name}")
        print(f"Mean: {mean_val}, Q1: {q1}, Median: {median_val}, Q3: {q3}")

        file_exists = os.path.isfile(output_csv)
    #     with open(output_csv, 'a', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         if not file_exists:
    #             writer.writerow(['Group', 'Model', 'Mean', 'Q1', 'Median', 'Q3'])
    #         writer.writerow([group_name, model_name, mean_val, q1, median_val, q3])
    #     print(f"Saved summary to {output_csv}")
    # else:
    #     print("Error: No numbers found.")
    #
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('argv error')
    else:
        input_path = '/workspaces/project/output/sis/similarity.txt'
        group = sys.argv[1]
        model = sys.argv[2]
        output_path = sys.argv[3]
        extract_summary(input_path, group, model, output_path)
