import csv
import sys

csv.field_size_limit(sys.maxsize)

input_file = "./data/computed_data/train_full.csv"
output_file = "./data/computed_data/train_full_modified.csv"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8", newline="") as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)
    
    for i, row in enumerate(reader, start=1):
        if i == 1527:
            print(f"Skipping row {i}")
            continue
        writer.writerow(row)

print(f"Finished processing. Modified file saved as {output_file}.")

