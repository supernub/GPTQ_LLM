import csv

def main():
    input_csv = "semantic_result.csv"
    count_ones = 0

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row["consistency"] == "1":
                count_ones += 1

    if total > 0:
        accuracy = count_ones / total
        print(f"accuracy: {accuracy:.2%}")
    else:
        print("no info")

if __name__ == "__main__":
    main()
