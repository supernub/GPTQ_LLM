import csv

def main():
    input_csv = "judge_result.csv"
    total = 0
    count_ones = 0

    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row["label"] == "1":
                count_ones += 1

    if total == 0:
        print("num error")
        return

    proportion = count_ones / total
    print(f"文件: {input_csv}")
    print(f"共 {total} 条记录，语义一致的有 {count_ones} 条。")
    print(f"accuracy: {proportion:.2%}")

if __name__ == "__main__":
    main()
