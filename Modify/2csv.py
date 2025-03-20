import re
import csv

def parse_samples(file_path):
    """
    从给定文件中解析出 `Sample X: ...` 形式的文本,
    返回 {样本编号: 对应文本} 的字典
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 用正则提取形如 "Sample 0: <内容>" 的段落
    pattern = re.compile(r"Sample\s+(\d+):\s*(.*?)(?=Sample\s+\d+:|$)", re.DOTALL)
    matches = pattern.findall(text)

    samples_dict = {}
    for idx_str, content in matches:
        idx = int(idx_str)
        content = content.strip()  # 去掉首尾空白
        samples_dict[idx] = content

    return samples_dict


def export_to_csv(samples_dict, output_csv):
    """
    将 {sample_id -> text} 的字典写入 CSV 文件。
    CSV 列为: sample_id, text
    """
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "text"])  # 表头

        for sample_id in sorted(samples_dict.keys()):
            text = samples_dict[sample_id]
            writer.writerow([sample_id, text])


def main():
    origin_file = "/mnt/data/origin_sample_384.txt"
    optimized_file = "/mnt/data/optimized_sample_384.txt"

    # 1. 解析 origin 文件，写入 origin.csv
    origin_samples = parse_samples(origin_file)
    export_to_csv(origin_samples, "origin_samples.csv")
    print("origin_samples.csv 文件已生成。")

    # 2. 解析 optimized 文件，写入 optimized.csv
    optimized_samples = parse_samples(optimized_file)
    export_to_csv(optimized_samples, "optimized_samples.csv")
    print("optimized_samples.csv 文件已生成。")


if __name__ == "__main__":
    main()
