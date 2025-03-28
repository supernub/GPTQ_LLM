import re

def word_count_in_range(line, min_words=21, max_words=49):
    """
    判断行的单词数是否在 [min_words, max_words] 之间
    """
    words = line.split()
    count = len(words)
    return (min_words <= count <= max_words)

def looks_meaningful(line, min_alpha_ratio=0.7):
    """
    简易“有意义”检测：
      - 行中字母（含空格）占比 >= min_alpha_ratio
      - 排除全是大写或全是小写的极端情况 (可选)
      - 不全是符号/数字/网址等
    """
    stripped = line.strip()
    if not stripped:
        return False  # 空行直接排除

    # 计算字母 + 空格的占比
    alpha_space_count = sum(1 for c in stripped if c.isalpha() or c.isspace())
    ratio = alpha_space_count / len(stripped)

    if ratio < min_alpha_ratio:
        return False

    # 可进一步排除纯网址、纯数字或过多符号等
    # 例如简单判断是否几乎全是字母或数字：
    # if re.fullmatch(r'https?://\S+', stripped):
    #     return False

    return True

def main():
    input_file = "c4_en_short50.txt"  
    output_file = "sample_1000.txt"
    
    selected_lines = []
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if word_count_in_range(line, min_words=21, max_words=49) and looks_meaningful(line):
                selected_lines.append(line)
                if len(selected_lines) >= 1000:
                    # 满足条件的行达到100条，就可以提前停止
                    break

    # 把筛选好的行写到输出文件
    with open(output_file, "w", encoding="utf-8") as fout:
        for item in selected_lines:
            fout.write(item + "\n")

    print(f"共找到 {len(selected_lines)} 条符合条件的文本，已写入：{output_file}")

if __name__ == "__main__":
    main()
