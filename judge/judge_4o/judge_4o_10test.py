
import csv
import os
import time
from openai import OpenAI   

# ========= 配置 =========
MODEL_NAME   = "gpt-4o"            
FEWSHOT_FILE = "fewshot.txt"
INPUT_CSV    = "../modify/output_4o_10test.csv"
OUTPUT_CSV   = "judge_4o_10test_result.csv"
RATE_LIMIT_S = 1.2                 

client = OpenAI(api_key="your-api-key")

def load_fewshot_prompt(path: str) -> str:
    """ few‑shot """
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def generate_response(prompt: str,
                      max_tokens: int = 8,
                      temperature: float = 0.0) -> str:
    """ GPT‑4o"""
    resp = client.chat.completions.create(
        model       = MODEL_NAME,
        messages    = [{"role": "user", "content": prompt}],
        max_tokens  = max_tokens,
        temperature = temperature,
    )
    return resp.choices[0].message.content.strip()

def judge_semantic_equivalence(origin: str,
                               changed: str,
                               fewshot: str) -> str:
    """
    few‑shot + 两段文本 => 返回 'YES' or 'NO'
    """
    prompt = f"""{fewshot}

Original: {origin}
Changed: {changed}
Answer:"""
    return generate_response(prompt)

# ========= 主流程 =========
def main() -> None:
    fewshot_text = load_fewshot_prompt(FEWSHOT_FILE)

    # 读入
    with open(INPUT_CSV, encoding="utf-8") as fin:
        rows = list(csv.DictReader(fin))

    new_rows = []
    for idx, row in enumerate(rows, 1):
        origin_text   = row["Original"]
        changed_text  = row["Modified"]

        print(f"[{idx}] Judging…", end=" ")

        # GPT‑4o 判定
        try:
            judgment = judge_semantic_equivalence(origin_text,
                                                  changed_text,
                                                  fewshot_text)
            label = 1 if "YES" in judgment.upper() else 0
            print("✓" if label else "✗")
        except Exception as e:
            # 出错时写 -1 
            print(f"ERROR: {e}")
            label = -1

        row["label"] = str(label)
        new_rows.append(row)
        time.sleep(RATE_LIMIT_S)

    # 写出
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as fout:
        fieldnames = ["Original", "Modified", "Label"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"\nDone. Results saved to {OUTPUT_CSV}.")

if __name__ == "__main__":
    main()
