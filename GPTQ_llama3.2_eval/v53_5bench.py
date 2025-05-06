
from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL
import sys, pathlib, json

MODEL_PATH = "Llama-3.2-1B-Instruct-gptq-v53-4bit"

def run_gptq_three(model_id: str, out_json: str):
    if not model_id:
        sys.exit("❌ model_id is None!  请检查 MODEL_PATH 是否填写正确。")

    print("→ Using model:", model_id)   # sanity‑check
    tasks = [
        EVAL.LM_EVAL.ARC_CHALLENGE,
        EVAL.LM_EVAL.MMLU,
        EVAL.LM_EVAL.HELLASWAG,
    ]

    GPTQModel.eval(
        model_id=model_id,
        framework=EVAL.LM_EVAL,
        tasks=tasks,
        output_file=out_json,
    )
    return pathlib.Path(out_json)

if __name__ == "__main__":
    if not pathlib.Path(MODEL_PATH).exists():
        sys.exit(f"❌ 模型目录不存在: {MODEL_PATH}")

    out_file = run_gptq_three(MODEL_PATH, "llama32_1b_v53_3bench.json")
    print("✓ 3‑bench finished →", out_file.resolve())
