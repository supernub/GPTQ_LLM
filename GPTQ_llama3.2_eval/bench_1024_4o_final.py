from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL
import json

def main():
    quant_path = "Llama-3.2-1B-Instruct-gptqmodel-4bit"

    print("Running lm-eval (ARC Challenge)...")
    lm_eval_results = GPTQModel.eval(
        quant_path,
        framework=EVAL.LM_EVAL,
        tasks=[EVAL.LM_EVAL.ARC_CHALLENGE]
    )
    with open("lm-eval_result.json", "w") as f:
        json.dump(lm_eval_results, f, indent=2, default=str)  # ← 加 default=str

    print("Running evalplus (HUMAN)...")
    evalplus_results = GPTQModel.eval(
        quant_path,
        framework=EVAL.EVALPLUS,
        tasks=[EVAL.EVALPLUS.HUMAN]
    )
    with open("evalplus_result.json", "w") as f:
        json.dump(evalplus_results, f, indent=2, default=str)  # ← 加 default=str

if __name__ == "__main__":
    main()
