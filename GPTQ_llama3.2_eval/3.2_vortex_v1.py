from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

model_id = "ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"

# ---------- lm‑eval ----------
lm_eval_results = GPTQModel.eval(
    model_id,
    framework=EVAL.LM_EVAL,
    tasks=[
        EVAL.LM_EVAL.ARC_CHALLENGE,
        EVAL.LM_EVAL.GSM8K_COT,
        EVAL.LM_EVAL.MMLU,
    ]
)
