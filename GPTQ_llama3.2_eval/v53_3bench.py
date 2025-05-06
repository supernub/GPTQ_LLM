from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL


model_id = "Llama-3.2-1B-Instruct-gptq-v53-4bit"     

# ---------- lm‑eval ----------
lm_eval_results = GPTQModel.eval(
    model_id,
    framework=EVAL.LM_EVAL,
    tasks=[
        EVAL.LM_EVAL.ARC_CHALLENGE,
        EVAL.LM_EVAL.MMLU,
        EVAL.LM_EVAL.HELLASWAG,
    ],
)