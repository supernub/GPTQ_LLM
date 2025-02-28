from gptqmodel import GPTQModel
from gptqmodel.utils.eval import EVAL

model_id = "ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"

# 2. 使用 evalplus 评测框架
evalplus_results = GPTQModel.eval(
    model_id,
    framework=EVAL.EVALPLUS,
    tasks=[EVAL.EVALPLUS.HUMAN],
    output_file='evalplus_result.json'
)

print("=== [evalplus] Results ===")
print(evalplus_results)