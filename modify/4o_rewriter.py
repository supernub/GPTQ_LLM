import openai
import csv
import time


openai.api_key = ""  # Replace your actual key

input_txt = "../data/clean_10_test.txt"
output_csv = "output_gpt4o.csv"
sleep_between_requests = 1.2  # sec

# ============ PROMPT  ============

REWRITE_PROMPT = """
Please rewrite the following text according to the rewriting principles used in calibration datasets for quantized LLMs:

Rewriting goals:
- Preserve all important information (e.g., names, locations, products)
- Make the text fluent, grammatically correct, and natural sounding
- Avoid repetitive or template-like structures
- Promote diversity in phrasing and sentence construction
- Do not remove or skip any critical details from the original

Text to rewrite:
"{text}"
"""

# ============ GPT-4o  ============

def rewrite_text_with_gpt4o(original_text):
    prompt = REWRITE_PROMPT.format(text=original_text)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        full_reply = response.choices[0].message.content.strip()

        # Step 1: Remove leading "Sure!", "Here's a rewrite:" etc.
        for prefix in ["Sure!", "Certainly!", "Here's a rewrite:", "Rewrite:", "Rewritten text:"]:
            if full_reply.lower().startswith(prefix.lower()):
                full_reply = full_reply[len(prefix):].strip()

        # Step 2: Return the last non-empty line
        lines = [line.strip() for line in full_reply.split("\n") if line.strip()]
        return lines[-1] if lines else full_reply

    except Exception as e:
        print(f"Error for input: {original_text[:50]}... \n{e}")
        return "[ERROR]"

# ============ MAIN  ============

with open(input_txt, "r", encoding="utf-8") as infile, open(output_csv, "w", newline='', encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Original", "Modified", "Label"])

    for i, line in enumerate(infile):
        original = line.strip()
        if not original:
            continue

        print(f" [{i+1}] Rewriting: {original[:50]}...")
        rewritten = rewrite_text_with_gpt4o(original)

        writer.writerow([original, rewritten, "0"])
        time.sleep(sleep_between_requests)

print(f"\n Finished! Output saved to: {output_csv}")
