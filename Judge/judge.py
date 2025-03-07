import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Replace with your actual model path

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

def generate_response(prompt, max_tokens=256, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Simple split to isolate the newly generated text from the prompt
    if prompt in decoded:
        response = decoded.split(prompt)[-1].strip()
    else:
        response = decoded
    return response

def extract_key_points(text, text_label="Original Text"):
    """
    Prompt the model to extract key points from a given text.
    """
    prompt = f"""
Please list the core information of the following {text_label} in bullet points.
Do not provide any extra analysis or conclusion, just the essential facts or events:

{text}

Key Points:
""".strip()
    return generate_response(prompt)

def judge_semantic_equivalence(keypointsA, keypointsB):
    """
    Prompt the model to compare two sets of key points and decide if they are semantically the same.
    """
    prompt = f"""
Here are two sets of key points from two texts. 
Please judge whether they express the same core meaning or not.

Key Points A:
{keypointsA}

Key Points B:
{keypointsB}

If they are essentially the same, answer "YES" and briefly explain.
If they differ significantly, answer "NO" and briefly explain.

Answer:
""".strip()

    return generate_response(prompt)

# Example usage
if __name__ == "__main__":
    original_text = """Toe was daar nie plek vir telling teen Ikeys | Netwerk24
Toe was daar nie plek vir telling teen Ikeys
Deur Stephen Nell 29 Maart 2016 00:00
Guy Alexander, Ikeys se agtsteman, soek ondersteuning van ’n spanmaat in gister se stryd teen Tuks. Foto: SASPA | Thys Lombard
Kaapstad. – ’n Hele paar groot geeste van Ikeys-rugby sou Maandag in hul grafte gedraai of hul koppe in skaamte laat hang het oor die nederlaag van 25-100 teen Tuks hier in die Kaapstad-stadion.
Die Ikeys sal nou hul status as Varsitybeker-span in ’n promosie-relegasie-uitspeelwedstryd moet verdedig. Te oordeel na Maandag se vertoning, is daar selfs teen swakker teenstand vir die Ikeys nie veel hoop nie.
Hulle het soms verdedig soos die skoner geslag in wedstryde wat laerskoolseuns en -meisies pouses teen mekaar speel.
Tuks verdien baie krediet vir ’n puik vertoning.
Die enigste kritiek is dat hulle in die tweede helfte ’n uiltjie gaan knip het nadat hulle met die omdraaislag reeds met 63-7 voor was.
Hul linkervleuel, Duhan van der Merwe, het ’n allemintige ses drieë gedruk en lyk soos ’n man wat op ’n hoër vlak sy man sal kan staan.
Dinge het elke keer gebeur wanneer hy in besit gestel is.
Sy eerste drie drieë is tussen die 12de en 22ste minuut in die eerste helfte aangeteken en die res in die laaste kwart.
Die grootste vernedering was nadat Marthinus de Beer se strafdoel in die doodsnikke die telbord laat aanrol het. Daar was egter nie plek vir drie syfers nie en die 1 moes bo twee nulle op die telbord pryk.
Vir die Ikeys het hul agtsteman en kaptein, Guy Alexander, en losskakel, Thomas Bednall, hard teen die oormag gespook.
Van ’n spanpoging was daar egter nie sprake nie – ’n hartseer stand van sake vir ’n universiteit wat twee seisoene gelede nog die Varsitybeker-kroon gedra het.
Duhan van der Merwe (7, 9, 9, 7, 7, 9), Andries Swanepoel (7), Ruan Steenkamp (5), Marthinus de Beer (5), Clyde Davids (9), Jan Enslin (5). Doelskoppe: Joshua Stander (12), De Beer (6). Strafdoele: De Beer (3). Ikeys 25 (7): Drieë: Mark Prior (7), Thomas Bednall (9, 9).
Meer oor: Varsitybeker | Ikeys"""
    modified_text = """Toe was daar geen plek vir 'n telling teen Ikeys | Netwerk24
Toe was daar geen plek vir 'n telling teen Ikeys
Deur Stephen Nell 29 Maart 2016 00:00
Kaapstad – Die Ikeys-rugbyspan het Maandag 'n swaar nederlaag van 25-100 teen Tuks in die Kaapstad-stadion gely. Hierdie uitslag beteken dat Ikeys hul Varsitybeker-status in 'n promosie-relegasie-wedstryd moet verdedig.
Ikeys se verdediging was swak, wat Tuks in staat gestel het om maklik drieë te druk. Tuks se linkervleuel, Duhan van der Merwe, het ses drieë gedruk en getoon dat hy op 'n hoër vlak kan meeding. Sy eerste drie drieë is tussen die 12de en 22ste minuut aangeteken, terwyl die res in die laaste kwart van die wedstryd gekom het.
Met die rustyd het Tuks reeds met 63-7 voorgeloop. In die tweede helfte het hulle effens verslap, maar die oorwinning was nooit in gevaar nie. Marthinus de Beer se strafdoel het in die laaste oomblikke die telbord tot 100 laat rol, maar die telbord kon nie drie syfers akkommodeer nie, en die "1" het bo die twee nulle gepryk.
Vir Ikeys het kaptein Guy Alexander en losskakel Thomas Bednall dapper probeer, maar die spanpoging was onvoldoende. Dit is 'n teleurstellende prestasie vir 'n universiteit wat net twee seisoene gelede die Varsitybeker gewen het.
Doele:Tuks: Duhan van der Merwe (6 drieë), Andries Swanepoel (1), Ruan Steenkamp (1), Marthinus de Beer (1), Clyde Davids (1), Jan Enslin (1). Doelskoppe: Joshua Stander (12), De Beer (6). Strafdoele: De Beer (3).
Ikeys: Drieë: Mark Prior (1), Thomas Bednall (2)."""

    # First round: get key points from each text
    keypoints_A = extract_key_points(original_text, text_label="Original Text")
    keypoints_B = extract_key_points(modified_text, text_label="Modified Text")

    print("=== Key Points A ===")
    print(keypoints_A)
    print("=== Key Points B ===")
    print(keypoints_B)

    # Second round: compare and decide
    final_judgment = judge_semantic_equivalence(keypoints_A, keypoints_B)
    print("=== Final Judgment ===")
    print(final_judgment)
