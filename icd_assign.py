import sys
import os 
from pydantic import BaseModel
from difflib import SequenceMatcher
import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from icd9 import ICD9
import time
import resource


def set_memory_limit(max_memory_gb):
    """
    Set a maximum memory usage limit (in GB) for the current process.
    """
    max_memory_bytes = max_memory_gb * 1024 ** 3
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, hard))

def main():

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    tree = ICD9('./codes_pretty_printed.json')

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    HF_TOKEN = ""


    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config = bnb_config,
        device_map = "auto",
        token=HF_TOKEN
    )


    def get_desc(candidate_codes):
        return_desc = []
        for code in candidate_codes:
            desc = tree.find(code).description
            return_desc.append(desc)
        return return_desc


    text_generator = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        max_new_tokens=100,
        return_full_text=False
    )

    def extract_codes_index(evaluations, codes, descs):
        predicted_codes = []
        inx = 0
        for eval in evaluations:
            if eval.judgement == 'Yes':
                #troviamo il testo più simile
                sim_array = []
                for desc in descs:
                    sim_array.append(SequenceMatcher(None, desc, eval.description).ratio())
                print(f"Added : {codes[sim_array.index(max(sim_array))]}")
                predicted_codes.append(codes[sim_array.index(max(sim_array))])
        return predicted_codes


    class ICD10Evaluation(BaseModel):
        description: str
        judgement: str  # "Yes" o "No"


    def extract_icd10_output(response_text):
        match = re.search(r'"(.+?)" \((Yes|No)\)', response_text)
        return match.groups() if match else (None, None)


    def evaluate_icd10(case_note: str, icd_descriptions: list):
        results = []

        for icd in icd_descriptions:
            prompt = f"""
            You are an expert medical assistant.
            Consider the following ICD-10 code description: "{icd}"
            Case note: "{case_note}"

            Your task is to determine if there is any mention of the ICD-10 code description in the case note.
            Follow this format **EXACTLY**:
            "<ICD-10 description>" (Yes|No)

            Example:
            "Fracture of femur" (Yes)
            "Diabetes mellitus" (No)

            Now respond for: "{icd}"
            """

            response = text_generator(prompt)
            response_text = response[0]["generated_text"].strip()
            # Estrarre i dati con regex
            description, judgement = extract_icd10_output(response_text)
            repeat = 1
            
            while description == None:
                repeat+=1
                if repeat >= 20:
                    return 'Destroy'
                response = text_generator(prompt)
                response_text = response[0]['generated_text'].strip()
                description, judgement = extract_icd10_output(response_text)
            
            while not description and judgement:
                description, judgement = extract_icd10_output(response_text)

            if description and judgement:
                try:
                    parsed_output = ICD10Evaluation(description=description, judgement=judgement)
                    results.append(parsed_output)
                except ValidationError as e:
                    print(f"Errore nella validazione Pydantic per '{icd}':", e)
            else:
                print(f"Errore nel parsing per '{icd}': Output non conforme - {response_text}")

        return results


    def assign_icd(case_note):
        
        i = 1
        assigned_codes = []
        parent_codes = []
        predicted_codes = []
        initial_codes = []
        candidate_codes = []
        
        toplevelnodes = tree.children
        candidate_codes = [node.code for node in toplevelnodes]
        code_desc = get_desc(candidate_codes)
        evaluations = evaluate_icd10(case_note, code_desc)
        if evaluations != 'Destroy':
            predicted_codes = extract_codes_index(evaluations, candidate_codes, code_desc)
        
        while True:
            if evaluations == 'Destroy':
                break
            for code in predicted_codes:
                    if len(tree.find(code).children) == 0:
                        print(code, " is a leaf, added.")
                        assigned_codes.append(code)
                    else:
                        parent_codes.append(code)
        
            if len(parent_codes) > 0:
                parent_code = parent_codes.pop(0)
                candidate_codes = [node.code for node in tree.find(parent_code).children]
            else:
                break
        
            code_desc = get_desc(candidate_codes)
            evaluations = evaluate_icd10(case_note, code_desc)
            if evaluations != 'Destroy':
                predicted_codes = extract_codes_index(evaluations, candidate_codes, code_desc)
            i = i+1
        
        print("Finished for this text")
        return assigned_codes


    # folder_path = './../data'


    # towrite = ""


    # for folder in os.listdir(folder_path):
    #     if os.path.isdir(folder_path + '/' + folder):
    #         for file_name in os.listdir(folder_path + '/' + folder):
    #             if not os.path.isdir(folder_path + '/' + folder + '/' + file_name):
    #                 with open(folder_path + '/' + folder + '/' + file_name, 'r', encoding='utf-8') as file:
    #                     content = file.read()
    #                     icd_codes = assign_icd(content)
    #                     towrite += folder + '/' + os.path.splitext(file_name)[0] + ' = ' + str(icd_codes) + '\n'


    data = []
    with open("./report_image_test.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))


    files = []
    currentS = data[0]['study_id']
    for sd in data:
        if currentS != sd['study_id']:
            currentS = sd['study_id']
            files.append('s' + str(sd['study_id']))


    folder_path = './../data'

    checkcount = 0
    countDone = 0
    towrite = ""


    for folder in os.listdir(folder_path):
        if os.path.isdir(folder_path + '/' + folder):
            for file_name in os.listdir(folder_path + '/' + folder):
                if not os.path.isdir(folder_path + '/' + folder + '/' + file_name):
                    if os.path.splitext(file_name)[0] in files:
                        countDone+=1
                        with open(folder_path + '/' + folder + '/' + file_name, 'r', encoding='utf-8') as file:
                            init_time = time.time()
                            content = file.read()
                            icd_codes = assign_icd(content)
                            towrite += folder + '/' + os.path.splitext(file_name)[0] + ' = ' + str(icd_codes) + '\n'
                            count += 1
                            checkcount += 1
                            finish_time = time.time()
                            with open('./output/timeInfo.txt', 'a') as file:
                                file.write(str(count) + "/" + str(len(files)) + " Elapsed time: " + str(finish_time - init_time) + "\n")
                                file.close()
                if checkcount == 20:
                    checkcount = 0
                    with open('./output/checkpoints/checkpoint' + str(time.time()) + '.txt', 'w') as file:
                        file.write('------- Checkpoint --------\n' + towrite)
                        file.close()

    with open('./output/result.txt', 'w') as file:
        file.write(towrite)
        file.close()

    

if __name__ == '__main__':
    MAX_RAM_MEMORY = 40
    try:
        set_memory_limit(MAX_RAM_MEMORY)
        main()
    except MemoryError:
        sys.stderr.write('Memory limit exceeded.\n')
        sys.exit(1)
 
