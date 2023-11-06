from transformers import AutoTokenizer
import transformers
import torch
import datasets
import csv
from sentence_transformers import CrossEncoder

# dataset = datasets.load_dataset("edmundtsou/conversation_keywords_test")


# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model = "meta-llama/Llama-2-7b-chat-hf" #"meta-llama/Llama-2-13b-chat-hf"
tokenStr = "higgingface-token"

tokenizer = AutoTokenizer.from_pretrained(model, token = tokenStr)
# model = AutoModelForCausalLM.from_pretrained(model, token = tokenStr)


pipeline = transformers.pipeline(
     "text-generation",
     model=model,
     torch_dtype=torch.float16,
     device_map="auto",
     token = tokenStr
)


system_message_1 = """You are an assistant helping a patient in a hospital to communicate with doctors and family members.
The patient will tell you a few keywords, and you will generate a complete sentence using those keywords. Give the generated sentence directly without explaining anything.
"""

system_message = """You are a patient in a hospital try to communicate with doctors and family members.
Generate a short, single sentence use the keywords provided as context. Make sure the provided keywords are used in the sentence and the sentence is short.
"""

system_message_0 = """You are an assistant helping generate sentence from given keywords. You will reply only the generated sentence and nothing else. 
For example, DO NOT say "Sure, here is a" """


def removeInstructionText(sent):
    sent = sent[sent.rfind("[/INST]")+8:]
    sent = sent[sent.rfind('\n')+1:]
    sent = sent.strip().strip('"')
    return sent.strip()


# Specify the CSV file name
output_filename1 = 'output_sentences.new.k2-both.3.csv'
output_filename2 = 'output_sentences.new.k2-max.3.csv'
input_filename = './data/hospital_conv.2.csv'

data_list = []
with open(input_filename, 'r') as in_file:
    # Create a DictReader object
    reader = csv.DictReader(in_file)
    # Iterate over the rows in the CSV file
    for row in reader:
        # Each row is a dictionary with header keys and corresponding values
        data_list.append(row)

with open(output_filename1, 'w') as out_file1, open(output_filename2, 'w') as out_file2:
    
    writer1 = csv.writer(out_file1)
    writer2 = csv.writer(out_file2)

    ss_model = CrossEncoder('cross-encoder/stsb-roberta-base')

    for i, row in enumerate(data_list):
        k1 = row["1 Keyword"]
        k2 = row["2 Keywords"].strip().replace(" ", ", ")
        k3 = row["3 Keywords"].strip().replace(" ", ", ")
        emotion = row["Emotion"]
        ori_sentence = row["Sentence"]

        keywords = k2
    
        #full_prompt = f'[INST]<<SYS>>{system_message}<</SYS>>The conversation is about {ct}, and you feel {emotion}. Use these keywords ({k2}), you say:[/INST]'
        #full_prompt = f'[INST]<<SYS>>{system_message}<</SYS>>Generate a single short sentence using this keyword ({k1}) to communicate with doctors, nurses, or family members, when you feel {emotion}.[/INST]'
        full_prompt1 = f'[INST]<<SYS>>{system_message}<</SYS>>Generate a single short sentence using these keywords ({keywords}) to communicate with doctors, nurses, or family members.[/INST]'
        full_prompt2 = f'[INST]<<SYS>>{system_message}<</SYS>>Generate a short question using these keywords ({keywords}) to communicate with doctors, nurses, or family members.[/INST]'

        sequences1 = pipeline(
        full_prompt1,
        do_sample=True,
        top_k=10,
        temperature=0.15,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=120,
        )        
        
        for seq in sequences1:
            gen_sentence1 = removeInstructionText(seq['generated_text'])
            ss_score1 = ss_model.predict((ori_sentence, gen_sentence1))
            print(f'{keywords}, {ori_sentence}, {gen_sentence1}, {ss_score1}')
            print("-"*120)
            # print(f"Result: {seq['generated_text']}")

            writer1.writerow([keywords, ori_sentence, gen_sentence1, ss_score1])

        sequences2 = pipeline(
        full_prompt2,
        do_sample=True,
        top_k=10,
        temperature=0.15,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=120,
        )        
        
        for seq in sequences2:
            gen_sentence2 = removeInstructionText(seq['generated_text'])
            ss_score2 = ss_model.predict((ori_sentence, gen_sentence2))
            print(f'{keywords}, {ori_sentence}, {gen_sentence2}, {ss_score2}')
            print("-"*120)
            # print(f"Result: {seq['generated_text']}")

            writer1.writerow([keywords, ori_sentence, gen_sentence2, ss_score2])

        if (ss_score1 > ss_score2):
            writer2.writerow([keywords, ori_sentence, gen_sentence1, ss_score1])
        else:
            writer2.writerow([keywords, ori_sentence, gen_sentence2, ss_score2])

        if i % 10 == 0:
            out_file1.flush()
            out_file2.flush()
   

    
    

# while True:
#     keywords = input("Enter your keywords: ")

#     full_prompt = f'[INST]<<SYS>>{system_message}<</SYS>>Use these keywords ({keywords}), the generated sentence is:[/INST]'

#     ## Generate
#     # inputs = tokenizer(full_prompt, return_tensors="pt")
#     # generate_ids = model.generate(inputs.input_ids, max_length=200)
#     # output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

#     # print(output)
#     # print("========================`n")

#     sequences = pipeline(
#     full_prompt,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=200,
#     )
 
#     for seq in sequences:
#         print(f"Result: {seq['generated_text']}")
