from openai import OpenAI
from tqdm import tqdm
import os
client = OpenAI(
  api_key='',
  base_url=''
)
def get_chat_completion_4o(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages, 
        temperature=0.1
    )
    return response.choices[0].message.content


gpt_prompt_4o = '''
The text below contains medical information. Your task is to create rules based on the medical knowledge presented, ensuring each rule comprehensively represents the information in the evidence text without omitting any significant details.

**Instructions**:
1. **Clear Distinction Between Conditions and Outcomes**: Each rule should clearly reflect when certain items are conditions, patient characteristics, or prerequisites (such as specific diagnoses, clinical findings, or circumstances) and when others are outcomes, actions, or recommendations. Always list conditions at the start of the rule, followed by the recommended actions or results.
2. **Multiple Rules for Complex Evidence**: If an evidence statement includes multiple distinct aspects, create separate rules for each part. This ensures that each concept is represented fully and clearly, preventing unrelated ideas from being merged into a single rule. Reuse the evidence as needed for each distinct rule.

3. **Parallel Items and Logical Structure**: Arrange items logically within each rule.
   - For items that have a natural sequence based on real-world logic (e.g., patient characteristics -> diagnostics -> treatment recommendations), follow this order.
   - If items are parallel without a strict sequence, indicate them at the same level within the rule without imposing order, such as "PET benefits -> Diagnosis, Staging, Treatment Response" if they all apply equally.

4. **Real-World Logical Order Over Sentence Structure**: When applicable, follow a natural sequence based on real-world relationships rather than sentence order. The preferred sequence generally follows:
   - **Conditions or Requirements** (e.g., patient characteristics or specific criteria)
   - **Diagnostics** (e.g., tests performed)
   - **Findings or Results** (e.g., outcomes of diagnostics)
   - **Treatment Recommendations or Actions** (e.g., proposed treatments based on findings)
   - **Follow-Up or Monitoring Plans** (if applicable)

   Present these elements in this logical order, regardless of how they are arranged in the original text.

5. **Thorough Coverage to Reconstruct Guideline**: Ensure that the generated set of evidence and rules covers the guideline comprehensively, allowing the guideline to be reconstructed accurately from these parts. Each evidence should be presented as close to the original guideline text as possible, using exact phrasing where feasible.

6. **Include Every Key Detail**: Represent every significant term, phrase, or concept from the evidence. 

7. **Logical Flow Reflecting Real-World Context, Not Sentence Order**: Arrange items based on their real-world logical relationships, not the sentence structure. Avoid linking items simply because they are close in the sentence; instead, show their true context and relationship based on medical reasoning.

8. **Concise but Complete Representation**: Each rule item should be as concise as possible while fully representing all critical details. Avoid unnecessary redundancy but ensure all essential information is included.

9. **Direct Language Without Conditionals**: Use clear and direct language without excessive use of conditionals like “if” or “when.” Each rule should be logically complete and self-contained, conveying the full context of the information.

**Key Terms to Include**: {key_words}

**Medical Text**: {medical_text}

**Example Rule Format**:
- **Correct**:
   - **Evidence**: "A patient presents with abdominal pain, followed by a CT scan, which shows an abscess. Treatment includes antibiotics."
   - **Rule**: Abdominal pain -> CT scan -> Abscess detected -> Treatment: Antibiotics.

- **Incorrect** (based purely on sentence order):
   - **Evidence**: "A patient presents with abdominal pain, followed by a CT scan, which shows an abscess. Treatment includes antibiotics."
   - **Rule**: Abdominal pain -> Abscess detected -> CT scan -> Treatment: Antibiotics.

**Return Format**:
1. Evidence:
   Rule:
2. Evidence:
   Rule:

Return only the specified output format without additional commentary or explanation.
'''

import json
import sys
import medspacy
from medspacy.util import DEFAULT_PIPENAMES
from spacy.tokens import Span
import re
if len(sys.argv) < 3:
    print("Usage: python create_rules.py <json_file>")
    sys.exit(1)

# JSON file path passed as an argument
json_file = sys.argv[1]

# Load JSON data from the file
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)


def generate_filenames(filename):
    # 正则表达式匹配文件名中的数字
    match = re.search(r"(\d+)", filename)
    if not match:
        raise ValueError("文件名中未找到数字")
    
    # 提取数字并转换为整数
    number = int(match.group(1))
    
    # 生成新的数字
    number_plus_1 = number + 1
    number_minus_1 = number - 1
    
    # 替换原数字生成新的文件名
    filename_plus_1 = re.sub(r"(\d+)", str(number_plus_1), filename, 1)
    filename_minus_1 = re.sub(r"(\d+)", str(number_minus_1), filename, 1)
    
    return filename_plus_1, filename_minus_1


guideline_json = data.get("content", "")
new_json_file = generate_filenames(json_file)
if os.path.exists(new_json_file[0]):
    with open(new_json_file[0], 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    sentences = re.split(r'(?<=\.\s|\?\s|!\s)(?=\S)', data1["content"])
    guideline_json = sentences[-1] + guideline_json

if os.path.exists(new_json_file[1]):
    with open(new_json_file[1], 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    sentences = re.split(r'(?<=\.\s|\?\s|!\s)(?=\S)', data2["content"])
    guideline_json = guideline_json + sentences[0]

output_file_path = sys.argv[2]


# Setup medspacy (only done once per script execution)
medspacy_pipes = set(DEFAULT_PIPENAMES)
if 'medspacy_quickumls' not in medspacy_pipes:
    medspacy_pipes.add('medspacy_quickumls')
nlp = medspacy.load(enable=medspacy_pipes, quickumls_path='/home/jwang/Project/medspacy_test')

# Define types to include
tret_sem_ids = ['T059', 'T060', 'T061', 'T058', 'T056']  # test and treatment
symp_sem_ids = ['T184', 'T034', 'T037', 'T033']  # symptom
dise_sem_ids = ['T020', 'T019','T046', 'T047', 'T048', 'T191', 'T049', 'T050']  # disease
drug_sem_ids = ['T073', 'T074', 'T203', 'T075', 'T200', 'T192']  # drug
sust_sem_ids = ['T120', 'T121', 'T195', 'T122', 'T123', 'T125', 'T126', 'T127', 'T129', 'T130', 'T131', 'T104', 'T109', 'T114', 'T116', 'T197', 'T196', 'T168']  # substance
cuitypes_toinclude = tret_sem_ids + symp_sem_ids + dise_sem_ids + drug_sem_ids + sust_sem_ids

# Define function to process CUI codes
def cui_code(text):
    doc = nlp(text)
    dict_vis, cui_code, cui_entity = {}, {}, {}

    for entity in doc.ents:
        if any(ent in cuitypes_toinclude for ent in entity._.semtypes):
            dict_vis[str(entity).lower()] = 1
            cui_code[entity.label_] = str(entity).lower()
            cui_entity[str(entity).lower()] = entity.label_

    return dict_vis, cui_code, cui_entity

# Process the guideline
def process_guideline(guideline_text, output_file_path):
    cui_word, _, _ = cui_code(guideline_text)
    new_prompt = gpt_prompt_4o.format(key_words=','.join(cui_word.keys()), medical_text=guideline_text)
    messages = [{"role": "user", "content": new_prompt}]
    rule = get_chat_completion_4o(messages)
    # Save to JSON file
    result = {"cui": list(cui_word.keys()), "rule": rule}
    with open(output_file_path, "w") as f:
        json.dump(result, f)


# Run the processing
process_guideline(guideline_json, output_file_path)
