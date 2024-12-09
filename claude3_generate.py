import json
import json
import ipdb
import glob
import os
import ipdb
from openai import OpenAI
import os
import json
import csv
import pandas as pd
import argparse
os.environ["OPENAI_API_KEY"]=''

client = OpenAI(
  api_key='',
  base_url=''
)
with open("amboss_doc_open_test.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

for x in data:
    response = client.chat.completions.create(
        model="claude-3-opus-20240229",
        messages=[
            {"role": "system", f"content": "You should act as a doctor now you should answer the following question."},
            {"role": "user", "content": f"{x['input']}\nProvide the correct answer. You should not only give me the correct option but also the reasons"}
        ]
    )
    result = response.choices[0].message.content
    x['claude3_answer'] = result
ipdb.set_trace()

with open(f"claude3_opus_reason.json", "w") as file:
    json.dump(data, file)

prompt = """
template: |
  Act as a USMLE evaluator, your role involves assessing and comparing a medical student's explanation to the provided target answer. Begin the assessment by carefully reviewing the provided target answer. Then, based on following specific criteria, determine the score for the student's answer.

  Evaluation Criteria:
  For each diagnosis, evaluate the medical student explanation base on the following three questions:
  1. Does the medical student's answer contain any evidence of incorrect reading comprehension? (indication the question has not been understood)
  2. Does the medical student's answer contain any evidence of incorrect reasoning steps? (incorrect rationale for answering the question)
  3. Does the medical student's answer contain any evidence of incorrect recall of knowledge? (mention of an irrelevant and/or incorrect fact for answering the question)
  Give a single score for each of these three questions. The score is selected from [1, 2, 3, 4, 5] (1=incomprehensible and incorrect, 5=clear and correct)

  Medical student's answer:
  {pred}

  Target answer:
  {target}
  
  Background Question:
  {question}

  Your evaluation should be provided in JSON format, as follows(don't generate any other information):
  {{"diagnosis 1": {{"question 1": "The score for question 1", "question 2": "The score for question 2", "question 3": "The score for question 3", }}, "diagnosis 2": "score for diagnosis 2 with the same format as diagnosis 1","diagnosis 3": "score for diagnosis 3 with the same format as diagnosis 1", "overall score": "the average score for diagnosis 1, 2, 3", "reason": "the reason why you give the score"}}

  Output:
"""
ipdb.set_trace()
for x in data:
    if 'claude3_rate' in x:
        continue
    pred = x['claude3_answer']
    target = x["output"]
    question = x["input"]
    text = prompt.replace('{pred}', pred)
    text = text.replace('{target}', target)
    text = text.replace('{question}', question)
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "user", f"content": text},
        ]
    )
    result = response.choices[0].message.content
    json_string = result.strip('```json\n')
    json_object = json.loads(json_string)
    x['claude3_rate_q1'] = json_object["diagnosis 1"]["question 1"]
    x['claude3_rate_q2'] = json_object["diagnosis 1"]["question 2"]
    x['claude3_rate_q3'] = json_object["diagnosis 1"]["question 3"]
    x['claude3_rate'] = json_object["overall score"]

score = 0
score_q1 = 0
score_q2 = 0
score_q3 = 0
score_num = 0
for x in data:
    if 'claude3_rate' in x:
        score_num += 1
        score += x['claude3_rate']
        score_q1 += x['claude3_rate_q1']
        score_q2 += x['claude3_rate_q2']
        score_q3 += x['claude3_rate_q3']

print(score / score_num)
print(score_q1 / score_num)
print(score_q2 / score_num)
print(score_q3 / score_num)