import re
import os
from openai import OpenAI
import json

client = OpenAI(
  api_key='sk-5iaHigKdjpoE0iWP3002512eEeAa45DdB730B6004b5fAd99',
  base_url='http://149.88.91.225:3000/v1',
)

prompt = '''
You have two tasks:
1. Judge whether the logical rule and evidence are valuable to generate very hard medical exam question. If not only return No.
2. You should construct a detailed QA problem based on a provided logical rule and evidence. The QA problem should include:

A realistic scenario involving a professional, medical, or ethical situation.
Clear options (A, B, C, etc.) for potential actions or answers.
A correct answer based on the logical rule and evidence provided.
A detailed explanation (Reason) that explains why the correct answer is the best choice.

Example Input
Logical Rule: "Excessive exercise in hotter temperatures -> Swelling of blood vessels in lower legs -> Condition: Golfer's vasculitis."
Evidence: "Golfer's vasculitis is a form of vasculitis (swelling of the blood vessels) experienced in the lower legs caused by excessive exercise in hotter temperatures."

Example Output
Scenario:
A 65-year-old man has recently taken up golfing as a way to stay active. Over the past few weeks, he has been playing golf for extended hours in the afternoons under the summer sun. After his last game, he noticed significant swelling and discomfort in his lower legs. Concerned, he visits his primary care physician. The doctor suspects that the patient may have Golfer's vasculitis and explains the condition to him.

Which of the following is the next best step for the patient to manage this condition?
A: Continue playing golf but take breaks in shaded areas.
B: Stop playing golf entirely to prevent further complications.
C: Reduce the amount of time spent playing golf and avoid playing during the hottest part of the day.
D: Seek emergency medical care to rule out serious vascular disease.
E: Apply ice and take over-the-counter pain medications after each game.

Correct Answer:
C: Reduce the amount of time spent playing golf and avoid playing during the hottest part of the day.

Reason:
The evidence indicates that Golfer's vasculitis is caused by excessive exercise in hotter temperatures. The logical rule suggests that avoiding excessive exercise during peak heat can reduce swelling and prevent further complications. While continuing to play golf with adjustments (C) is safe, options like stopping entirely (B) are unnecessary, and seeking emergency care (D) is not warranted unless symptoms worsen.


Now please geenrate the QA based on the following logical rule and evidence:
Logical Rule: {logical_chain}
Evidence: {evidence}
'''

def generate_qa(logical_rule, evidence):
    new_prompt = prompt.format(logical_chain=logical_rule, evidence=evidence)
    messages = [{"role": "user", "content": new_prompt}]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating QA: {e}")
        return "No"

# Function to process all JSON files in a directory
def process_json_file(file_path, output_dir):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    output = []
    for item in data:
        rule = item.get("rule")
        evidence = item.get("evidence")
        if not rule or not evidence:
            continue
        
        qa = generate_qa(rule, evidence)
        if qa.lower() != "no" and qa.lower() != "no." and len(qa) > 40:  # Filter out "No" results
            output.append({"rule": rule, "evidence": evidence, "qa": qa})
    
    # Save the filtered results
    if output:
        output_file = os.path.join(output_dir, os.path.basename(file_path))
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w') as out_f:
            json.dump(output, out_f, indent=4)

# Main execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python process_medical_json.py <input_json_file> <output_dir>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    process_json_file(input_file, output_dir)