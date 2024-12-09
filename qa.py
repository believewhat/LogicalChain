import os
import json
import re

def extract_evidence_and_rule(text):
    """
    使用正则表达式提取 Evidence 和 Rule 对，并将其合并为单条记录。
    """
    pattern = r"\d+\.\s*Evidence:\s*(.+?)\s*Rule:\s*(.+?)(?=\d+\.|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    combined_records = []
    for evidence, rule in matches:
        combined_text = f"Evidence: {evidence.strip()}  Rule: {rule.strip()}"
        combined_records.append({"evidence": evidence.strip(), "rule": rule.strip()})
    return combined_records

def generate_qa_for_rule(evidence, rule):
    """
    使用 ChatGPT 生成基于 Evidence 和 Rule 的 QA。
    """
    qa = {
        "rule": rule,
        "evidence": evidence,
    }
    return qa

def process_rules_from_folders(folders, output_dir):
    """
    读取文件夹中的 JSON 文件，提取规则并生成 QA，保存为 JSON。
    """
    os.makedirs(output_dir, exist_ok=True)
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist, skipping.")
            continue

        for filename in os.listdir(folder):
            if filename.endswith('.json'):
                filepath = os.path.join(folder, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                if "rule" in data:
                    extracted_rules = extract_evidence_and_rule(data["rule"])
                    qa_records = []
                    for record in extracted_rules:
                        evidence = record["evidence"]
                        rule = record["rule"]
                        qa = generate_qa_for_rule(evidence, rule)
                        qa_records.append(qa)

                    output_file = os.path.join(output_dir, f"qa_{filename}")
                    with open(output_file, 'w', encoding='utf-8') as out_file:
                        json.dump(qa_records, out_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 当前文件夹下的目标文件夹
    input_folders = ["pubmed_rules_json", "textbook_rules_json", "rules_json"]
    output_folder = "processed_rules"

    process_rules_from_folders(input_folders, output_folder)
    print(f"Processed QA files saved in {output_folder}")
