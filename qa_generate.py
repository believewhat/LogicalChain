import os
import json
import re
import sys

def extract_evidence_and_rule_from_text(text):
    """
    使用正则表达式提取 Evidence 和 Rule 对。
    """
    pattern = r"Evidence:\s*(.+?)\s*Rule:\s*(.+?)(?=\n\d+\.|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    return [{"evidence": evidence.strip(), "rule": rule.strip()} for evidence, rule in matches]

def process_json_files(json_dir, output_dir, start_index=0):
    """
    处理单个 JSON 文件目录中的所有 JSON 文件，并保存为单独的 JSON 文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]
    total_processed = 0

    for i, file_path in enumerate(files):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                if "rule" in data and data['rule']:
                    rule_text = data["rule"]
                    extracted_records = extract_evidence_and_rule_from_text(rule_text)
                    for idx, record in enumerate(extracted_records):
                        output_file = os.path.join(output_dir, f"{start_index + total_processed}.json")
                        with open(output_file, "w") as json_file:
                            json.dump(record, json_file, indent=4)
                        total_processed += 1
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path}")

    print(f"Processed {total_processed} records from {json_dir} into {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_json_directory> <output_directory>")
        sys.exit(1)

    input_json_directory = sys.argv[1]
    output_directory = sys.argv[2]

    process_json_files(input_json_directory, output_directory)
