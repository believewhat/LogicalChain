import json
import os
import re
def process_json_files(folder_path, output_file):
    combined_data = []
    def remove_first_percentage(text):
        return re.sub(r'\n\d+%','', text, count=1)
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)

            # Read the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Combine question and options into text
            question = data.get('question', '')
            options_text = '\n'.join([remove_first_percentage(option['option']) for option in data.get('options', [])])
            combined_text = question + '\n' + options_text

            # Extract answers and combine into output
            answers = [option['answer'] for option in data.get('options', [])]
            combined_output = '\n'.join(answers)

            # Add the combined text and output to the list
            combined_data.append({"text": combined_text, "output": f"Correct Answer: {remove_first_percentage(data['correct_answer'])}" + '\n' + combined_output})

    # Save the combined data to a JSON file
    with open(output_file, 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)

# Example usage
folder_path = '/home/jwang/Project/uptodata/amboss/results/amboss_result/d1_1'
output_file = 'amboss.json'
process_json_files(folder_path, output_file)
