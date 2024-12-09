import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path to the JSONL file
#jsonl_file = "/home/jwang/Project/doctorrobot/LongLoRA/open_guideline/open_guidelines.jsonl"
#jsonl_file = "/home/jwang/Project/doctorrobot/LongLoRA/open_guideline/textbook"
#jsonl_file_root = "/home/jwang/Project/doctorrobot/LongLoRA/open_guideline/textbook"
jsonl_file_root = "/data/experiment_data/junda/datasets--MedRAG--pubmed/snapshots/33da3593d5756bc04c8909f170003c0b14197957/chunk/"
jsonl_files = os.listdir(jsonl_file_root)
print(jsonl_files)
# Directory to save the individual JSON files
output_dir = "input_pubmed"
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

# Function to process and save a single JSON entry
def save_entry(index, line):
    try:
        # Parse the line as JSON
        entry = json.loads(line.strip())
        
        # Define the output file path, using the 'id' if available, or the index as filename
        output_file = os.path.join(output_dir, f"{entry.get('id', index)}.json")
        
        # Save the entry as a separate JSON file
        with open(output_file, 'w', encoding='utf-8') as out_f:
            json.dump(entry, out_f, ensure_ascii=False, indent=4)
        
        return f"Saved {output_file}"
    except Exception as e:
        return f"Failed to save entry {index}: {e}"

for jsonl_file in jsonl_files:
    # Process the JSONL file in parallel
    with open(os.path.join(jsonl_file_root, jsonl_file), 'r', encoding='utf-8') as f:
        lines = list(f)  # Read all lines in the JSONL file

    # Set the maximum number of workers for parallel processing
    max_workers = 64  # Adjust based on available resources

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(save_entry, i, line) for i, line in enumerate(lines)]
    
        # Track progress and handle results
        for future in as_completed(futures):
            result = future.result()
            print(result)
