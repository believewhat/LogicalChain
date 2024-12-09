import json
import random
from openai import OpenAI
import os
os.environ["OPENAI_API_KEY"] = 'sk-5iaHigKdjpoE0iWP3002512eEeAa45DdB730B6004b5fAd99'

client = OpenAI(
    api_key='sk-5iaHigKdjpoE0iWP3002512eEeAa45DdB730B6004b5fAd99',
    base_url='https://pro.xiaoai.plus/v1',
    timeout=40
)
# 加载 usmle_train_doc.json 数据
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 保存修改后的数据
def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# 调用 GPT-4 API 来重写文档
def rephrase_with_gpt4(doc_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please rephrase the following text:\n\n{doc_text}",
                        },
                    ],
                }
            ],
            max_tokens=2000,
            temperature=0.0,
        )
        rephrased_text = response.choices[0].message.content
        return rephrased_text
    except Exception as e:
        print(f"Error in GPT-4 API call: {e}")
        return doc_text  # 如果失败，返回原始文本

# 处理数据
def process_data(data):
    # 抽取 2.5% 的 doc
    sample_size = max(1, int(0.05 * len(data)))
    sampled_indices = random.sample(range(len(data)), sample_size)

    for i, entry in enumerate(data):
        truncated_docs = entry['doc'][:3]  # 只保留前3个doc
        joined_docs = '\nReference:\n'.join(truncated_docs)  # 连接doc为字符串
        
        # 如果是被选中的2.5%的数据，进行rephrase
        if i in sampled_indices:
            rephrased_docs = rephrase_with_gpt4(joined_docs)
            entry['input'] = entry['input'] + '\nBioReference:'
            entry['output'] = rephrased_docs  # 修改output为rephrase后的doc
        
        # 更新doc为只保留前3个
        entry['doc'] = truncated_docs
        if i % 100 == 0:
            save_json(data, 'modified_usmle_train_doc.json')
    
    return data

def main():
    input_file = 'usmle_train_doc.json'
    output_file = 'modified_usmle_train_doc.json'

    # 加载数据
    data = load_json(input_file)

    # 处理数据，保留前3个doc，并对2.5%的数据修改output

    processed_data = process_data(data)

    # 保存结果
    save_json(processed_data, output_file)
    print(f"Modified data saved to {output_file}")

if __name__ == "__main__":
    main()
