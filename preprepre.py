import json
import re
file_path = 'amboss_test.json'

# 读取JSON数据
with open(file_path, 'r') as file:
    json_data = json.load(file)

# 准备材料的前缀
materials_prefix = "Here are some materials to help you to answer. Select the materials that you think it is useful."

# 遍历每个项目并修改
for item in json_data:
    # 对每个材料编号并加前缀
    materials = [f"Material {i + 1}: {mat}" for i, mat in enumerate(item['doc'][:10])]
    # 将材料合并成一个字符串，每个材料之间用空格隔开
    materials_str = '\n'.join(materials)
    # 将材料字符串添加到'input'的前面
    item['input_ir'] = materials_prefix + '\n' + materials_str + '\n\n' + item['input']
    item['output_option'] = re.findall('(Correct Answer: [A-Z]\n)', item['output'])[0]

with open('amboss_test2.json', 'w') as file:
    json.dump(json_data, file, indent=4)



file_path = 'amboss_train.json'

# 读取JSON数据
with open(file_path, 'r') as file:
    json_data = json.load(file)

# 准备材料的前缀
materials_prefix = "Here are some materials to help you to answer. Select the materials that you think it is useful."

# 遍历每个项目并修改
for item in json_data:
    # 对每个材料编号并加前缀
    materials = [f"Material {i + 1}: {mat}" for i, mat in enumerate(item['doc'][:10])]
    # 将材料合并成一个字符串，每个材料之间用空格隔开
    materials_str = '\n'.join(materials)
    # 将材料字符串添加到'input'的前面
    item['input_ir'] = materials_prefix + '\n' + materials_str + '\n\n' + item['input']
    item['output_option'] = re.findall('(Correct Answer: [A-Z]\n)', item['output'])[0]

with open('amboss_train2.json', 'w') as file:
    json.dump(json_data, file, indent=4)
import ipdb
ipdb.set_trace()