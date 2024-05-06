import os
import uuid
import requests
folder = r'/Users/lingoace/Desktop/paper'  # 首先定义文件夹的路径
with open('1012.txt','r') as reader:
    lines=reader.readlines()
for index,line in enumerate(lines):
    if index<16800:
        continue
    print(index)
    line=line.strip('\n').replace('|','').strip()
    if 'http' in line:
        houzhui=line.split('.')[-1]
        down_res = requests.get(url=line)
        name=str(uuid.uuid1())
        file_name=os.path.join(folder, name)+"."+houzhui
        with open(file_name,"wb") as code:
            code.write(down_res.content)

print('okokokok')