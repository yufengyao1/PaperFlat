'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-10-13 11:22:38
LastEditors: yufengyao yufegnyao1@gmail.com
LastEditTime: 2023-10-28 12:32:22
FilePath: \PFLD-MINE\rename.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import uuid
folder = r'C:\Users\yufen\Desktop\tmpimg\back'  # 首先定义文件夹的路径
file_names = os.listdir(folder)  # 创建一个所有文件名的列表

for name in file_names:
    try:
        new_name=str(uuid.uuid4())+".jpg"
        os.rename(os.path.join(folder, name), os.path.join(folder, new_name))  # 执行重命名
    except:
        # os.remove(os.path.join(folder, name))
        continue

print('ok')