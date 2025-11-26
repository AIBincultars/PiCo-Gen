ORI_FOLDER = "raw/mxl/SymphonyNet_dataset"  # Replace with the path to your folder containing XML (.xml, .mxl, .musicxml) files
DES_FOLDER = "preprocessed/abc/SymphonyNet_dataset"   # The script will convert the musicxml files and output standard abc notation files to this folder

import os
import math
import random
import subprocess
from tqdm import tqdm
from multiprocessing import Pool


def convert_xml2abc(file_list):
    cmd = 'python xml2abc.py -d 8 -c 6 -x '
    for file in tqdm(file_list):
        filename = os.path.basename(file)
        os.makedirs(DES_FOLDER, exist_ok=True)

        try:
            p = subprocess.Popen(cmd + '"' + file + '"', stdout=subprocess.PIPE, shell=True)
            result = p.communicate()
            # 增加 errors='ignore' 增强容错
            output = result[0].decode('utf-8', errors='ignore')

            if output.strip() == '':
                with open("logs/xml2abc_error_log.txt", "a", encoding="utf-8") as f:
                    f.write(file + '\n')
                continue
            else:
                # ================= [新增：风格注入逻辑] =================
                # 1. 判断风格 (根据文件路径字符串)
                style_tag = None
                path_lower = file.replace("\\", "/").lower()

                if "classical" in path_lower:
                    style_tag = "Classical"
                elif "contemporary" in path_lower:
                    style_tag = "Contemporary"

                # 2. 如果识别出风格，注入 N: Style: 标签
                if style_tag:
                    lines = output.splitlines()
                    insert_idx = 0
                    # 寻找 X: 字段，插在其后（ABC标准建议）
                    for idx, line in enumerate(lines):
                        if line.startswith("X:"):
                            insert_idx = idx + 1
                            break

                    lines.insert(insert_idx, f"R: {style_tag}")
                    output = "\n".join(lines) + "\n"
                # ======================================================

                # 保持原有的写入逻辑
                with open(os.path.join(DES_FOLDER, filename.rsplit('.', 1)[0] + '.abc'), 'w', encoding='utf-8') as f:
                    f.write(output)

        except Exception as e:
            with open("logs/xml2abc_error_log.txt", "a", encoding="utf-8") as f:
                f.write(file + ' ' + str(e) + '\n')

if __name__ == '__main__':
    file_list = []
    os.makedirs("logs", exist_ok=True)

    # Traverse the specified folder for XML/MXL files
    for root, dirs, files in os.walk(os.path.abspath(ORI_FOLDER)):
        for file in files:
            if file.endswith((".mxl", ".xml", ".musicxml")):
                    # 排除临时文件
                if file.endswith(".tmp.xml"): continue
                filename = os.path.join(root, file).replace("\\", "/")
                file_list.append(filename)

        # Shuffle and prepare for multiprocessing
    random.shuffle(file_list)
    num_files = len(file_list)

        # 修复 range(0) 报错：增加空列表检查
    if num_files == 0:
        print(f"No files found in {os.path.abspath(ORI_FOLDER)}")
        exit()

    num_processes = max(1, os.cpu_count() - 2)

        # 修复切分逻辑，防止除零错误
    chunk_size = math.ceil(num_files / num_processes)
    file_lists = [file_list[i:i + chunk_size] for i in range(0, num_files, chunk_size)]

        # Create a pool for processing
    with Pool(processes=num_processes) as pool:
        pool.map(convert_xml2abc, file_lists)