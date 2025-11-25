import os
import json
import argparse
import subprocess
import sys
import re
import signal
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool

# =================配置区=================
DEFAULT_DATASET_ROOT = "raw/SymphonyNet_dataset/SymphonyNet_dataset"
OUTPUT_XML_ROOT = "raw/mxl/SymphonyNet_dataset"
MAPPING_FILE = "instrument_mapping_master.json"
DEFAULT_TIMEOUT = 300  # 给足时间

# Windows MuseScore 路径 (WSL格式)
# 请确保这是你刚才确认过的路径
MUSESCORE_CMD = "/mnt/c/Program Files/MuseScore 4/bin/MuseScore4.exe"
# =======================================

# 全局变量
GLOBAL_MAPPING = None


def load_mapping():
    if not os.path.exists(MAPPING_FILE):
        raise FileNotFoundError(f"Mapping file {MAPPING_FILE} not found!")
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def init_worker(mapping):
    global GLOBAL_MAPPING
    GLOBAL_MAPPING = mapping
    import warnings
    warnings.filterwarnings("ignore")


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError


# --- [核心修复] 路径转换函数 ---
def wsl_to_win_path(wsl_path):
    """
    调用 wslpath 工具，将 WSL 的 /mnt/d/... 转换为 Windows 的 D:\...
    """
    try:
        # 获取绝对路径
        abs_path = os.path.abspath(wsl_path)
        # 调用系统命令转换
        result = subprocess.check_output(['wslpath', '-w', abs_path], text=True)
        return result.strip()
    except Exception as e:
        return None


def get_target_instrument_name(midi_path):
    try:
        import mido
        mid = mido.MidiFile(midi_path, clip=True)
    except Exception:
        return None

    prog_counts = Counter()
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'program_change':
                if msg.channel == 9:
                    key = "DRUM_Channel_10"
                else:
                    key = f"PROG_{msg.program}"
                if key in GLOBAL_MAPPING:
                    prog_counts[key] += 1

    if not prog_counts:
        return "Piano"

    most_common_key = prog_counts.most_common(1)[0][0]
    return GLOBAL_MAPPING[most_common_key]["target_xml_name"]


def inject_name_into_xml(xml_path, target_name):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        part_list = root.find('part-list')
        if part_list is not None:
            for score_part in part_list.findall('score-part'):
                pn = score_part.find('part-name')
                if pn is not None:
                    pn.text = target_name
                else:
                    new_pn = ET.SubElement(score_part, 'part-name')
                    new_pn.text = target_name
                pa = score_part.find('part-abbreviation')
                if pa is not None:
                    pa.text = target_name[:3]
        tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
        return True
    except:
        return False


def process_file(file_info):
    src_path, rel_path, timeout_limit = file_info

    # 这里的路径操作依然使用 Linux 路径
    dest_path = os.path.join(OUTPUT_XML_ROOT, os.path.splitext(rel_path)[0] + ".xml")
    temp_dest_path = dest_path + ".tmp.xml"  # 加 .xml 后缀，MuseScore 对后缀敏感

    if os.path.exists(dest_path):
        return "skipped_exists"

    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_limit)

    try:
        # 1. 分析 (Python 读 Linux 路径，没问题)
        target_name = get_target_instrument_name(src_path)
        if not target_name:
            return "error_midi_read"

        # 2. 转换 (关键修复！)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # [修复点] 将 Linux 路径转换为 Windows 路径传给 EXE
        win_src_path = wsl_to_win_path(src_path)
        win_out_path = wsl_to_win_path(os.path.abspath(temp_dest_path))  # 必须是绝对路径才能转换

        if not win_src_path or not win_out_path:
            return "error_path_conversion"

        # 调用 Windows 程序，传入 Windows 路径
        cmd = [MUSESCORE_CMD, "-o", win_out_path, win_src_path]

        # 这里的 subprocess 是在 WSL 里跑，但它启动的是 Windows 进程
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_limit - 10
        )

        if result.returncode != 0:
            if os.path.exists(temp_dest_path): os.remove(temp_dest_path)
            return f"error_musescore_code_{result.returncode}"

        # 检查输出是否存在 (Python 检查 Linux 路径，因为它们指向同一个文件)
        if not os.path.exists(temp_dest_path):
            return "error_musescore_no_output"

        # 3. 注入 (Python 读 Linux 路径，没问题)
        success = inject_name_into_xml(temp_dest_path, target_name)
        if not success:
            if os.path.exists(temp_dest_path): os.remove(temp_dest_path)
            return "error_xml_inject"

        # 4. 完成
        os.rename(temp_dest_path, dest_path)
        return "success"

    except (TimeoutError, subprocess.TimeoutExpired):
        if os.path.exists(temp_dest_path): os.remove(temp_dest_path)
        return "timeout"
    except Exception as e:
        if os.path.exists(temp_dest_path): os.remove(temp_dest_path)
        return f"error_unknown: {str(e)}"
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def main():
    global MUSESCORE_CMD  # 声明全局变量

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--mscore_cmd", type=str, default=MUSESCORE_CMD)
    args = parser.parse_args()

    MUSESCORE_CMD = args.mscore_cmd

    # 验证 MuseScore (简单检查文件是否存在)
    if not os.path.exists(MUSESCORE_CMD):
        print(f"Error: MuseScore not found at '{MUSESCORE_CMD}'")
        print("Please check the path.")
        return

    try:
        mapping = load_mapping()
    except Exception as e:
        print(f"Mapping error: {e}")
        return

    print("Preparing file list...")
    tasks = []
    sub_dirs = ['classical', 'contemporary']

    for sub in sub_dirs:
        dir_path = os.path.join(args.input_dir, sub)
        if not os.path.exists(dir_path): continue

        for root, _, files in os.walk(dir_path):
            for f in files:
                if f.lower().endswith(('.mid', '.midi')):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, args.input_dir)
                    tasks.append((full_path, rel_path, args.timeout))

    print(f" Starting conversion for {len(tasks)} files...")
    print(f" Using: {MUSESCORE_CMD}")
    print(f" Workers: {args.workers}")

    stats = {"success": 0, "skipped": 0, "timeout": 0, "error": 0}
    error_log = open("convert_errors_mscore.log", "a", encoding="utf-8")

    with Pool(processes=args.workers, initializer=init_worker, initargs=(mapping,)) as pool:
        for res in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks)):
            if res == "success":
                stats["success"] += 1
            elif res == "skipped_exists":
                stats["skipped"] += 1
            elif "timeout" in res:
                stats["timeout"] += 1
                error_log.write(f"TIMEOUT: {res}\n")
            else:
                stats["error"] += 1
                error_log.write(f"FAIL: {res}\n")

            error_log.flush()

    error_log.close()
    print("\nSummary:", stats)


if __name__ == "__main__":
    main()