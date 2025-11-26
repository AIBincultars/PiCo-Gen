import os
import json
import argparse
import subprocess
import sys
import re
import signal
import shutil
import xml.etree.ElementTree as ET
from tqdm import tqdm
from multiprocessing import Pool

# =================配置区=================
DEFAULT_DATASET_ROOT = "raw/SymphonyNet_dataset/SymphonyNet_dataset"
OUTPUT_XML_ROOT = "raw/mxl/SymphonyNet_dataset"
MAPPING_FILE = "instrument_mapping_master.json"  # 确保路径正确
DEFAULT_TIMEOUT = 300

# Windows MuseScore 路径 (WSL格式)
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


def wsl_to_win_path(wsl_path):
    try:
        abs_path = os.path.abspath(wsl_path)
        result = subprocess.check_output(['wslpath', '-w', abs_path], text=True)
        return result.strip()
    except Exception:
        return None


# --- [核心修复] 移除旧的 get_target_instrument_name，新增以下函数 ---

def standardize_xml_parts(xml_path):
    """
    读取 MuseScore 生成的 XML，遍历每个 score-part。
    根据该 Part 内部保留的 midi-instrument/midi-program 信息，
    去查 GLOBAL_MAPPING，实现“每个声部独立命名”。
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        part_list = root.find('part-list')

        if part_list is None:
            return False

        changed = False

        for score_part in part_list.findall('score-part'):
            # 1. 获取该声部的 MIDI 信息
            midi_inst = score_part.find('midi-instrument')
            if midi_inst is None:
                continue

            channel_elem = midi_inst.find('midi-channel')
            program_elem = midi_inst.find('midi-program')

            mapping_key = None

            # 2. 确定 Mapping Key
            # A. 检查是否为打击乐 (Channel 10)
            if channel_elem is not None:
                try:
                    chan = int(channel_elem.text)
                    # MusicXML channel 通常是 1-based (10) 或 0-based (9)
                    if chan == 10 or chan == 9:
                        mapping_key = "DRUM"
                except:
                    pass

            # B. 如果不是打击乐，读取 Program Change
            if not mapping_key and program_elem is not None:
                try:
                    prog_id = int(program_elem.text)
                    # MuseScore 输出通常对应 MIDI (1-based)，所以可能需要 -1
                    # 先尝试 -1 (匹配 0-127 标准)
                    key_try = f"PROG_{prog_id - 1}"
                    if key_try in GLOBAL_MAPPING:
                        mapping_key = key_try
                    else:
                        # 备选：尝试不减 1
                        key_try = f"PROG_{prog_id}"
                        if key_try in GLOBAL_MAPPING:
                            mapping_key = key_try
                except:
                    pass

            # 3. 执行映射 (如果找到了对应的 Key)
            if mapping_key and mapping_key in GLOBAL_MAPPING:
                target_info = GLOBAL_MAPPING[mapping_key]
                target_name = target_info["target_xml_name"]  # e.g., "Violin"
                target_abbr = target_info["target_xml_abbr"]  # e.g., "Vln."

                # 修改 XML 中的 part-name
                pn = score_part.find('part-name')
                if pn is None:
                    pn = ET.SubElement(score_part, 'part-name')
                pn.text = target_name

                # 修改 XML 中的 part-abbreviation
                pa = score_part.find('part-abbreviation')
                if pa is None:
                    pa = ET.SubElement(score_part, 'part-abbreviation')
                pa.text = target_abbr

                changed = True

        if changed:
            tree.write(xml_path, encoding="UTF-8", xml_declaration=True)
            return True
        return True  # 即使没改动也算成功（可能是没匹配到）

    except Exception as e:
        return False


def process_file(file_info):
    src_path, rel_path, timeout_limit = file_info
    dest_path = os.path.join(OUTPUT_XML_ROOT, os.path.splitext(rel_path)[0] + ".xml")
    temp_dest_path = dest_path + ".tmp.xml"

    if os.path.exists(dest_path):
        return "skipped_exists"

    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_limit)

    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        # 1. 转换路径
        win_src_path = wsl_to_win_path(src_path)
        win_out_path = wsl_to_win_path(os.path.abspath(temp_dest_path))

        if not win_src_path or not win_out_path:
            return "error_path_conversion"

        # 2. 调用 MuseScore 转换
        # 注意：不再预先读取 MIDI 分析 instrument，直接转！
        cmd = [MUSESCORE_CMD, "-o", win_out_path, win_src_path]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_limit - 10
        )

        if result.returncode != 0 or not os.path.exists(temp_dest_path):
            if os.path.exists(temp_dest_path): os.remove(temp_dest_path)
            return f"error_musescore_code_{result.returncode}"

        # 3. 【关键步骤】标准化 XML 中的声部名称
        success = standardize_xml_parts(temp_dest_path)

        if not success:
            if os.path.exists(temp_dest_path): os.remove(temp_dest_path)
            return "error_xml_standardize"

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
    global MUSESCORE_CMD

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--mscore_cmd", type=str, default=MUSESCORE_CMD)
    args = parser.parse_args()

    MUSESCORE_CMD = args.mscore_cmd

    if not os.path.exists(MUSESCORE_CMD):
        print(f"Error: MuseScore not found at '{MUSESCORE_CMD}'")
        return

    try:
        mapping = load_mapping()
    except Exception as e:
        print(f"Mapping error: {e}")
        print("Make sure you have run 'data/scan_instruments.py' first!")
        return

    print("Preparing file list...")
    tasks = []
    # 自动扫描所有子目录
    for root, _, files in os.walk(args.input_dir):
        for f in files:
            if f.lower().endswith(('.mid', '.midi')):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, args.input_dir)
                tasks.append((full_path, rel_path, args.timeout))

    print(f" Starting conversion for {len(tasks)} files...")

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