# -*- coding: utf-8 -*-
ORI_FOLDER = 'preprocessed/SymphonyNet_dataset/abc'  # Replace with the path to your folder containing standard ABC notation files
INTERLEAVED_FOLDER = 'preprocessed/SymphonyNet_dataset/abc_interleaved'   # Output interleaved ABC notation files to this folder
AUGMENTED_FOLDER = 'preprocessed/SymphonyNet_dataset/abc_aug'   # Output key-augmented and rest-omitted ABC notation files to this folder
EVAL_SPLIT = 0.1    # The ratio of eval data

import os
import re
import json
import shutil
import random
from tqdm import tqdm
from abctoolkit.utils import (
    remove_information_field,
    remove_bar_no_annotations,
    Quote_re,
    Barlines,
    extract_metadata_and_parts,
    extract_global_and_local_metadata,
    extract_barline_and_bartext_dict)
from abctoolkit.convert import unidecode_abc_lines
from abctoolkit.rotate import rotate_abc
from abctoolkit.check import check_alignment_unrotated
from abctoolkit.transpose import Key2index, transpose_an_abc_text

os.makedirs(INTERLEAVED_FOLDER, exist_ok=True)
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)
for key in Key2index.keys():
    key_folder = os.path.join(AUGMENTED_FOLDER, key)
    os.makedirs(key_folder, exist_ok=True)

from instruction_generator import PromptGenerator, extract_abc_metadata, save_cache
# [新增] 导入“目标时长/小节”计算函数
from instruction_generator import get_target_duration_and_bars  # [新增]

# [新增] 并行与断点续转需要
import hashlib
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

# ====== 断点续转与失败日志配置 ======
RESUME_STATE_PATH = 'preprocessed/SymphonyNet_dataset/.resume_state.json'
FAILED_LOG = 'preprocessed/SymphonyNet_dataset/.failed.log'
SAVE_RESUME_EVERY = 100  # 每处理多少个 future 保存一次断点

prompt_gen = PromptGenerator()


# ======================= 工具函数（新增） =======================
def _append_failed_log(msg: str):
    """失败日志落盘"""
    os.makedirs(os.path.dirname(FAILED_LOG), exist_ok=True)
    with open(FAILED_LOG, 'a', encoding='utf-8') as f:
        f.write(msg.rstrip() + '\n')


def _load_resume() -> dict:
    """加载断点状态"""
    if not os.path.exists(RESUME_STATE_PATH):
        return {}
    try:
        with open(RESUME_STATE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        _append_failed_log(f"[RESUME_LOAD_FAIL] {e}\n{traceback.format_exc()}")
        return {}


def _save_resume_atomic(state: dict):
    """原子写断点状态，防止中断导致损坏"""
    try:
        os.makedirs(os.path.dirname(RESUME_STATE_PATH), exist_ok=True)
        tmp = RESUME_STATE_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, RESUME_STATE_PATH)
    except Exception as e:
        _append_failed_log(f"[RESUME_SAVE_FAIL] {e}\n{traceback.format_exc()}")


def _file_signature(abs_path: str, root: str) -> str:
    """
    计算文件签名（路径+mtime+size），当源文件变更时自动失效重做。
    如果你想更激进的命中率（不随 mtime 变化），可改成仅用相对路径做 md5。
    """
    try:
        st = os.stat(abs_path)
        rel = os.path.relpath(abs_path, start=root).replace('\\', '/')
        sig = f"{rel}|{int(st.st_mtime)}|{st.st_size}"
        return hashlib.md5(sig.encode('utf-8')).hexdigest()
    except Exception:
        return hashlib.md5(abs_path.encode('utf-8')).hexdigest()


def _remaining_keys_for_file(abc_path: str, ori_root: str, resume: dict) -> list:
    """
    计算该文件还需要处理的 keys（结合断点状态与磁盘已有产物）
    """
    abc_name = os.path.splitext(os.path.basename(abc_path))[0]
    file_id = _file_signature(abc_path, ori_root)
    done_keys = set(resume.get(file_id, {}).get('done_keys', []))

    remains = []
    for k in Key2index.keys():
        outp = os.path.join(AUGMENTED_FOLDER, k, f"{abc_name}_{k}.abc")
        if (k not in done_keys) and not (os.path.exists(outp) and os.path.getsize(outp) > 0):
            remains.append(k)
    return remains


def _resume_mark_done(source_path: str, ori_root: str, keys_done: list, resume: dict):
    """把该文件完成的 key 写入断点状态"""
    if not source_path or not keys_done:
        return
    file_id = _file_signature(source_path, ori_root)
    entry = resume.get(file_id, {"rel_path": os.path.relpath(source_path, start=ori_ROOT).replace('\\', '/'),
                                 "done_keys": []})
    s = set(entry.get("done_keys", []))
    s.update(keys_done)
    entry["done_keys"] = sorted(list(s))
    resume[file_id] = entry


# ======================= 主处理函数（仅小改） =======================
# [修改] 增加 allowed_keys 参数，以支持“只处理剩余 keys”
def abc_preprocess_pipeline(abc_path, allowed_keys=None):
    with open(abc_path, 'r', encoding='utf-8') as f:
        abc_lines = f.readlines()

    # ================= [插入] 智能指令生成逻辑 =================
    # 1. 提取元数据 (含乐器、风格、调性等)
    # 注意：这一步要在 remove_information_field 之前做，否则头部信息就被删了
    metadata = extract_abc_metadata(abc_lines)

    # 2. 生成或获取缓存的 Prompt
    instruction = prompt_gen.get_instruction(metadata)
    # ==========================================================

    # --- 以下是原版清洗逻辑 (保持不动) ---
    abc_lines = [line for line in abc_lines if line.strip() != '']
    abc_lines = unidecode_abc_lines(abc_lines)
    # 注意：清洗会移除 N: Style: 等字段，所以上面必须先提取
    abc_lines = remove_information_field(abc_lines=abc_lines,
                                         info_fields=['X:', 'T:', 'C:', 'W:', 'w:', 'Z:', '%%MIDI', 'N:', 'R:'])
    abc_lines = remove_bar_no_annotations(abc_lines)

    for i, line in enumerate(abc_lines):
        if re.search(r'^[A-Za-z]:', line) or line.startswith('%'):
            continue
        else:
            if r'\"' in line:
                abc_lines[i] = abc_lines[i].replace(r'\"', '')

    for i, line in enumerate(abc_lines):
        quote_contents = re.findall(Quote_re, line)
        for quote_content in quote_contents:
            for barline in Barlines:
                if barline in quote_content:
                    line = line.replace(quote_content, '')
                    abc_lines[i] = line

    try:
        _, bar_no_equal_flag, _ = check_alignment_unrotated(abc_lines)
        if not bar_no_equal_flag:
            print(abc_path, 'Unequal bar number')  # 调试时可打开
            raise Exception
    except:
        raise Exception

    for i, line in enumerate(abc_lines):
        quote_matches = re.findall(r'"[^"]*"', line)
        for match in quote_matches:
            if match == '""':
                line = line.replace(match, '')
            if match[1] in ['^', '_']:
                sub_string = match
                pattern = r'([^a-zA-Z0-9])\1+'
                sub_string = re.sub(pattern, r'\1', sub_string)
                if len(sub_string) <= 40:
                    line = line.replace(match, sub_string)
                else:
                    line = line.replace(match, '')
        abc_lines[i] = line
    # --- 原版清洗逻辑结束 ---

    abc_name = os.path.splitext(os.path.split(abc_path)[-1])[0]

    # Transpose Prep
    metadata_lines, part_text_dict = extract_metadata_and_parts(abc_lines)
    # global_metadata_dict, local_metadata_dict = extract_global_and_local_metadata(metadata_lines)
    # if global_metadata_dict['K'][0] == 'none':
    #     global_metadata_dict['K'][0] = 'C'
    # ori_key = global_metadata_dict['K'][0] # 原代码提取的key，现在已经在 metadata 里了

    # Interleaving
    interleaved_abc = rotate_abc(abc_lines)
    interleaved_path = os.path.join(INTERLEAVED_FOLDER, abc_name + '.abc')
    with open(interleaved_path, 'w') as w:
        w.writelines(interleaved_abc)

    # Augmentation Loop & Info Collection
    processed_files_info = []  # 用于返回所有生成的增强文件信息

    # [修改] 仅处理“剩余 keys”
    keys_to_do = list(allowed_keys) if allowed_keys else list(Key2index.keys())

    for key in keys_to_do:
        try:
            transposed_abc_text = transpose_an_abc_text(abc_lines, key)
            transposed_abc_lines = transposed_abc_text.split('\n')
            transposed_abc_lines = list(filter(None, transposed_abc_lines))
            transposed_abc_lines = [line + '\n' for line in transposed_abc_lines]

            # rest reduction
            metadata_lines, prefix_dict, left_barline_dict, bar_text_dict, right_barline_dict = \
                extract_barline_and_bartext_dict(transposed_abc_lines)

            reduced_abc_lines = list(metadata_lines)  # 复制一份头部

            # 重组 Body
            for i in range(len(bar_text_dict['V:1'])):
                line = ''
                for symbol in prefix_dict.keys():
                    valid_flag = False
                    for char in bar_text_dict[symbol][i]:
                        if char.isalpha() and not char in ['Z', 'z', 'X', 'x']:
                            valid_flag = True
                            break
                    if valid_flag:
                        if i == 0:
                            part_patch = '[' + symbol + ']' + prefix_dict[symbol] + left_barline_dict[symbol][0] + \
                                         bar_text_dict[symbol][0] + right_barline_dict[symbol][0]
                        else:
                            part_patch = '[' + symbol + ']' + bar_text_dict[symbol][i] + right_barline_dict[symbol][i]
                        line += part_patch
                line += '\n'
                reduced_abc_lines.append(line)

            reduced_abc_name = abc_name + '_' + key
            reduced_abc_path = os.path.join(AUGMENTED_FOLDER, key, reduced_abc_name + '.abc')

            with open(reduced_abc_path, 'w', encoding='utf-8') as w:
                w.writelines(reduced_abc_lines)

            # [新增] 为“该增强版本”的实际内容重新抽取元数据，用于精确时长/小节
            meta_for_aug = extract_abc_metadata(reduced_abc_lines)  # [新增]
            tgt_sec, tgt_bars = get_target_duration_and_bars(meta_for_aug)  # [新增]

            # [核心] 收集该文件信息，包含 Instruction
            processed_files_info.append({
                'path': reduced_abc_path,
                'source_path': abc_path,              # [新增] 源文件路径，父进程用于更新断点
                'key': key,
                'instruction': instruction,           # 仍用最上方生成的 Prompt
                'style': metadata.get('style', 'classical'),
                'target_duration_sec': int(tgt_sec),  # [新增]
                'target_bars': int(tgt_bars)          # [新增]
            })
        except Exception as e:
            _append_failed_log(f"[PROC_FAIL] {abc_path} @ Key={key} :: {e}\n{traceback.format_exc()}")
            continue

    return processed_files_info  # 返回列表，而非单个元组


# =========================== 主程序 ===========================
if __name__ == '__main__':

    # [新增] Windows/WSL 推荐 spawn，防止 fork 带来的崩溃/死锁
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 已设置过启动方式则跳过
        pass

    data = []

    # 检查输入目录
    if not os.path.exists(ORI_FOLDER):
        print(f"Error: Input folder '{ORI_FOLDER}' does not exist.")
        exit()

    print(f"Scanning ABC files in {ORI_FOLDER}...")

    # 递归获取所有 .abc 文件 (因为 1_batch_xml2abc.py 可能生成了子目录)
    file_list = []
    for root, dirs, files in os.walk(ORI_FOLDER):
        for file in files:
            if file.endswith('.abc'):
                file_list.append(os.path.join(root, file))

    # ========= 这里加入“构建任务 + 断点恢复”的进度条 =========
    print(f"Found {len(file_list)} files. Building tasks with resume...")

    # [新增] 加载断点
    resume_state = _load_resume()

    # [新增] 构建 (file, remains) 任务，仅提交仍需处理的 keys —— 带进度条
    tasks = []
    skipped = 0
    for p in tqdm(file_list, desc="[Build] scan & resume", unit="file", leave=False):
        remains = _remaining_keys_for_file(p, ORI_FOLDER, resume_state)
        if remains:
            tasks.append((p, remains))
        else:
            skipped += 1

    print(f"[Resume] {skipped} files already fully done. Will process {len(tasks)} files.")
    # =======================================================

    # --------------- [修改] 单进程 -> 多进程并行 ---------------
    # 建议：子进程禁用 LLM（避免网络/不可序列化问题）
    os.environ["SYM_WORKER"] = "1"

    # 合理并发（默认 4~8）
    workers = max(4, min(8, (os.cpu_count() or 8) - 2))
    print(f"[Parallel] Using {workers} workers...")

    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(abc_preprocess_pipeline, p, remains) for (p, remains) in tasks]
            for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="[Process] files", unit="job")):
                try:
                    files_info = fut.result()
                    data.extend(files_info)

                    # [新增] 从子进程返回中统计完成的 keys，并更新断点
                    done_keys = []
                    source_path = None
                    for item in files_info:
                        if 'key' in item:
                            done_keys.append(item['key'])
                        if not source_path and 'source_path' in item:
                            source_path = item['source_path']

                    if source_path and done_keys:
                        _resume_mark_done(source_path, ORI_FOLDER, done_keys, resume_state)

                except Exception as e:
                    _append_failed_log(f"[FUTURE_FAIL] {e}\n{traceback.format_exc()}")
                    print(f"Failed: <future> - {e}")
                    continue

                # 定期保存 Prompt 缓存与断点
                if i % SAVE_RESUME_EVERY == 0:
                    save_cache()
                    _save_resume_atomic(resume_state)

    except KeyboardInterrupt:
        print("\nInterrupted! Saving current progress...")
    finally:
        save_cache()
        _save_resume_atomic(resume_state)
    # --------------- [修改] 结束 ---------------

    # 划分数据集
    random.shuffle(data)
    eval_len = int(EVAL_SPLIT * len(data))
    eval_data = data[:eval_len]
    train_data = data[eval_len:]

    print(f"Total Samples: {len(data)}. Train: {len(train_data)}, Eval: {len(eval_data)}")

    # 写入 JSONL (注意路径更新)
    data_index_path = 'preprocessed/SymphonyNet_dataset/symphonynet_augmented.jsonl'
    eval_index_path = 'preprocessed/SymphonyNet_dataset/symphonynet_augmented_eval.jsonl'
    train_index_path = 'preprocessed/SymphonyNet_dataset/symphonynet_augmented_train.jsonl'

    # 确保输出目录存在
    os.makedirs('preprocessed', exist_ok=True)

    with open(data_index_path, 'w', encoding='utf-8') as w:
        for d in data:
            w.write(json.dumps(d, ensure_ascii=False) + '\n')
    with open(eval_index_path, 'w', encoding='utf-8') as w:
        for d in eval_data:
            w.write(json.dumps(d, ensure_ascii=False) + '\n')
    with open(train_index_path, 'w', encoding='utf-8') as w:
        for d in train_data:
            w.write(json.dumps(d, ensure_ascii=False) + '\n')
