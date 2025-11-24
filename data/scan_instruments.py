import os
import json
import argparse
import mido
from tqdm import tqdm
from collections import Counter, defaultdict
from gm_constants import GM_INSTRUMENTS


# GM 标准名参考 (仅用于生成建议名)
GM_REF = GM_INSTRUMENTS

def scan_all_midi(input_dir):
    print(f"Scanning MIDI files in {input_dir}...")

    # 统计器：Key 是 "prog_{id}" 或 "drum"
    stats_counter = Counter()
    # 采样器：记录每个 ID 对应的文件名 (方便你核查)
    example_files = defaultdict(list)

    files = [os.path.join(r, f) for r, _, fs in os.walk(input_dir) for f in fs if f.endswith(('.mid', '.midi'))]

    for f_path in tqdm(files):
        try:
            mid = mido.MidiFile(f_path)
            seen_in_this_file = set()

            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'program_change':
                        if msg.channel == 9:  # Channel 10
                            key = "drum"
                        else:
                            key = f"prog_{msg.program}"

                        seen_in_this_file.add(key)

            # 统计文件覆盖率 (即：有多少个文件包含这个乐器)
            for key in seen_in_this_file:
                stats_counter[key] += 1
                if len(example_files[key]) < 3:
                    example_files[key].append(os.path.basename(f_path))

        except:
            continue

    return stats_counter, example_files


def generate_mapping_template(counter, examples, output_file):
    """
    生成一个 JSON 配置文件，供用户修改
    """
    mapping_config = {}

    # 按出现频率排序
    for key, count in counter.most_common():
        # 1. 原始信息
        if key == "drum":
            raw_desc = "Percussion (Channel 10)"
            suggested_name = "Percussion"
            suggested_id = 8  # 假设 8 是打击乐 ID
        else:
            prog_id = int(key.split('_')[1])
            raw_desc = GM_REF.get(prog_id, f"Unknown Program {prog_id}")

            # 2. 智能建议 (根据 GM 范围粗略归类)
            if 0 <= prog_id <= 7:
                suggested_name, suggested_id = "Piano", 0
            elif 40 <= prog_id <= 44:
                suggested_name, suggested_id = "Strings", 1  # Violin/Viola...
            elif 56 <= prog_id <= 63:
                suggested_name, suggested_id = "Brass", 5
            elif 72 <= prog_id <= 79:
                suggested_name, suggested_id = "Woodwind", 3
            else:
                suggested_name, suggested_id = "Other", 99

        # 3. 构建配置项
        mapping_config[key] = {
            "frequency": count,  # 出现次数
            "raw_description": raw_desc,  # 原始描述 (GM标准)
            "example_files": examples[key],  # 包含该乐器的文件示例

            # [重点] 下面这两个是你需要手动确认/修改的
            "target_name": suggested_name,
            "target_id": suggested_id
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_config, f, indent=4)

    print(f"\n[Important] Mapping template saved to: {output_file}")
    print("-> Please open this JSON file and edit 'target_name' & 'target_id' to match your PiCo-Gen requirements.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../raw')
    parser.add_argument('--output_map', type=str, default='instrument_mapping.json')
    args = parser.parse_args()

    stats, examples = scan_all_midi(args.input_dir)
    generate_mapping_template(stats, examples, args.output_map)