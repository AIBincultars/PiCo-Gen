import os
import json
import argparse
import mido
from tqdm import tqdm
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 1. GM 标准乐器表
# ==========================================
GM_INSTRUMENTS = {
    0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano", 2: "Electric Grand Piano", 3: "Honky-tonk Piano",
    4: "Electric Piano 1", 5: "Electric Piano 2", 6: "Harpsichord", 7: "Clavinet",
    8: "Celesta", 9: "Glockenspiel", 10: "Music Box", 11: "Vibraphone", 12: "Marimba", 13: "Xylophone",
    14: "Tubular Bells", 15: "Dulcimer",
    16: "Drawbar Organ", 17: "Percussive Organ", 18: "Rock Organ", 19: "Church Organ", 20: "Reed Organ",
    21: "Accordion", 22: "Harmonica", 23: "Tango Accordion",
    24: "Acoustic Guitar (nylon)", 25: "Acoustic Guitar (steel)", 26: "Electric Guitar (jazz)",
    27: "Electric Guitar (clean)", 28: "Electric Guitar (muted)", 29: "Overdriven Guitar", 30: "Distortion Guitar",
    31: "Guitar harmonics",
    32: "Acoustic Bass", 33: "Electric Bass (finger)", 34: "Electric Bass (pick)", 35: "Fretless Bass",
    36: "Slap Bass 1", 37: "Slap Bass 2", 38: "Synth Bass 1", 39: "Synth Bass 2",
    40: "Violin", 41: "Viola", 42: "Cello", 43: "Contrabass", 44: "Tremolo Strings", 45: "Pizzicato Strings",
    46: "Orchestral Harp", 47: "Timpani",
    48: "String Ensemble 1", 49: "String Ensemble 2", 50: "SynthStrings 1", 51: "SynthStrings 2", 52: "Choir Aahs",
    53: "Voice Oohs", 54: "Synth Voice", 55: "Orchestra Hit",
    56: "Trumpet", 57: "Trombone", 58: "Tuba", 59: "Muted Trumpet", 60: "French Horn", 61: "Brass Section",
    62: "SynthBrass 1", 63: "SynthBrass 2",
    64: "Soprano Sax", 65: "Alto Sax", 66: "Tenor Sax", 67: "Baritone Sax", 68: "Oboe", 69: "English Horn",
    70: "Bassoon", 71: "Clarinet",
    72: "Piccolo", 73: "Flute", 74: "Recorder", 75: "Pan Flute", 76: "Blown Bottle", 77: "Shakuhachi", 78: "Whistle",
    79: "Ocarina",
    80: "Lead 1 (square)", 81: "Lead 2 (sawtooth)", 82: "Lead 3 (calliope)", 83: "Lead 4 (chiff)",
    84: "Lead 5 (charang)", 85: "Lead 6 (voice)", 86: "Lead 7 (fifths)", 87: "Lead 8 (bass + lead)",
    88: "Pad 1 (new age)", 89: "Pad 2 (warm)", 90: "Pad 3 (polysynth)", 91: "Pad 4 (choir)",
    92: "Pad 5 (bowed)", 93: "Pad 6 (metallic)", 94: "Pad 7 (halo)", 95: "Pad 8 (sweep)",
    96: "FX 1 (rain)", 97: "FX 2 (soundtrack)", 98: "FX 3 (crystal)", 99: "FX 4 (atmosphere)",
    100: "FX 5 (brightness)", 101: "FX 6 (goblins)", 102: "FX 7 (echoes)", 103: "FX 8 (sci-fi)",
    104: "Sitar", 105: "Banjo", 106: "Shamisen", 107: "Koto", 108: "Kalimba", 109: "Bag pipe", 110: "Fiddle",
    111: "Shanai",
    112: "Tinkle Bell", 113: "Agogo", 114: "Steel Drums", 115: "Woodblock", 116: "Taiko Drum", 117: "Melodic Tom",
    118: "Synth Drum", 119: "Reverse Cymbal",
    120: "Guitar Fret Noise", 121: "Breath Noise", 122: "Seashore", 123: "Bird Tweet", 124: "Telephone Ring",
    125: "Helicopter", 126: "Applause", 127: "Gunshot"
}


def get_gm_name(prog_id):
    return GM_INSTRUMENTS.get(prog_id, f"General MIDI {prog_id}")


# ==========================================
# 2. 智能归类逻辑 (决定 Base Name)
# ==========================================
def get_smart_category(prog_id):
    """
    输入: MIDI Program ID
    输出: (Base Name, Abbreviation)
    用于确定清洗后的'基础'乐器名，比如把所有 Guitar 归为 'Nylon Guitar'
    """
    # Keyboards
    if prog_id in [0, 1, 2, 3]:       return "Piano", "Pno."
    if prog_id == 6:                  return "Harpsichord", "Hpschd."
    if prog_id in [4, 5]:             return "Electric Piano", "E.Pno."
    if 16 <= prog_id <= 20:           return "Organ", "Org."
    if prog_id == 8:                  return "Celesta", "Cel."

    # Strings (Solo)
    if prog_id == 40:                 return "Violin", "Vln."
    if prog_id == 41:                 return "Viola", "Vla."
    if prog_id == 42:                 return "Cello", "Vc."
    if prog_id == 43:                 return "Contrabass", "Cb."
    if prog_id == 46:                 return "Harp", "Hrp."

    # Strings (Ensemble/Tech)
    if prog_id == 45:                 return "Pizzicato Strings", "Pizz."
    if prog_id == 44:                 return "Tremolo Strings", "Trem."
    if prog_id in [48, 49]:           return "String Ensemble", "Str."

    # Brass
    if prog_id in [56, 59]:           return "Trumpet", "Tpt."
    if prog_id == 57:                 return "Trombone", "Tbn."
    if prog_id == 58:                 return "Tuba", "Tuba"
    if prog_id == 60:                 return "French Horn", "Hn."
    if 61 <= prog_id <= 63:           return "Brass Section", "Brass"

    # Woodwinds
    if prog_id in [73, 72]:           return "Flute", "Fl."
    if prog_id == 68:                 return "Oboe", "Ob."
    if prog_id == 71:                 return "Clarinet", "Cl."
    if prog_id == 70:                 return "Bassoon", "Bsn."
    if 64 <= prog_id <= 67:           return "Saxophone", "Sax."
    if prog_id == 69:                 return "English Horn", "E.Hn."

    # Percussion
    if prog_id == 47:                 return "Timpani", "Timp."
    if 9 <= prog_id <= 15:            return "Mallet Percussion", "Mallet"

    # Synth / Other (SymphonyNet 清洗重点)
    if prog_id in [50, 51, 55]:       return "Synth Strings", "Syn.Str."  # 隔离电子弦乐
    if 80 <= prog_id <= 103:          return "Synthesizer", "Synth."

    # Default fallback
    return get_gm_name(prog_id), "Inst."


# ==========================================
# 3. 核心扫描任务 (单文件)
# ==========================================
def scan_file_task(args):
    """
    扫描单个文件，返回：
    1. 文件路径
    2. 乐器列表: [(key, track_name), ...]
       注意：这里我们把 track_name 也带出来，做深度统计
    """
    f_path, input_dir = args
    instruments_found = []

    try:
        # clip=True 防止非法 velocity 报错
        mid = mido.MidiFile(f_path, clip=True)

        # 遍历每个轨道
        for track in mid.tracks:
            current_track_name = "Unknown"
            track_has_notes = False
            program_change_events = set()

            for msg in track:
                if msg.type == 'track_name':
                    # 清洗一下名字，去掉乱码或多余空格
                    current_track_name = msg.name.strip()

                elif msg.type == 'program_change':
                    if msg.channel == 9:  # Channel 10 is drums
                        program_change_events.add("DRUM")
                    else:
                        program_change_events.add(f"PROG_{msg.program}")

                elif msg.type == 'note_on' and msg.velocity > 0:
                    track_has_notes = True

            # 如果这个轨道有音符，我们才记录它的乐器信息
            # 避免记录那些只有控制信息没有音符的空轨道
            if track_has_notes and program_change_events:
                for key in program_change_events:
                    # 记录 (乐器ID, 该轨道的名字)
                    # 例如: ('PROG_40', 'Violin I')
                    instruments_found.append((key, current_track_name))

    except Exception:
        return None

    return f_path, instruments_found


# ==========================================
# 4. 主程序
# ==========================================
def scan_dataset(input_dir, output_file):
    print(f"Deep Scanning MIDI files in: {input_dir}")

    # 1. 收集文件
    all_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.mid', '.midi')):
                all_files.append(os.path.join(root, f))

    print(f"Found {len(all_files)} MIDI files. Launching parallel scouts...")

    # 2. 并行处理
    # 统计器：{ "PROG_40": count }
    prog_counter = Counter()
    # 名字统计器：{ "PROG_40": Counter({"Violin I": 500, "Violin II": 400...}) }
    name_dist_counter = defaultdict(Counter)
    # 样本文件：{ "PROG_40": [file1, file2...] }
    example_files = defaultdict(list)

    tasks = [(f, input_dir) for f in all_files]

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(scan_file_task, tasks), total=len(tasks), unit="file"))

    # 3. 汇总数据
    print(" Aggregating statistics...")
    for res in results:
        if res is None: continue
        f_path, inst_list = res

        # 计算相对路径
        try:
            rel_path = os.path.relpath(f_path, input_dir)
        except:
            rel_path = os.path.basename(f_path)

        for key, track_name in inst_list:
            # 统计 Program ID 出现次数
            prog_counter[key] += 1

            # 统计该 ID 下各种 Track Name 的分布
            # 只有当名字不是默认的 Unknown 或空时才记录，减少噪音
            if track_name and track_name != "Unknown":
                name_dist_counter[key][track_name] += 1

            # 记录样本
            if len(example_files[key]) < 5:
                example_files[key].append(rel_path)

    # 4. 生成详细报告 (JSON)
    mapping_data = {}

    for key, count in prog_counter.most_common():
        # 获取基本信息
        if key == "DRUM":
            raw_desc = "Percussion (Channel 10)"
            pid = -1
            base_name, base_abbr = "Percussion", "Perc."
        else:
            pid = int(key.split('_')[1])
            raw_desc = get_gm_name(pid)
            base_name, base_abbr = get_smart_category(pid)

        # 获取该乐器最常出现的 Top 5 轨道名
        top_names = name_dist_counter[key].most_common(5)
        # 格式化一下: "Violin I (120 files)"
        top_names_str = [f"{n} ({c})" for n, c in top_names]

        mapping_data[key] = {
            "frequency": count,
            "gm_description": raw_desc,

            # === 核心配置区 ===
            "target_xml_name": base_name,  # 基础名 (Violin)
            "target_xml_abbr": base_abbr,  # 基础缩写 (Vln.)

            # === 深度侦察结果 ===
            # 让你确认是否需要开启“细粒度控制”
            "top_track_names": top_names_str,
            "example_files": example_files[key]
        }

    # 5. 保存
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=4)

    print(f"\n [Success] Master Mapping generated: {output_file}")
    print("-" * 60)
    print("💡 How to use this file:")
    print("1. Open the JSON.")
    print("2. Look at 'top_track_names'. If you see 'Violin I', 'Violin II' there,")
    print("   it proves your dataset HAS fine-grained labels.")
    print("3. 'target_xml_name' is the BASE name. The next script will use this")
    print("   PLUS the track numbers (I/II) found in the file to generate final names.")
    print("-" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='raw/SymphonyNet_dataset/SymphonyNet_dataset')
    parser.add_argument('--output_map', type=str, default='instrument_mapping_master.json')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f" Error: Input directory '{args.input_dir}' not found.")
    else:
        scan_dataset(args.input_dir, args.output_map)