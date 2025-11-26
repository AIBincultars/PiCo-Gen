import os
import json
import argparse
import mido
from tqdm import tqdm
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# 1. GM 标准乐器表 (Reference)
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
# 2. 科学分类映射表 (Scientific Taxonomy)
# ==========================================
# 依据：发声物理机制、频响范围、关键演奏技法
SCIENTIFIC_MAPPING = {
    # --- Keyboards (Excitation: Struck/Plucked/Wind) ---
    0: ("Piano", "Pno."), 1: ("Piano", "Pno."), 2: ("Piano", "Pno."), 3: ("Piano", "Pno."),
    4: ("Electric Piano", "E.Pno."), 5: ("Electric Piano", "E.Pno."),
    6: ("Harpsichord", "Hpschd."), 7: ("Harpsichord", "Hpschd."),
    16: ("Organ", "Org."), 17: ("Organ", "Org."), 18: ("Organ", "Org."), 19: ("Organ", "Org."), 20: ("Organ", "Org."),
    21: ("Accordion", "Acc."), 22: ("Harmonica", "Harm."), 23: ("Accordion", "Acc."),

    # --- Plucked Strings (Excitation: Plucked, Transient) ---
    24: ("Nylon Guitar", "N.Gtr."),
    25: ("Steel Guitar", "S.Gtr."),
    26: ("Clean Elec Guitar", "E.Gtr."), 27: ("Clean Elec Guitar", "E.Gtr."), 28: ("Clean Elec Guitar", "E.Gtr."),
    29: ("Distorted Guitar", "Dist.Gtr."), 30: ("Distorted Guitar", "Dist.Gtr."),
    31: ("Clean Elec Guitar", "Harm.Gtr."),  # Guitar harmonics usually mapped to Clean
    46: ("Harp", "Hrp."),

    # --- Bass (Function: Low Frequency Foundation) ---
    32: ("Acoustic Bass", "Ac.Bass"),
    33: ("Electric Bass", "E.Bass"), 34: ("Electric Bass", "E.Bass"),
    35: ("Electric Bass", "Fretless"),  # Fretless is technically Electric
    36: ("Electric Bass", "Slap.Bass"), 37: ("Electric Bass", "Slap.Bass"),
    38: ("Synth Bass", "Syn.Bass"), 39: ("Synth Bass", "Syn.Bass"),

    # --- Bowed Strings (Excitation: Bowed, Sustained) ---
    # 严格区分四重奏乐器
    40: ("Violin", "Vln."),
    41: ("Viola", "Vla."),
    42: ("Violoncello", "Vc."),
    43: ("Contrabass", "Cb."),
    # 特殊技法独立
    44: ("Tremolo Strings", "Trem."),
    45: ("Pizzicato Strings", "Pizz."),
    # 合奏
    48: ("String Ensemble", "Str."), 49: ("String Ensemble", "Str."),
    50: ("String Ensemble", "Syn.Str."), 51: ("String Ensemble", "Syn.Str."),
    # Synth Strings mapped to Ensemble context
    55: ("String Ensemble", "Hit"),  # Orchestra Hit often used as staccato strings

    # --- Brass (Excitation: Lip Reed) ---
    56: ("Trumpet", "Tpt."),
    57: ("Trombone", "Tbn."),
    58: ("Tuba", "Tuba"),
    59: ("Muted Trumpet", "Mut.Tpt."),  # 独立分类，共振峰不同
    60: ("French Horn", "Hn."),
    61: ("Brass Section", "Brass"), 62: ("Brass Section", "Syn.Brass"), 63: ("Brass Section", "Syn.Brass"),

    # --- Woodwinds (Excitation: Air Reed/Cane Reed) ---
    64: ("Saxophone", "Sax."), 65: ("Saxophone", "Sax."), 66: ("Saxophone", "Sax."), 67: ("Saxophone", "Sax."),
    68: ("Oboe", "Ob."),
    69: ("English Horn", "E.Hn."),
    70: ("Bassoon", "Bsn."),
    71: ("Clarinet", "Cl."),
    72: ("Piccolo", "Picc."),  # 独立分类，音域极高
    73: ("Flute", "Fl."),
    74: ("Flute", "Rec."),  # Recorder mapped to Flute family
    75: ("Flute", "Pan.Fl."), 76: ("Flute", "Bottle"), 77: ("Flute", "Shaku."), 78: ("Flute", "Whistle"),
    79: ("Flute", "Ocarina"),

    # --- Percussion & Chromatic (Function: Rhythm & Color) ---
    47: ("Timpani", "Timp."),
    8: ("Celesta", "Cel."),
    9: ("Glockenspiel", "Glk."),
    10: ("Music Box", "M.Box"),
    11: ("Vibraphone", "Vib."),
    12: ("Marimba", "Mar."),
    13: ("Xylophone", "Xyl."),
    14: ("Tubular Bells", "T.Bells"),
    15: ("Dulcimer", "Dulc."),
    112: ("Mallet Percussion", "Tink.Bell"), 113: ("Mallet Percussion", "Agogo"),
    114: ("Mallet Percussion", "Steel.Dr"), 115: ("Mallet Percussion", "Woodblk"),

    # --- Synthesizer (Function: Electronic Texture) ---
    # Lead (Melody)
    80: ("Synth Lead", "Lead"), 81: ("Synth Lead", "Lead"), 82: ("Synth Lead", "Lead"), 83: ("Synth Lead", "Lead"),
    84: ("Synth Lead", "Lead"), 85: ("Synth Lead", "Lead"), 86: ("Synth Lead", "Lead"), 87: ("Synth Lead", "Lead"),
    # Pad (Atmosphere)
    88: ("Synth Pad", "Pad"), 89: ("Synth Pad", "Pad"), 90: ("Synth Pad", "Pad"), 91: ("Synth Pad", "Pad"),
    92: ("Synth Pad", "Pad"), 93: ("Synth Pad", "Pad"), 94: ("Synth Pad", "Pad"), 95: ("Synth Pad", "Pad"),
    # FX (Effects)
    96: ("Synth Pad", "FX"), 97: ("Synth Pad", "FX"), 98: ("Synth Pad", "FX"), 99: ("Synth Pad", "FX"),
    100: ("Synth Pad", "FX"), 101: ("Synth Pad", "FX"), 102: ("Synth Pad", "FX"), 103: ("Synth Pad", "FX"),

    # --- Ethnic / SFX ---
    104: ("Ethnic", "Eth."), 105: ("Ethnic", "Banjo"), 106: ("Ethnic", "Eth."), 107: ("Ethnic", "Koto"),
    108: ("Ethnic", "Kalimba"), 109: ("Ethnic", "Bagpipe"), 110: ("Violin", "Fiddle"), 111: ("Ethnic", "Shanai"),

    # Percussion placeholders (non-chromatic usually handled by Channel 10)
    116: ("Percussion", "Taiko"), 117: ("Percussion", "Tom"), 118: ("Percussion", "Syn.Drum"),
    119: ("Percussion", "Rev.Cym"),

    # SFX
    120: ("Clean Elec Guitar", "Gtr.Fret"),  # Noise
    121: ("Ethnic", "Breath"), 122: ("Ethnic", "Seashore"), 123: ("Ethnic", "Bird"), 124: ("Ethnic", "Tel."),
    125: ("Ethnic", "Heli."), 126: ("Ethnic", "Applause"), 127: ("Ethnic", "Gunshot")
}


def get_smart_category(prog_id):
    """
    根据 MIDI Program ID 返回科学分类后的 (XML Name, XML Abbreviation)
    """
    if prog_id in SCIENTIFIC_MAPPING:
        return SCIENTIFIC_MAPPING[prog_id]

    # Fallback
    return GM_INSTRUMENTS.get(prog_id, "Instrument"), "Inst."


# ==========================================
# 3. 核心扫描任务 (单文件)
# ==========================================
def scan_file_task(args):
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
                    current_track_name = msg.name.strip()

                elif msg.type == 'program_change':
                    if msg.channel == 9:  # Channel 10 is drums
                        program_change_events.add("DRUM")
                    else:
                        program_change_events.add(f"PROG_{msg.program}")

                elif msg.type == 'note_on' and msg.velocity > 0:
                    track_has_notes = True

            if track_has_notes and program_change_events:
                for key in program_change_events:
                    instruments_found.append((key, current_track_name))

    except Exception:
        return None

    return f_path, instruments_found


# ==========================================
# 4. 主程序
# ==========================================
def scan_dataset(input_dir, output_file):
    print(f"Scientific Scanning MIDI files in: {input_dir}")

    all_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.mid', '.midi')):
                all_files.append(os.path.join(root, f))

    print(f"Found {len(all_files)} MIDI files. Launching analysis...")

    prog_counter = Counter()
    name_dist_counter = defaultdict(Counter)
    example_files = defaultdict(list)

    tasks = [(f, input_dir) for f in all_files]

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(scan_file_task, tasks), total=len(tasks), unit="file"))

    print(" Aggregating statistics and applying scientific taxonomy...")
    for res in results:
        if res is None: continue
        f_path, inst_list = res

        try:
            rel_path = os.path.relpath(f_path, input_dir)
        except:
            rel_path = os.path.basename(f_path)

        for key, track_name in inst_list:
            prog_counter[key] += 1
            if track_name and track_name != "Unknown":
                name_dist_counter[key][track_name] += 1
            if len(example_files[key]) < 5:
                example_files[key].append(rel_path)

    # 生成映射报告
    mapping_data = {}

    # 遍历所有出现过的 Program ID，或者遍历整个 SCIENTIFIC_MAPPING 确保全覆盖
    # 这里我们遍历 SCIENTIFIC_MAPPING 确保 JSON 完整，即便数据集中没出现的乐器也有定义
    # 同时合并扫描到的统计信息

    # 1. 先处理 DRUM
    drum_key = "DRUM"
    mapping_data[drum_key] = {
        "frequency": prog_counter[drum_key],
        "gm_description": "Percussion (Channel 10)",
        "target_xml_name": "Percussion",
        "target_xml_abbr": "Perc.",
        "top_track_names": [f"{n} ({c})" for n, c in name_dist_counter[drum_key].most_common(5)],
        "example_files": example_files[drum_key]
    }

    # 2. 处理所有 Program ID (0-127)
    for pid in range(128):
        key = f"PROG_{pid}"
        base_name, base_abbr = get_smart_category(pid)

        mapping_data[key] = {
            "target_xml_name": base_name,
            "target_xml_abbr": base_abbr
        }

        # 如果数据集中出现了这个乐器，补充统计信息
        if prog_counter[key] > 0:
            mapping_data[key]["frequency"] = prog_counter[key]
            mapping_data[key]["gm_description"] = get_gm_name(pid)
            mapping_data[key]["top_track_names"] = [f"{n} ({c})" for n, c in name_dist_counter[key].most_common(5)]
            mapping_data[key]["example_files"] = example_files[key]

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=4)

    print(f"\n [Success] Scientific Mapping generated: {output_file}")
    print("-" * 60)
    print("Rationale applied:")
    print("1. Separation of excitation mechanisms (Piano vs Guitar vs Harpsichord)")
    print("2. Distinction of frequency formants (Violin vs Viola vs Cello)")
    print("3. Independence of articulations (Pizzicato/Tremolo preserved)")
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