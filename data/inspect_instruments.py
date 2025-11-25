import os
import mido
import argparse
from tqdm import tqdm
from collections import Counter

# GM 标准乐器表 (用于将 ID 转为可读名称)

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

def get_instrument_names(midi_path):
    """
    从单个 MIDI 文件中提取所有乐器信息
    返回一个集合，包含 (Program_ID, GM_Name, Track_Name)
    """
    instruments = set()
    try:
        # clip=True 忽略 velocity 溢出错误
        mid = mido.MidiFile(midi_path, clip=True)

        # 遍历所有轨道
        for track in mid.tracks:
            current_program = None
            track_name = "Unknown"

            # 先扫一遍轨道名 (MetaMessage)
            for msg in track:
                if msg.type == 'track_name':
                    try:
                        # 处理可能的编码问题
                        track_name = msg.name.strip()
                    except:
                        track_name = "Unknown"

            # 再扫一遍乐器定义 (Message)
            for msg in track:
                if msg.type == 'program_change':
                    prog = msg.program
                    is_drum = (msg.channel == 9)  # Channel 10 是打击乐

                    if is_drum:
                        gm_name = "Percussion (Channel 10)"
                        prog_id_str = "DRUM"
                    else:
                        gm_name = GM_INSTRUMENTS.get(prog, f"Program {prog}")
                        prog_id_str = str(prog)

                    # 记录格式: "ID | GM标准名 | 轨道原名(如果有)"
                    # 这样你可以看到比如: "40 | Violin | Violin I"
                    info_str = f"[{prog_id_str}] {gm_name} (Track: {track_name})"
                    instruments.add(info_str)

    except Exception as e:
        # print(f"Error reading {midi_path}: {e}")
        pass

    return instruments


def main():
    parser = argparse.ArgumentParser(description="扫描 MIDI 数据集中的所有乐器")
    parser.add_argument('--input_dir', type=str, default='raw/SymphonyNet_dataset/SymphonyNet_dataset', help='数据集根目录')
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Path {args.input_dir} does not exist!")
        return

    print(f"Scanning {args.input_dir} ...")

    all_instruments = Counter()

    # 递归遍历
    files = []
    for root, _, filenames in os.walk(args.input_dir):
        for f in filenames:
            if f.lower().endswith(('.mid', '.midi')):
                files.append(os.path.join(root, f))

    print(f"Found {len(files)} MIDI files. Analyzing...")

    for f in tqdm(files):
        insts = get_instrument_names(f)
        all_instruments.update(insts)

    print("\n" + "=" * 60)
    print(f"SCAN RESULT: Found {len(all_instruments)} unique instrument/track combinations")
    print("=" * 60)
    print(f"{'Count':<8} | Instrument Info")
    print("-" * 60)

    # 按出现频率排序打印
    for name, count in all_instruments.most_common():
        print(f"{count:<8} | {name}")

    # 保存到文件方便查看
    output_file = "all_instruments_list.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for name, count in all_instruments.most_common():
            f.write(f"{count}\t{name}\n")

    print(f"\n[Done] Full list saved to {output_file}")


if __name__ == '__main__':
    main()