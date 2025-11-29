# instruction_generator.py
# -*- coding: utf-8 -*-
import os
import json
import random
import time
import re
from typing import Dict, Any, List, Tuple, Optional

# =============== LLM 配置（保留硬编码） ===============
USE_LLM = True
API_KEY = "sk-a1f0220095ba49dea7ca9b162cef5916"
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
PROMPT_CACHE_FILE = "prompt_cache.json"
# =====================================================

# ============ 目标时长/小节控制（供外部调用） ============
# 可选：keep / bucket / fixed
DURATION_MODE = "bucket"
DURATION_BUCKETS = [15, 30, 45, 60, 90, 120]
FIXED_DURATION_SEC = 60

BARS_MODE = "bucket"
BARS_BUCKETS = [8, 12, 16, 24, 32]
FIXED_BARS = 16

# 冲突协调策略：duration_over_bars / bars_over_duration / average
CONFLICT_POLICY = "duration_over_bars"
GLOBAL_SEED = None
# =====================================================

# ===== openai 兼容 DeepSeek 客户端 =====
try:
    from openai import OpenAI
except ImportError:
    print("Warning: 'openai' module not found. LLM features disabled.")
    print("Please run: pip install openai")
    USE_LLM = False


# ==================== 常量与工具 ====================
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

_NOTE_TO_PC = {
    'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11,
    'c':0,'d':2,'e':4,'f':5,'g':7,'a':9,'b':11
}
_Q_UNIT_ALIAS = {'C': '1/4', 'C|': '1/2'}

def _rotate(lst, n):
    n = n % len(lst)
    return lst[n:] + lst[:n]

def _corr(a: List[float], b: List[float]) -> float:
    import math
    if not a or not b:
        return 0.0
    ma = sum(a) / len(a); mb = sum(b) / len(b)
    da = [x - ma for x in a]; db = [y - mb for y in b]
    num = sum(x*y for x,y in zip(da,db))
    den = math.sqrt(sum(x*x for x in da) * sum(y*y for y in db))
    return (num/den) if den > 0 else 0.0

def _parse_fraction(s: str) -> Optional[Tuple[int,int]]:
    m = re.match(r'^\s*(\d+)\s*/\s*(\d+)\s*$', s)
    return (int(m.group(1)), int(m.group(2))) if m else None

def _bucketize(x: float, buckets: List[int]) -> int:
    return min(buckets, key=lambda v: abs(v - x)) if buckets else int(round(x))

def _beats_per_bar_from_meter(meter: str, fallback: int = 4) -> int:
    """复合拍修正：6/8, 9/8, 12/8 通常按 2/3/4 拍处理"""
    try:
        top = int(str(meter).split("/")[0])
        bot = int(str(meter).split("/")[1])
        if bot == 8 and top % 3 == 0:
            return max(1, top // 3)
        return max(1, top)
    except:
        return fallback


# ===================== 头部解析 =====================
def _parse_headers(lines: List[str]) -> Dict[str, Any]:
    meter = '4/4'
    default_len = '1/8'
    q_beat_unit = None
    bpm = 120
    key_text = 'C'
    style = 'classical'

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith('M:'):
            meter = line[2:].strip() or meter
        elif line.startswith('L:'):
            default_len = line[2:].strip() or default_len
        elif line.startswith('Q:'):
            q_line = line[2:].strip()
            if '=' in q_line:
                left, right = q_line.split('=', 1)
                left = left.strip()
                if left in _Q_UNIT_ALIAS:
                    q_beat_unit = _Q_UNIT_ALIAS[left]
                else:
                    if _parse_fraction(left):
                        q_beat_unit = left
                try:
                    bpm = int(re.findall(r'\d+', right)[0])
                except Exception:
                    pass
            else:
                try:
                    bpm = int(re.findall(r'\d+', q_line)[0])
                except Exception:
                    pass
        elif line.startswith('K:'):
            k = line[2:].strip()
            if k and k.lower() != 'none':
                key_text = k
        elif line.startswith('R:'):
            s = line[2:].strip().lower()
            if s:
                style = s
        elif line.startswith('N:') and "Style:" in line:
            parts = line.split("Style:")
            if len(parts) > 1:
                style = parts[1].strip().lower()

    if not q_beat_unit:
        q_beat_unit = default_len or '1/4'

    beats_per_bar = _beats_per_bar_from_meter(meter, 4)

    return {
        'meter': meter,
        'default_length': default_len,
        'q_beat_unit': q_beat_unit,
        'bpm': int(bpm),
        'beats_per_bar': int(beats_per_bar),
        'key': key_text,
        'style': style
    }


# =============== 正文 token 化（含小节与声部统计） ===============
def _tokenize_body(lines: List[str]) -> Tuple[List[List[str]], int, Dict[str,int]]:
    body = []
    voice_counter: Dict[str,int] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith('%'):
            continue
        if re.match(r'^[A-Za-z]:', line):
            if line.startswith('V:'):
                vname = line[2:].strip()
                voice_counter[vname] = voice_counter.get(vname, 0) + 1
            continue
        body.append(line)

    joined = ' '.join(body)
    toks = re.split(r'(\|\:|\:\||\|)', joined)

    bars_tokens: List[List[str]] = [[]]
    for t in toks:
        t = t.strip()
        if not t:
            continue
        if t in ('|', '|:', ':|'):
            bars_tokens.append([t])
            bars_tokens.append([])
        else:
            bars_tokens[-1].append(t)
    bars_tokens = [b for b in bars_tokens if b]

    bars_notated = sum(1 for b in bars_tokens if not (len(b)==1 and b[0] in ('|:',':|')))
    return bars_tokens, bars_notated, voice_counter


# =============== 和弦/连音/破音（> <）的事件解析 ===============
def _length_value_of_unit(frac: str) -> float:
    frac = frac.strip()
    m = _parse_fraction(frac)
    if not m:
        return 0.25 if frac == '1/4' else 0.5 if frac=='1/2' else 0.125
    num, den = m
    return float(num)/float(den)

def _parse_note_length_suffix(s: str) -> float:
    if not s: return 1.0
    s = s.strip()
    m = _parse_fraction(s)
    if m: return float(m[0])/float(m[1])
    if s.startswith('/'):
        if s == '/': return 0.5
        try: return 1.0/float(s[1:])
        except: return 1.0
    if s.isdigit(): return float(s)
    return 1.0

def _tuplet_factor(n: int, meter: str) -> float:
    if n == 3: return 2.0/3.0
    if n == 4: return 3.0/4.0
    if n == 2:
        try:
            top,bot = meter.split('/')
            top=int(top); bot=int(bot)
            if bot==8 and top in (6,9,12):
                return 3.0/2.0
        except: pass
        return 1.0
    return 1.0

def _add_pitch_to_hist(token: str, hist: List[int]):
    base = token.strip()[-1]
    if base not in _NOTE_TO_PC: return
    pc = _NOTE_TO_PC[base]
    acc = token[:-1]
    semitones = 0
    # 支持 ^^ __（两次叠加优先）
    semitones += acc.count('^^')*2
    semitones += acc.count('^') - acc.count('^^')*2
    semitones -= acc.count('__')*2
    semitones -= acc.count('_') - acc.count('__')*2
    pc = (pc + semitones) % 12
    hist[pc] += 1


def _estimate_bars_performed(bars_tokens: List[List[str]]) -> int:
    """近似考虑 |: :| 重复，估计实际演奏小节数"""
    performed = 0
    i = 0
    in_repeat = False
    repeat_start_idx = None

    while i < len(bars_tokens):
        bar = bars_tokens[i]
        if len(bar)==1 and bar[0] == '|:':
            in_repeat = True
            repeat_start_idx = i+1
            i += 1
            continue
        if len(bar)==1 and bar[0] == ':|':
            if in_repeat and repeat_start_idx is not None:
                seg = bars_tokens[repeat_start_idx:i]
                seg_count = sum(1 for b in seg if not (len(b)==1 and b[0] in ('|:',':|')))
                performed += seg_count  # 第二遍
            in_repeat = False
            repeat_start_idx = None
            i += 1
            continue

        if not (len(bar)==1 and bar[0] in ('|:',':|')):
            performed += 1
        i += 1

    return max(1, performed)


def _parse_events_and_time(lines: List[str]) -> Dict[str, Any]:
    headers = _parse_headers(lines)
    meter = headers['meter']
    L_val = _length_value_of_unit(headers['default_length'])
    q_val = _length_value_of_unit(headers['q_beat_unit'])
    sec_per_qbeat = 60.0 / float(max(1, headers['bpm']))

    bars_tokens, bars_notated, voice_counter = _tokenize_body(lines)
    bars_performed = _estimate_bars_performed(bars_tokens)

    # 支持和弦 + 连音 + 破音 ><
    token_pat = re.compile(
        r"\((\d+)|"                  # 1. Tuplet
        r"([<>])|"                   # 2. Broken
        r"(?:\[([^\]]*)\]|"          # 3. Chord content inside []
        r"([_=^]*[A-Ga-gzZ]))"       # 4. Single note/rest
        r"(\d*(?:/\d+)?)"            # 5. Length suffix
    )

    events_L: List[Tuple[float, bool]] = []  # (dur_in_L, is_note)：一个“事件”（和弦整体当作一个时间事件）
    pitch_hist = [0]*12
    onsets_count = 0

    tuplet_rem = 0
    tuplet_scale = 1.0
    pending_broken = None  # None / '>' / '<'

    def _apply_broken_to_last_and_mark_next(symbol: str):
        nonlocal pending_broken
        if events_L:
            d_last, is_note = events_L[-1]
            if is_note:
                if symbol == '>':
                    events_L[-1] = (d_last*1.5, True)
                else:
                    events_L[-1] = (d_last*0.5, True)
        pending_broken = symbol

    for bar in bars_tokens:
        if len(bar)==1 and bar[0] in ('|:', ':|'):
            continue
        text = ''.join(bar)

        for m in token_pat.finditer(text):
            if m.group(1):  # Tuplet 开始
                n = int(m.group(1))
                tuplet_rem = n
                tuplet_scale = _tuplet_factor(n, meter)
                continue

            if m.group(2):  # Broken rhythm 符号
                sym = m.group(2)
                _apply_broken_to_last_and_mark_next(sym)
                continue

            chord_inner = m.group(3)
            single_note = m.group(4)
            suffix = m.group(5)

            mult = _parse_note_length_suffix(suffix)
            dur_in_L = mult * (tuplet_scale if tuplet_rem > 0 else 1.0)

            # 连音“按事件”扣减
            if tuplet_rem > 0:
                tuplet_rem -= 1
                if tuplet_rem == 0:
                    tuplet_scale = 1.0

            if chord_inner is not None:
                notes = re.findall(r'[_=^]*[A-Ga-g]', chord_inner)
                is_note = len(notes) > 0
                if pending_broken and is_note:
                    if pending_broken == '>':
                        dur_in_L *= 0.5
                    else:
                        dur_in_L *= 1.5
                    pending_broken = None

                events_L.append((dur_in_L, is_note))
                if is_note:
                    onsets_count += len(notes)     # 密度计数：和弦内每个音都算 onset
                    for n in notes:
                        _add_pitch_to_hist(n, pitch_hist)
                continue

            if single_note:
                is_rest = 'z' in single_note.lower()
                if pending_broken and not is_rest:
                    if pending_broken == '>':
                        dur_in_L *= 0.5
                    else:
                        dur_in_L *= 1.5
                    pending_broken = None

                events_L.append((dur_in_L, not is_rest))
                if not is_rest:
                    onsets_count += 1
                    _add_pitch_to_hist(single_note, pitch_hist)
                continue

    # L 单位 -> 秒
    total_beats_q = sum(d * (L_val / max(1e-9, q_val)) for d,_ in events_L)
    performed_duration_sec = max(0.001, total_beats_q * sec_per_qbeat)

    note_beats_q = sum(d * (L_val / max(1e-9, q_val)) for d,isn in events_L if isn)
    duration_density = (note_beats_q * sec_per_qbeat) / performed_duration_sec
    onset_density = onsets_count / performed_duration_sec

    polyphony_index = float(max(1, len(voice_counter) or 1))

    key_text = headers['key']
    major = not re.search(r'(min|aeo|dor|phr|loc|m)\b', key_text.lower())
    tonic_pc = 0
    mk = re.match(r'^([A-Ga-g])([#b]?)(?:\s|:|$)', key_text.strip())
    if mk:
        name = mk.group(1); acc = mk.group(2)
        base = _NOTE_TO_PC[name]
        if acc == '#': base = (base+1)%12
        elif acc == 'b': base = (base-1)%12
        tonic_pc = base
    ref = _KS_MAJOR if major else _KS_MINOR
    ref_rot = _rotate(ref, tonic_pc)
    key_consistency = _corr(pitch_hist, ref_rot) if sum(pitch_hist)>0 else 0.0

    return {
        'headers': headers,
        'performed_duration_sec': float(performed_duration_sec),
        'bars_notated': int(max(1, bars_notated)),
        'bars_performed': int(max(1, bars_performed)),
        'onset_density': float(onset_density),
        'duration_density': float(duration_density),
        'polyphony_index': float(polyphony_index),
        'pitch_class_hist': pitch_hist,
        'key_consistency': float(key_consistency),
        'register': 'mid-range'
    }


# =================== 对外：元数据提取 ===================
def extract_abc_metadata(content) -> Dict[str, Any]:
    if isinstance(content, str):
        lines = content.splitlines()
    else:
        lines = content

    headers = _parse_headers(lines)
    timing = _parse_events_and_time(lines)

    # 乐器：V: name/nm/snm
    instruments: List[str] = []
    for raw in lines:
        line = str(raw).strip()
        if line.startswith('V:'):
            m_nm = re.search(r'(?:name|nm)="([^"]+)"', line)
            m_snm = re.search(r'(?:subname|snm)="([^"]+)"', line)
            if m_nm:
                instruments.append(m_nm.group(1))
            elif m_snm:
                instruments.append(m_snm.group(1))
    if not instruments:
        instruments = ["Piano"]
    # 去重保序
    seen=set(); ordered=[]
    for v in instruments:
        if v and v not in seen:
            ordered.append(v); seen.add(v)

    dens = timing['onset_density']
    if dens < 2:
        complexity = "simple"
    elif dens < 6:
        complexity = "moderate"
    else:
        complexity = "virtuoso"

    dur = timing['performed_duration_sec']
    if dur < 30:
        duration_tag = "short"
    elif dur < 120:
        duration_tag = "medium"
    else:
        duration_tag = "long"

    meta = {
        "instruments": ordered,
        "key": headers['key'],
        "meter": headers['meter'],
        "tempo": "slow" if headers['bpm']<70 else "moderate" if headers['bpm']<110 else "fast" if headers['bpm']<140 else "very fast",
        "bpm": headers['bpm'],
        "beats_per_bar": headers['beats_per_bar'],
        "style": headers['style'],

        "duration_sec": dur,
        "duration_tag": duration_tag,
        "complexity": complexity,
        "note_density": timing['onset_density'],
        "register": timing['register'],
        "num_voices": max(1, len(ordered)),

        "bars_notated": timing['bars_notated'],
        "bars_performed": timing['bars_performed'],
        "duration_density": timing['duration_density'],
        "polyphony_index": timing['polyphony_index'],
        "key_consistency": timing['key_consistency'],
    }
    return meta


# =================== 目标控制（供外部调用） ===================
def get_target_duration_and_bars(meta: Dict[str, Any]) -> Tuple[int, int]:
    bpm = int(meta.get("bpm", 120) or 120)
    meter = str(meta.get("meter", "4/4") or "4/4")
    bpb = int(meta.get("beats_per_bar", _beats_per_bar_from_meter(meter)) or 4)

    est_sec = float(meta.get("duration_sec", 30.0) or 30.0)
    est_bars = int(meta.get("bars_performed", max(1, round((est_sec/(60.0/bpm))/max(1,bpb)))))

    if DURATION_MODE == "fixed":
        tgt_sec = int(FIXED_DURATION_SEC)
    elif DURATION_MODE == "bucket":
        tgt_sec = int(_bucketize(est_sec, DURATION_BUCKETS))
    else:
        tgt_sec = int(round(est_sec))

    if BARS_MODE == "fixed":
        tgt_bars = int(FIXED_BARS)
    elif BARS_MODE == "bucket":
        tgt_bars = int(_bucketize(est_bars, BARS_BUCKETS))
    else:
        tgt_bars = int(est_bars)

    sec_per_beat = 60.0 / max(1, bpm)
    bars_from_sec = max(1, round((tgt_sec/sec_per_beat)/max(1,bpb)))

    if CONFLICT_POLICY == "bars_over_duration":
        beats_need = tgt_bars * max(1, bpb)
        sec_from_bars = round(beats_need * sec_per_beat)
        return int(sec_from_bars), int(tgt_bars)
    elif CONFLICT_POLICY == "duration_over_bars":
        return int(tgt_sec), int(bars_from_sec)
    else:
        avg_bars = int(max(1, round((tgt_bars + bars_from_sec)/2)))
        beats_need = avg_bars * max(1, bpb)
        sec_from_bars = round(beats_need * sec_per_beat)
        return int(sec_from_bars), int(avg_bars)


# ==================== Prompt 生成器 ====================
class PromptGenerator:
    def __init__(self):
        if GLOBAL_SEED is not None:
            random.seed(int(GLOBAL_SEED))
        self.cache: Dict[str, List[str]] = {}
        self._load_cache()
        self.client = None
        if USE_LLM:
            try:
                self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            except Exception as e:
                print(f"Failed to init OpenAI client: {e}")

    # ---- Cache I/O ----
    def _load_cache(self):
        if os.path.exists(PROMPT_CACHE_FILE):
            try:
                with open(PROMPT_CACHE_FILE, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception:
                print("[Instruction Engine] Cache corrupted, starting fresh.")

    def save_cache(self):
        try:
            d = os.path.dirname(PROMPT_CACHE_FILE)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(PROMPT_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    # ---- 入口 ----
    def get_instruction(self, meta: Dict[str, Any]) -> str:
        tgt_sec, tgt_bars = get_target_duration_and_bars(meta)

        inst_key = ",".join(sorted(meta.get('instruments', []) or ["various"]))
        cache_key = "|".join([
            f"inst={inst_key}",
            f"style={meta.get('style','classical')}",
            f"key={meta.get('key','C')}",
            f"meter={meta.get('meter','4/4')}",
            f"tempo={meta.get('tempo','moderate')}",
            f"bpm={meta.get('bpm',120)}",
            f"complexity={meta.get('complexity','moderate')}",
            f"register={meta.get('register','mid-range')}",
            f"voices={meta.get('num_voices',1)}",
            f"tgtSec={tgt_sec}",
            f"tgtBars={tgt_bars}",
        ])

        if cache_key in self.cache and self.cache[cache_key]:
            return random.choice(self.cache[cache_key])

        prompts: List[str] = []
        if self.client and USE_LLM:
            try:
                prompts = self._call_llm(meta, tgt_sec, tgt_bars)
            except Exception:
                prompts = []

        if not prompts:
            prompts = [self._template(meta, tgt_sec, tgt_bars) for _ in range(3)]

        self.cache[cache_key] = prompts
        return random.choice(prompts)

    def _call_llm(self, meta: Dict[str, Any], tgt_sec: int, tgt_bars: int) -> List[str]:
        inst = ", ".join(meta.get('instruments', []) or ["various instruments"])
        style = meta.get("style", "classical")
        key = meta.get("key", "C")
        meter = meta.get("meter", "4/4")
        bpm = int(meta.get("bpm", 120) or 120)
        tempo = meta.get("tempo", "moderate")
        complexity = meta.get("complexity", "moderate")
        register = meta.get("register", "mid-range")
        voices = int(meta.get("num_voices", max(1, len(meta.get('instruments', [])) or 1)))

        system = (
            "You are a meticulous music data prompt writer.\n"
            "- Output EXACTLY 3 lines, no intro/outro, no numbering.\n"
            "- Each line must explicitly state target duration (in seconds) AND target bars.\n"
            "- One sentence per line; concise but specific.\n"
            "- No bullets, no lists, no explanations."
        )
        user = (
            "Attributes:\n"
            f"- Instruments: {inst}\n"
            f"- Style: {style}\n"
            f"- Key: {key}\n"
            f"- Meter: {meter}\n"
            f"- Tempo: {tempo} ({bpm} BPM)\n"
            f"- Complexity: {complexity}\n"
            f"- Register: {register}\n"
            f"- Voices: {voices}\n\n"
            "Targets:\n"
            f"- Target duration ≈ {tgt_sec} seconds\n"
            f"- Target bars ≈ {tgt_bars} bars\n\n"
            "Constraints:\n"
            "1) Mention both seconds and bars explicitly (e.g., “~60s (~16 bars at 120 BPM in 4/4)”).\n"
            "2) Vary phrasing across lines.\n"
            "3) Exactly 3 lines; one sentence each."
        )

        backoff = 0.8
        last_err = None
        for attempt in range(1, 4):
            try:
                t0 = time.perf_counter()
                resp = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=0.7,
                    stream=False,
                )
                dt = (time.perf_counter() - t0) * 1000.0  # ms

                # —— 仅打印“成功调用”的日志 —— #
                rid = getattr(resp, "id", None)
                usage = getattr(resp, "usage", None)
                finish_reason = None
                try:
                    finish_reason = resp.choices[0].finish_reason
                except Exception:
                    pass

                if usage:
                    try:
                        pt = getattr(usage, "prompt_tokens", None) or usage.get("prompt_tokens")
                        ct = getattr(usage, "completion_tokens", None) or usage.get("completion_tokens")
                        tt = getattr(usage, "total_tokens", None) or usage.get("total_tokens")
                    except Exception:
                        pt = ct = tt = None
                else:
                    pt = ct = tt = None

                print(f"[LLM][SUCCESS] id={rid} model={MODEL_NAME} latency={dt:.1f}ms "
                      f"tokens(p/c/t)={pt}/{ct}/{tt} finish={finish_reason}", flush=True)

                content = (resp.choices[0].message.content or "").strip()

                # 清洗
                lines_raw = [ln.strip() for ln in content.splitlines() if ln.strip()]
                cleaned = []
                for ln in lines_raw:
                    if re.search(r'^(here are|the following|outputs?:)\b', ln.lower()):
                        continue
                    ln = re.sub(r'^\s*(?:[\-\*\u2022]+|\d+[\.\)]|[a-zA-Z][\.\)])\s*', '', ln)
                    if len(ln.split()) <= 3:
                        continue
                    if not (re.search(r'\b\d+\s*s\b', ln.lower()) or re.search(r'\bbar', ln.lower())):
                        continue
                    cleaned.append(ln)

                if len(cleaned) >= 3:
                    return cleaned[:3]
            except Exception as e:
                last_err = e
            time.sleep(backoff)
            backoff *= 2
        if last_err:
            raise last_err
        return []

    def _template(self, meta: Dict[str, Any], tgt_sec: int, tgt_bars: int) -> str:
        inst = ", ".join(meta.get('instruments', []) or ["orchestra"])
        style = meta.get("style", "classical")
        key = meta.get("key", "C")
        meter = meta.get("meter", "4/4")
        bpm = int(meta.get("bpm", 120) or 120)
        tempo = meta.get("tempo", "moderate")
        complexity = meta.get("complexity", "moderate")
        register = meta.get("register", "mid-range")
        voices = int(meta.get("num_voices", max(1, len(meta.get('instruments', [])) or 1)))

        pool = [tempo, style, complexity, register]
        try:
            adjs = random.sample(pool, 2)
        except ValueError:
            adjs = pool[:2]
        adj_phrase = ", ".join(adjs)

        variants = [
            f"Compose a {adj_phrase} {style} piece in {key} for {inst} (voices={voices}), targeting ~{tgt_sec}s (~{tgt_bars} bars at {bpm} BPM in {meter}).",
            f"A {style} track for {inst} (voices={voices}) in {key}, {complexity} texture and {register} register, aiming for ~{tgt_sec}s total (~{tgt_bars} bars at {bpm} BPM, {meter}).",
            f"Write a {tempo} {style} composition in {key} for {inst} (voices={voices}) with {complexity} writing and {register} register; duration ~{tgt_sec}s (~{tgt_bars} bars at {bpm} BPM in {meter}).",
        ]
        return random.choice(variants)


# ================ 文件级接口（与你现有代码一致） ================
_generator = PromptGenerator()

def generate_instruction_from_file(file_path: str) -> Tuple[str, Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        meta = extract_abc_metadata(content)
        inst = _generator.get_instruction(meta)
        return inst, meta
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return "Compose a concise piece.", {}

def save_cache():
    _generator.save_cache()
