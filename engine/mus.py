# -*- coding: utf-8 -*-
"""mus.py — 音乐分析数据的运行时查询层（拍、频带、歌词、结构）。
所有查询都是 t 的纯函数，便于多进程分段渲染。
"""
import json, os, math
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


def _lrc(path):
    """读 `[mm:ss.xx]English|中文` 歌词文件 -> [(t, en, zh), ...]"""
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            ts, body = line[1:].split("]", 1)
            m, sec = ts.split(":")
            t = int(m) * 60 + float(sec)
        except Exception:
            continue
        parts = body.split("|")
        en = parts[0].strip()
        zh = parts[1].strip() if len(parts) > 1 else ""
        if en:
            out.append((t, en, zh))
    return out


A = json.load(open(os.path.join(ROOT, "analysis3.json"), encoding="utf-8"))

# 仓库里的 analysis3.json 只带时间轴（歌词正文因版权移除）。
# 在同目录放一份 lyrics.lrc，正文就会被合并回来。
_LRC = _lrc(os.path.join(ROOT, "lyrics.lrc"))
if _LRC:
    for i, (t, en, zh) in enumerate(_LRC):
        if i >= len(A["lyrics"]):
            break
        A["lyrics"][i]["en"] = en
        A["lyrics"][i]["zh"] = zh
        ws = A["lyrics"][i].get("words") or []
        toks = en.split()
        for w, wt in enumerate(ws):
            if isinstance(wt, list) and len(wt) >= 3 and w < len(toks):
                wt[0] = toks[w]

DUR = A["duration"]
BPM = A["bpm"]
BEAT = A["beat"]
BEATS = np.array(A["beats"], np.float32)
DOWN = np.array(A["downbeats"], np.float32)
FDT = A["frame_dt"]
BANDS = {k: np.array(v, np.float32) for k, v in A["bands"].items()}
RMS = np.array(A["rms"], np.float32)
FLUX = np.array(A["flux"], np.float32)
HFC = np.array(A["hfc"], np.float32)
CENT = np.array(A["centroid"], np.float32)
STEREO = np.array(A["stereo"], np.float32)
NOV_T = np.array(A["novelty_t"], np.float32)
NOV = np.array(A["novelty"], np.float32)
KICKS = np.array([k[0] for k in A["kicks"]], np.float32)
KICK_S = np.array([k[1] for k in A["kicks"]], np.float32)
LINES = A["lyrics"]
KEY = A.get("key", "A minor")


def _at(curve, t):
    i = int(t / FDT)
    if i < 0:
        return 0.0
    if i >= len(curve):
        return float(curve[-1])
    return float(curve[i])


def band(name, t):
    return _at(BANDS[name], t)


def rms(t):
    return _at(RMS, t)


def flux(t):
    return _at(FLUX, t)


def hfc(t):
    return _at(HFC, t)


def centroid(t):
    return _at(CENT, t)


def stereo(t):
    return _at(STEREO, t)


def env(name, t, win=0.35):
    """频带包络：返回最近 win 秒内的峰值（比瞬时值更稳，适合驱动动作）。"""
    j = int(win / FDT)
    i = int(t / FDT)
    a = max(0, i - j); b = min(len(BANDS[name]), i + 1)
    return float(BANDS[name][a:b].max()) if b > a else 0.0


def rms_env(t, win=0.25):
    j = int(win / FDT); i = int(t / FDT)
    a = max(0, i - j); b = min(len(RMS), i + 1)
    return float(RMS[a:b].max()) if b > a else 0.0


def novelty(t, win=1.2):
    i = int(np.searchsorted(NOV_T, t))
    a = max(0, i - int(win * 20)); b = min(len(NOV), i + 1)
    return float(NOV[a:b].max()) if b > a else 0.0


def beat_index(t):
    return int(np.searchsorted(BEATS, t))


def beat_phase(t):
    """0..1 当前拍内相位。"""
    i = beat_index(t)
    if i == 0:
        return 0.0
    b0 = BEATS[i - 1]
    return float(np.clip((t - b0) / BEAT, 0, 1))


def beat_pulse(t, decay=9.0):
    """拍点冲击包络：拍上为 1，之后指数衰减。"""
    i = beat_index(t)
    if i == 0:
        return 0.0
    dt = t - BEATS[i - 1]
    return float(math.exp(-dt * decay))


def bar_phase(t):
    i = beat_index(t)
    return (i % 4) / 4.0


def is_down(t, tol=0.06):
    if len(DOWN) == 0:
        return False
    i = int(np.searchsorted(DOWN, t))
    for j in (i - 1, i):
        if 0 <= j < len(DOWN) and abs(DOWN[j] - t) < tol:
            return True
    return False


def kick_near(t, lookback=0.22, lookahead=0.06):
    """最近一次 kick 的强度（带衰减）。"""
    i = np.searchsorted(KICKS, t + lookahead)
    best = 0.0
    j = i - 1
    while j >= 0 and KICKS[j] > t - lookback:
        dt = t - KICKS[j]
        best = max(best, float(KICK_S[j]) * math.exp(-dt * 11.0))
        j -= 1
    return best


# ---------------------------------------------------------------- 歌词
def line_index(t):
    """当前正在唱的歌词行（-1 = 还没开始）。"""
    i = -1
    for k, ln in enumerate(LINES):
        if ln["t"] <= t + 0.02:
            i = k
        else:
            break
    return i


def line_at(t):
    i = line_index(t)
    return LINES[i] if i >= 0 else None


def find(text, nth=0):
    """按英文行文本查时间（前缀匹配，忽略大小写）。"""
    t = text.lower().strip()
    hits = [ln for ln in LINES if ln["en"].lower().strip().startswith(t)]
    if not hits:
        hits = [ln for ln in LINES if t in ln["en"].lower()]
    if not hits:
        raise KeyError(text)
    return hits[min(nth, len(hits) - 1)]["t"]


def lines_between(t0, t1):
    return [ln for ln in LINES if t0 <= ln["t"] < t1]


def word_progress(ln, t):
    """行内逐词进度：返回 (已到时词数, 当前词内进度, 全行进度)。"""
    ws = ln.get("words") or []
    if not ws:
        return 0, 1.0, 1.0
    n = 0
    for w in ws:
        if t >= w[1]:
            n += 1
    if n < len(ws):
        w = ws[n]
        p = float(np.clip((t - w[1]) / max(1e-3, w[2] - w[1]), 0, 1))
    else:
        p = 1.0
    tot = float(np.clip((t - ln["t"]) / max(0.2, ws[-1][1] - ln["t"] + 0.4), 0, 1))
    return n, p, tot


def line_life(t, ln):
    """(可见度, 行内进度 0..1)。"""
    if ln is None:
        return 0.0, 0.0
    dur = 2.2
    k = LINES.index(ln) if ln in LINES else 0
    nxt = LINES[k + 1]["t"] if k + 1 < len(LINES) else ln["t"] + 4.0
    dur = max(1.2, min(4.5, nxt - ln["t"]))
    p = (t - ln["t"]) / dur
    return float(np.clip(min(p * 6, (1 - p) * 4, 1)), 0), float(np.clip(p, 0, 1))
