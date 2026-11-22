# -*- coding: utf-8 -*-
"""lyralign.py — 人声带精细对齐：把 LRC 行/词时间吸附到 1.5–6kHz 中央声道起音上。
输出 analysis3.json（结构与 analysis2.json 相同，只改 lyrics 时间 + 附带 vocal 曲线）。"""
import json, os, numpy as np

ROOT = r"D:\AI_SS\Ying_Shi\mv2"
SR, NFFT, HOP = 44100, 1024, 128          # 2.90ms 帧
A2 = json.load(open(os.path.join(ROOT, "analysis2.json"), encoding="utf-8"))

raw = np.fromfile(os.path.join(ROOT, "audio.f32"), dtype=np.float32).reshape(-1, 2)
mid = 0.5 * (raw[:, 0] + raw[:, 1])
side = 0.5 * (raw[:, 0] - raw[:, 1])
# 中央声道突出：mid 减去 0.75*side 的幅度，压掉立体声铺底
sig = mid - 0.6 * side
sig = sig / (np.abs(sig).max() + 1e-9)

win = np.hanning(NFFT).astype(np.float32)
nfr = 1 + (len(sig) - NFFT) // HOP
freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
band = ((freqs >= 1400) & (freqs <= 6200)).astype(np.float32)
band2 = ((freqs >= 220) & (freqs <= 1400)).astype(np.float32)   # 人声基频辅助
print("frames", nfr)

mag_b = np.empty(nfr, np.float32)
mag_l = np.empty(nfr, np.float32)
CH = 4096
for s in range(0, nfr, CH):
    e = min(nfr, s + CH)
    idx = np.arange(s, e)[:, None] * HOP + np.arange(NFFT)[None, :]
    fr = sig[idx] * win
    sp = np.abs(np.fft.rfft(fr, axis=1)).astype(np.float32)
    mag_b[s:e] = (sp[:, :] @ band) / band.sum()
    mag_l[s:e] = (sp[:, :] @ band2) / band2.sum()
times = np.arange(nfr) * HOP / SR

def smo(x, k):
    k = np.ones(k, np.float32) / k
    return np.convolve(x, k, mode="same")

# 起音：正向差分 + 轻度平滑（保留 3ms 精度）
on = np.maximum(np.diff(mag_b, prepend=mag_b[0]), 0)
on = 0.75 * on / (on.max() + 1e-9) + 0.25 * np.maximum(np.diff(mag_l, prepend=mag_l[0]), 0) / (np.maximum(np.diff(mag_l, prepend=mag_l[0]), 0).max() + 1e-9)
on = smo(on, 3)
on = np.concatenate([[0], np.maximum(np.diff(on), 0)])  # 再取一次差分：尖峰更突出
on = smo(on, 3)
on /= (on.max() + 1e-9)

lines = A2["lyrics"]
N = len(lines)
# ---------- 行级 DP 对齐 ----------
R = 0.85           # 搜索半径
lam = 1.6          # 偏离先验的惩罚
cands = []
for i, ln in enumerate(lines):
    t0 = ln["t"]
    j0, j1 = max(0, int((t0 - R) / (HOP / SR))), min(nfr, int((t0 + R) / (HOP / SR)))
    seg = on[j0:j1]
    # 取局部峰
    pk = np.where((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]) & (seg[1:-1] > 0.06))[0] + 1
    if len(pk) == 0:
        pk = np.array([int(np.argmax(seg))])
    cands.append([(j0 + p) * HOP / SR for p in pk])

def score(i, tt):
    j = int(tt * SR / HOP)
    return on[max(0, j - 1):j + 2].max() - lam * abs(tt - lines[i]["t"]) / 0.85

INF = -1e9
rows = [[score(i, c) for c in cands[i]] for i in range(N)]
DP = [None] * N; DP[0] = np.array(rows[0])
parent = [np.zeros(len(cands[0]), int)]
for i in range(1, N):
    cur = np.full(len(cands[i]), INF)
    par_i = np.zeros(len(cands[i]), int)
    for a, tc in enumerate(cands[i]):
        lim = tc - 0.10
        vals = np.full(len(cands[i - 1]), INF)
        for b, tp in enumerate(cands[i - 1]):
            if tp < lim:
                vals[b] = DP[i - 1][b]
        b = int(np.argmax(vals))
        cur[a] = rows[i][a] + vals[b]
        par_i[a] = b
    DP[i] = cur; parent.append(par_i)
path = [0] * N
path[N - 1] = int(np.argmax(DP[N - 1]))
for i in range(N - 1, 0, -1):
    path[i - 1] = int(parent[i][path[i]])
new_t = [cands[i][path[i]] for i in range(N)]
shift = np.array(new_t) - np.array([l["t"] for l in lines])
print("line shift: mean %.3f  median %.3f  |shift|>0.25: %.0f%%  max %.2f" %
      (shift.mean(), np.median(shift), (np.abs(shift) > 0.25).mean() * 100, np.abs(shift).max()))
prev_on = np.array([on[max(0, int(l["t"] * SR / HOP) - 2):int(l["t"] * SR / HOP) + 3].max() for l in lines])
new_on = np.array([on[max(0, int(t * SR / HOP) - 2):int(t * SR / HOP) + 3].max() for t in new_t])
print("vocal onset at anchor: before %.3f -> after %.3f" % (prev_on.mean(), new_on.mean()))

# ---------- 词级吸附（单调、行内） ----------
for i, ln in enumerate(lines):
    t0 = new_t[i]
    t1 = new_t[i + 1] if i + 1 < N else t0 + 4.5
    words = ln.get("words") or [[w, t0, t0 + 0.1] for w in ln["en"].split()]
    span_end = min(t1 - 0.10, t0 + 4.6)
    span_end = max(span_end, t0 + 0.30)
    prev = t0 - 0.02
    outw = []
    for k, (w, wa, wb) in enumerate(words):
        est = t0 + (k + 0.5) / len(words) * (span_end - t0) if len(words) > 1 else t0
        lo = max(prev + 0.02, est - 0.28)
        hi = max(est + 0.28, lo + 0.05)
        j0, j1 = int(lo * SR / HOP), min(nfr, int(hi * SR / HOP))
        tt = est
        if j1 > j0 + 1:
            p = j0 + int(np.argmax(on[j0:j1]))
            if on[p] > 0.05:
                tt = p * HOP / SR
        tt = min(tt, span_end - 0.06)
        outw.append([w, round(float(tt), 3), None])
        prev = tt
    for k in range(len(outw)):
        nxt = outw[k + 1][1] if k + 1 < len(outw) else span_end
        outw[k][2] = round(float(max(nxt, outw[k][1] + 0.08)), 3)
    ln["words"] = outw
    ln["t"] = round(float(t0), 3)

out = dict(A2)
out["lyrics"] = lines
out["vocal_step"] = round(HOP / SR, 5)
out["vocal_onset"] = [round(float(v), 4) for v in on[::16]]
out["vocal_onset_dt"] = round(16 * HOP / SR, 5)
out["meta"] = "lyric lines/words DP-snapped to 1.4-6.2kHz centered-channel onsets (hop 128)"
with open(os.path.join(ROOT, "analysis3.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
print("wrote analysis3.json")
for i in (0, 1, 12, 20, 40, 60, 74, 90, 95):
    if i < N:
        print("%6.2f  %s" % (lines[i]["t"], lines[i]["en"]))
