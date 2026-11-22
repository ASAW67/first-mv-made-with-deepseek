# -*- coding: utf-8 -*-
"""audio2.py — 纯 numpy DSP 的深度听觉分析。

产出 analysis2.json：
  * STFT 频谱（对数幅度）派生的逐帧特征：频段能量、质心、平坦度、噪声度、立体声宽度
  * 起音强度曲线 + 峰值表（自适应中值阈值）
  * 速度 / 拍网格（含逐拍吸附）/ 拍内相位 / 强拍
  * 色度 / 调性估计
  * 结构新颖度曲线（SSM + 棋盘核）
  * RMS / 响度 / 低频冲击 / 高频哒声
  * 歌词行时间（LRC 锚点 + 人声带起音微调）+ 词级时间
"""
import os, sys, json, math
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SR = 44100
NFFT = 2048
HOP = 512
WIN = np.hanning(NFFT).astype(np.float32)

RAW = os.path.join(ROOT, "audio.f32")
LRC = os.path.join(os.path.dirname(ROOT), "mv", "lyrics.lrc")


def load_audio():
    x = np.fromfile(RAW, np.float32).reshape(-1, 2)
    return x[:, 0].copy(), x[:, 1].copy()


def stft(mono):
    pad = NFFT // 2
    x = np.pad(mono, (pad, pad), mode="reflect")
    n = 1 + (len(x) - NFFT) // HOP
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    frames = x[idx] * WIN
    return np.fft.rfft(frames, axis=1).astype(np.complex64)


def hz2bin(f):
    return int(round(f / (SR / NFFT)))


def band(S, lo, hi):
    a, b = hz2bin(lo), max(hz2bin(hi), hz2bin(lo) + 1)
    return S[:, a:b].mean(axis=1)


def smooth(x, k):
    if k <= 1:
        return x
    ker = np.ones(k, np.float32) / k
    return np.convolve(x, ker, mode="same")


def norm01(x, lo=1.0, hi=99.0):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / max(1e-9, b - a), 0, 1).astype(np.float32)


def main():
    L, R = load_audio()
    mono = 0.5 * (L + R)
    n_samples = len(mono)
    dur = n_samples / SR
    Z = stft(mono)
    mag = np.abs(Z)
    logmag = np.log1p(mag * 8.0)
    nfr = mag.shape[0]
    times = (np.arange(nfr) * HOP) / SR

    # ---- 频段能量（dB 域，再做百分位归一）
    bands = {
        "sub":    (20, 60),
        "bass":   (60, 160),
        "lowmid": (160, 400),
        "mid":    (400, 1200),
        "pres":   (1200, 3500),   # 人声主要存在区
        "brill":  (3500, 8000),
        "air":    (8000, 16000),
    }
    E = {}
    for k, (a, b) in bands.items():
        e = band(mag, a, b)
        E[k] = norm01(np.log1p(e * 10.0))

    # ---- 谱质心 / 平坦度 / 噪声度
    freqs = np.fft.rfftfreq(NFFT, 1 / SR).astype(np.float32)
    p = mag ** 2
    tot = p.sum(1) + 1e-9
    centroid = (p * freqs[None, :]).sum(1) / tot
    centroid_n = norm01(np.log1p(centroid / 400.0))
    flat = np.exp(np.log(mag + 1e-6).mean(1)) / (mag.mean(1) + 1e-6)
    flat_n = norm01(flat)
    # 高频通量（"嘶"声/镲片）
    hfc = (mag[:, hz2bin(4000):] ** 1.2).sum(1)
    hfc = norm01(np.log1p(hfc))

    # ---- 起音强度（对数谱通量，正向差）
    d = np.diff(logmag, axis=0, prepend=logmag[:1])
    flux = np.maximum(d, 0).sum(1)
    flux = flux / (logmag.sum(1) + 1e-6)
    flux_s = smooth(flux, 3)
    # 自适应中值阈值
    med = np.empty_like(flux_s)
    k = 21
    pad = np.pad(flux_s, (k, k), mode="edge")
    for i in range(nfr):
        med[i] = np.median(pad[i:i + 2 * k + 1])
    thr = med * 1.35 + 0.001
    peak = (flux_s > thr)
    # 局部极大 + 最小间隔 4 帧
    cand = []
    for i in range(2, nfr - 2):
        if peak[i] and flux_s[i] >= flux_s[i - 1] and flux_s[i] >= flux_s[i + 1]:
            if cand and (i - cand[-1]) < 4:
                if flux_s[i] > flux_s[cand[-1]]:
                    cand[-1] = i
            else:
                cand.append(i)
    on_t = times[cand]
    on_s = flux_s[cand]
    # 只保留明显峰
    keep = on_s > np.percentile(on_s, 45)
    on_t, on_s = on_t[keep], on_s[keep]

    # ---- 速度估计：低频起音峰 + 相位离散最小化（PDM），比自相关稳
    low = 0.7 * E["sub"] + 0.3 * E["bass"]
    dl = np.maximum(np.diff(low, prepend=low[0]), 0)
    dl = smooth(dl, 3)
    kp = []
    for i in range(2, nfr - 2):
        if dl[i] > 0.05 and dl[i] >= dl[i - 1] and dl[i] > dl[i + 1]:
            if kp and times[i] - kp[-1][0] < 0.16:
                if dl[i] > kp[-1][1]:
                    kp[-1] = (times[i], dl[i])
            else:
                kp.append((times[i], dl[i]))
    kt = np.array([p[0] for p in kp], np.float64)
    kw = np.array([p[1] for p in kp], np.float64)
    best_p, best_r = 0.4615, -1
    for p in np.arange(0.44, 0.49, 0.00005):
        ph = (kt / p) % 1.0
        C = (kw * np.cos(2 * np.pi * ph)).sum()
        S = (kw * np.sin(2 * np.pi * ph)).sum()
        R = math.hypot(C, S) / kw.sum()
        if R > best_r:
            best_r, best_p = R, p
    period = float(best_p)
    bpm = 60.0 / period
    ph = (kt / period) % 1.0
    C = (kw * np.cos(2 * np.pi * ph)).sum()
    S = (kw * np.sin(2 * np.pi * ph)).sum()
    phase = (math.atan2(S, C) / (2 * math.pi)) * period
    while phase < 0:
        phase += period

    # ---- 拍网格（相位起点 + 逐步吸附到最强低频起音）
    beats = []
    b = phase
    while b < dur - 0.02:
        beats.append(b)
        b += period
    beats = np.array(beats, np.float32)
    snapped = beats.copy()
    kt_arr = kt
    for i, bt in enumerate(beats):
        j = np.searchsorted(kt_arr, bt)
        best_t, bd = None, 0.045
        for jj in (j - 1, j, j + 1):
            if 0 <= jj < len(kt_arr):
                dd = abs(kt_arr[jj] - bt)
                if dd < bd:
                    bd, best_t = dd, kt_arr[jj]
        if best_t is not None:
            snapped[i] = best_t
    # 强拍：低频能量在 4 拍周期里最重的一相
    bi = np.clip(np.searchsorted(times, snapped), 0, nfr - 1)
    low = E["sub"][bi] + 0.7 * E["bass"][bi]
    best_ph, best_v = 0, -1
    for o in range(4):
        v = low[o::4].mean()
        if v > best_v:
            best_v, best_ph = v, o
    downbeats = [float(snapped[i]) for i in range(best_ph, len(snapped), 4)]

    # ---- 色度 / 调性
    pc = np.zeros((nfr, 12), np.float32)
    fbin = freqs[1:]
    for b in range(1, len(freqs)):
        f = freqs[b]
        if f < 55 or f > 2200:
            continue
        midi = 69 + 12 * math.log2(f / 440.0)
        pc[:, int(round(midi)) % 12] += mag[:, b]
    pc = pc / (pc.sum(1, keepdims=True) + 1e-9)
    chroma = pc.mean(0)
    KSHARP = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    KMINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    best = ("C major", -9)
    for r in range(12):
        for prof, mode in ((KSHARP, "major"), (KMINOR, "minor")):
            c = np.corrcoef(np.roll(prof, r), chroma)[0, 1]
            if c > best[1]:
                best = (f"{names[r]} {mode}", float(c))
    key = best[0]
    key_corr = best[1]

    # ---- 结构新颖度（SSM + 棋盘核）
    feat = np.stack([norm01(np.log1p(band(mag, 200, 400))),
                     norm01(np.log1p(band(mag, 400, 900))),
                     norm01(np.log1p(band(mag, 900, 2000))),
                     norm01(np.log1p(band(mag, 2000, 4500))),
                     norm01(np.log1p(band(mag, 4500, 9000))),
                     centroid_n, flat_n], axis=1)
    step = 43  # ~0.5s
    fs = feat[::step]
    fs = (fs - fs.mean(0)) / (fs.std(0) + 1e-6)
    fs = fs / (np.linalg.norm(fs, axis=1, keepdims=True) + 1e-6)
    ss = fs @ fs.T
    ker = np.outer(np.r_[np.ones(8), -np.ones(8)], np.r_[np.ones(8), -np.ones(8)]).astype(np.float32)
    nov = np.zeros(len(ss), np.float32)
    for i in range(8, len(ss) - 8):
        patch = ss[i - 8:i + 8, i - 8:i + 8]
        nov[i] = float((patch * ker).sum())
    nov = nov - nov.min()
    nov = nov / (nov.max() + 1e-9)
    nov_t = (np.arange(len(nov)) * step * HOP) / SR

    # ---- 歌词解析与词级时间
    lines = []
    with open(LRC, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw.startswith("["):
                continue
            try:
                mm = int(raw[1:3]); ss = float(raw[4:9])
            except Exception:
                continue
            body = raw[10:]
            en, zh = (body.split("|", 1) + [""])[:2]
            lines.append({"t": mm * 60 + ss, "en": en.strip(), "zh": zh.strip()})

    # 人声带起音在行首附近微调
    pres = E["pres"]
    pres_flux = np.maximum(np.diff(pres, prepend=pres[0]), 0)
    pres_flux = smooth(pres_flux, 3)
    for ln in lines:
        t0 = ln["t"]
        j0 = int(max(0, (t0 - 0.12) * SR / HOP))
        j1 = int(min(nfr - 1, (t0 + 0.20) * SR / HOP))
        if j1 > j0 + 1:
            j = j0 + int(np.argmax(pres_flux[j0:j1]))
            dt = times[j] - t0
            if -0.10 < dt < 0.16:
                ln["t"] = float(t0 + dt * 0.55)
        # 词级
        words = ln["en"].split()
        ln["words"] = []
        if words:
            # 下一行开始作为上界
            nxt = None
            for other in lines:
                if other["t"] > ln["t"] + 0.25:
                    nxt = other["t"]; break
            span_end = min(nxt or (ln["t"] + 4.0), ln["t"] + 4.5) - 0.12
            span_end = max(span_end, ln["t"] + 0.35)
            # 音节权重
            def syl(w):
                w = w.lower().strip(",.!?;:")
                n = len([c for c in w if c in "aeiouy"])
                return max(1, n)
            wts = np.array([syl(w) for w in words], np.float32)
            wts = wts / wts.sum()
            acc = 0.0
            for w, wt in zip(words, wts):
                w0 = ln["t"] + acc * (span_end - ln["t"])
                acc += wt
                w1 = ln["t"] + acc * (span_end - ln["t"])
                # 吸附到附近起音
                j0 = int(max(0, (w0 - 0.08) * SR / HOP)); j1 = int(min(nfr - 1, (w0 + 0.12) * SR / HOP))
                if j1 > j0 + 1:
                    j = j0 + int(np.argmax(pres_flux[j0:j1]))
                    dt = times[j] - w0
                    if -0.06 < dt < 0.10:
                        w0 += dt * 0.6
                ln["words"].append([w, round(float(w0), 3), round(float(max(w1, w0 + 0.08)), 3)])

    # ---- 立体声宽度
    dsm = np.abs(L - R)
    dsm = np.convolve(dsm, np.ones(1024, np.float32) / 1024, mode="same")
    width = norm01(np.interp(times, np.arange(len(dsm)) / SR, dsm))

    # ---- 导出（每帧特征降采样到 ~86fps 太占空间；用 20ms 步长）
    ds = 2  # 每 2 帧 ≈ 23ms
    def pack(a):
        return [round(float(v), 4) for v in a[::ds]]
    out = {
        "duration": round(dur, 3),
        "sr": SR, "hop": HOP, "nfft": NFFT,
        "frame_dt": round(HOP * ds / SR, 5),
        "bpm": round(float(bpm), 3),
        "beat": round(float(period), 4),
        "beats": [round(float(b), 3) for b in snapped],
        "downbeats": [round(b, 3) for b in downbeats],
        "key": key, "key_corr": round(key_corr, 3),
        "chroma": [round(float(c), 4) for c in chroma],
        "bands": {k: pack(v) for k, v in E.items()},
        "centroid": pack(centroid_n),
        "flatness": pack(flat_n),
        "hfc": pack(hfc),
        "flux": pack(norm01(flux_s)),
        "stereo": pack(width),
        "rms": pack(norm01(np.sqrt(np.convolve(mono ** 2, np.ones(1024, np.float32) / 1024, mode="same"))[::HOP][:nfr])),
        "onsets": [round(float(t), 3) for t in on_t],
        "onset_strength": [round(float(s), 4) for s in on_s],
        "kicks": [[round(float(a), 3), round(float(b), 4)] for a, b in kp],
        "novelty_t": [round(float(t), 3) for t in nov_t],
        "novelty": [round(float(v), 4) for v in nov],
        "lyrics": lines,
    }
    with open(os.path.join(ROOT, "analysis2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"dur={dur:.3f}s  bpm={bpm:.2f}  beat={period:.4f}  beats={len(snapped)}  "
          f"downbeats={len(downbeats)}  key={key}({key_corr:.2f})  onsets={len(on_t)}  "
          f"lines={len(lines)}  frames={nfr}")
    # 粗略结构：新颖度前 12 峰
    order = np.argsort(nov)[::-1]
    picks = []
    for i in order:
        if all(abs(nov_t[i] - p) > 3.0 for p in picks):
            picks.append(float(nov_t[i]))
        if len(picks) >= 14:
            break
    print("novelty peaks:", sorted(round(p, 1) for p in picks))


if __name__ == "__main__":
    main()
