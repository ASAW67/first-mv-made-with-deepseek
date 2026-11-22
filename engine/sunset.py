# -*- coding: utf-8 -*-
"""
sunset.py — 5 秒日落，全部由代码算出来。

画面分两层：
  · 氛围层（天空 / 太阳 / 云 / 光轴 / 远岛 / 海面 / 日光带 / 雾）在**半分辨率**算，
    再上采样——天光本来就是柔的，半分辨率反而更干净、没有色带。
  · 结构层（前景芦苇、礁石、飞鸟、镜头光斑）在全分辨率画，保证边缘锐利。

用法：
  python sunset.py --still 2.4                 # 出一张静帧检查
  python sunset.py --video                     # 300 帧 4K60 成片
"""
import argparse
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np

import kit

F32 = np.float32
DUR = 5.0
FPS = 60

# ────────────────────────────── 参数 ──────────────────────────────
SUN_X = 0.545          # 相对画宽
SUN_Y = 0.556          # 相对画高（地平线上一点点）
HORIZON = 0.625        # 海平线
SUN_R = 0.0210         # 太阳半径（相对画高）

SKY = [
    (0.00, "0A1730"), (0.17, "17224C"), (0.33, "3A2A5E"),
    (0.47, "6C3A6A"), (0.58, "A84A5C"), (0.67, "DA663F"),
    (0.75, "F39B3B"), (0.82, "FFC96B"),
]


def _ramp(stops, n):
    """把 [(t, 'RRGGBB')] 变成 (n,3) 的查色表。"""
    ts = np.array([s[0] for s in stops], F32)
    cs = np.stack([kit.hx(s[1]) for s in stops])          # (k,3)
    x = np.clip(np.linspace(0, 1, n, dtype=F32), 0, 1)
    out = np.empty((n, 3), F32)
    for c in range(3):
        out[:, c] = np.interp(x, ts, cs[:, c])
    return out


def vnoise(w, h, cx, cy, rng, interp=cv2.INTER_CUBIC):
    """值噪声：小网格随机数 + 平滑上采样（带环绕，平移后无接缝）。"""
    cx = max(2, int(cx))
    cy = max(2, int(cy))
    g = rng.random((cy, cx)).astype(F32)
    g = np.concatenate([g, g[:1]], 0)
    g = np.concatenate([g, g[:, :1]], 1)
    return cv2.resize(g, (w, h), interpolation=interp)


def fbm(w, h, sx, sy, rng, octaves=5, persist=0.52, lac=2.0):
    out = np.zeros((h, w), F32)
    tot = 0.0
    amp = 1.0
    for _ in range(octaves):
        out += amp * vnoise(w, h, w / sx, h / sy, rng)
        tot += amp
        amp *= persist
        sx /= lac
        sy /= lac
    out /= tot
    lo, hi = np.percentile(out, 1.5), np.percentile(out, 98.5)
    return np.clip((out - lo) / max(1e-5, hi - lo), 0, 1)


def warp(img, dx, dy, wrap=True):
    """亚像素平移（np.roll 只能整数，会抖动）。"""
    h, w = img.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], F32)
    border = cv2.BORDER_WRAP if wrap else cv2.BORDER_REFLECT
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border)


def sstep(a, b, x):
    den = b - a
    if abs(den) < 1e-6:
        den = 1e-6
    t = np.clip((x - a) / den, 0, 1)
    return t * t * (3 - 2 * t)


# ────────────────────────────── 静态噪声 ──────────────────────────────
class Assets:
    """每个进程只算一次：云的形状、雾气、水纹，全部预生成再按时间平移。"""

    def __init__(self, w, h):
        self.w, self.h = w, h
        rng = np.random.default_rng(20221122)

        # 云：横向拉长
        self.c1 = fbm(w, h, w / 2.3, h / 11.0, rng, octaves=5)
        self.c2 = fbm(w, h, w / 6.5, h / 26.0, rng, octaves=4)
        # 云的密度：低频 + 细节，靠近地平线压扁
        dens = 0.62 * self.c1 + 0.38 * self.c2
        dens = np.clip((dens - 0.34) * 1.9, 0, 1)
        yy = np.linspace(0, 1, h, dtype=F32)[:, None]
        band = np.exp(-((yy - 0.30) / 0.20) ** 2) * 0.85 + \
               np.exp(-((yy - 0.53) / 0.075) ** 2) * 1.0 + \
               np.exp(-((yy - 0.15) / 0.16) ** 2) * 0.55
        self.dens = np.clip(dens * np.clip(band, 0, 1.25), 0, 1)

        # 光轴用的角度噪声 / 雾气 / 水纹
        self.haze = fbm(w, h, w / 9.0, h / 3.2, rng, octaves=3)
        self.water = fbm(w, h, w / 26.0, h / 3.0, rng, octaves=4)
        # 横向长条噪声：用来把水面的反光切成一截一截
        self.streak = fbm(w, h, w / 2.6, h / 34.0, rng, octaves=4)
        self.grain = np.random.default_rng(7).random((h, w)).astype(F32)


# ────────────────────────────── 氛围层 ──────────────────────────────
def atmosphere(t, A, w, h):
    """返回 (h,w,3) 的半分辨率天与海。"""
    u = t / DUR
    f = np.zeros((h, w, 3), F32)
    xs = np.arange(w, dtype=F32)[None, :]
    ys = np.arange(h, dtype=F32)[:, None]
    hy = HORIZON * h
    sx = SUN_X * w
    sy = (SUN_Y + 0.004 * u) * h          # 太阳缓慢下沉
    sr = SUN_R * h
    # ── 天空渐变 ─────────────────────────────────────────────
    stops = [
        (0.00, "0A1730"), (0.17, "17224C"), (0.33, "3A2A5E"),
        (0.47, "6C3A6A"), (0.58, "A84A5C"), (0.67, "DA663F"),
        (0.75, "F39B3B"), (0.82, "FFC96B"),
    ]
    ramp = _ramp(stops, 512)
    yy = np.clip(ys / max(1.0, hy), 0, 1)
    idx = (yy * 511).astype(np.int32)[:, 0]
    sky = ramp[idx][:, None, :] * np.ones((1, w, 1), F32)

    # 太阳周围的暖雾：把地平线附近染热
    d = np.sqrt((xs - sx) ** 2 + ((ys - sy) * 1.28) ** 2)
    warmth = np.exp(-(d / (sr * 12.0)) ** 1.7).astype(F32)
    sky += warmth[..., None] * kit.hx("FF8A3C") * 0.34

    # 高空压暗（自然的暗角式天顶）
    sky *= (1.0 - 0.32 * np.clip(1.0 - ys / max(1.0, hy), 0, 1) ** 1.4)[..., None]

    f[:] = sky

    # ── 太阳本体 + 日冕 ───────────────────────────────────────
    core = sstep(sr * 1.02, sr * 0.955, d)
    limb = sstep(sr * 1.10, sr * 0.50, d)          # 临边昏暗
    halo = (np.exp(-(d / (sr * 1.35)) ** 2.1) * 0.40 +
            np.exp(-(d / (sr * 4.2)) ** 1.45) * 0.26 +
            np.exp(-(d / (sr * 14.0)) ** 1.15) * 0.13)
    sc = kit.mixc(kit.hx("FFF3D2"), kit.hx("FFA845"), np.clip(0.18 + 1.2 * (SUN_Y - 0.40) / 0.25, 0, 1))
    f += halo[..., None] * sc * 0.34
    f += core[..., None] * sc * (1.55 + 1.10 * limb)[..., None]

    # ── 云 ───────────────────────────────────────────────────
    drift_x = 4.0 + 24.0 * u
    drift_y = 1.2 * u
    dens = warp(A.dens, -drift_x, drift_y)
    dens *= (1.0 - 0.55 * sstep(hy * 0.96, hy * 1.14, ys))      # 云不落到海面
    dens = np.clip(dens * 1.05, 0, 1)

    # 朝向太阳的一侧被照亮：比较「当前点」与「朝太阳方向偏移后」的密度
    px = (sin_shift(dens, sx, sy, ys, xs, 9.0))
    rim = np.clip(dens - px, 0, 1) ** 0.8                # 迎光边缘
    body = dens ** 1.15

    # 薄云挡在太阳前面时会「透光发亮」，而不是变灰
    veil = np.clip(1.0 - d / (sr * 4.2), 0, 1) ** 1.5

    cloud_col = kit.mixc(kit.hx("2A1B3A"), kit.hx("8C3E4A"), np.clip(warmth * 1.5, 0, 1))
    cloud_col = kit.mixc(cloud_col, kit.hx("F0A055"), np.clip(rim * 1.6 + warmth * 0.55, 0, 1))

    f *= (1.0 - body[..., None] * 0.82 * (1.0 - 0.88 * veil[..., None]))
    f += cloud_col * (body[..., None] * (0.30 + 0.55 * np.clip(warmth * 1.4, 0, 1))[..., None])
    f += kit.mixc(kit.hx("FFCB7A"), kit.WHITE, 0.35) * (rim[..., None] * 1.15)
    f += sc * (body[..., None] * veil[..., None] * 1.9)

    # ── 光轴（穿过云隙的金色射线）─────────────────────────────
    ang = np.arctan2(ys - sy, xs - sx)
    ray_n = 0.5 + 0.5 * np.sin(ang * 26.0 + 2.2)
    ray_n = ray_n ** 3.2
    ray_d = np.exp(-d / (sr * 11.0)).astype(F32)
    rays = ray_n * ray_d * (1.0 - dens * 0.85) * 0.55
    f += rays[..., None] * kit.hx("FFCf86") * (0.5 + 0.5 * math.sin(t * 2.0))

    # ── 远岛 ─────────────────────────────────────────────────
    isl = np.zeros((h, w), F32)
    pts1 = [(0.00, 0.640), (0.05, 0.622), (0.095, 0.632), (0.13, 0.616),
            (0.165, 0.630), (0.205, 0.640), (0.24, 0.660)]
    pts2 = [(0.00, 0.632), (0.03, 0.626), (0.055, 0.634), (0.07, 0.642)]
    for pts in (pts1, pts2):
        tmp = np.zeros((h, w), F32)
        poly = np.array([[p[0] * w, p[1] * h] for p in pts], F32)
        poly = np.vstack([poly, [[0.0, h * 1.2], [max(p[0] for p in pts) * w, h * 1.2]]])
        cv2.fillPoly(tmp, [poly.astype(np.int32)], 1.0)
        isl = np.maximum(isl, tmp)
    island_col = kit.mixc(kit.hx("1A1526"), kit.hx("5A2A3A"), np.clip(warmth * 0.9, 0, 1))
    f *= (1.0 - isl[..., None] * 0.94)
    f += island_col * (isl[..., None] * 0.30)

    # ── 海面 ─────────────────────────────────────────────────
    sea_mask = sstep(hy - 1.2, hy + 1.2, ys)
    yl = np.clip((ys - hy) / max(1.0, h - hy), 0, 1)     # 0 在地平线，1 在脚边

    refl = _ramp([(0.0, "FFC46B"), (0.05, "E06A44"), (0.14, "8A3557"),
                  (0.34, "33214A"), (0.70, "131B36"), (1.0, "070C1C")], 256)
    ridx = (np.clip(yl, 0, 1) * 255).astype(np.int32)[:, 0]
    sea = refl[ridx][:, None, :] * np.ones((1, w, 1), F32)

    # 波浪：几乎水平的条纹，频率向地平线压缩（透视），并带轻微起伏
    wv = warp(A.water, 3.0 * u, -8.0 * u)
    sk = warp(A.streak, 2.0 * u, -4.0 * u)
    depth = yl + 0.055
    undul = 2.4 * np.sin(xs / w * 9.0 + yl * 7.0 + t * 0.7)
    band = 0.5 + 0.5 * np.sin(9.4 / depth + undul - t * 2.1)
    band2 = 0.5 + 0.5 * np.sin(21.0 / depth + t * 3.1 + wv * 1.6)
    fade_in = sstep(0.0, 0.05, yl)          # 贴近地平线的条纹太密，淡出避免摩尔纹
    sea *= (0.80 + (0.11 * band + 0.06 * band2) * fade_in)[..., None]

    # 日光带：柱状倒影，只有波峰反光，且被横向噪声切碎
    colw = (0.017 + 0.105 * yl ** 1.25) * w
    cw = np.exp(-np.abs(xs - sx) / colw)
    crest = np.clip(band * 2.0 - 0.62, 0, 1) ** 1.1
    crest = crest * np.clip(0.30 + 1.05 * sk, 0, 1.4)
    fine = np.clip((warp(A.grain, 0, -30.0 * u) - 0.76) * 4.6, 0, 1)
    path = cw * (0.07 + 1.60 * crest) * (0.55 + 0.45 * (1 - yl ** 0.8))
    sea += kit.hx("FFD79A") * (path[..., None] * 1.0)
    sea += kit.hx("FFF0C8") * ((cw * crest * fine)[..., None] * 1.25)
    sea += kit.hx("FF9A4A") * (cw[..., None] * 0.12 * (1 - yl * 0.8)[..., None])

    sea_mask = kit.blur(sea_mask, 1.6)
    f = f * (1 - sea_mask[..., None]) + sea * sea_mask[..., None]

    # 地平线那道亮线（靠近太阳更亮）
    hl = np.exp(-((ys - hy) / (h * 0.0035)) ** 2) * (0.35 + 0.65 * np.exp(-np.abs(xs - sx) / (w * 0.30)))
    f += kit.hx("FFE0B0") * (hl[..., None] * 0.42)

    # ── 地平线雾带 ────────────────────────────────────────────
    hxz = np.exp(-np.abs(xs - sx) / (w * 0.42))
    haze = np.exp(-((ys - hy) / (h * 0.017)) ** 2) * (0.45 + 0.55 * warp(A.haze, -2.0 * u, 0))
    f += kit.hx("FFC98A") * ((haze * 0.15 * np.clip(0.25 + warmth, 0, 1.3) * (0.30 + 0.70 * hxz))[..., None])

    # ── 太阳上方的光柱（低太阳时的天顶光）──────────────────────
    pillar = np.exp(-np.abs(xs - sx) / (w * 0.020)) * np.exp(-np.clip(sy - ys, 0, None) / (h * 0.22))
    pillar = pillar * (ys < sy)
    f += kit.hx("FFD9A4") * (pillar[..., None] * 0.10)

    # ── 逆光横向拉丝（变形宽银幕的味道）───────────────────────
    streak = np.exp(-np.abs(ys - sy) / (sr * 0.55)) * np.exp(-np.abs(xs - sx) / (w * 0.30))
    f += kit.hx("FFD9A8") * (streak[..., None] * 0.24)

    return np.clip(f, 0, 6.0)


def sin_shift(dens, sx, sy, ys, xs, k):
    """取「朝太阳方向偏移 k 像素」处的密度（用于算迎光边缘）。"""
    dy = np.sign(sy - ys) * k
    dx = np.sign(sx - xs) * k * 0.6
    mx = np.broadcast_to((xs - dx).astype(F32), dens.shape).copy()
    my = np.broadcast_to((ys - dy).astype(F32), dens.shape).copy()
    return cv2.remap(dens, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ────────────────────────────── 前景 ──────────────────────────────
def bez(p0, p1, p2, n=40):
    s = np.linspace(0, 1, n, dtype=F32)[:, None]
    return (1 - s) ** 2 * np.array(p0) + 2 * (1 - s) * s * np.array(p1) + s ** 2 * np.array(p2)


def blade(mask, pts, w0, w1, S):
    """把一条贝塞尔曲线画成两端粗细不同的叶片（芦苇叶 / 草叶）。"""
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-6)
    k = np.linspace(0, 1, len(pts), dtype=F32)[:, None]
    w = (w0 + (w1 - w0) * k ** 1.9) * S * 0.5
    poly = np.concatenate([pts + n * w, (pts - n * w)[::-1]], 0)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1.0, cv2.LINE_AA)


def foreground(f, t, S, W, H):
    """全分辨率画：前景草叶、礁石、飞鸟。"""
    u = t / DUR
    ink = kit.hx("05070C")
    rim_c = kit.hx("FF8B3D")

    def blade_layer(count, base, spread, hmin, hmax, wmin, wmax, phase, blur=0.0, seed=0):
        m = np.zeros((H, W), F32)
        rnd = np.random.default_rng(seed)
        for i in range(count):
            uu = i / max(1.0, count - 1.0)
            bx = (base[0] + spread * (uu - 0.5) + 0.008 * math.sin(i * 3.7 + seed)) * W
            by = base[1] * H
            tipp = (hmin + (hmax - hmin) * rnd.random()) * H
            lean = (0.028 + 0.115 * rnd.random()) * W * (1 if i % 2 else -1) * (1.0 + 0.4 * uu)
            sway = math.sin(t * (0.95 + 0.2 * rnd.random()) + i * 0.9 + phase) * (0.006 + 0.005 * rnd.random()) * W
            p0 = (bx, by)
            p1 = (bx + lean * 0.10, by - (by - tipp) * 0.58)
            p2 = (bx + lean + sway, tipp)
            pts = bez(p0, p1, p2, 40)
            blade(m, pts, wmin + (wmax - wmin) * rnd.random(), 1.0, S)
        if blur > 0:
            m = kit.blur(m, blur * S)
        return m

    # ── 极前景：两片虚焦的大叶（做景深）──────────────────────
    far = blade_layer(2, (0.020, 1.20), 0.05, 0.30, 0.46, 46.0, 64.0, 0.4, blur=3.2, seed=11)
    f *= (1 - far[..., None] * 0.96)

    # ── 主前景草丛 ───────────────────────────────────────────
    main = blade_layer(7, (0.075, 1.10), 0.10, 0.42, 0.66, 20.0, 32.0, 0.0, blur=0.5, seed=3)
    # 右下礁石
    rock = np.array([[0.760, 1.04], [0.800, 0.988], [0.845, 0.972], [0.890, 0.984],
                     [0.934, 0.968], [0.968, 0.980], [1.03, 0.972], [1.03, 1.04]], F32)
    rock[:, 0] *= W
    rock[:, 1] *= H
    cv2.fillPoly(main, [rock.astype(np.int32)], 1.0)
    # 底部暗岸线，避免海面在画面下缘被硬切
    cv2.rectangle(main, (0, int(H * 0.995)), (W, H), 1.0, -1)

    sh = warp(main, 1.1 * S, 0.8 * S, wrap=False)
    rim = kit.blur(np.clip(sh - main, 0, 1), 0.6 * S)

    f *= (1 - main[..., None] * 0.99)
    f += ink * (main[..., None] * 0.9)
    f += rim_c * (rim[..., None] * 0.40)

    # ── 飞鸟（背光剪影）───────────────────────────────────────
    bird = np.zeros((H, W), F32)
    for i in range(6):
        bx = (0.245 + 0.072 * i + 0.014 * u * (6 + i % 3)) * W
        by = (0.410 - 0.018 * i + 0.010 * math.sin(i * 2.2 + u * 3)) * H
        sz = (0.0135 - 0.0010 * i) * H
        fl = math.sin(t * 7.2 + i * 1.35)
        curve = 0.72 * fl
        segs = [((bx - sz * 1.0, by - sz * curve * 0.9), (bx, by)),
                ((bx, by), (bx + sz * 1.0, by - sz * curve * 0.9))]
        for p, q in segs:
            cv2.line(bird, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                     1.0, max(2, int(2.3 * S)), cv2.LINE_AA)
    bird = kit.blur(bird, 0.5 * S)
    f *= (1 - np.clip(bird, 0, 1)[..., None] * 0.95)

    # ── 飞鸟（背光剪影）───────────────────────────────────────
    bird = np.zeros((H, W), F32)
    for i in range(7):
        bx = (0.235 + 0.070 * i + 0.014 * u * (6 + i % 3)) * W
        by = (0.415 - 0.016 * i + 0.010 * math.sin(i * 2.2 + u * 3)) * H
        sz = (0.0135 - 0.0009 * i) * H
        fl = math.sin(t * 7.5 + i * 1.3)
        curve = 0.72 * fl
        segs = [((bx - sz * 1.0, by - sz * curve * 0.9), (bx, by)),
                ((bx, by), (bx + sz * 1.0, by - sz * curve * 0.9))]
        for p, q in segs:
            cv2.line(bird, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                     1.0, max(2, int(2.4 * S)), cv2.LINE_AA)
    bird = kit.blur(bird, 0.5 * S)
    f *= (1 - np.clip(bird, 0, 1)[..., None] * 0.95)


# ────────────────────────────── 单帧 ──────────────────────────────
_ASSETS = {}


def frame(t, S, W, H):
    key = (W, H)
    if key not in _ASSETS:
        hw, hh = W // 2, H // 2
        _ASSETS[key] = (Assets(hw, hh), hw, hh)
    A, hw, hh = _ASSETS[key]

    plate = atmosphere(t, A, hw, hh)
    f = kit.up(plate, W, H)
    f = kit.blur(f, 0.9 * S)

    foreground(f, t, S, W, H)

    # ── 调色 ─────────────────────────────────────────────────
    f = np.clip(f, 0, None)
    f = kit.tonemap_soft(f, 1.05)
    f = kit.tone(f, exposure=1.02, contrast=1.08, sat=1.07, lift=0.004, gamma=0.97)
    # 暗部偏青、亮部偏暖的分色调
    lum = (f[..., 0] * 0.30 + f[..., 1] * 0.59 + f[..., 2] * 0.11)[..., None]
    f = f + (1 - lum) * kit.hx("0A2030") * 0.10 + lum * kit.hx("FFC08A") * 0.055

    # ── 摄影机：极缓的推近（5 秒里 1.4%）──────────────────────
    u = t / DUR
    M = cv2.getRotationMatrix2D((SUN_X * W, 0.60 * H), 0.0, 1.0 + 0.014 * u)
    M[0, 2] -= 0.004 * W * u
    f = cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    f = kit.bloom(f, thresh=0.70, radius=30 * S, strength=0.42, downscale=3)
    f = kit.vignette(f, strength=0.30, power=2.2)
    f = kit.grain(f, amt=0.016, seed=int(t * 1000) % 9973)
    return np.clip(f, 0, 1)


# ────────────────────────────── 渲染 ──────────────────────────────
def make_ambience(path, sr=48000):
    """用噪声合成 5 秒海浪 + 风的环境声（无采样、无素材）。"""
    n = int(DUR * sr)
    t = np.arange(n, dtype=F32) / sr
    rng = np.random.default_rng(4242)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)

    def shaped(noise, lo, hi, tilt=1.0):
        F = np.fft.rfft(noise)
        band = np.exp(-((np.log(freqs + 1.0) - np.log((lo + hi) * 0.5)) ** 2) /
                      (2 * (np.log(hi / lo) / 2.2) ** 2)) * (freqs / ((lo + hi) * 0.5)) ** (-tilt * 0.0)
        band[freqs < lo] *= np.exp(-((np.log(lo / (freqs[freqs < lo] + 1e-6))) ** 2) * 2.0)
        band[freqs > hi] *= np.exp(-((np.log(freqs[freqs > hi] / hi)) ** 2) * 1.4)
        return np.fft.irfft(F * band, n)

    # 浪：三段不同周期的涌动
    swell = (0.55 + 0.45 * np.sin(2 * np.pi * t / 2.35 - 0.6)) \
          * (0.60 + 0.40 * np.sin(2 * np.pi * t / 3.71 + 1.3)) \
          * (0.70 + 0.30 * np.sin(2 * np.pi * t / 1.43 + 2.6))
    swell = 0.35 + 0.65 * swell / max(1e-6, swell.max())

    out = np.zeros((n, 2), F32)
    for ch in range(2):
        r = np.random.default_rng(1000 + ch)
        low = shaped(r.standard_normal(n), 90.0, 620.0)      # 浪的低频轰鸣
        mid = shaped(r.standard_normal(n), 500.0, 2600.0)    # 浪花
        air = shaped(r.standard_normal(n), 3200.0, 11000.0)  # 空气感
        wind = shaped(r.standard_normal(n), 220.0, 900.0) * (0.6 + 0.4 * np.sin(2 * np.pi * t / 4.3 + ch))
        sig = low * (0.55 + 0.55 * swell) + mid * (0.16 + 0.30 * swell) * 0.9 \
              + air * 0.05 * (0.4 + 0.6 * swell) + wind * 0.10
        out[:, ch] = sig / (np.abs(sig).max() + 1e-6)

    fade = np.ones(n, F32)
    fi = int(0.40 * sr)
    fo = int(1.10 * sr)
    fade[:fi] = np.linspace(0, 1, fi) ** 1.6
    fade[-fo:] = np.linspace(1, 0, fo) ** 1.3
    out *= fade[:, None] * 0.55

    pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16)
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def render_video(out, scale, workers, crf):
    total = int(DUR * FPS)
    sw, sh = 1920 * scale, 1080 * scale
    ff = kit.FFMPEG
    tmp = os.path.join(kit.ROOT, "_sunset_frames")
    os.makedirs(tmp, exist_ok=True)

    t0 = time.time()
    kit.init(scale)
    if workers <= 1:
        idxs = range(total)
        for i in idxs:
            f = frame(i / FPS, scale, sw, sh)
            cv2.imwrite(os.path.join(tmp, "f%04d.png" % i),
                        cv2.cvtColor((f * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            if i % 20 == 0:
                print("  %d/%d  %.1fs" % (i, total, time.time() - t0), flush=True)
    else:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_worker, [(i, scale, sw, sh) for i in range(total)])):
                if i % 40 == 0:
                    print("  %d/%d  %.1fs" % (i, total, time.time() - t0), flush=True)

    wav = os.path.join(kit.ROOT, "_sunset_audio.wav")
    make_ambience(wav)
    cmd = [ff, "-y", "-v", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "f%04d.png"),
           "-i", wav,
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-x264-params", "aq-mode=3:psy-rd=1.0,0.15",
           "-c:a", "aac", "-b:a", "192k", "-shortest",
           "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    print("done %.1fs -> %s  %.1f MB" % (time.time() - t0, out, os.path.getsize(out) / 1e6))


def _worker(args):
    i, scale, sw, sh = args
    if not hasattr(_worker, "k"):
        kit.init(scale)
        _worker.k = 1
    f = frame(i / FPS, scale, sw, sh)
    cv2.imwrite(os.path.join(kit.ROOT, "_sunset_frames", "f%04d.png" % i),
                cv2.cvtColor((f * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", type=float, default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--out", default=None)
    ap.add_argument("--times", default=None, help="逗号分隔的多个静帧时间")
    a = ap.parse_args()

    kit.init(a.scale)
    sw, sh = 1920 * a.scale, 1080 * a.scale

    if a.times:
        ts = [float(x) for x in a.times.split(",")]
        tiles = []
        for i, t in enumerate(ts):
            f = frame(t, a.scale, sw, sh)
            img = (f * 255 + 0.5).astype(np.uint8)
            cv2.imwrite(os.path.join(kit.ROOT, "out", "sunset_t%04.1f.png" % t),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            tiles.append(cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA))
        sheet = np.concatenate(tiles, 0)
        cv2.imwrite(os.path.join(kit.ROOT, "sunset_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print("still sheet -> sunset_sheet.png")

    if a.still is not None:
        f = frame(a.still, a.scale, sw, sh)
        img = (f * 255 + 0.5).astype(np.uint8)
        p = os.path.join(kit.ROOT, "out", "sunset_still.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print("still ->", p)

    if a.video:
        out = a.out or os.path.join(kit.ROOT, "out", "sunset_5s_4K60.mp4")
        render_video(out, a.scale, a.workers, a.crf)


if __name__ == "__main__":
    main()
