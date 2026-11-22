# -*- coding: utf-8 -*-
"""ch_b.py — 第二幕「几何」30.00 ─ 51.93

她把身体重组成各种形状：
  POINTS   她散成点，点再铺成一个球面
  CIRCLE   球塌成完美圆环（先侧视成一条线，再转正）
  SINE     圆环被拉成正弦波，振幅跟着 bass 呼吸
  INFINITY 正弦卷成无穷符号，粒子沿 ∞ 双向流动
  CURRENT  电流穿过她，粒子被拉成一根根竖线
  BLIND    去色 / 只剩剪影，两块黑斑盖住眼睛
  DIZZY    整帧滚动 + 双重曝光 + 倾斜的文字

只依赖 kit / mus / stage / cloud / film2；不改引擎。
"""
import math
import numpy as np
import cv2
import kit, mus, stage, cloud
from film2 import scene

F32 = np.float32
PI = math.pi
TAU = 2.0 * math.pi

MINT, CORAL, VIOLET, GOLD = kit.MINT, kit.CORAL, kit.VIOLET, kit.GOLD
CREAM, ASH, PALE, ICE = kit.CREAM, kit.ASH, kit.PALE, kit.ICE
BLACK, WHITE = kit.BLACK, kit.WHITE
TEAL = kit.hx("2FA6A0")
DEEP = kit.hx("02060B")
SLATE = kit.hx("0B2231")

K_DIST = 1500.0
K_FOCAL = 1150.0


# ============================================================ 私有工具
def deep_base(f, c0="050C14", c1="010306", power=1.3):
    """深海蓝黑底：上深下更黑（1/4 分辨率算完放大，省时间）。"""
    g = kit.vgrad(kit.SH // 4, kit.SW // 4, kit.hx(c0), kit.hx(c1), power)
    f += kit.up(g, kit.SW, kit.SH)
    return f


def glow(f, cx, cy, radius, color, strength=1.0, power=1.8, ratio=1.0):
    """1/4 分辨率的柔光团：颜色/强度都在小图上乘好，再放大叠上去。"""
    if strength <= 0.004 or radius <= 1.0:
        return f
    h, w = max(4, kit.SH // 4), max(4, kit.SW // 4)
    yy = (np.arange(h, dtype=F32) * (kit.H / h) - cy)[:, None]
    xx = ((np.arange(w, dtype=F32) * (kit.W / w) - cx) / max(ratio, 1e-3))[None, :]
    d = np.sqrt(xx * xx + yy * yy) / radius
    m = np.clip(1.0 - d, 0, 1) ** power
    m3 = m[..., None] * (np.asarray(color, F32) * strength)
    f += kit.up(m3, kit.SW, kit.SH)
    return f


def proj(P, dist=K_DIST, focal=K_FOCAL, cx=None, cy=None, oy=0.0):
    """和 cloud.Cloud.project 同一套投影，方便把 2D 图形钉在粒子上。"""
    cx = kit.W * 0.5 if cx is None else cx
    cy = kit.H * 0.5 if cy is None else cy
    z = np.maximum(P[:, 2] + dist, 40.0)
    k = focal / z
    return cx + P[:, 0] * k, cy + oy - P[:, 1] * k, k


def disc(f, cx, cy, r, color, alpha=1.0, soft=0.28, mul=False):
    """本地小圆斑（只碰周围几十像素，便宜）。mul=True 用来压黑。"""
    if r <= 0.2 or alpha <= 0.004:
        return f
    cxp, cyp, rp = cx * kit.S, cy * kit.S, max(1.0, r * kit.S)
    x0 = max(0, int(cxp - rp * 1.7) - 1); x1 = min(kit.SW, int(cxp + rp * 1.7) + 2)
    y0 = max(0, int(cyp - rp * 1.7) - 1); y1 = min(kit.SH, int(cyp + rp * 1.7) + 2)
    if x1 <= x0 or y1 <= y0:
        return f
    yy = (np.arange(y0, y1, dtype=F32) + 0.5 - cyp)[:, None]
    xx = (np.arange(x0, x1, dtype=F32) + 0.5 - cxp)[None, :]
    d = np.sqrt(xx * xx + yy * yy)
    m = np.clip((rp - d) / max(1.0, rp * soft), 0, 1)
    m = (m * m * (3 - 2 * m)) * alpha
    sub = f[y0:y1, x0:x1]
    if mul:
        sub *= (1 - m)[..., None]
    else:
        sub += m[..., None] * np.asarray(color, F32)
    return f


def add_lines(f, segs, color, alpha=1.0, thick=1.4, blur=0.0):
    """一批折线画进同一张半分辨率 mask，颜色在小图上乘好，只合成一次。"""
    if alpha <= 0.004 or not segs:
        return f
    h, w = max(2, kit.SH // 2), max(2, kit.SW // 2)
    m = np.zeros((h, w), F32)
    for pts in segs:
        p = (np.asarray(pts, F32) * (kit.S * 0.5)).astype(np.int32)
        if len(p) < 2:
            continue
        cv2.polylines(m, [p], False, 1.0, max(1, int(thick * kit.S * 0.5)), cv2.LINE_AA)
    if blur > 0:
        m = kit.blur(m, max(0.5, blur * kit.S * 0.5))
    m3 = m[..., None] * (np.asarray(color, F32) * (alpha * 1.55))
    f += kit.up(m3, kit.SW, kit.SH)
    return f


def soft_ring(f, cx, cy, r, thick, color, alpha, ratio=1.0):
    """1/4 分辨率的柔光环（拍点扩散、起振点用）。"""
    if alpha <= 0.004 or r <= 0.5:
        return f
    h, w = max(4, kit.SH // 4), max(4, kit.SW // 4)
    sc = h / kit.H
    yy = (np.arange(h, dtype=F32) - cy * sc)[:, None]
    xx = (np.arange(w, dtype=F32) - cx * (w / kit.W))[None, :] / max(ratio, 1e-3)
    d = np.sqrt(xx * xx + yy * yy)
    m = np.exp(-((d - r * sc) / max(0.9, thick * sc)) ** 2)
    m3 = m[..., None] * (np.asarray(color, F32) * (alpha * 1.2))
    f += kit.up(m3, kit.SW, kit.SH)
    return f


def band_h(f, pos, thick, color, alpha):
    """一条横向软光带（电流扫描），只碰它自己那几行。"""
    if alpha <= 0.004:
        return f
    r0 = max(0, int((pos - 3.4 * thick) * kit.SH))
    r1 = min(kit.SH, int((pos + 3.4 * thick) * kit.SH) + 2)
    if r1 <= r0:
        return f
    y = (np.arange(r0, r1, dtype=F32) / kit.SH - pos) / max(thick, 1e-4)
    m = np.exp(-y * y) * alpha
    f[r0:r1] += m[:, None, None] * np.asarray(color, F32)
    return f


def vline(f, x, color, alpha, wpx=1):
    if alpha <= 0.004:
        return f
    xi = int(x * kit.S)
    if xi < 0 or xi >= kit.SW:
        return f
    f[:, max(0, xi):min(kit.SW, xi + max(1, int(wpx * kit.S)))] += np.asarray(color, F32) * alpha
    return f


def term_rows(f, t, rows, x, y, t0, step, size=19, lead=1.62, alpha=1.0, cps=0.34, caret=True):
    """终端逐行逐字打印。rows = [[(txt, color, font), ...], ...]"""
    for i, parts in enumerate(rows):
        a0 = t0 + i * step
        if t < a0:
            break
        total = sum(len(p[0]) for p in parts)
        n = max(1, int(total * kit.cl((t - a0) / cps)))
        xx = x
        for txt, col, fp in parts:
            if n <= 0:
                break
            s = txt[:n]
            n -= len(s)
            kit.draw_text(f, s, xx, y + i * size * lead, size, col, fp, alpha, 0.6)
            xx += kit.text_width(fp, size, s, 0.6)
        if caret and i == len(rows) - 1 and int((t - a0) * 3.2) % 2 == 0:
            kit.draw_text(f, "_", xx + 1, y + i * size * lead, size, MINT, kit.F_MONO, alpha * 0.85, 0.0)
    return f


def tilted_text(f, txt, cx, cy, size, color, fp, ang=0.0, alpha=1.0, tracking=1.6):
    """逐字排版并整体倾斜（比 warp 一张大图便宜，也不会糊）。"""
    total = kit.text_width(fp, size, txt, tracking)
    x = cx - total * 0.5
    tn = math.tan(ang)
    for ch in txt:
        w = kit.text_width(fp, size, ch, tracking)
        if ch.strip():
            kit.draw_text(f, ch, x, cy + (x + w * 0.5 - cx) * tn, size, color, fp, alpha, tracking)
        x += w
    return f


def threads(cl, R=150.0, Hh=780.0, cols=168):
    """把粒子绑到一根根竖线上（电流段的「竖线」形态）。"""
    idx = np.floor(cl.u * cols)
    ang = idx / cols * TAU * 7.0
    rad = R * (0.34 + 0.66 * cl.rnd)
    x = np.cos(ang) * rad
    z = np.sin(ang) * rad
    y = (cl.v - 0.5) * Hh
    return np.stack([x, y, z], 1).astype(F32)


_SFC = {}
_BL = {}


def sf(cl, path, h, depth):
    """sprite_field 跨帧缓存：立绘粒子化很贵，形状每帧完全一样。"""
    key = (path, round(h, 1), round(depth, 2))
    r = _SFC.get(key)
    if r is None:
        r = cloud.sprite_field(cl.n, cl.rng, path, h, depth)
        _SFC[key] = r
    return r


def cv2line(m, p0, p1, v, thick):
    cv2.line(m, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), float(v), thick, cv2.LINE_AA)


def grid_floor(f, t, y=0.80, horizon=0.56, color=None, alpha=0.30, lines=18, persp=3.2):
    """透视网格地面（半分辨率绘制后放大）。"""
    color = MINT if color is None else color
    bw, bh = kit.SW // 2, kit.SH // 2
    m = np.zeros((bh, bw), F32)
    yh = horizon * bh; yb = y * bh + bh * 0.35
    cx = bw / 2
    for i in range(-lines, lines + 1):
        x0 = cx + i * bw * 0.055
        cv2line(m, (cx + i * bw * 0.015, yh), (x0 + i * bw * 0.10, yb), 1.0, 1)
    for j in range(1, 15):
        q = (j / 15.0) ** persp
        yy = yh + (yb - yh) * q
        cv2line(m, (0, yy), (bw, yy), 1.0, 1)
    m = kit.blur(m, 0.5 * kit.S)
    fade = np.clip((np.arange(bh, dtype=F32) / bh - horizon) * 2.6, 0, 1)[:, None]
    m = m * fade * alpha
    f += kit.up(m, kit.SW, kit.SH)[..., None] * np.asarray(color, F32)
    return f


# ================================================================ 30.00 点
@scene(30.00, 33.92, "POINTS", "B")
def points(c):
    t = c.t
    f = c.f
    lt = t - 30.00
    pl = c.pulse(7.0)
    c.post.update(grain=0.030, vignette=0.54, bloom=(0.58, 22, 0.52), chroma=0.35)
    deep_base(f, "050C15", "010306", 1.35)
    glow(f, kit.W * 0.5, kit.H * 0.46, 780, kit.hx("0C3540"), 0.17, 1.9)

    # 开场：一个点被点亮，然后整个人散开
    if lt < 0.55:
        a = 1.0 - kit.cl(lt / 0.55)
        disc(f, kit.W * 0.5, kit.H * 0.46, 3.0 + 46 * a, CREAM, 0.55 * a)
        vline(f, kit.W * 0.5, MINT, 0.5 * a, 1)
        vline(f, kit.W * 0.5 - 90 * (1 - a), MINT, 0.25 * a, 1)
        vline(f, kit.W * 0.5 + 90 * (1 - a), MINT, 0.25 * a, 1)

    # 立绘粒子 → 球面点云
    body = sf(c.cl, stage.MAID_L, 1000.0, 26.0)
    lum = body["lum"]
    sphere = c.cl.sh(cloud.shape_sphere, 300.0)
    yaw = 0.16 + lt * 0.40
    pit = 0.07 * math.sin(lt * 0.9)
    A = c.cl.transform(body["P"], yaw=0.05, pitch=pit, scale=0.72, offset=(0, -16, 0))
    B = c.cl.transform(sphere, offset=(0, -16, 0))
    p = kit.cl((lt - 0.86) / 1.15)
    P = c.cl.morph(A, B, p, stagger=0.60, curve=lambda q: q * q * (3 - 2 * q))
    P = c.cl.transform(P, yaw=yaw, pitch=pit)

    # 球面点云打个光：亮的那面朝左上，边上暗下来，才看得出是球不是一块饼
    nrm = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-3)
    lam = np.clip(nrm[:, 0] * -0.42 + nrm[:, 1] * 0.34 + nrm[:, 2] * 0.72, 0, 1) ** 0.85
    shade = (1.0 - p) + p * (0.16 + 1.25 * lam)
    depth = np.clip(1.28 - (P[:, 2] + 130.0) / 700.0, 0.15, 1.6)
    br = (0.16 + 1.60 * lum) * depth * shade * (0.86 + 0.50 * pl)
    km = K_FOCAL / K_DIST

    def cmap(xs, ys, zz, kk):
        q = np.clip((kk / km - 0.88) / 0.46, 0, 1)
        return kit.mixc(TEAL, CREAM, q * 0.92)

    c.cl.draw(f, P, dist=K_DIST, focal=K_FOCAL, color_map=cmap, bright=br,
              size=0.90 + 0.14 * p, blur_r=1.0, glow_gain=0.30 + 0.16 * p,
              core_gain=1.05, shade=0.55,
              target=0.41 + 0.20 * p + 0.08 * pl, tm=0.80 + 0.50 * p)

    # 炸开那一下：白闪 + 冲击环
    fl = math.exp(-((lt - 1.02) / 0.135) ** 2)
    if fl > 0.01:
        f += WHITE * (0.24 * fl)
        soft_ring(f, kit.W * 0.5, kit.H * 0.46, 30 + 380 * (1 - fl), 26, CREAM, 0.40 * fl)

    # 拍点上扩散的光环
    soft_ring(f, kit.W * 0.5, kit.H * 0.46, 110 + 430 * (1 - pl), 26, MINT, 0.30 * pl * pl)

    # 左下角终端：p ∈ R³
    rows = [
        [(">>> self = PointCloud(n=62000)", MINT, kit.F_MONO)],
        [(">>> for p in self: p", CREAM, kit.F_MONO), (" \u2208 ", MINT, kit.F_SANS_SC),
         ("R\u00b3", CREAM, kit.F_MONO)],
        [("       x, y, z = p.coords", PALE, kit.F_MONO)],
        [("       assert kind(p) == 'point'", ASH, kit.F_MONO)],
        [(">>> self.morph(Sphere(r=300))", MINT, kit.F_MONO)],
    ]
    term_rows(f, t, rows, 88, kit.H * 0.585, 30.04, 0.42, size=19, lead=1.62)
    a = kit.cl((lt - 2.30) / 0.5)
    if a > 0:
        kit.draw_text(f, "\u5982\u679c\u6211\u662f\u4e00\u7ec4\u70b9", 92, kit.H * 0.585 + 5 * 19 * 1.62 + 12,
                      22, MINT, kit.F_SERIF_SC, a * 0.62, 3.0)

    kit.draw_text(f, "SPHERE  r=300  n=62000", kit.W - 92, kit.H * 0.145, 18, ASH, kit.F_MONO,
                  0.42, 1.6, "right")
    kit.draw_text(f, "yaw %.2f" % yaw, kit.W - 92, kit.H * 0.145 + 26, 18, TEAL, kit.F_MONO,
                  0.42, 1.6, "right")
    c.lyric(size=42, y=kit.H * 0.855)


# ================================================================ 33.92 圆
@scene(33.92, 37.60, "CIRCLE", "B")
def circle(c):
    t = c.t
    f = c.f
    lt = t - 33.92
    pl = c.pulse(8.0)
    c.post.update(grain=0.028, vignette=0.50, bloom=(0.55, 24, 0.50), chroma=0.25)
    stage.bg_image(f, stage.BG_PALACE, zoom=1.24 + lt * 0.020, dy=-12, dim=0.35,
                   tint=kit.hx("08202E"), tint_k=0.45)
    stage.caustics(f, t, strength=0.055, speed=0.26)
    glow(f, kit.W * 0.5, kit.H * 0.46, 520, kit.hx("0E3E4C"), 0.20, 1.8)

    ring = c.cl.sh(cloud.shape_ring, 300.0, 5.6, "xy")
    yaw = (PI * 0.5) * (1.0 - kit.ease_out(kit.cl((lt - 0.10) / 1.10))) + 0.05
    Rk = 1.0 + 0.026 * pl
    cx, cy = kit.W * 0.5, kit.H * 0.455
    P = c.cl.transform(ring, yaw=yaw, scale=Rk)
    ang_r = np.arctan2(P[:, 1], P[:, 0])
    sweep = 0.30 + 1.35 * np.clip(np.cos(ang_r - t * 1.35), 0, 1) ** 2
    depth = np.clip(1.20 - P[:, 2] / 620.0, 0.24, 1.45)
    br = sweep * depth * (0.88 + 0.30 * pl)

    def cmap_c(xs, ys, zz, kk):
        qq = np.clip((zz + 160.0) / 320.0, 0, 1)
        return kit.mixc(CREAM, TEAL, qq)

    c.cl.draw(f, P, dist=K_DIST, focal=K_FOCAL, cx=cx, cy=cy, color_map=cmap_c, bright=br,
              size=1.20, blur_r=1.4, glow_gain=0.50, shade=0.45, target=0.40 + 0.14 * pl)

    # 圆心站着一个小小的她
    stage.character(f, stage.MAID_L, cx, kit.H * 0.705, 412, alpha=0.72,
                    tint=kit.hx("8CBECC"), tint_k=0.38, rim=MINT, rim_k=0.26, glow=0.02)

    # 刻度环 + 正在画出来的周长
    n_t = 72
    aa = np.linspace(0, TAU, n_t, endpoint=False)
    r0, r1 = 300.0 * 1.040, 300.0 * 1.088
    p0 = np.stack([np.cos(aa) * r0 * Rk, np.sin(aa) * r0 * Rk, np.zeros(n_t, F32)], 1)
    p1 = np.stack([np.cos(aa) * r1 * Rk, np.sin(aa) * r1 * Rk, np.zeros(n_t, F32)], 1)
    q = np.concatenate([p0, p1], 0)
    q = c.cl.transform(q, yaw=yaw)
    sx, sy, _ = proj(q, K_DIST, K_FOCAL, cx=cx, cy=cy)
    segs = [[(sx[i], sy[i]), (sx[i + n_t], sy[i + n_t])] for i in range(n_t)]
    add_lines(f, segs, MINT, 0.34 + 0.30 * pl, thick=1.2)

    prog = kit.cl((lt - 0.95) / 1.85)
    if prog > 0:
        na = max(2, int(160 * prog))
        at = np.linspace(0, TAU * prog, na)
        pa = np.stack([np.cos(at) * 300.0 * 1.016, np.sin(at) * 300.0 * 1.016, np.zeros(na, F32)], 1)
        pa = c.cl.transform(pa, yaw=yaw)
        ax_, ay_, _ = proj(pa, K_DIST, K_FOCAL, cx=cx, cy=cy)
        add_lines(f, [list(zip(ax_, ay_))], CREAM, 0.60 + 0.25 * pl, thick=2.0, blur=1.2)
        disc(f, ax_[-1], ay_[-1], 5.0, CREAM, 0.85)

    # 半径标注
    rq = np.array([[0.0, 0.0, 0.0], [300.0 * Rk, 0.0, 0.0]], F32)
    rq = c.cl.transform(rq, yaw=yaw)
    rx, ry, _ = proj(rq, K_DIST, K_FOCAL, cx=cx, cy=cy)
    add_lines(f, [[(rx[0], ry[0]), (rx[1], ry[1])]], ASH, 0.45, thick=1.1)
    kit.draw_text(f, "r", (rx[0] + rx[1]) * 0.5 + 6, (ry[0] + ry[1]) * 0.5 - 30, 24, PALE,
                  kit.F_SERIF_I, 0.65, 2.0)

    tx = kit.W * 0.735
    kit.draw_text(f, "C = 2\u03c0r", tx, kit.H * 0.255, 30, CREAM, kit.F_MONO, 0.72, 1.4)
    kit.draw_text(f, "\u03b8 = %5.3f rad" % (TAU * prog), tx, kit.H * 0.255 + 48, 20, MINT,
                  kit.F_MONO, 0.55, 1.2)
    kit.draw_text(f, "perimeter = %6.2f" % (TAU * prog), tx, kit.H * 0.255 + 82, 20, ASH,
                  kit.F_MONO, 0.45, 1.2)
    c.lyric(size=42, y=kit.H * 0.855)


# ================================================================ 37.60 正弦
@scene(37.60, 41.30, "SINE", "B")
def sine(c):
    t = c.t
    f = c.f
    lt = t - 37.60
    pl = c.pulse(8.5)
    c.post.update(grain=0.028, vignette=0.52, bloom=(0.56, 24, 0.52), chroma=0.3)
    deep_base(f, "061019", "010407", 1.25)
    glow(f, kit.W * 0.5, kit.H * 0.26, 940, kit.hx("0E3A4C"), 0.18, 1.8)
    grid_floor(f, t, y=0.90, horizon=0.56, alpha=0.22 + 0.12 * pl, color=kit.hx("5FD0C6"))

    bass = c.band("bass")
    A = 150.0 + 110.0 * bass
    ph = -t * 2.0
    pitch = 0.30
    oy = (0, 58, 0)
    ring = c.cl.sh(cloud.shape_ring, 300.0, 4.2, "xy")
    wav = c.cl.sh(cloud.shape_sine, 1350.0, A, 2.0, ph, 10.0)
    p = kit.cl((lt - 0.03) / 0.60)
    P = c.cl.morph(ring, wav, p, stagger=0.5, ease="out")
    P = c.cl.transform(P, pitch=pitch, offset=oy)

    flow = 0.22 + 0.90 * np.clip(np.sin(P[:, 0] * 0.013 - t * 2.6), 0, 1) ** 1.4
    depth = np.clip(1.25 - (P[:, 2] + 60.0) / 620.0, 0.20, 1.5)
    br = flow * depth * (0.90 + 0.30 * pl)

    def cmap(xs, ys, zz, kk):
        qq = np.clip((xs / kit.W - 0.05) / 0.90, 0, 1)
        return kit.mixc(MINT, VIOLET, qq * 0.85)

    c.cl.draw(f, P, dist=K_DIST, focal=K_FOCAL, color_map=cmap, bright=br,
              size=1.42, blur_r=1.6, glow_gain=0.75, shade=0.45, target=0.52 + 0.14 * pl)

    # 中轴 + 采样线 + 切线（39.4 起）
    vline(f, kit.W * 0.5, MINT, 0.10, 1)
    add_lines(f, [[(80, kit.H * 0.5 - 58 * 0 + 0), (kit.W - 80, kit.H * 0.5)]], ASH, 0.18, thick=1.0)

    if lt > 1.55:
        qa = kit.cl((lt - 1.55) / 0.85)
        xs0 = np.linspace(-0.40, 0.40, 7) * 1350.0
        kk = 2.0 / 675.0 * PI
        yy0 = A * np.sin(xs0 * kk + ph)
        slp = np.clip(A * kk * np.cos(xs0 * kk + ph), -1.15, 1.15)
        w0 = np.stack([xs0, yy0, np.zeros_like(xs0)], 1)
        w1 = np.stack([xs0 + 86.0, yy0 + slp * 86.0, np.zeros_like(xs0)], 1)
        both = c.cl.transform(np.concatenate([w0, w1], 0), pitch=pitch, offset=oy)
        tx_, ty_, _ = proj(both, K_DIST, K_FOCAL)
        n9 = len(xs0)
        segs = [[(tx_[i], ty_[i]), (tx_[i + n9], ty_[i + n9])] for i in range(n9)]
        add_lines(f, segs, GOLD, 0.34 * qa, thick=1.0)
        for i in range(n9):
            disc(f, tx_[i], ty_[i], 2.4, GOLD, 0.45 * qa)

    # 沿波跑的光点
    hx = ((t * 0.42) % 1.0) * 1350.0 - 675.0
    hy = A * math.sin(hx * (2.0 / 675.0 * PI) + ph)
    hp = c.cl.transform(np.array([[hx, hy, 0.0]], F32), pitch=pitch, offset=oy)
    hx_, hy_, _ = proj(hp, K_DIST, K_FOCAL)
    disc(f, float(hx_[0]), float(hy_[0]), 7.0, CREAM, 0.8)
    soft_ring(f, float(hx_[0]), float(hy_[0]), 26, 10, MINT, 0.30)
    # 起振点：左端那一个亮结
    o0 = c.cl.transform(np.array([[-675.0, A * math.sin(-675.0 * (2.0 / 675.0 * PI) + ph), 0.0]], F32),
                        pitch=pitch, offset=oy)
    ox, oy2, _ = proj(o0, K_DIST, K_FOCAL)
    disc(f, float(ox[0]), float(oy2[0]), 9.0 + 3.0 * pl, CREAM, 0.85)
    soft_ring(f, float(ox[0]), float(oy2[0]), 30, 12, MINT, 0.26 + 0.14 * pl)

    kit.draw_text(f, "y = A\u00b7sin(kx + \u03c6)", 88, kit.H * 0.185, 22, PALE, kit.F_MONO, 0.60, 1.0)
    kit.draw_text(f, "A = %6.2f   \u03c6 = %.2f" % (A, ph), 88, kit.H * 0.185 + 34, 20, MINT,
                  kit.F_MONO, 0.55, 1.0)
    c.lyric(size=42, y=kit.H * 0.855)


# ================================================================ 41.30 无穷
@scene(41.30, 45.01, "INFINITY", "B")
def infinity(c):
    t = c.t
    f = c.f
    lt = t - 41.30
    pl = c.pulse(8.5)
    c.post.update(grain=0.030, vignette=0.56, bloom=(0.58, 26, 0.55), chroma=0.4)
    deep_base(f, "050A14", "010307", 1.3)
    glow(f, kit.W * 0.5, kit.H * 0.48, 860, kit.hx("241C44"), 0.30, 1.7)

    lem = c.cl.sh(cloud.shape_lemniscate, 430.0, 11.0)
    wav = c.cl.sh(cloud.shape_sine, 700.0, 130.0, 2.0, -41.30 * 2.0, 9.0)
    p = kit.cl((lt - 0.04) / 0.78)
    P = c.cl.morph(wav, lem, p, stagger=0.55, ease="io")
    pitch = 0.62
    roll = 0.10 * math.sin(lt * 0.55) + 0.05
    P = c.cl.transform(P, pitch=pitch, roll=roll, offset=(0, 8, 0))

    flow = np.clip(np.sin(c.cl.u * TAU * 4.0 - t * 6.0), 0, 1) ** 2
    depth = np.clip(1.30 - (P[:, 2] + 40.0) / 620.0, 0.18, 1.6)
    br = (0.34 + 1.45 * flow) * depth * (0.90 + 0.25 * pl)

    def cmap(xs, ys, zz, kk):
        qq = np.clip((ys / kit.H - 0.18) / 0.66, 0, 1)
        return kit.mixc(VIOLET, CREAM, qq * 0.7)

    c.cl.draw(f, P, dist=K_DIST, focal=K_FOCAL, color_map=cmap, bright=br,
              size=1.16, blur_r=1.3, glow_gain=0.72, shade=0.40, target=0.50 + 0.14 * pl)

    # 三颗沿 ∞ 跑的光点 + 拖尾
    heads = []
    trails = []
    for i in range(3):
        s0 = (t * 0.40 + i * 0.333) % 1.0
        pts = []
        for k in range(10):
            uu = (s0 - k * 0.016) % 1.0
            th = uu * TAU
            dd = 1.0 + math.sin(th) ** 2
            pts.append((430.0 * math.cos(th) / dd, 430.0 * math.sin(th) * math.cos(th) / dd, 0.0))
        pts = c.cl.transform(np.asarray(pts, F32), pitch=pitch, roll=roll, offset=(0, 8, 0))
        px, py, _ = proj(pts, K_DIST, K_FOCAL)
        trails.append(list(zip(px, py)))
        heads.append((px[0], py[0]))
    add_lines(f, trails, MINT, 0.34, thick=1.4, blur=1.2)
    for k, (hx, hy) in enumerate(heads):
        disc(f, float(hx), float(hy), 6.0 - k * 0.8, CREAM, 0.85 - k * 0.16)
        soft_ring(f, float(hx), float(hy), 22, 9, MINT, 0.26)

    # 「你来做我的界限」：绕着她的一圈虚线框，随拍呼吸
    qd = kit.cl((lt - 0.60) / 0.90)
    if qd > 0:
        rw, rh = 660.0 * (1 + 0.018 * pl), 360.0 * (1 + 0.018 * pl)
        segs = []
        ns = 26
        for i in range(ns):
            for j in (0, 1):
                sa = (i + 0.12 + j * 0.5) / ns
                sb = sa + 0.30 / ns

                def rpt(u):
                    u = u % 1.0
                    if u < 0.25:
                        return (-rw + 4 * rw * u, -rh, 0.0)
                    if u < 0.5:
                        return (rw, -rh + 4 * rh * (u - 0.25), 0.0)
                    if u < 0.75:
                        return (rw - 4 * rw * (u - 0.5), rh, 0.0)
                    return (-rw, rh - 4 * rh * (u - 0.75), 0.0)

                segs.append([rpt(sa), rpt(sb)])
        segs = c.cl.transform(np.asarray(segs, F32).reshape(-1, 3), pitch=pitch, roll=roll,
                              offset=(0, 8, 0)).reshape(-1, 2, 3)
        pts_all = segs.reshape(-1, 3)
        px_, py_, _ = proj(pts_all, K_DIST, K_FOCAL)
        px_ = px_.reshape(-1, 2); py_ = py_.reshape(-1, 2)
        add_lines(f, [list(zip(px_[i], py_[i])) for i in range(len(px_))], VIOLET, 0.74 * qd, thick=1.5)

    kit.draw_text(f, "(x\u00b2+y\u00b2)\u00b2 = a\u00b2(x\u00b2\u2212y\u00b2)", 88, kit.H * 0.175, 21, PALE,
                  kit.F_MONO, 0.55, 1.0)
    kit.draw_text(f, "a = %.1f   t \u2192 \u221e" % (430.0 + 8 * math.sin(lt)), 88, kit.H * 0.175 + 32,
                  19, VIOLET, kit.F_MONO, 0.52, 1.0)
    c.lyric(size=42, y=kit.H * 0.855)


# ================================================================ 45.01 电流
@scene(45.01, 48.67, "CURRENT", "B")
def current(c):
    t = c.t
    f = c.f
    lt = t - 45.01
    pl = c.pulse(9.0)
    c.post.update(grain=0.032, vignette=0.55, bloom=(0.60, 22, 0.55), chroma=0.55)
    deep_base(f, "050D12", "010305", 1.25)
    glow(f, kit.W * 0.5, kit.H * 0.52, 720, kit.hx("0B3A3C"), 0.28, 1.6)

    lem = c.cl.sh(cloud.shape_lemniscate, 320.0, 4.5)
    col = threads(c.cl, R=246.0, Hh=980.0)
    p = kit.cl((lt - 0.05) / 0.55)
    P = c.cl.morph(lem, col, p, stagger=0.5, ease="out")
    P[:, 1] *= 1.0 + 0.05 * pl

    line_h = 0.5 + 0.5 * np.sin(np.floor(c.cl.u * 168.0) * 2.399 + 1.3)
    run = np.clip(np.sin((c.cl.v - 0.5) * 8.0 - t * 4.4), 0, 1) ** 4
    br = (0.16 + 0.55 * line_h + 1.7 * line_h * run) * (0.85 + 0.40 * pl)

    def cmap(xs, ys, zz, kk):
        qq = np.clip((zz - 1200.0) / 620.0, 0, 1)
        return kit.mixc(CREAM, kit.hx("3FD0C8"), qq)

    c.cl.draw(f, P, dist=K_DIST, focal=K_FOCAL, color_map=cmap, bright=br,
              size=1.20, blur_r=1.7, glow_gain=0.58, shade=0.35, target=0.38 + 0.16 * pl)

    # 她站在电流里
    stage.character(f, stage.MAID_L, kit.W * 0.5, kit.H * 1.01, 810, alpha=0.86,
                    tint=kit.hx("7E9FB4"), tint_k=0.44, rim=MINT, rim_k=0.22,
                    glow=0.0)

    # 竖向电场线（半分辨率 + 逐行抖动）
    h, w = kit.SH // 2, kit.SW // 2
    fm = np.zeros((h, w), F32)
    rows = np.arange(h, dtype=F32)[:, None]
    for i, bx in enumerate(np.linspace(0.055, 0.945, 13) * w):
        ix = (bx + 3.4 * np.sin(rows * 0.034 + t * 5.2 + i * 1.7)).astype(np.int32)
        np.put_along_axis(fm, np.clip(ix, 0, w - 1), np.full((h, 1), 1.0, F32), axis=1)
    fm = kit.blur(fm, 0.8 * kit.S)
    fm = kit.up(fm, kit.SW, kit.SH)
    f += fm[..., None] * ICE * (0.10 + 0.08 * pl)

    # 横向扫描光带
    by = 0.5 + 0.34 * math.sin(t * 1.9 - 1.1)
    band_h(f, by, 0.050, MINT, 0.07 + 0.05 * pl)
    band_h(f, by, 0.006, ICE, 0.26 + 0.16 * pl)

    # 电弧
    rs = c.rng(17)
    segs = []
    for k in range(3):
        y0 = kit.H * (0.26 + 0.22 * k) + 22 * math.sin(t * 3.0 + k * 2.0)
        pts = []
        x = kit.W * 0.17
        while x < kit.W * 0.83:
            pts.append((x, y0 + float(rs.normal(0, 13))))
            x += float(rs.uniform(16.0, 40.0))
        segs.append(pts)
    add_lines(f, segs, CREAM, 0.42 + 0.45 * pl, thick=1.7, blur=2.0)

    # 强拍切行
    g = 0.55 * pl ** 2.2
    if g > 0.02:
        f = kit.glitch_rows(f, t, g, seed=5, band=14 + int(18 * pl))

    kit.draw_text(f, "AC", kit.W * 0.155, kit.H * 0.185, 40, ICE if lt < 1.4 else ASH,
                  kit.F_MONO_B, 0.92, 6.0)
    kit.draw_text(f, "DC", kit.W * 0.845, kit.H * 0.185, 40, MINT if lt >= 1.4 else ASH,
                  kit.F_MONO_B, 0.92, 6.0, "right")
    kit.draw_text(f, "switch(current)", 88, kit.H * 0.885, 19, TEAL, kit.F_MONO, 0.55, 1.0)
    # 词放左下角：她站在画面正中，中下方会让字压在她的裙子上
    c.lyric(size=38, y=kit.H * 0.788, x=kit.W * 0.088, align="left", alpha=0.95)


# ================================================================ 48.67 盲
@scene(48.67, 50.52, "BLIND", "B")
def blind(c):
    t = c.t
    f = c.f
    lt = t - 48.67
    pl = c.pulse(9.0)
    c.post.update(grain=0.036, vignette=0.62, bloom=(0.64, 20, 0.34))
    c.post["sat"] = 0.16
    c.post["contrast"] = 1.10

    deep_base(f, "05080C", "010203", 1.2)
    glow(f, kit.W * 0.5, kit.H * 0.56, 760, kit.hx("132A38"), 0.24 + 0.06 * pl, 1.9)

    # 竖线 → 立绘剪影
    col = threads(c.cl, R=246.0, Hh=980.0)
    body = sf(c.cl, stage.MAID_L, 1000.0, 26.0)
    p = kit.cl((lt - 0.02) / 0.55)
    P = c.cl.morph(col, body["P"], p, stagger=0.5, ease="out")

    cy = kit.H * 0.52
    # 先把它画进空帧拿到密度，再用密度「挖」出纯黑剪影
    if p < 1.0:
        # 还在成形：电流被一点点吸进她的身体（照常画成光，不做剪影）
        fade = 1.0 - 0.55 * p
        c.cl.draw(f, P, dist=K_DIST, focal=K_FOCAL, cy=cy, color=MINT,
                  bright=0.34 * fade, size=1.00, blur_r=1.3, glow_gain=0.40,
                  shade=0.2, target=0.52)
    else:
        if "m" in _BL:
            m = _BL["m"]
        else:
            tmp = kit.new_frame(kit.BLACK)
            c.cl.draw(tmp, P, dist=K_DIST, focal=K_FOCAL, cy=cy, color=kit.WHITE, bright=1.0,
                      size=1.85, blur_r=1.0, glow_gain=0.0, additive=True, shade=0.0,
                      target=0.92)
            m = np.clip(tmp.max(axis=2), 0, 1)[..., None]
            _BL["m"] = m
            _BL["mk"] = 1.0 - m * 0.90
            _BL["lay"] = m * kit.hx("10283A")

        # 密度图 → 纯黑剪影 + 描边
        np.multiply(f, _BL["mk"], out=f)
        np.add(f, _BL["lay"], out=f)
        eg = np.clip(body["edge"] * 2.6, 0, 1)
        c.cl.draw(f, P, dist=K_DIST, focal=K_FOCAL, cy=cy, color=CREAM,
                  bright=eg * (0.85 + 0.3 * pl), size=0.95, blur_r=1.1, glow_gain=0.38,
                  shade=0.2, target=0.85)

        # 两块黑斑盖住眼睛
        eyes = np.array([[-38.7, 369.4, 0.0], [23.3, 369.4, 0.0]], F32)
        ex, ey, ek = proj(eyes, K_DIST, K_FOCAL, cy=cy)
        for i in range(2):
            rr = 42.0 * ek[i]
            disc(f, float(ex[i]), float(ey[i]), rr * 1.55, kit.hx("0B1A26"), 0.70)
            disc(f, float(ex[i]), float(ey[i]), rr, BLACK, 1.0, soft=0.10, mul=True)
            disc(f, float(ex[i]), float(ey[i]), rr * 0.62, BLACK, 1.0)
            soft_ring(f, float(ex[i]), float(ey[i]), rr * 0.99, 2.4, CREAM, 0.24 + 0.10 * pl)

    # 视野被夺：上下合拢
    sh = 0.30 * kit.cl((lt - 0.85) / 0.85)
    yv = np.abs(np.arange(kit.SH, dtype=F32) / kit.SH - 0.5)
    m2 = np.clip((yv - (0.5 - sh)) / 0.05, 0, 1)
    f *= (1.0 - m2 * 0.94)[:, None, None]

    c.lyric(size=36, y=kit.H * 0.055, alpha=0.92)
    glow(f, kit.W * 0.5, kit.H * 0.60, 520, kit.hx("16323E"), 0.09 + 0.04 * pl, 2.4, 0.7)
    kit.draw_text(f, "VISION  0x00", 88, kit.H * 0.905, 19, ASH, kit.F_MONO, 0.45, 1.2)
    kit.draw_text(f, "blind(my.vision)", kit.W - 92, kit.H * 0.905, 19, ASH, kit.F_MONO,
                  0.45, 1.2, "right")


# ================================================================ 50.52 眩晕
@scene(50.52, 51.93, "DIZZY", "B")
def dizzy(c):
    t = c.t
    f = c.f
    lt = t - 50.52
    pl = c.pulse(10.0)
    c.post.update(grain=0.042, vignette=0.52, bloom=(0.52, 24, 0.60), chroma=1.5)
    deep_base(f, "0A0A12", "020206", 1.2)
    glow(f, kit.W * 0.5, kit.H * 0.5, 640 + 130 * math.sin(lt * 6.0),
                   kit.hx("2A2350"), 0.34, 1.6)

    # 螺旋（程序化）：越往外越松的旋涡
    cx, cyv = kit.W * 0.5, kit.H * 0.5
    pts = []
    for i in range(360):
        a = i * 0.118 + lt * 4.4
        r = 4.0 + (i ** 1.22) * 0.55
        pts.append((cx + math.cos(a) * r, cyv + math.sin(a) * r * 0.82))
    add_lines(f, [pts], VIOLET, 0.50 + 0.24 * pl, thick=1.8, blur=2.0)
    pts2 = []
    for i in range(260):
        a = -i * 0.136 - lt * 2.9
        r = 10.0 + (i ** 1.18) * 0.52
        pts2.append((cx + math.cos(a) * r * 1.06, cyv + math.sin(a) * r * 0.84))
    add_lines(f, [pts2], MINT, 0.28 + 0.12 * pl, thick=1.3, blur=1.8)

    # 三圈错位残影：整帧轻轻转一点点叠回去（眩晕感全靠这个）
    for k in range(3):
        ang_k = (0.028 + 0.011 * k) * (1.0 if k % 2 == 0 else -1.0) * (1.0 + 0.45 * pl)
        f += kit.scale_about(f, cx, cyv, 1.0 + 0.010 * k, ang_k) * (0.17 - 0.035 * k)
    for i in range(3):
        aa = lt * 4.4 + i * 2.1
        rr = 235 + 42 * math.sin(lt * 5.0 + i)
        disc(f, kit.W * 0.5 + math.cos(aa) * rr, kit.H * 0.5 + math.sin(aa) * rr * 0.80,
             5.0 + 2.0 * pl, CREAM, 0.75)

    ln = mus.line_at(t)
    txt = ln["en"] if ln else "so dizzy"
    ang = 0.155 + 0.055 * math.sin(lt * 5.4)
    tx, ty = kit.W * 0.5, kit.H * 0.415
    tilted_text(f, txt, tx + 8, ty + 4, 86, CORAL, kit.F_SERIF_I, ang, 0.34, 2.0)
    tilted_text(f, txt, tx - 8, ty - 4, 86, MINT, kit.F_SERIF_I, ang, 0.34, 2.0)
    tilted_text(f, txt, tx, ty, 86, CREAM, kit.F_SERIF_I, ang, 0.97, 2.0)
    tilted_text(f, txt, tx + 12, ty + 106, 27, VIOLET, kit.F_SERIF_I, -ang * 1.5, 0.55, 4.0)
    if ln and ln.get("zh"):
        kit.draw_text(f, ln["zh"], tx + 4, ty + 146, 25, MINT, kit.F_SERIF_SC,
                      0.58, 8.0, "center")

    # 双重曝光：模糊副本叠加
    g = kit.blur(kit.shift(f, 21.0 * math.sin(lt * 2.8), -13.0), 3.4)
    f *= 0.84
    f += g * kit.mixc(kit.WHITE, VIOLET, 0.30) * 0.60

    # 整帧滚动
    c.f = f
    c.roll(0.085 * math.sin(lt * 6.6) + 0.028 * math.sin(lt * 15.0))
    f = c.f

    # 强拍白闪
    a = 0.34 * pl ** 3
    if a > 0.004:
        f += kit.WHITE * a
    c.post["vignette"] = 0.50
