"""ch_g.py — 章节 G：103.84 – 129.46（Mili / world.execute(me);）

CHORUS2    103.84–107.52  两具粒子身体互相穿过 + 同心圆波纹 + 强拍白闪
COMPLETION 107.52–111.24  鲸被吸进她的身体，合成为一个人形，光从内部亮起
LEFT       111.24–116.65  空舞台：她一个人站着，地板残响，五盏灯逐盏熄灭
ISOLATION  116.65–119.98  缩到一点，巨大黑场，逐字打出 isolation
ERASE      119.98–125.99  碎片剥离被风吹走 + 扫描带上扫清空 → 只剩光标
CHALLENGE  125.99–129.46  仰视神像 + 程序化裂纹 + 白闪硬切

情绪由满到空：亮度一路下降，亮点只留给强拍、内部光和扫描线。
"""
import math
import numpy as np
import cv2

import kit
import cloud
import stage
import mus
from film2 import scene

W, H, S = kit.W, kit.H, kit.S
SW, SH = kit.SW, kit.SH
F32 = np.float32

MINT, CORAL, VIOLET = kit.MINT, kit.CORAL, kit.VIOLET
GOLD, CREAM, RUST = kit.GOLD, kit.CREAM, kit.RUST

T_CH1 = 103.84
T_CH2 = 107.52
T_LEFT = 111.24
T_ISO = 116.65
T_ERASE = 119.98
T_CHAL = 125.99
T_END = 129.46


# ══════════════════════════════════════════════════════════ 基础工具
_SC = {}
_SF = {}


def _scratch():
    """半分辨率 uint8 画布（复用，每次调用前清零）。"""
    k = (SW, SH)
    m = _SC.get(k)
    if m is None:
        m = np.zeros((max(2, SH // 2), max(2, SW // 2)), np.uint8)
        _SC[k] = m
    return m


def _fscratch():
    k = (SW, SH)
    m = _SF.get(k)
    if m is None:
        m = np.zeros((max(2, SH // 2), max(2, SW // 2)), F32)
        _SF[k] = m
    return m


def _ell(m, cx, cy, r, thick, ratio, val):
    bh, bw = m.shape
    cv2.ellipse(m, (int(round(cx * bw / W)), int(round(cy * bh / H))),
                (max(1, int(round(r * ratio * bw / W))), max(1, int(round(r * bh / H)))),
                0, 0, 360, val, max(1, int(round(thick * bw / W))), cv2.LINE_AA)


def rings(f, items, color=None, glow=0.0):
    """批量同心圆：同色合成一张半分辨率图，只做一次 up+add。

    items: [(cx, cy, r, thick, ratio, strength), ...]
    """
    if not items:
        return
    m = _fscratch()
    m[:] = 0
    hit = 0.0
    for (cx, cy, r, thick, ratio, strength) in items:
        if strength <= 0.004 or r <= 1.0:
            continue
        _ell(m, cx, cy, r, thick, ratio, float(strength))
        hit = max(hit, float(strength))
    if hit <= 0.004:
        return
    if glow > 0:
        m += cv2.GaussianBlur(m, (0, 0), max(0.9, glow * (SH // 2) / H)) * 1.25
    f += kit.up(m[..., None], SW, SH) * np.asarray(color, F32)


def ring(f, cx, cy, r, thick=2.0, color=None, strength=1.0, ratio=1.0, glow=0.0):
    rings(f, [(cx, cy, r, thick, ratio, strength)], color, glow)


def vscan(f, x, color=None, strength=1.0, soft=9.0):
    """整高的软竖线（光缝）。"""
    xx = x * S
    c0 = max(0, int(xx - 3.2 * soft * S))
    c1 = min(SW, int(xx + 3.2 * soft * S) + 1)
    if c1 <= c0 or strength <= 0.004:
        return
    cols = np.arange(c0, c1, dtype=F32)
    prof = np.exp(-((cols - xx) / max(1e-3, soft * S)) ** 2) * strength
    f[:, c0:c1] += prof[None, :, None] * np.asarray(color, F32)


def hscan(f, y, color=None, strength=1.0, soft=9.0):
    """整宽的软横线（扫描带）。"""
    yy = y * S
    r0 = max(0, int(yy - 3.2 * soft * S))
    r1 = min(SH, int(yy + 3.2 * soft * S) + 1)
    if r1 <= r0 or strength <= 0.004:
        return
    rows = np.arange(r0, r1, dtype=F32)
    prof = np.exp(-((rows - yy) / max(1e-3, soft * S)) ** 2) * strength
    f[r0:r1] += prof[:, None, None] * np.asarray(color, F32)


def grid_floor(f, y0f=0.86, color=None, alpha=0.10, rows=9, span=14, vx=0.5):
    """透视网格地板（半分辨率绘制）。"""
    m = _scratch()
    m[:] = 0
    bh, bw = m.shape
    y0 = y0f * bh / H
    for i in range(rows):
        q = (i + 1.0) / rows
        y = y0 + (bh * 1.02 - y0) * (q ** 2.45)
        cv2.line(m, (0, int(y)), (bw, int(y)), 255, 1, cv2.LINE_AA)
    vxp = vx * bw / W
    for j in range(-span, span + 1):
        xf = vxp + j * (bw * 0.075)
        cv2.line(m, (int(vxp + (xf - vxp) * 0.05), int(y0)),
                 (int(xf), int(bh * 1.05)), 255, 1, cv2.LINE_AA)
    f += kit.up((m.astype(F32) * (1 / 255.0))[..., None], SW, SH) * (np.asarray(color, F32) * alpha)


_PF = {}


def pfield(c, path, hw, depth=32.0):
    """sprite_field 跨帧缓存（每帧新建 Cloud，sh 缓存只在帧内有效）。"""
    k = (path, round(float(hw), 2), round(float(depth), 2))
    v = _PF.get(k)
    if v is None:
        v = c.cl.sh(cloud.sprite_field, path, float(hw), float(depth))
        _PF[k] = v
    return v


def pdraw(c, P, col, bright=None, dist=1500.0, focal=1150.0, cx=None, cy=None, oy=0.0,
          size=1.5, blur_r=1.5, target=0.9, glow=0.5, core=1.0, shade=0.0,
          exposure=1.0, tm=2.0, twinkle=None):
    """包一层 draw：color_map 预先按 keep 裁剪，长度与 draw 内部严格一致。"""
    sx, sy, z, kk = c.cl.project(P, dist, focal, cx, cy, oy)
    bw, bh = SW // 2, SH // 2
    xs = sx * (bw / W)
    ys = sy * (bh / H)
    keep = (xs > -30) & (xs < bw + 30) & (ys > -30) & (ys < bh + 30)
    if col is not None and getattr(col, "ndim", 1) == 2:
        ck = col[keep]
        cm = (lambda *a: ck)
        cval = 1.0
    else:
        cm = None
        cval = 1.0 if col is None else col
    c.cl.draw(c.f, P, color=cval, color_map=cm, bright=bright, dist=dist, focal=focal,
              cx=cx, cy=cy, oy=oy, size=size, blur_r=blur_r, target=target,
              glow_gain=glow, core_gain=core, shade=shade, exposure=exposure,
              tm=tm, twinkle=twinkle)
    return sx, sy


def shade_of(lum, lo=0.26, hi=0.90, gamma=1.5):
    """把立绘亮度压成高对比结构光（防白团：暗部沉下去）。"""
    return np.clip((lum - lo) / max(1e-4, hi - lo), 0, 1) ** gamma


def fig_col(lum, edge, lo, hi, accent, ek=1.5, top=0.78):
    """暗部→深色，亮部→中调，轮廓边→强调色（keeps 立绘可读，不成白团）。"""
    k = shade_of(lum)
    col = kit.mixc(np.asarray(lo, F32), np.asarray(hi, F32), k * top)
    a = np.clip(edge * ek, 0, 0.92)[:, None]
    return col + (np.asarray(accent, F32)[None, :] - col) * a


def body_bright(lum, edge, glow_k=1.0):
    """结构光：填充暗、轮廓亮 —— 只有代码能做到的「描边粒子人形」。"""
    k = shade_of(lum)
    return (0.03 + 0.55 * k + 2.35 * np.clip(edge, 0, 1) ** 0.9) * glow_k


def beats_in(t0, t1):
    return [float(b) for b in mus.BEATS if t0 <= b <= t1]


# ══════════════════════════════════════════════════════════ 1. CHORUS2
@scene(T_CH1, T_CH2, "CHORUS2", "G")
def chorus2(c):
    f = c.f
    t = c.t
    u = t - T_CH1
    p = kit.cl(u / (T_CH2 - T_CH1))
    pl = c.pulse(10.0)
    c.post.update(bloom=(0.60, 22, 0.62), vignette=0.40, grain=0.017,
                  chroma=0.0016, exposure=1.05)

    # ── 底：深海 + 中央光轴 + 极淡地板
    f += kit.vgrad(SH, SW, kit.hx("02060C"), kit.hx("081C2C"), 1.25)
    kit.add_radial(f, W * 0.50, H * 0.46, 580, kit.hx("0C3044"), 0.85, 2.6)
    grid_floor(f, 0.845, MINT, 0.035 + 0.03 * pl, rows=8, span=12)

    # ── 两具粒子身体：她 ←→ 鲸，互相穿过
    e = kit.ease_io(p)
    xh = W * kit.lerp(0.335, 0.625, e)
    xw = W * kit.lerp(0.700, 0.330, e)
    kick = 0.80 + 0.40 * c.kick()

    F1 = pfield(c, stage.MAID_L, 950.0, 34.0)
    P1 = cloud.Cloud.transform(F1["P"], yaw=0.26 + 0.07 * math.sin(u * 1.4))
    col1 = fig_col(F1["lum"], F1["edge"], kit.hx("0B2E4A"), kit.hx("6FB6C8"), MINT, 1.7)
    b1 = body_bright(F1["lum"], F1["edge"]) * kick
    pdraw(c, P1, col1, bright=b1, cx=xh, cy=H * 0.503, dist=1500.0,
          size=1.30, blur_r=0.95, target=0.62, glow=0.30, core=0.9, shade=0.5,
          twinkle=(t * 2.3, 0.10))

    F2 = pfield(c, stage.WHALE["serious"], 2580.0, 60.0)
    P2 = cloud.Cloud.transform(F2["P"], yaw=-0.28 + 0.06 * math.sin(u * 1.1))
    # 紫金：身体紫、头顶与下摆镶金（靠高度而不是靠 edge，避免整具变土黄）
    hy2 = np.clip((P2[:, 1] + 0.80 * 2580.0) / (1.60 * 2580.0), 0, 1)
    k2 = shade_of(F2["lum"], 0.30, 0.88, 1.7)
    col2 = kit.mixc(kit.hx("160C2E"), kit.hx("523383"), k2)
    gld = np.clip((hy2 - 0.80) / 0.20, 0, 1) * 0.85 + np.clip(0.30 - hy2, 0, 1) * 0.55
    gld = np.clip(gld + np.clip(c.cl.rnd - 0.86, 0, 1) * 5.0, 0, 1)[:, None]
    col2 = col2 + (np.asarray(GOLD, F32)[None, :] - col2) * gld
    b2 = (0.05 + 0.85 * k2 + 2.0 * np.clip(F2["edge"], 0, 1) ** 0.9) * (0.85 + 0.30 * kick)
    pdraw(c, P2, col2, bright=b2, cx=xw, cy=H * 0.935, dist=1500.0,
          size=1.45, blur_r=1.05, target=0.56, glow=0.32, core=0.9, shade=0.6,
          twinkle=(t * 1.7 + 1.3, 0.13))

    # ── 同心圆波纹：每个拍点发射一圈（批量合成，两种颜色两张图）
    rx, ry = W * 0.50, H * 0.505
    batch, batch2 = [], []
    for bt in beats_in(T_CH1, T_CH2):
        age = t - bt
        if 0.0 <= age < 1.85:
            a = (1.0 - age / 1.85) ** 2.0
            r = 70 + 780 * (age / 1.85) ** 0.58
            batch.append((rx, ry, r, 2.2 + 1.6 * (1.0 - a), 1.0, 0.30 * a))
            batch.append((rx, ry, r + 16, 1.0, 1.0, 0.14 * a))
        if 0.0 <= age < 1.55:
            a2 = (1.0 - age / 1.55) ** 2.4
            batch2.append((xh, H * 0.44, 30 + 300 * (age / 1.55) ** 0.65,
                           1.4, 0.70, 0.16 * a2))
    rings(f, batch, MINT, glow=1.9)
    rings(f, batch2, CREAM, glow=2.6)

    # ── 穿过时的竖光缝
    gap = abs(xh - xw)
    seam = kit.cl(1.0 - gap / (W * 0.28))
    if seam > 0.02:
        vscan(f, (xh + xw) * 0.5, MINT, 0.13 * seam + 0.10 * seam * pl, soft=12.0)
        vscan(f, (xh + xw) * 0.5, CREAM, 0.09 * seam, soft=3.0)

    # ── 底部反光 + 尘
    stage.floor_reflection(f, t, 0.905, 0.16, MINT)
    stage.dust(f, t, 90, CREAM, 0.18, (0.0, 1.0, 0.0, 0.90))

    # ── 字幕 / 角标
    stage.hud_corner(f, t, left="CHORUS 02 // vibration lock", right="2 bodies · 124000 pts",
                     alpha=0.28, color=MINT, y=H * 0.072)
    c.lyric(size=42, y=H * 0.858, glow=0.26, alpha=0.95)

    # ── 强拍：白闪 + 轻微 scale
    fl = kit.cl(pl * 1.35)
    c.f = f
    if fl > 0.012:
        c.f *= (1.0 - 0.125 * fl)
        c.f += kit.WHITE * (0.125 * fl)
        c.zoom_frame(1.0 + 0.016 * fl)


# ══════════════════════════════════════════════════════════ 2. COMPLETION
@scene(T_CH2, T_LEFT, "COMPLETION", "G")
def completion(c):
    f = c.f
    t = c.t
    u = t - T_CH2
    p = kit.cl(u / (T_LEFT - T_CH2))
    pl = c.pulse(9.0)
    c.post.update(bloom=(0.58, 26, 0.64), vignette=0.44, grain=0.016,
                  exposure=1.05, chroma=0.0012)

    f += kit.vgrad(SH, SW, kit.hx("03070E"), kit.hx("0A2033"), 1.3)
    kit.add_radial(f, W * 0.50, H * 0.54, 560, kit.hx("0C3145"), 0.80, 2.5)

    # ── 合一进度
    mp = kit.ease_io(kit.cl((u - 0.55) / 1.85))
    fin = kit.cl((u - 2.05) / 0.95)
    kick = 0.85 + 0.30 * c.kick()

    F1 = pfield(c, stage.MAID_L, 780.0, 34.0)
    F2 = pfield(c, stage.WHALE["serious"], 2580.0, 60.0)
    P1 = cloud.Cloud.transform(F1["P"], yaw=0.20 + 0.05 * math.sin(u * 0.9))
    P2 = cloud.Cloud.transform(F2["P"], yaw=-0.28)
    PM = c.cl.morph(P2, P1, mp, stagger=0.62, ease="io")

    cxh, cyh = W * 0.500, H * 0.545
    cxw, cyw = W * 0.330, H * 0.935
    cx = kit.lerp(cxw, cxh, mp)
    cy = kit.lerp(cyw, cyh, mp)

    # 她的本体（合并一开始就让位给合成体，避免双倍密度糊成白团）
    col1 = fig_col(F1["lum"], F1["edge"], kit.hx("0B2E4A"), kit.hx("6FB6C8"), MINT, 1.7)
    b1 = body_bright(F1["lum"], F1["edge"]) * (0.85 + 0.30 * kick)
    pdraw(c, P1, col1, bright=b1, cx=cxh, cy=cyh, dist=1500.0, size=1.30,
          blur_r=0.95, target=0.62 * (1.0 - 0.94 * kit.cl(mp * 1.5)), glow=0.30,
          core=0.9, shade=0.5, twinkle=(t * 2.1, 0.09))

    # 合成体：鲸的粒子一路搬进她的身体
    hyw = np.clip((P2[:, 1] + 0.80 * 2580.0) / (1.60 * 2580.0), 0, 1)
    kw = shade_of(F2["lum"], 0.30, 0.88, 1.7)
    col_w = kit.mixc(kit.hx("160C2E"), kit.hx("523383"), kw)
    gld = np.clip(np.clip((hyw - 0.84) / 0.16, 0, 1) * 0.8, 0, 1)[:, None]
    col_w = col_w + (np.asarray(GOLD, F32)[None, :] - col_w) * gld
    col_h = fig_col(F1["lum"], F1["edge"], kit.hx("12455F"), kit.hx("8FD3DC"), MINT, 1.9)
    colm = kit.mixc(col_w, col_h, mp)
    bm = kit.lerp(body_bright(F2["lum"], F2["edge"], 0.9),
                  body_bright(F1["lum"], F1["edge"], 1.15), mp) * (0.9 + 0.3 * kick)

    # 内部光：胸口为中心把粒子点亮
    sx, sy, _z, _k = c.cl.project(PM, 1500.0, 1150.0, cx, cy, 0.0)
    gm = np.exp(-(((sx - cxh) / 170.0) ** 2 + ((sy - (cyh - 30.0)) / 200.0) ** 2))
    bm = bm * (1.0 + 1.20 * fin * gm)

    pdraw(c, PM, colm, bright=bm, cx=cx, cy=cy, dist=1500.0, size=1.30,
          blur_r=0.95, target=0.34 + 0.14 * mp * mp, glow=0.32, core=0.95,
          shade=0.60, twinkle=(t * 1.9 + 0.7, 0.12))

    # ── 胸腔的内部光
    core_y = cyh - 60.0
    kit.add_radial(f, cxh, core_y, 50 + 120 * fin, CREAM, 0.10 + 0.30 * fin, 2.1)
    kit.add_radial(f, cxh, core_y, 260 + 340 * fin, MINT, 0.04 + 0.09 * fin, 2.6)
    if fin > 0.02:
        ring(f, cxh, core_y, 110 + 420 * fin, thick=1.4, color=MINT,
             strength=0.14 * fin * (0.5 + 0.5 * pl), ratio=1.0, glow=2.2)

    stage.dust(f, t, 80, CREAM, 0.12 + 0.10 * fin, (0.0, 1.0, 0.0, 0.9))

    # ── 大字 COMPLETION（逐字浮现）
    word = "COMPLETION"
    kit.draw_text(f, word, W * 0.5, H * 0.062, 108, CREAM, kit.F_SERIF_B, 1.0,
                  tracking=30.0 - 12.0 * p, align="center", glow=0.30,
                  glow_color=MINT,
                  char_alpha=lambda i, rel: kit.cl((u - 0.80 - i * 0.075) / 0.42))
    kit.draw_text(f, "// be completion", W * 0.5, H * 0.062 + 150, 24,
                  MINT, kit.F_MONO, 0.32 * kit.cl((u - 1.9) / 0.6), 6.0,
                  align="center", glow=0.2)

    stage.hud_corner(f, t, left="MERGE  body(2) -> body(1)",
                     right="density %.3f" % (0.42 + 0.58 * mp), alpha=0.24,
                     color=MINT, y=H * 0.935)
    c.lyric(size=42, y=H * 0.858, glow=0.26, alpha=0.95)
    c.f = f


# ══════════════════════════════════════════════════════════ 3. LEFT
_LAMPS = [(0.150, 0.168), (0.336, 0.124), (0.505, 0.148), (0.672, 0.122), (0.848, 0.174)]
# 五盏灯逐盏熄灭 —— 时间全部取自歌词（Though / You have left x4），不自己编
_LAMP_T = [mus.LINES[53]["t"]] + [mus.find("You have left", i) for i in range(4)]


@scene(T_LEFT, T_ISO, "LEFT", "G")
def left(c):
    f = c.f
    t = c.t
    u = t - T_LEFT
    p = kit.cl(u / (T_ISO - T_LEFT))
    pl = c.pulse(9.0)
    c.post.update(bloom=(0.64, 20, 0.44), vignette=0.47, grain=0.020, exposure=0.98)

    dead = sum(1 for d in _LAMP_T if t >= d)
    zoom = 1.02 + 0.115 * kit.ease_io(p)
    stage.bg_image(f, zoom=zoom, dx=5.5 * math.sin(u * 0.32),
                   dy=2.0 * math.cos(u * 0.27), tint=kit.hx("0A2030"),
                   tint_k=0.34, dim=0.54, blur_r=0.7)

    stage.caustics(f, t, MINT, 0.040, 1.0, 0.22, 0.60)
    # 便宜的光柱（不用 lightbeam：那个每帧 80ms）
    vscan(f, W * 0.442, MINT, 0.035, soft=110.0)
    vscan(f, W * 0.442, MINT, 0.030, soft=26.0)
    stage.floor_reflection(f, t, 0.855, 0.18, MINT)

    # ── 地板残留的一圈粒子（另一半的位置，缓慢消散）
    ringP = c.cl.sh(cloud.shape_ring, 170.0, 7.0, "xz")
    rn = c.cl.rnd
    rn2 = np.mod(rn * 91.7, 1.0)
    prog = kit.ease_io(p)
    P = cloud.Cloud.transform(ringP, pitch=0.34, scale=1.0 + 0.34 * prog,
                              offset=(0, 14.0 * prog, 0))
    P = P.copy()
    P[:, 1] += 64.0 * prog * np.clip(rn - 0.55, 0, 1) * 3.0
    sc = 1.0 + 0.10 * (rn2 - 0.5) + 0.22 * prog * (rn2 - 0.3)
    P[:, 0] *= sc
    P[:, 2] *= sc
    fade = 0.16 * (1.0 - 0.88 * prog) * (0.85 + 0.15 * pl)
    colr = kit.mixc(kit.hx("16485A"), CREAM, np.clip((rn - 0.55) * 2.0, 0, 1))
    pdraw(c, P, colr, bright=(0.06 + 2.2 * rn ** 4) * (0.55 + 0.45 * (1.0 - prog)),
          cx=W * 0.655, cy=H * 0.845, dist=1500.0, size=1.25, blur_r=1.05,
          target=fade, glow=0.28, shade=0.4)

    # ── 她一个人站在中间
    stage.character(f, stage.MAID_L, W * 0.442, H * 0.845, 600 + 14 * prog,
                    alpha=0.95, k=1.0, tint=kit.hx("0B1E2C"), tint_k=0.40,
                    glow=0.12, rim=MINT, rim_k=0.24, shadow=0.55)

    # ── 五盏灯：一盏一盏灭
    lit_rings = []
    for i, (lx, ly) in enumerate(_LAMPS):
        d = _LAMP_T[i]
        a = t - d
        x, y = W * lx, H * ly
        if a < 0:
            br = 0.90 + 0.12 * math.sin(t * 3.1 + i * 1.7)
            kit.add_radial(f, x, y, 300, kit.hx("4A3418"), 0.85 * br, 2.4)
            kit.add_radial(f, x, y, 120, GOLD, 0.55 * br, 2.0)
            kit.add_radial(f, x, y, 30, CREAM, 0.70 * br, 1.7)
            lit_rings.append((x, y, 34 + 22 * (0.5 + 0.5 * math.sin(t * 2.2 + i)),
                              1.2, 1.0, 0.13))
        else:
            fl = math.exp(-a * 5.0)
            if fl > 0.015:
                kit.add_radial(f, x, y, 90, GOLD, 0.60 * fl, 1.7)
                kit.add_radial(f, x, y, 26, CREAM, 0.55 * fl, 1.6)
            kit.add_radial(f, x, y, 240 + 160 * min(a, 1.2),
                           np.array([-0.14, -0.132, -0.118], F32), 1.0, 2.0)
    rings(f, lit_rings, GOLD, glow=2.6)

    # ── 全局随灭灯变暗（亮度阶梯）
    f *= (1.0 - 0.085 * dead)
    stage.dust(f, t, 70, CREAM, 0.16 * (1.0 - 0.5 * dead / 5.0), (0.0, 1.0, 0.05, 0.95))

    stage.hud_corner(f, t, left="-- empty stage // residual: 1 body",
                     right="lamps %d/5" % (5 - dead), alpha=0.28, color=MINT,
                     y=H * 0.072)
    c.lyric(size=40, y=H * 0.872, glow=0.20, alpha=0.75 * (1.0 - 0.10 * dead))
    c.f = f


# ══════════════════════════════════════════════════════════ 4. ISOLATION
@scene(T_ISO, T_ERASE, "ISOLATION", "G")
def isolation(c):
    f = c.f
    t = c.t
    u = t - T_ISO
    p = kit.cl(u / (T_ERASE - T_ISO))
    pl = c.pulse(7.0)
    c.post.update(bloom=(0.70, 24, 0.38), vignette=0.60, grain=0.024, exposure=0.94)

    f += kit.vgrad(SH, SW, kit.hx("01040A"), kit.hx("02090F"), 1.6)

    # 只有一小片地面光
    fy = H * 0.742
    kit.add_radial(f, W * 0.5, fy + 6, 250, kit.hx("1E6E7A"), 0.42, 2.4, ratio=0.26)
    kit.add_radial(f, W * 0.5, fy + 2, 110, kit.hx("2FA0A4"), 0.22 + 0.10 * pl, 2.0, ratio=0.24)
    kit.add_radial(f, W * 0.5, fy - 6, 46, CREAM, 0.10 + 0.06 * pl, 1.8, ratio=0.5)
    hscan(f, fy + 1.0, kit.hx("3A9A96"), 0.13 + 0.05 * pl, soft=4.0)
    # 地面线，锚定她站的地方
    hscan(f, fy + 12.0, kit.hx("123C46"), 0.16, soft=26.0)

    # 缩到极小
    stage.character(f, stage.MAID_L, W * 0.5, fy, 220, alpha=0.94,
                    tint=kit.hx("0A1B28"), tint_k=0.46, glow=0.20,
                    rim=MINT, rim_k=0.30, shadow=0.3)

    # ── 一个字一个字打出 isolation
    word = "isolation"
    k = int(kit.cl((t - 117.02) / 0.215) * (len(word) + 0.999))
    k = min(k, len(word))
    fw = kit.text_width(kit.F_MONO, 104, word, 14.0)
    x0 = W * 0.5 - fw * 0.5
    ty = H * 0.190
    if k > 0:
        kit.draw_text(f, word[:k], x0, ty, 104, MINT, kit.F_MONO, 0.92, 14.0,
                      "left", glow=0.26, glow_color=MINT)
        prog_w = kit.text_width(kit.F_MONO, 104, word[:k], 14.0)
        r0 = int(max(0, min(SH - 2, (ty + 132) * S)))
        a0 = int(max(0, min(SW - 1, x0 * S)))
        a1 = int(max(a0 + 1, min(SW, (x0 + prog_w) * S)))
        f[r0:r0 + 2, a0:a1] += MINT * 0.32
        f[r0 - 2:r0, a0:a1] += MINT * 0.09
    # 光标
    cxp = x0 + kit.text_width(kit.F_MONO, 104, word[:k], 14.0) + 8
    stage.cursor(f, cxp, ty + 10, t, size=86, color=CREAM,
                 alpha=0.90 if k < len(word) else 0.80,
                 style="block", blink=(k >= len(word)), glow=0.5)

    # 落定之后：一个低亮度的圈把她框住
    if k >= len(word):
        ring(f, W * 0.5, fy + 2, 150 + 44 * kit.cl((t - 118.95) / 1.0), 1.1,
             MINT, 0.09, 1.84, 2.8)

    stage.dust(f, t, 45, MINT, 0.09, (0.3, 0.7, 0.3, 0.86))
    stage.hud_corner(f, t, left="field: 1  ·  extent: 0.02",
                     right="chars %d/%d" % (k, len(word)), alpha=0.24,
                     color=MINT, y=H * 0.072)
    c.lyric(size=38, y=H * 0.878, glow=0.16, alpha=0.50)
    # 极慢推镜
    c.f = f
    c.zoom_frame(1.0 + 0.014 * kit.ease_io(p))


# ══════════════════════════════════════════════════════════ 5. ERASE
@scene(T_ERASE, T_CHAL, "ERASE", "G")
def erase(c):
    f = c.f
    t = c.t
    u = t - T_ERASE
    p = kit.cl(u / (T_CHAL - T_ERASE))
    pw = c.pulse(9.0)
    c.post.update(bloom=(0.64, 20, 0.46), vignette=0.54, grain=0.021, exposure=0.86)

    f += kit.vgrad(SH, SW, kit.hx("02060B"), kit.hx("04101A"), 1.4)
    kit.add_radial(f, W * 0.50, H * 0.48, 520, kit.hx("0A2634"), 0.30, 2.5)

    cx0, cy0 = W * 0.500, H * 0.500
    F1 = pfield(c, stage.MAID_L, 860.0, 32.0)
    P0 = cloud.Cloud.transform(F1["P"], yaw=0.18 + 0.06 * math.sin(u * 0.6))

    rnd = c.cl.rnd
    rnd2 = np.mod(rnd * 137.03, 1.0)
    # 何时剥离：中上部 / 右半先走
    bias = 0.42 * np.clip(P0[:, 0] / 250.0, -1, 1) - 0.52 * np.clip(P0[:, 1] / 440.0, -1, 1)
    tp = 120.30 + 3.35 * np.mod(rnd * 3.71, 1.0) - bias
    frag = rnd < 0.46
    tau = t - tp
    active = frag & (tau > 0.0)

    # 风：速度衰减的位移 + 卷曲
    disp = 900.0 * (1.0 - np.exp(-np.clip(tau, 0, 99) / 0.55))
    dx = 0.92 + 0.20 * (rnd2 - 0.5)
    dy = -0.56 + 0.26 * (rnd2 - 0.5)
    curl = 78.0 * np.sin(np.clip(tau, 0, 9) * 2.2 + rnd * 6.283)
    Pd = P0.copy()
    Pd[:, 0] += (disp * dx + curl * 0.45) * active
    Pd[:, 1] += (disp * dy + curl) * active
    Pd[:, 2] += (disp * 0.16 * (rnd2 - 0.5)) * active

    # ── 扫描带：从下往上，扫到即清空
    scan_p = kit.ease_io(kit.cl((t - 120.58) / 3.85))
    sy_scan = (1.12 - 1.18 * scan_p) * H
    sx_, sy_, _z, _k = c.cl.project(P0, 1500.0, 1150.0, cx0, cy0, 0.0)
    alive = (sy_ < sy_scan)
    near = np.exp(-((sy_ - sy_scan) / 22.0) ** 2)
    fa = np.clip(1.0 - tau / 2.30, 0, 1) ** 1.5 * active

    colb = fig_col(F1["lum"], F1["edge"], kit.hx("0B2E4A"), kit.hx("6FB6C8"), MINT, 1.7)
    colf = kit.mixc(CORAL, GOLD, np.clip(rnd2 * 1.4, 0, 1))
    col = kit.mixc(colb, colf, np.clip(active * 1.0, 0, 1))
    bright = (body_bright(F1["lum"], F1["edge"], 1.15) * alive * (1.0 - active)
              * (1.0 + 2.8 * near) + fa * (0.10 + 2.7 * rnd2 ** 1.8))
    bright = bright * (0.9 + 0.25 * pw)

    frac = float(np.clip(alive.sum() / float(max(1, alive.size)), 0.04, 1.0))
    pdraw(c, Pd, col, bright=bright, cx=cx0, cy=cy0, dist=1500.0, size=1.42,
          blur_r=1.00, target=0.60 * (0.30 + 0.70 * frac), glow=0.32, core=0.9,
          shade=0.5)

    # ── 扫描线本体
    if -10 < sy_scan < H + 10:
        inten = 0.75 + 0.25 * c.kick()
        hscan(f, sy_scan, CREAM, 0.05 * inten, soft=24.0)
        hscan(f, sy_scan, kit.WHITE, 0.52 * inten, soft=1.5)
        hscan(f, sy_scan - 3.0, MINT, 0.10 * inten, soft=3.0)
        rg = c.rng(41)
        for _ in range(5):
            gx = rg.uniform(0.0, W)
            gw = rg.uniform(40.0, 220.0)
            gy = sy_scan + rg.normal(0.0, 7.0)
            a0 = int(max(0, min(SW - 1, gx * S)))
            a1 = int(max(a0 + 1, min(SW, (gx + gw) * S)))
            r0 = int(max(0, min(SH - 1, gy * S)))
            f[r0:r0 + 2, a0:a1] += CORAL * (0.45 * inten)

    # ── 剩余计数
    rem = int(np.count_nonzero(alive & ~active))
    kit.draw_text(f, "fragments %05d" % rem, W * 0.50, H * 0.056, 24, MINT,
                  kit.F_MONO, 0.34, 5.0, "center", glow=0.16)
    kit.draw_text(f, "erase_pointless()", W * 0.50, H * 0.056 + 32, 19, CREAM,
                  kit.F_MONO, 0.18, 3.0, "center")

    # ── 尾巴：只剩一个白色光标（周围还有几粒没散尽的余烬）
    after = kit.cl((t - 124.35) / 0.7)
    if after > 0.01:
        gy = H * 0.452
        kit.add_radial(f, cx0, gy + 20, 210, kit.hx("0A2430"), 0.30 * after, 2.6)
        rg = c.rng(77)
        motes = []
        for _ in range(9):
            ang = rg.uniform(-2.6, -0.5)
            sp = rg.uniform(0.4, 1.0)
            age = rg.uniform(0.0, 2.2)
            motes.append((cx0 + math.cos(ang) * 120 * sp * (1.0 + 0.5 * age),
                          gy - 40 - math.sin(-ang) * 90 * sp * (1.0 + 0.5 * age),
                          math.exp(-age * 0.9)))
        rings(f, [(mx, my, 2.0, 1.0, 1.0, 0.30 * ma * after) for (mx, my, ma) in motes],
              CREAM, glow=2.4)
        stage.cursor_trail(f, [(cx0 - 6, gy - 44), (cx0, gy - 14), (cx0, gy)],
                           CREAM, 0.20 * after, 1.4)
        stage.cursor(f, cx0, gy, t, size=30, color=CREAM, alpha=0.95 * after,
                     style="beam", blink=True, glow=0.85)
        kit.draw_text(f, "0 fragments", W * 0.50, H * 0.056, 24, MINT,
                      kit.F_MONO, 0.42 * after, 5.0, "center", glow=0.2)

    stage.hud_corner(f, t, left="SCAN up // wipe",
                     right="%.0f%% erased" % (100.0 * scan_p), alpha=0.24,
                     color=MINT, y=H * 0.115)
    c.lyric(size=40, y=H * 0.878, glow=0.18, alpha=0.88)
    c.f = f


# ══════════════════════════════════════════════════════════ 6. CHALLENGE
_CRACKS = None


def _cracks():
    """固定的裂纹折线（跨帧一致）：从神的胸口炸开，另两条从画幅边缘爬进来。"""
    global _CRACKS
    if _CRACKS is not None:
        return _CRACKS
    rg = np.random.default_rng(12946)
    ox, oy = W * 0.520, H * 0.415
    out = []
    for i in range(11):
        if i < 9:
            ang = -math.pi * 0.5 + (i - 4) * 0.34 + rg.normal(0, 0.14)
            x, y = ox + math.cos(ang) * rg.uniform(20, 52), oy + math.sin(ang) * rg.uniform(20, 52)
        else:
            side = i - 9
            x = 4.0 if side == 0 else W - 4.0
            y = H * rg.uniform(0.30, 0.78)
            ang = (0.0 if side == 0 else math.pi) + rg.normal(0, 0.5)
        x, y = float(x), float(y)
        pts = [(x, y)]
        for _ in range(int(rg.integers(9, 15))):
            ang += rg.normal(0, 0.30)
            L = rg.uniform(30, 74)
            x += math.cos(ang) * L
            y += math.sin(ang) * L
            pts.append((x, y))
        out.append((i * 1.0, pts))
        # 分叉
        for _b in range(2 if i < 6 else 1):
            if rg.random() < 0.45:
                continue
            j = int(rg.integers(2, max(3, len(pts) - 1)))
            bx, by = pts[j]
            a2 = ang + float(rg.choice([-1.0, 1.0])) * rg.uniform(0.55, 1.05)
            br = [(bx, by)]
            for _ in range(int(rg.integers(3, 7))):
                a2 += rg.normal(0, 0.30)
                L2 = rg.uniform(20, 48)
                bx += math.cos(a2) * L2
                by += math.sin(a2) * L2
                br.append((bx, by))
            out.append((i * 1.0 + 0.34, br))
    _CRACKS = out
    return out


def _draw_cracks(f, t):
    m = _scratch()
    m[:] = 0
    bh, bw = m.shape
    tips = []
    for (i0, pts) in _cracks():
        t0 = 126.86 + i0 * 0.10
        q = kit.cl((t - t0) / 0.95)
        if q <= 0.0:
            continue
        n = max(2, int(round(len(pts) * q)))
        pp = np.array([(x * bw / W, y * bh / H) for (x, y) in pts[:n]], np.int32)
        cv2.polylines(m, [pp], False, 255, 1, cv2.LINE_AA)
        if q < 1.0:
            tips.append(pts[n - 1])
    mf = m.astype(F32) * (1 / 255.0)
    if mf.max() < 0.02:
        return
    f += kit.up(mf[..., None], SW, SH) * (CREAM * 0.90)
    gl = cv2.GaussianBlur(mf, (0, 0), max(1.0, 2.6 * bh / H))
    f += kit.up(gl[..., None], SW, SH) * (CORAL * 1.25)
    for (tx, ty) in tips:
        kit.add_radial(f, tx, ty, 26, CORAL, 0.55, 1.8)
        kit.add_radial(f, tx, ty, 7, kit.WHITE, 0.85, 1.4)


@scene(T_CHAL, T_END, "CHALLENGE", "G")
def challenge(c):
    f = c.f
    t = c.t
    u = t - T_CHAL
    p = kit.cl(u / (T_END - T_CHAL))
    pl = c.pulse(8.0)
    c.post.update(bloom=(0.60, 22, 0.54), vignette=0.46, grain=0.020,
                  exposure=0.86, chroma=0.0018 + 0.010 * kit.cl((u - 2.6) / 0.9))

    f += kit.vgrad(SH, SW, kit.hx("01030A"), kit.hx("0A0716"), 1.25)

    # ── 神像（仰视：pitch 让头部朝镜头压下来，尺寸压满画幅）
    cxg, cyg = W * 0.520, H * 0.575
    G = c.cl.sh(cloud.shape_god, 1180.0)
    Pg = cloud.Cloud.transform(G, pitch=-0.18, yaw=0.05 + 0.05 * math.sin(u * 0.85))
    hy = np.clip((Pg[:, 1] + 800.0) / 1600.0, 0, 1)

    cg = kit.mixc(kit.hx("2A1046"), VIOLET, np.clip(hy * 1.7, 0, 1))
    cg = kit.mixc(cg, GOLD, np.clip((hy - 0.74) / 0.26, 0, 1) * 0.9)
    err = kit.cl((u - 1.55) / 1.1)
    cg = kit.mixc(cg, RUST, 0.40 * err * np.clip(c.cl.rnd * 1.4, 0, 1))

    # 光环（细、暗，只做指示）
    hyy = cyg - 300
    rings(f, [(cxg, hyy, 214 + 5 * math.sin(u * 1.7), 1.3, 1.0,
               0.09 + 0.04 * pl),
              (cxg, hyy, 243, 1.0, 1.0, 0.05)], GOLD, glow=3.0)

    bg = (0.12 + 1.6 * hy ** 1.6 + 2.2 * np.clip(c.cl.rnd - 0.62, 0, 1) * 2.4)
    bg = bg * (0.85 + 0.30 * pl) * (1.0 - 0.30 * err)
    pdraw(c, Pg, cg, bright=bg, cx=cxg, cy=cyg, dist=1500.0, size=1.40,
          blur_r=1.25, target=0.66, glow=0.38, core=0.95, shade=0.55,
          twinkle=(t * 1.6, 0.16))
    kit.add_radial(f, cxg, hyy, 420, kit.hx("2A1B4E"), 0.30, 2.4)
    kit.add_radial(f, cxg, cyg - 200, 640, kit.hx("14081C"), 0.30, 2.6)
    # 神脚下：错误色渗出
    kit.add_radial(f, cxg, H * 0.975, 520, RUST, 0.10 + 0.20 * err, 2.4, ratio=1.7)

    # ── 她的剪影 + 举起的手（身后一片光，让她成为真正的黑剪影）
    hx_, hy_ = W * 0.352, H * 0.952
    vscan(f, hx_ + 6, kit.hx("7A2A24"), 0.16 + 0.06 * pl, soft=88.0)
    vscan(f, hx_ + 6, CORAL, 0.05, soft=20.0)
    kit.add_radial(f, hx_, hy_ - 120, 240, kit.hx("4A1E1A"), 0.50, 2.2, ratio=0.85)
    kit.add_radial(f, hx_ + 10, hy_ + 6, 130, RUST, 0.30, 2.0, ratio=0.35)
    stage.character(f, stage.MAID_L, hx_, hy_, 292, alpha=1.0,
                    tint=kit.hx("01050A"), tint_k=0.95, glow=0.0,
                    rim=CORAL, rim_k=1.05, shadow=0.0)
    arm = kit.cl((u - 0.60) / 0.85)
    if arm > 0.02:
        ax = 1.0 - (1.0 - arm) ** 2
        sh = (hx_ + 12, hy_ - 208)
        el = (hx_ + 38 * ax, hy_ - 208 - 44 * ax)
        hd = (hx_ + 38 * ax - 10 * ax, hy_ - 208 - 44 * ax - 56 * ax)
        m = _scratch()
        m[:] = 0
        bh, bw = m.shape
        pp = np.array([(x * bw / W, y * bh / H) for (x, y) in [sh, el, hd]], np.int32)
        cv2.polylines(m, [pp], False, 255, 2, cv2.LINE_AA)
        mf = m.astype(F32) * (1 / 255.0)
        f += kit.up(mf[..., None], SW, SH) * (CORAL * 0.95)
        f += kit.up(cv2.GaussianBlur(mf, (0, 0), 2.6)[..., None], SW, SH) * (CORAL * 1.1)
        if ax > 0.6:
            g = kit.cl((ax - 0.6) / 0.4)
            kit.add_radial(f, hd[0], hd[1], 44, CORAL, 0.32 * g, 2.0)
            kit.add_radial(f, hd[0], hd[1], 9, kit.WHITE, 0.80 * g, 1.5)
            vscan(f, hd[0], CORAL, 0.05 * g, soft=8.0)

    # ── 裂纹 + 系统错误
    _draw_cracks(f, t)
    err2 = kit.cl((u - 1.50) / 0.62)
    if err2 > 0.02:
        kit.glitch_rows(f, t, 0.26 * err2, 9, 10)
        kit.draw_text(f, "FATAL: unhandled argument", W * 0.055, H * 0.605, 26,
                      CORAL, kit.F_MONO, 0.80 * err2, 3.0, "left", glow=0.32)
        kit.draw_text(f, "  raise Challenge(god=0x%04X)" % (int(4096 * (1.0 - p)) + 59049),
                      W * 0.055, H * 0.605 + 40, 20, RUST, kit.F_MONO, 0.62 * err2,
                      2.0, "left")

    stage.hud_corner(f, t, left="RPC  challenge(your_god)",
                     right="stack: illegal", alpha=0.30, color=CORAL, y=H * 0.028)
    c.lyric(size=42, y=H * 0.858, glow=0.24, alpha=0.95)

    # ── 结尾 0.3s 白闪，硬切
    w = kit.cl((t - (T_END - 0.30)) / 0.30)
    c.f = f
    if w > 0.001:
        c.f *= (1.0 - 0.94 * w)
        c.f += kit.WHITE * (0.94 * w)
        c.zoom_frame(1.0 + 0.03 * w)
