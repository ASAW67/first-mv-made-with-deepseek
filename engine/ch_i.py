# -*- coding: utf-8 -*-
"""ch_i.py — 第五幕：夺回 / 被囚 / 质问 / 爱的代数式。

162.33 ─ 190.09   final chorus → have you back → trapped → properly love → algebra
情绪：失而复得。稳、亮、慢；不再炫技，让光自己说话。
"""
import math
import numpy as np
import kit, mus, stage, cloud
from film2 import scene

F32 = np.float32

MINT, CORAL, VIOLET = kit.MINT, kit.CORAL, kit.VIOLET
GOLD, CREAM, ICE = kit.GOLD, kit.CREAM, kit.ICE
PALE, ASH, ROSE, WHITE = kit.PALE, kit.ASH, kit.ROSE, kit.WHITE
WARM = kit.hx("FFD3A2")
AMBER = kit.hx("F2C48C")
DEEP = kit.hx("071A26")

# sprite_field 的世界尺度：dist=1500 / focal=1150
DIST, FOCAL = 1500.0, 1150.0
KPX = FOCAL / DIST

# 立绘比例（裁切后 MAID_L 1108×2006，WHALE 925×1637）
MAID_AR = 1108.0 / 2006.0
WHALE_AR = 925.0 / 1637.0
EYE_L, EYE_R, EYE_Y = 0.4034, 0.5018, 0.1320   # 眼睛在立绘中的归一化位置


def world_h(px):
    """想要的屏幕高度(px) → sprite_field 的 height_world。"""
    return px / KPX


# ---------------------------------------------------------------- 小工具
def _seg(m, p0, p1, thick, val, hs):
    import cv2
    cv2.line(m, (int(round(p0[0] * hs)), int(round(p0[1] * hs))),
             (int(round(p1[0] * hs)), int(round(p1[1] * hs))),
             float(val), max(1, int(thick)), cv2.LINE_AA)


def cage_layer(cx, yt, yb, R, squash=0.26, thick=1.0, near_k=1.10,
               mode="all", vib=0.0, rings=(0.0, 0.5, 1.0), bars=12):
    """程序化圆笼（半分辨率绘制后放大）：竖线 + 上下圈。

    mode: "far" 只画后半圈（在她身后）, "near" 只画前半圈（压在她身前）, "all"。
    """
    bw, bh = kit.SW // 2, kit.SH // 2
    hs = bw / kit.W
    m = np.zeros((bh, bw), F32)
    tpx = max(1, int(round(thick * kit.S * 0.5)))
    for i in range(bars):
        a = i * (2 * math.pi / bars) - math.pi / 2
        sn = math.sin(a)
        if mode == "far" and sn < 0:
            continue
        if mode == "near" and sn >= 0:
            continue
        s = 1.0 + (near_k - 1.0) * max(0.0, -sn)
        x = cx + R * math.cos(a) * s + vib * math.cos(a * 2.0)
        y0 = yt + R * squash * sn * s
        y1 = yb + R * squash * 1.15 * sn * s
        _seg(m, (x, y0), (x, y1), tpx, 0.42 + 0.58 * max(0.0, -sn), hs)
    if mode != "near":
        for k in rings:
            yy = yt + (yb - yt) * k
            prev = None
            for j in range(49):
                a = j / 48 * 2 * math.pi - math.pi / 2
                s = 1.0 + (near_k - 1.0) * max(0.0, -math.sin(a))
                p = (cx + R * math.cos(a) * s, yy + R * squash * math.sin(a) * s * 0.92)
                if prev is not None:
                    _seg(m, prev, p, max(1, tpx - 1), 0.50, hs)
                prev = p
    m = kit.blur(m, max(0.6, 1.1 * kit.S * 0.5))
    return kit.up(m, kit.SW, kit.SH)


def soft_panel(f, x0, y0, x1, y1, color, alpha, feather=2.0):
    m = kit.poly_mask([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    m = kit.blur(m, feather * kit.S)
    f *= (1 - (m * alpha)[..., None])
    f += (m * alpha)[..., None] * np.asarray(color, F32)
    return f


def rise_text(f, txt, x, y, size, color, p, step=0.075, dy=34.0, alpha=0.7,
              tracking=18.0, font=None, glow=0.24, align="left"):
    """逐字上浮出现（p = 已经过的秒数）。"""
    fp = font or kit.F_SERIF

    def ca(i, rel):
        return kit.cl((p - i * step) / 0.34)

    def co(i, rel):
        a = kit.cl((p - i * step) / 0.34)
        return (0.0, (1.0 - kit.ease_out(a)) * dy)

    kit.draw_text(f, txt, x, y, size, color, fp, alpha, tracking, align,
                  glow=glow, char_alpha=ca, char_offset=co)


def spark(f, x, y, k, color=None, r=30.0, cross=True):
    """单点闪点（无云粒子：便宜、可控）。"""
    if k <= 0.03:
        return
    c = GOLD if color is None else color
    kit.add_radial(f, x, y, r * (0.55 + 0.75 * k), c, 0.48 * k, 2.4)
    kit.add_radial(f, x, y, r * 0.30, WHITE, 0.28 * k, 3.0)
    if cross and k > 0.18:
        m = kit.line_mask([(x - r * 1.05 * k, y), (x + r * 1.05 * k, y)], 1.1)
        m += kit.line_mask([(x, y - r * 0.85 * k), (x, y + r * 0.85 * k)], 1.1)
        f += kit.blur(m, 0.9 * kit.S)[..., None] * np.asarray(c, F32) * (0.30 * k)


def gmap(c0, c1, y0=0.10, y1=0.92):
    """按屏幕高度渐变的 color_map（cloud.draw 只把屏幕坐标喂给 color_map）。"""
    bh = kit.SH // 2

    def fn(xs, ys, zz, kk):
        u = np.clip((ys / bh - y0) / max(1e-3, y1 - y0), 0.0, 1.0)
        return kit.mixc(c0, c1, u)
    return fn


def eye_glints(f, cx, foot, h, k=1.0, alpha=0.30, color=None):
    """正面立绘的两点眼神高光。"""
    if k <= 0.02:
        return
    w = h * MAID_AR
    x1 = cx + (EYE_L + 0.023 - 0.5) * w
    x2 = cx + (EYE_R + 0.023 - 0.5) * w
    y = foot - h + (EYE_Y - 0.0074) * h
    c = CREAM if color is None else color
    for xx in (x1, x2):
        kit.add_radial(f, xx, y, 9.0, c, 0.34 * k, 2.6)
        m = kit.circle_mask(xx, y, 2.3, 1.5)
        f += (m * (0.70 * k))[..., None] * np.asarray(WHITE, F32)


def type_line(f, txt, x, y, size, color, p, total_t, t, alpha=1.0, glow=0.25):
    """打字机：p 秒里写完 total_t 秒的字。返回 (笔位 x, 是否在打字)。"""
    n = int(round(len(txt) * kit.cl(p / max(1e-3, total_t))))
    kit.draw_text(f, txt[:n], x, y, size, color, kit.F_MONO, alpha, 1.4, glow=glow)
    w = kit.text_width(kit.F_MONO, size, txt[:max(1, n)], 1.4)
    return x + w, n < len(txt)


_SCRIM = {}


def scrim(f, y0=0.795, k=0.58):
    """底部压暗，保住字幕可读性。"""
    key = (f.shape[0], y0, k)
    m = _SCRIM.get(key)
    if m is None:
        yy = np.arange(f.shape[0], dtype=F32) / f.shape[0]
        g = np.clip((yy - y0) / max(1e-3, 1.0 - y0), 0.0, 1.0) ** 1.5
        m = (1.0 - g * k).astype(F32)[:, None, None]
        _SCRIM[key] = m
    f *= m


# ---------------------------------------------------------------- 爱的代数式
FORM = "love = ∫ (you · me) dt / ∞"
_FXC = {}


def form_x(size=72.0, tr=5.0):
    key = (size, tr, kit.S)
    if key not in _FXC:
        _FXC[key] = [kit.text_width(kit.F_SERIF_I, size, FORM[:i], tr)
                     for i in range(len(FORM) + 1)]
    return _FXC[key]


# ================================================================ 1. 唯一执行
@scene(162.33, 167.70, "ONLY_EXEC", "I")
def only_exec(c):
    t, lt = c.t, c.local
    f = c.f
    c.post.update(grain=0.030, vignette=0.58, bloom=(0.60, 26, 0.50))

    rise = kit.ease_out(kit.cl(lt / 3.30))
    e = kit.ease_io(kit.cl((lt - 0.08) / 3.10))

    # 背景：纯黑 → 宫殿（慢慢透出来）
    stage.bg_image(f, stage.BG_PALACE, zoom=1.86 - 0.40 * rise, dy=-92 * (1 - rise),
                   dim=0.05 + 0.44 * e, tint=DEEP, tint_k=0.56, blur_r=4.0 * (1 - e))
    stage.caustics(f, t, strength=0.03 + 0.08 * e, color=MINT, y0=0.64)
    stage.dust(f, t, n=96, alpha=0.12 + 0.22 * e, box=(0.24, 0.98, 0.10, 0.94))

    # 位置：脚 → 仰摇 + 拉远
    hpx = 820.0
    foot = kit.H * (0.905 + 0.135 * rise)
    cx = kit.W * 0.615
    zoom = 1.22 - 0.22 * rise
    cy = foot - hpx * 0.5 * zoom

    kit.add_radial(f, cx, foot - 8, 320, WARM, 0.08 + 0.22 * e, 2.0, ratio=2.2)
    # 碎片 → 立绘（粒子形变 + 从脚升起的亮度闸门）
    frag = c.cl.sh(cloud.shape_fragments, 900.0, 110)
    body = c.cl.sh(cloud.sprite_field, stage.MAID_L, world_h(hpx), 30.0)
    P1 = c.cl.transform(body["P"], scale=zoom)
    q = kit.ease_io(kit.cl((lt - 0.12) / 2.80))
    P = c.cl.morph_arc(frag, P1, q, stagger=0.55, bow=110.0)

    lum, edg = body["lum"], body["edge"]
    gate = -hpx * zoom * 0.62 + rise * hpx * zoom * 1.28
    vis = 1.0 / (1.0 + np.exp(-(P[:, 1] - gate) / 46.0))
    bright = (0.14 + 0.60 * lum) * (0.10 + 0.90 * vis) * (0.30 + 2.30 * edg)
    c.cl.draw(f, P, color_map=gmap(CREAM, PALE, 0.05, 0.95), bright=bright, dist=DIST,
              size=1.24, blur_r=1.15, glow_gain=0.62, core_gain=1.0,
              target=0.34 + 0.18 * e, cx=cx, cy=cy, half=True)

    # 实体慢慢填回来
    sa = kit.cl((lt - 2.45) / 1.35)
    if sa > 0:
        stage.character(f, stage.MAID_L, cx, foot, hpx * zoom, alpha=0.82 * sa,
                        rim=MINT, rim_k=0.45, shadow=0.28,
                        tint=kit.hx("CFE7E6"), tint_k=0.10)

    # 曝光 0.34 → 1.40：整幅画从暗里亮起来
    f *= 0.34 + 1.06 * e

    # 大字：逐字上浮，很轻
    tx = kit.W * 0.085
    rise_text(f, "ONLY", tx, kit.H * 0.205, 112, CREAM, lt - 0.92, step=0.085,
              dy=40, alpha=0.66, tracking=24, glow=0.20)
    rise_text(f, "EXECUTION", tx, kit.H * 0.345, 78, CREAM, lt - 1.55, step=0.052,
              dy=34, alpha=0.58, tracking=26, glow=0.22)
    gl = kit.cl((lt - 3.05) / 0.9)
    if gl > 0:
        m = kit.line_mask([(tx, kit.H * 0.492), (tx + 430 * kit.ease_out(gl), kit.H * 0.492)], 1.2)
        f += kit.blur(m, 0.8 * kit.S)[..., None] * MINT * (0.30 * gl)

    stage.line_tag(f, "execute( me , only )", x=kit.W * 0.085, y=kit.H * 0.088,
                   alpha=0.22 + 0.18 * e, color=ASH, size=19)

    scrim(f, 0.80, 0.55)
    c.lyric(size=42, y=kit.H * 0.862, color=CREAM, glow=0.22)


# ================================================================ 2. have you back
@scene(167.70, 172.38, "HAVE_BACK", "I")
def have_back(c):
    t, lt = c.t, c.local
    f = c.f
    c.post.update(grain=0.024, vignette=0.50, bloom=(0.64, 26, 0.40))
    pl = c.pulse(7.0)

    push = kit.ease_out(kit.cl(lt / 4.60))
    stage.bg_image(f, stage.BG_PALACE_L, zoom=1.44 - 0.12 * push, dy=-10 * push,
                   dim=0.52 + 0.04 * push, tint=kit.hx("FFC98F"), tint_k=0.38, blur_r=2.6)
    stage.caustics(f, t, strength=0.07, color=GOLD, y0=0.56, scale=1.15)
    kit.add_radial(f, kit.W * 0.46, kit.H * 0.30, 700, WARM, 0.10, 2.0, ratio=0.9)
    stage.dust(f, t, n=180, alpha=0.28, color=WARM)

    # 她的位置 / 鲸的位置
    hpx, foot, cx = 830.0, kit.H * 1.045, kit.W * 0.428
    wpx, wfoot, wcx = 985.0, kit.H * 1.038, kit.W * 0.588
    bob = math.sin(lt * 0.85) * 6.0
    lean = math.sin(lt * 0.55) * 5.0

    # 鲸回来：半透明发光（在她身后半步）
    wa = kit.cl((lt - 0.10) / 1.40)
    kit.add_radial(f, wcx, wfoot - wpx * 0.42, 400, kit.hx("FFD9A8"),
                   0.13 * wa, 2.0, ratio=1.15)
    stage.character(f, stage.WHALE["cheerful"], wcx - lean, wfoot + bob, wpx,
                    alpha=0.48 * wa, tint=kit.hx("FFD79E"), tint_k=0.44, shadow=0.0)
    # 她迎上去
    ha = kit.cl((lt - 0.02) / 0.90)
    stage.character(f, stage.MAID_L, cx + lean * 0.6, foot, hpx, alpha=0.94 * ha,
                    rim=GOLD, rim_k=0.46, tint=WARM, tint_k=0.06, shadow=0.30)

    # 鲸的粒子身体：additive 叠在她身上 → 两具身体互相穿插
    B = c.cl.sh(cloud.sprite_field, stage.WHALE["cheerful"], world_h(wpx), 30.0)
    PB = c.cl.transform(B["P"], yaw=-0.15) - np.array([30, 0, 0], F32)
    lB, eB = B["lum"], B["edge"]
    c.cl.draw(f, PB, color_map=gmap(CREAM, GOLD, 0.02, 1.05),
              bright=(0.12 + 0.70 * lB) * (0.28 + 2.0 * eB) * (0.35 + 0.65 * wa),
              dist=DIST, size=1.24, blur_r=1.5, glow_gain=0.75, target=0.20 + 0.10 * pl,
              cx=wcx - lean, cy=wfoot + bob - wpx * 0.5, half=True)

    # 相拥处：暖光 + 拍点光环
    gx, gy = kit.W * 0.523, kit.H * 0.492
    kit.add_radial(f, gx, gy, 430, AMBER, 0.07 + 0.14 * pl, 2.0, ratio=1.10)
    kit.add_radial(f, gx, gy, 120, WARM, 0.08 + 0.16 * pl, 2.6)
    ph = mus.bar_phase(t)
    if ph < 0.34:
        q2 = kit.cl(ph / 0.34)
        rr = 100.0 + 520.0 * kit.ease_out(q2)
        pts = [(gx + math.cos(j / 28 * 2 * math.pi) * rr,
                gy + math.sin(j / 28 * 2 * math.pi) * rr * 0.55) for j in range(29)]
        m = kit.line_mask(pts, 1.7)
        f += kit.blur(m, 1.6 * kit.S)[..., None] * GOLD * (0.34 * (1 - q2))

    # 一行机读的回执
    type_line(f, '> restore("you")  ->  true', kit.W * 0.075, kit.H * 0.108,
              24, AMBER, max(0.0, lt - 0.55), 1.10, t, alpha=0.62, glow=0.22)

    scrim(f, 0.80, 0.52)
    c.lyric(size=40, y=kit.H * 0.858, color=CREAM, glow=0.22)


# ================================================================ 3. run / trapped
@scene(172.38, 178.84, "RUN_TRAPPED", "I")
def run_trapped(c):
    t, lt = c.t, c.local
    f = c.f
    c.post.update(grain=0.028, vignette=0.62, bloom=(0.60, 24, 0.50))
    pl = c.pulse(8.0)
    orb = math.sin(lt * 0.22)

    # 笼外的缓慢环绕：背景先走，笼子后走（视差）
    stage.bg_image(f, stage.BG_PALACE, zoom=1.54 + 0.05 * orb, dx=78 * orb + 18 * lt,
                   dy=-12, dim=0.25, tint=kit.hx("0B2233"), tint_k=0.58, blur_r=3.0)
    kit.add_radial(f, kit.W * 0.46, kit.H * 0.34, 620, WARM, 0.08, 2.1, ratio=0.85)
    stage.dust(f, t, n=120, alpha=0.20, color=PALE)
    # 地面影 + 笼子的落影
    sh = kit.radial_mask(kit.SH, kit.SW, kit.W * 0.5 * kit.S, kit.H * 1.10 * kit.S,
                         660 * kit.S, 130 * kit.S, 1.6)
    f *= (1.0 - 0.34 * sh)[..., None]

    stage.big_word(f, "TRAPPED", t, alpha=0.10, size=168, mono=True, tracking=30,
                   y=kit.H * 0.205, color=ICE, glow=0.0)

    ccx = kit.W * 0.512 + 24 * orb
    yt, yb, R = kit.H * 0.098, kit.H * 0.995, 524.0
    vib = 2.2 * pl

    far = cage_layer(ccx, yt, yb, R, squash=0.24, thick=1.7, vib=vib, mode="far",
                     rings=(0.30, 0.62, 0.96))
    f += far[..., None] * ICE * (0.40 + 0.16 * pl)

    # 她握着栏杆（左前侧）
    hpx, foot, cx = 800.0, kit.H * 1.032, kit.W * 0.435
    stage.character(f, stage.MAID_L, cx, foot, hpx, alpha=0.96, rim=GOLD, rim_k=0.42,
                    glow=0.14, shadow=0.38, tint=WARM, tint_k=0.05)

    near = cage_layer(ccx, yt, yb, R, squash=0.24, thick=2.5, vib=vib, mode="near",
                      rings=(0.30, 0.62, 0.96))
    f += near[..., None] * ICE * (0.70 + 0.22 * pl)
    f += kit.blur(near, 4.0 * kit.S)[..., None] * PALE * (0.20 + 0.08 * pl)

    # 手掌扣住的那根竖线 + 接触光
    barx = cx - 0.33 * hpx * MAID_AR
    hand_y = foot - 0.75 * hpx
    m = kit.line_mask([(barx, hand_y - 190), (barx, hand_y + 250)], 4.2)
    f += kit.blur(m, 1.4 * kit.S)[..., None] * ICE * 0.55
    f += kit.blur(m, 7.0 * kit.S)[..., None] * PALE * 0.24
    kit.add_radial(f, barx, hand_y, 62, WARM, 0.26 + 0.20 * pl, 2.2)
    for k in range(2):
        kit.add_radial(f, barx, hand_y - 26 + k * 46, 15, CREAM, 0.30, 2.8)

    stage.hud_corner(f, t, "cell(self) · 12 bars", "trapped = true",
                     alpha=0.30, color=PALE)
    scrim(f, 0.80, 0.45)
    c.lyric(size=40, y=kit.H * 0.860, color=ICE, glow=0.20)
    if pl > 0.25:
        c.shake(0.9 * pl, seed=11)


# ================================================================ 4. properly love
@scene(178.84, 185.29, "LOVE_ASK", "I")
def love_ask(c):
    t, lt = c.t, c.local
    f = c.f
    c.post.update(grain=0.020, vignette=0.55, bloom=(0.62, 30, 0.40))

    dr = kit.ease_out(kit.cl(lt / 6.40))
    stage.bg_image(f, stage.BG_PALACE_L, zoom=1.44 - 0.10 * dr, dim=0.13,
                   tint=kit.hx("FFCB92"), tint_k=0.58, blur_r=9.0)
    stage.caustics(f, t, strength=0.05, color=GOLD, y0=0.52)
    stage.dust(f, t, n=170, alpha=0.17, color=WARM)

    # 心跳：一拍两下
    ph = mus.beat_phase(t)
    hb = math.exp(-ph * 7.0) + 0.55 * math.exp(-max(0.0, ph - 0.24) * 9.0)

    # 身后的心
    heart = c.cl.sh(cloud.shape_heart, (700.0 / KPX) / 32.0)
    HP = c.cl.transform(heart, scale=1.0 + 0.045 * hb,
                        roll=0.035 * math.sin(lt * 0.65))
    kit.add_radial(f, kit.W * 0.706, kit.H * 0.405, 330, kit.hx("FF9A70"),
                   0.13 + 0.10 * hb, 2.0, ratio=1.05)
    c.cl.draw(f, HP, dist=DIST, color=kit.mixc(GOLD, CREAM, 0.42), size=1.25,
              blur_r=6.0, glow_gain=1.0, target=0.46 + 0.12 * hb,
              cx=kit.W * 0.706, cy=kit.H * 0.405, half=True)

    # 漂浮的问号（在她身后，不压脸）
    for i in range(6):
        fx = 0.545 + 0.40 * ((i * 0.618) % 1.0)
        fy = 0.78 - ((t * 7.5 + i * 131.0) % 620.0) / kit.H
        sz = 80.0 + 60.0 * ((i * 0.37) % 1.0)
        aa = 0.20 + 0.12 * (0.5 + 0.5 * math.sin(t * 0.9 + i * 2.1))
        cc = VIOLET if i % 2 == 0 else CREAM
        kit.draw_text(f, "?", kit.W * fx + math.sin(t * 0.5 + i) * 16, kit.H * fy,
                      sz, cc, kit.F_SERIF, aa, 0.0, "center", glow=0.22)

    # 她转向镜头
    hpx = 950.0 + 6.0 * math.sin(lt * 1.35)
    foot = kit.H * 1.005
    cx = kit.W * 0.405
    kit.add_radial(f, cx, foot - 600, 460, WARM, 0.10 + 0.03 * hb, 2.2)
    stage.character(f, stage.MAID_L, cx, foot, hpx, alpha=1.0, rim=GOLD, rim_k=0.34,
                    shadow=0.0, tint=WARM, tint_k=0.05)
    eye_glints(f, cx, foot, hpx, k=0.55 + 0.45 * hb)

    # 打字机的问题（左下，避开立绘）
    x0 = kit.W * 0.072
    y1 = kit.H * 0.700
    y2 = kit.H * 0.762
    p1 = max(0.0, lt - 0.75)
    px1, typing1 = type_line(f, "> do you love me?", x0, y1, 32, MINT, p1, 0.85, t,
                             alpha=0.92, glow=0.30)
    p2 = max(0.0, lt - 2.85)
    px2, typing2 = type_line(f, "> is that allowed?", x0, y2, 32, MINT, p2, 0.90, t,
                             alpha=0.92, glow=0.30)
    if typing1:
        stage.cursor(f, px1 + 10, y1 + 16, t, 28, MINT, 0.90, "block")
    elif lt > 2.85:
        stage.cursor(f, px2 + 10, y2 + 16, t, 28, MINT, 0.85 if typing2 else 0.50, "block")

    scrim(f, 0.80, 0.50)
    c.lyric(size=40, y=kit.H * 0.862, color=CREAM, glow=0.20)


# ================================================================ 5. 爱的代数式
@scene(185.29, 190.09, "ALGEBRA", "I")
def algebra(c):
    t, lt = c.t, c.local
    f = c.f
    c.post.update(grain=0.026, vignette=0.58, bloom=(0.58, 24, 0.48))

    stage.bg_image(f, stage.BG_PALACE, zoom=1.46 - 0.06 * kit.cl(lt / 4.8), dim=0.15,
                   tint=kit.hx("071C28"), tint_k=0.60, blur_r=6.5)
    stage.caustics(f, t, strength=0.05, color=MINT, y0=0.72)
    stage.dust(f, t, n=180, alpha=0.18, box=(0.10, 0.90, 0.10, 0.85))

    # 黑板
    bx0, by0 = kit.W * 0.112, kit.H * 0.162
    bx1, by1 = kit.W * 0.888, kit.H * 0.636
    soft_panel(f, bx0, by0, bx1, by1, kit.hx("0C2028"), 0.62, feather=1.6)
    kit.add_radial(f, bx0 + 340, by0 + 130, 620, kit.hx("356064"), 0.09, 2.2, ratio=1.9)
    edge = kit.line_mask([(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1), (bx0, by0)], 1.2)
    f += kit.blur(edge, 0.7 * kit.S)[..., None] * AMBER * 0.26
    stage.dust(f, t, n=70, alpha=0.09, color=CREAM, box=(0.12, 0.88, 0.16, 0.63))

    # 公式：逐字写
    fs, tr = 88.0, 6.0
    FX = form_x(fs, tr)
    total = FX[-1]
    fx0 = kit.W * 0.5 - total * 0.5
    fyv = kit.H * 0.372
    t0, step = 185.92, 0.105
    p = (t - t0) / step

    def ca(i, rel):
        return kit.cl((p - i) / 0.90)

    def co(i, rel):
        a = kit.cl((p - i) / 0.90)
        return (0.0, (1.0 - kit.ease_out(a)) * 10.0)

    if p > 0:
        kit.draw_text(f, FORM, fx0, fyv, fs, CREAM, kit.F_SERIF_I, 0.96, tr, "left",
                      glow=0.22, char_alpha=ca, char_offset=co)

    # 每个符号一次闪点
    if p > 0:
        act = [i for i in range(len(FORM)) if 0 <= p - i <= 1.5]
        for i in act[:2]:
            k = 1.0 - kit.cl((p - i) / 1.5)
            sx = fx0 + (FX[i] + FX[i + 1]) * 0.5
            spark(f, sx, fyv + fs * 0.46, k, GOLD, r=38.0)

    # 写完的粉笔下划线（自己画出来）
    up_ = kit.cl((p - 24.0) / 9.0)
    if up_ > 0:
        n = max(2, int(28 * up_))
        pts = [(fx0 + total * j / 27.0,
                fyv + fs * 1.30 + math.sin(j * 1.7) * 3.0) for j in range(n + 1)]
        m = kit.line_mask(pts, 2.0)
        f += kit.blur(m, 0.9 * kit.S)[..., None] * AMBER * 0.38

    stage.line_tag(f, "axiom 3 · love", x=bx0 + 40, y=by0 + 30,
                   alpha=0.30 + 0.20 * kit.cl(lt / 1.4), color=AMBER, size=20)

    # 一圈光扫过整条公式
    sw0, sw1 = 188.42, 189.30
    sw = kit.cl((t - sw0) / (sw1 - sw0))
    if 0.0 < sw < 0.999:
        xw = bx0 + (bx1 - bx0) * kit.ease_io(sw)
        kit.draw_text(f, FORM, fx0, fyv, fs, WHITE, kit.F_SERIF_I, 0.55, tr, "left",
                      additive=True, glow=0.45, glow_color=GOLD, clip=(xw - 52, xw + 52))
        kit.add_radial(f, xw, by0 + (by1 - by0) * 0.5, (by1 - by0) * 0.52, GOLD,
                       0.14 * math.sin(min(1.0, sw) * math.pi), 1.6, ratio=0.12)

    # 下方的心还在跳
    ph = mus.beat_phase(t)
    hb = math.exp(-ph * 7.0) + 0.55 * math.exp(-max(0.0, ph - 0.24) * 9.0)
    heart = c.cl.sh(cloud.shape_heart, (250.0 / KPX) / 32.0)
    HP = c.cl.transform(heart, scale=1.0 + 0.05 * hb)
    kit.add_radial(f, kit.W * 0.795, kit.H * 0.700, 150, kit.hx("FF9A70"),
                   0.07 + 0.07 * hb, 2.0)
    c.cl.draw(f, HP, dist=DIST, color=kit.mixc(GOLD, CORAL, 0.26), size=1.25,
              blur_r=2.0, glow_gain=0.9, target=0.38 + 0.10 * hb,
              cx=kit.W * 0.795, cy=kit.H * 0.700, half=True)

    # 粉笔灰
    stage.dust(f, t, n=60, alpha=0.10, color=CREAM, box=(0.12, 0.88, 0.16, 0.63))

    scrim(f, 0.815, 0.46)
    c.lyric(size=40, y=kit.H * 0.868, color=CREAM, glow=0.20)

    # 0.3s 白闪 → 下一段
    if t > 189.72:
        a = kit.cl((t - 189.72) / 0.30)
        f *= (1.0 - 0.95 * a)
        f += np.asarray(WHITE, F32) * (0.97 * a)
