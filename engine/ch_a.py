# -*- coding: utf-8 -*-
"""ch_a.py — 第一幕：开机 / 物件创建 / 世界初始化 / 前奏。
0.00 ─ 30.00  （点它是谁之前，先让她被创建出来）
"""
import math
import numpy as np
import kit, mus, stage, cloud
from film2 import scene

F32 = np.float32
S_ = stage

PD = dict(dist=1500, blur_r=1.5, size=1.5, glow_gain=0.55, core_gain=1.0, shade=0.8)


# ---------------------------------------------------------------- 工具
def aperture(f, open_amt, soft=40.0, cy=None):
    """CRT 开机：按 open_amt(0..1) 垂直展开可见区域。"""
    if open_amt >= 0.999:
        return f
    cy = kit.SH / 2 if cy is None else cy
    h = max(1.0, open_amt * kit.SH * 0.5)
    y = np.abs(np.arange(kit.SH, dtype=F32) - cy)
    m = np.clip((h - y) / max(1.0, soft * kit.S), 0, 1)
    f *= m[:, None, None]
    return f


def scan_flicker(f, t, amt=0.12):
    v = 1.0 - amt * (0.5 + 0.5 * math.sin(t * 61.0)) * (0.4 + 0.6 * (0.5 + 0.5 * math.sin(t * 13.0)))
    f *= v
    return f


def grid_floor(f, t, y=0.80, horizon=0.56, color=None, alpha=0.30, lines=18, persp=3.2):
    """透视网格地面（半分辨率绘制后放大）。"""
    color = kit.MINT if color is None else color
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
    m = kit.blur(m, 0.6 * kit.S)
    m = kit.up(m, kit.SW, kit.SH)
    fade = np.clip((np.arange(kit.SH, dtype=F32) / kit.SH - horizon) * 2.6, 0, 1)[:, None]
    f += (m * fade * alpha)[..., None] * np.asarray(color, F32)
    return f


def cv2line(m, p0, p1, v, thick):
    import cv2
    cv2.line(m, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), float(v), thick, cv2.LINE_AA)


# ================================================================ 0.00 开机
@scene(0.00, 2.53, "BOOT_0", "A")
def boot_0(c):
    t = c.t
    f = c.f
    c.post.update(grain=0.055, vignette=0.60, bloom=(0.60, 20, 0.55), scan=0.10)
    op = kit.cl((t - 0.42) / 0.50)
    if t < 0.46:
        p = kit.cl(t / 0.46)
        m = kit.circle_mask(kit.W / 2, kit.H / 2, 2.2 + 20 * p * p)
        f += (m * (0.55 + 0.45 * math.sin(t * 120)))[..., None] * kit.WHITE * 1.5
        kit.add_radial(f, kit.W / 2, kit.H / 2, 300 * p, kit.hx("7FE9FF"), 0.40 * p, 2.0)
        return
    # 开机展开：先是一条线，再张开
    aperture(f, op, 26.0)
    # 全屏极暗底噪 + 电流横纹
    f += kit.vgrad(kit.SH, kit.SW, kit.hx("061420"), kit.hx("010407"), 1.2) * (0.5 + 0.5 * op)
    for i in range(5):
        yy = (kit.H * (0.14 + 0.18 * i) + t * 40 * (1 + i)) % kit.H
        m = kit.poly_mask([(0, yy), (kit.W, yy), (kit.W, yy + 1.6), (0, yy + 1.6)])
        f += kit.blur(m, 2 * kit.S)[..., None] * kit.hx("2E7E96") * (0.05 + 0.05 * (i % 2))
    # 居中的机读大字：逐字敲出
    typed = kit.cl((t - 0.55) / 0.95)
    txt = "BOOT SEQUENCE"
    kit.draw_text(f, txt, kit.W / 2, kit.H * 0.40, 66, kit.MINT, kit.F_MONO_B,
                  0.85, 16, "center", glow=0.45)
    n = int(len(txt) * typed)
    kit.draw_text(f, txt[:n] + ("_" if int(t * 12) % 2 else ""), kit.W / 2, kit.H * 0.40, 66,
                  kit.WHITE, kit.F_MONO_B, 1.0, 16, "center")
    a2 = kit.cl((t - 1.15) / 0.5)
    if a2 > 0:
        stage.split_lines(f, kit.W * 0.36, kit.H * 0.53,
                          [("mem", "8.0 GB   "), ("line", "power  ON"),
                           ("cpu", "130.00 BPM"), ("id", "world.execute(me)")],
                          size=22, lead=1.7, alpha=a2 * 0.85, color=kit.PALE, accent=kit.MINT)
    # 歌词以终端行显示
    a3 = kit.cl((t - 0.66) / 0.4)
    ln = mus.LINES[0]
    kit.draw_text(f, "> " + ln["en"], kit.W * 0.30, kit.H * 0.78, 34, kit.CREAM, kit.F_MONO,
                  a3 * 0.95, 2.0, glow=0.25)
    kit.draw_text(f, ln["zh"], kit.W * 0.30, kit.H * 0.78 + 46, 22, kit.MINT, kit.F_SERIF_SC,
                  a3 * 0.5, 4.0)
    stage.hud_corner(f, t, "SESSION 0x0", "REC ●", 0.35 * a3)
    c.post["vignette"] = 0.55


@scene(2.53, 4.38, "BOOT_1", "A")
def boot_1(c):
    t = c.t
    f = c.f
    c.post.update(grain=0.03, vignette=0.6, bloom=(0.55, 20, 0.45), scan=0.04)
    stage.bg_image(f, stage.BG_PALACE, zoom=1.25, dim=0.10, blur_r=8, tint=kit.hx("08202E"), tint_k=0.7)
    # 代码面板：把「开机 → 保护」写成源码
    rows = [
        ("kw", "switch_on(power_line)"),
        ("cm", "  # Switch on the power line"),
        ("kw", "remember_to_put_on"),
        ("fn", "  protection = Shield(level=MAX)"),
        ("cm", "  # Remember to put on / Protection"),
        ("pl", "assert protection.active"),
        ("kw", "return"),
        ("pl", "  Ready"),
    ]
    typed = kit.cl((t - 2.55) / 1.7)
    stage.code_panel(f, t, rows, x=kit.W * 0.115, y=kit.H * 0.20, w=kit.W * 0.55,
                     alpha=1.0, size=21, typed=typed)
    # 右侧：逐渐成形的六边形护盾
    k = kit.cl((t - 3.4) / 0.9)
    if k > 0:
        cx, cy = kit.W * 0.78, kit.H * 0.50
        pts = [(cx + math.cos(a * math.pi / 3 + t * 0.3) * 150 * k,
                cy + math.sin(a * math.pi / 3 + t * 0.3) * 150 * k) for a in range(7)]
        m = kit.line_mask(pts, 1.6)
        f += m[..., None] * kit.MINT * 1.4
        m2 = kit.line_mask([(cx + math.cos(a * math.pi / 3 + t * 0.3 + 0.4) * 108 * k,
                             cy + math.sin(a * math.pi / 3 + t * 0.3 + 0.4) * 108 * k) for a in range(7)], 1.1)
        f += m2[..., None] * kit.MINT * 0.7
    c.lyric(ln=mus.LINES[1], y=kit.H * 0.80, size=40, mono=True, color=kit.MINT, glow=0.15)
    a = kit.cl((t - 3.9) / 0.35)
    if a > 0:
        kit.draw_text(f, "PROTECTION", kit.W / 2, kit.H * 0.88, 74, kit.MINT, kit.F_MONO_B,
                      a * 0.16, 22, "center")


@scene(4.38, 7.61, "LAY_DOWN", "A")
def lay_down(c):
    t = c.t
    f = c.f
    c.post.update(grain=0.035, vignette=0.55, bloom=(0.5, 24, 0.6))
    f += kit.vgrad(kit.SH, kit.SW, kit.hx("07131F"), kit.hx("02060B"), 1.4)
    kit.add_radial(f, kit.W / 2, kit.H * 0.85, 900, kit.hx("123B4E"), 0.30, 1.6)
    p = kit.cl((t - 4.38) / 1.5)
    A = c.cl.sh(cloud.shape_dots, (1000, 620, 340))
    A[:, 1] += 460
    B = c.cl.sh(cloud.shape_lattice, 900.0, -290.0, 60.0, 8.0, 0.10)
    P = c.cl.morph(A, B, p, stagger=0.75, curve=lambda q: q ** 2)
    col = kit.mixc(kit.PALE, kit.MINT, np.clip(0.4 + (t - 4.4) * 0.2, 0, 1))
    c.cl.draw(f, P, dist=1350, color=col, blur_r=0.9, size=1.0, glow_gain=0.5, shade=1.0,
              bright=(0.4 + 0.9 * (c.cl.v ** 2)), target=0.11)
    grid_floor(f, t, y=0.90, horizon=0.58, alpha=0.22 + 0.10 * c.pulse())
    # 物件开始聚合：一条竖起的轮廓
    if t > 5.85:
        q = kit.cl((t - 5.85) / 1.5)
        Pb = c.cl.sh(cloud.shape_figure, 620.0)
        Pb = c.cl.transform(Pb, offset=(0, 130, 0))
        M = c.cl.morph(B, Pb, q, stagger=0.6, ease="quint")
        c.cl.draw(f, M, dist=1350, color=kit.mixc(kit.MINT, kit.CREAM, 0.5), blur_r=1.6,
                  size=1.5, glow_gain=0.6, bright=0.9, target=0.55)
    c.lyric(size=42, y=kit.H * 0.80)
    c.post["vignette"] = 0.55


@scene(7.61, 10.38, "PARAMS", "A")
def params(c):
    t = c.t
    f = c.f
    c.post.update(grain=0.03, vignette=0.5, bloom=(0.55, 22, 0.5))
    stage.bg_image(f, stage.BG_PALACE, zoom=1.35, dx=40, dim=0.16, blur_r=10, tint=kit.hx("071A26"), tint_k=0.72)
    kit.add_radial(f, kit.W * 0.30, kit.H * 0.55, 620, kit.hx("0F3A4E"), 0.35, 1.7)
    # 左侧：粒子身体（低透明度）+ 本体闪烁
    F = c.cl_field if False else None
    body = c.cl.sh(cloud.sprite_field, stage.MAID_L, 700.0, 36.0)
    P = c.cl.transform(body["P"], yaw=0.16, offset=(-40, 40, 0))
    lum = body["lum"]
    col = kit.mixc(kit.hx("3E7FA8"), kit.CREAM, lum)
    flick = 0.55 + 0.45 * math.sin(t * 37.0) * math.sin(t * 11.0)
    c.cl.draw(f, P, dist=1400, color_map=lambda *a: col, bright=(0.25 + 1.4 * lum) * (0.7 + 0.5 * flick),
              blur_r=1.7, size=1.5, glow_gain=0.55, cx=kit.W * 0.30, cy=kit.H * 0.52)
    # 右侧：参数表
    typed = kit.cl((t - 7.75) / 2.0)
    items = [("class", "Human.Love", True),
             ("name", "\"world.execute(me)\"", True),
             ("height", "1.62 m", True),
             ("mass", "4.7e1 kg", True),
             ("affection", "Infinity", True),
             ("permission", "root", False),
             ("status", "initializing…", False)]
    stage.param_list(f, t, items, x=kit.W * 0.56, y=kit.H * 0.24, alpha=1.0, size=25,
                     theme=c.post.get("th", stage.TH_DEEP), typed=typed)
    c.lyric(size=40, y=kit.H * 0.84)
    # 初始化提示
    a = kit.cl((t - 9.6) / 0.7)
    if a > 0:
        kit.draw_text(f, "> INITIALIZATION", kit.W * 0.56, kit.H * 0.80, 30, kit.MINT, kit.F_MONO,
                      a, 3.0, glow=0.3)


@scene(10.38, 11.66, "INIT", "A")
def init(c):
    t = c.t
    f = c.f
    c.post.update(grain=0.03, vignette=0.55, bloom=(0.6, 20, 0.5), scan=0.05)
    stage.bg_image(f, stage.BG_PALACE, zoom=1.2, dim=0.12, blur_r=14, tint=kit.hx("061622"), tint_k=0.7)
    p = kit.cl((t - 10.4) / 1.1)
    # 进度条
    bx, by = kit.W * 0.30, kit.H * 0.52
    bw, bh = kit.W * 0.40, 6
    m = kit.poly_mask([(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)])
    f += (m * 0.25)[..., None] * kit.MINT
    m2 = kit.poly_mask([(bx, by), (bx + bw * p, by), (bx + bw * p, by + bh), (bx, by + bh)])
    f += m2[..., None] * kit.MINT * 1.8
    kit.add_radial(f, bx + bw * p, by + bh / 2, 120, kit.MINT, 0.5, 2.0)
    kit.draw_text(f, "INITIALIZATION", bx, by - 64, 34, kit.CREAM, kit.F_MONO_B, 0.95, 8.0)
    kit.draw_text(f, "%3d%%" % int(p * 100), bx + bw, by - 58, 30, kit.MINT, kit.F_MONO, 0.9, 2.0, "right")
    # 滚动的十六进制流
    rs = c.rng(5)
    for i in range(9):
        y = by + 40 + i * 34
        s = " ".join("%04X" % int(v) for v in (rs.random(10) * 65535))
        kit.draw_text(f, s, bx, y, 18, kit.MINT, kit.F_MONO, 0.10 + 0.10 * (i / 9), 2.0)
    c.lyric(size=40, y=kit.H * 0.80)


@scene(11.66, 13.61, "WORLD", "A")
def world(c):
    t = c.t
    f = c.f
    c.post.update(grain=0.028, vignette=0.5, bloom=(0.55, 24, 0.55))
    p = kit.cl((t - 11.66) / 1.5)
    zoom = 1.55 - 0.35 * kit.ease_out(p)
    stage.bg_image(f, stage.BG_PALACE, zoom=zoom, dy=-20 * (1 - p), dim=0.35 + 0.5 * p,
                   tint=kit.hx("0A2231"), tint_k=0.45 * (1 - p))
    stage.caustics(f, t, strength=0.05 + 0.08 * p)
    stage.lightbeam(f, t, x=0.42, width=0.20, strength=0.10 + 0.12 * p, tilt=0.22)
    stage.dust(f, t, n=200, alpha=0.35 * p)
    stage.floor_reflection(f, t, y=0.80, strength=0.16 * p)
    # 远处小小的她
    hh = 260 + 120 * p
    stage.character(f, stage.MAID_L, kit.W * 0.50, kit.H * 0.985, hh, alpha=0.55 + 0.45 * p,
                    rim=kit.MINT, rim_k=0.35, glow=0.10 * p, shadow=0.5)
    stage.cursor(f, kit.W * 0.50 + 40, kit.H * 0.70, t, 26, kit.WHITE, 0.0 + 0.9 * kit.cl((t - 12.8) / 0.5))
    a = kit.cl((t - 12.9) / 0.7)
    if a > 0:
        stage.title_card(f, "world.execute(me);", "set up our new world", t, a * 0.95,
                         size=86, tracking=6, glow=0.5, y=kit.H * 0.30)
    c.lyric(ln=mus.LINES[8], size=40, y=kit.H * 0.84, alpha=kit.cl((t - 13.6) / 0.4))


@scene(13.61, 15.20, "SIM", "A")
def sim(c):
    t = c.t
    f = c.f
    c.post.update(grain=0.03, vignette=0.55, bloom=(0.55, 26, 0.6), chroma=0.6)
    p = kit.cl((t - 13.61) / 1.2)
    stage.bg_image(f, stage.BG_PALACE, zoom=1.8 - 0.3 * p, dim=0.6, blur_r=2 + 6 * (1 - p))
    stage.caustics(f, t, strength=0.12)
    stage.character(f, stage.MAID_L, kit.W * 0.5, kit.H * 1.02, 700 + 200 * p, alpha=0.85,
                    rim=kit.MINT, rim_k=0.5, glow=0.12, shadow=0.6)
    # 模拟网格从地平线升起
    grid_floor(f, t, y=0.92, horizon=0.52, alpha=0.34 * p, color=kit.hx("6EFFF0"))
    a = kit.cl((t - 14.9) / 0.25)
    if a > 0:
        f = f * (1 - a) + kit.WHITE * a
    c.lyric(size=44, y=kit.H * 0.84)


# ================================================================ 前奏
@scene(15.20, 19.00, "PRELUDE_1", "A")
def prelude_1(c):
    t = c.t
    f = c.f
    lt = t - 15.2
    c.post.update(grain=0.028, vignette=0.52, bloom=(0.55, 26, 0.55))
    stage.bg_image(f, stage.BG_PALACE, zoom=1.28 + 0.10 * lt / 3.8, dx=30 * math.sin(lt * 0.20),
                   dim=0.72, tint=kit.hx("0B2534"), tint_k=0.30)
    stage.lightbeam(f, t, x=0.36, width=0.18, strength=0.16, tilt=0.26)
    stage.lightbeam(f, t, x=0.66, width=0.12, strength=0.10, tilt=-0.18)
    stage.caustics(f, t, strength=0.11)
    # 远处的鲸（"你"的化身）
    wh = kit.sprite(stage.WHALE["serious"], 760)
    bob = math.sin(lt * 0.55) * 16
    kit.place(f, wh, kit.W * 0.74, kit.H * 0.30 + bob, 1.0, 0.30, tint=kit.hx("0C2E42"),
              tint_k=0.55, glow=0.25, rot=math.sin(lt * 0.4) * 0.03)
    stage.dust(f, t, n=240, alpha=0.4)
    stage.floor_reflection(f, t, y=0.82, strength=0.20)
    stage.character(f, stage.MAID_L, kit.W * 0.34, kit.H * 1.0, 760, alpha=0.96,
                    rim=kit.MINT, rim_k=0.55, glow=0.12, shadow=0.55)
    stage.cursor(f, kit.W * 0.60, kit.H * 0.72 + math.sin(lt * 0.8) * 6, t, 24, kit.WHITE, 0.55)


@scene(19.00, 23.00, "PRELUDE_2", "A")
def prelude_2(c):
    t = c.t
    f = c.f
    lt = t - 19.0
    c.post.update(grain=0.028, vignette=0.55, bloom=(0.5, 28, 0.6))
    stage.bg_image(f, stage.BG_PALACE, zoom=1.18, dy=-10, dim=0.60, tint=kit.hx("0A2030"), tint_k=0.35)
    stage.lightbeam(f, t, x=0.50, width=0.26, strength=0.20, tilt=0.10)
    stage.caustics(f, t, strength=0.09)
    # 双人对称站位，缓慢推进
    push = 1.0 + 0.05 * lt / 4.0
    stage.chars_duo(f, t, h=700 * push, gap=520, alpha=0.97, dy=6,
                    rim=kit.MINT, rim_k=0.45, glow=0.10)
    stage.floor_reflection(f, t, y=0.84, strength=0.18)
    stage.dust(f, t, n=180, alpha=0.30)
    # 拍点上的细碎光点
    c.cl.draw(f, c.cl.sh(cloud.shape_dots, (1500, 900, 520)), dist=1500,
              color=kit.MINT, blur_r=4.0, size=1.0, glow_gain=0.5,
              target=0.30, exposure=0.55 + 0.25 * c.pulse())
    c.post["vignette"] = 0.52


@scene(23.00, 27.00, "PRELUDE_3", "A")
def prelude_3(c):
    t = c.t
    f = c.f
    lt = t - 23.0
    c.post.update(grain=0.03, vignette=0.6, bloom=(0.52, 26, 0.62), chroma=0.4)
    # 近景：手与怀表（把背景放大到局部）
    stage.bg_image(f, stage.BG_PALACE, zoom=2.9 + 0.16 * lt / 4.0, dx=-260, dy=120, dim=0.55,
                   blur_r=5, tint=kit.hx("0B2634"), tint_k=0.4)
    stage.character(f, stage.MAID_R, kit.W * 0.58, kit.H * 1.10, 1250, alpha=0.95,
                    rim=kit.MINT, rim_k=0.30, glow=0.08, tint=kit.hx("C8DCE8"), tint_k=0.08)
    # 环绕她的粒子环：小、斜、贴着怀表
    ring = c.cl.sh(cloud.shape_ring, 150.0, 3.0, "xz")
    ring = c.cl.transform(ring, pitch=1.25, roll=lt * 0.30 + 0.3, offset=(-150, 150, 60))
    c.cl.draw(f, ring, dist=1500, color=kit.mixc(kit.MINT, kit.CREAM, 0.5), blur_r=1.4,
              size=1.3, glow_gain=0.9, bright=(0.5 + 0.5 * c.cl.rnd), target=0.8,
              cx=kit.W * 0.50, cy=kit.H * 0.52)
    # 表盘刻度：程序化小刻度环
    cx0, cy0 = kit.W * 0.42, kit.H * 0.62
    for i in range(12):
        a = i / 12 * 6.283 + lt * 0.20
        r0, r1 = 46, 58 if i % 3 == 0 else 52
        m = kit.line_mask([(cx0 + math.cos(a) * r0, cy0 + math.sin(a) * r0),
                           (cx0 + math.cos(a) * r1, cy0 + math.sin(a) * r1)], 1.4)
        f += m[..., None] * kit.MINT * 0.8
    stage.cursor(f, kit.W * 0.44, kit.H * 0.58, t, 30, kit.WHITE, 0.8)
    c.post["vignette"] = 0.62


@scene(27.00, 30.00, "COLLAPSE", "A")
def collapse(c):
    t = c.t
    f = c.f
    p = kit.cl((t - 27.0) / 3.0)
    c.post.update(grain=0.035, vignette=0.66, bloom=(0.45, 24, 0.7), chroma=1.0 * p)
    # 世界线框化：亮度提高、色彩抽离
    stage.bg_image(f, stage.BG_PALACE, zoom=1.5 + 0.5 * p, dim=0.45 * (1 - 0.6 * p), blur_r=1 + 6 * p,
                   tint=kit.hx("0E3C52"), tint_k=0.5)
    grid_floor(f, t, y=0.95, horizon=0.45, alpha=0.5 * p, color=kit.MINT)
    stage.chars_duo(f, t, h=860, gap=470, alpha=0.9 * (1 - 0.7 * p), dy=10,
                    rim=kit.MINT, rim_k=0.6, glow=0.12 * (1 - p))
    # 所有东西被吸成一个点
    body = c.cl.sh(cloud.sprite_field, stage.MAID_L, 700.0, 36.0)
    P = c.cl.transform(body["P"], offset=(0, 20, 0))
    q = kit.cl((t - 27.4) / 2.0)
    target = np.zeros_like(P) * 1.0
    M = c.cl.morph(P, target, q, stagger=0.7, ease="in")
    c.cl.draw(f, M, dist=1400, color_map=lambda *a: kit.mixc(kit.MINT, kit.CREAM, 0.6),
              bright=1.0, blur_r=1.4 + 2.6 * q, size=1.5, glow_gain=0.9,
              target=0.85, exposure=1.0 - 0.35 * q)
    # 吸向中心的拖尾
    if q > 0.02:
        for i in range(7):
            ang = i / 7 * 6.283 + t * 1.4
            ln = 620 * (1 - q)
            m = kit.line_mask([(kit.W / 2, kit.H * 0.52),
                               (kit.W / 2 + math.cos(ang) * ln, kit.H * 0.52 + math.sin(ang) * ln * 0.62)], 1.4)
            f += kit.blur(m, 1.5 * kit.S)[..., None] * kit.MINT * (0.35 * q)
    # 拍点上的横向撕裂
    if p > 0.55:
        k = (p - 0.55) / 0.45
        f = kit.glitch_rows(f, t, k * 0.7, seed=4, band=10 + int(30 * k))
    # 最后一击：白闪 → 黑
    if t > 29.55:
        a = kit.cl((t - 29.55) / 0.12)
        b = kit.cl((t - 29.75) / 0.24)
        f *= (1 - b)
        f += kit.WHITE * (a * (1 - b)) * 0.8
    c.post["grain"] = 0.045
