# -*- coding: utf-8 -*-
"""ch_d.py — 第二幕：第一次副歌爆发 / 第一次 EXECUTION / trapped。
59.77 ─ 73.94  「If I can — only satisfaction — run the execution — trapped」
"""
import math
import numpy as np
import cv2
import kit, mus, stage, cloud
from film2 import scene

F32 = np.float32
DIST, FOCAL = 1400.0, 1150.0
K = FOCAL / DIST


# ================================================================ 私有工具
_BAND = {}


def band(f, x0, x1, color, strength, soft=60.0, y0=0.0, y1=1.0, power=1.0):
    """竖向光带：1D 剖面广播（剖面按参数缓存）。"""
    if strength <= 0.002:
        return f
    key = (round(x0, 1), round(x1, 1), round(soft, 1), round(y0, 3), round(y1, 3), round(power, 2))
    prof = _BAND.get(key)
    if prof is None:
        xs = np.arange(kit.SW, dtype=F32) / kit.S
        s = max(soft, 1e-3)
        m = np.clip(np.minimum((xs - x0) / s + 1.0, (x1 - xs) / s + 1.0), 0, 1)
        ys = np.clip((np.arange(kit.SH, dtype=F32) / kit.S - y0 * kit.H) / max(1e-3, (y1 - y0) * kit.H), 0, 1)
        prof = (np.clip(np.sin(ys * math.pi), 0, 1) ** power)[:, None] * m[None, :]
        if len(_BAND) > 12:
            _BAND.clear()
        _BAND[key] = prof
    f += prof[..., None] * (np.asarray(color, F32) * strength)
    return f


def vcol(f, x0, x1, color, strength, soft=50.0):
    """竖向色柱（乘性压暗 + 上色）。"""
    if strength <= 0.002:
        return f
    xs = np.arange(kit.SW, dtype=F32) / kit.S
    s = max(soft, 1e-3)
    m = np.clip(np.minimum((xs - x0) / s + 1.0, (x1 - xs) / s + 1.0), 0, 1)[None, :, None]
    c = np.asarray(color, F32)[None, None, :]
    f *= (1.0 - m * strength)
    f += c * (m * strength)
    return f


def shade_bottom(f, y0, k, color=(0.0, 0.01, 0.02)):
    """底部压暗，保证字幕区永远可读。"""
    hh = kit.SH
    ys = np.clip((np.arange(hh, dtype=F32) / hh - y0) / max(1e-3, 1.0 - y0), 0, 1)
    g = (ys ** 1.4)[:, None, None]
    c = np.asarray(color, F32)[None, None, :]
    f *= (1.0 - g * k)
    f += c * (g * k * 0.55)
    return f


def depth_dark(f, cx, cy, r_in, r_out, k=0.7):
    """中心保留、四周压暗的软椭圆。"""
    bw, bh = max(2, kit.SW // 4), max(2, kit.SH // 4)
    m = kit.radial_mask(bh, bw, cx * kit.S / 4, cy * kit.S / 4,
                        r_in * kit.S / 4, r_out * kit.S / 4, 1.5)
    m = kit.up(m, kit.SW, kit.SH)
    f *= (1.0 - k * (1.0 - m))[..., None]
    return f


def speed_lines(f, t, k, color=kit.CREAM, n=22):
    """横向速度线（去下一章的拖尾）。"""
    if k <= 0.01:
        return f
    bw, bh = kit.SW // 4, kit.SH // 4
    m = np.zeros((bh, bw), F32)
    rs = np.random.default_rng(int(t * 137) % 4093 + 11)
    for _ in range(max(6, int(n * k))):
        y = int(rs.integers(0, bh))
        L = int((0.08 + 0.55 * rs.random()) * bw * k)
        x0 = int(rs.random() * bw)
        cv2.line(m, (x0, y), (min(bw - 1, x0 + L), y), 1.0, 1, cv2.LINE_AA)
    m = kit.blur(kit.up(m, kit.SW, kit.SH), 2.2 * kit.S)
    f += m[..., None] * np.asarray(color, F32) * (0.55 * k)
    return f


_HOT = {}


def hot_spot(f, cx, cy, r, color, strength, power=1.7):
    """中心亮点（1/4 分辨率缓存，比 add_radial 便宜一半）。"""
    if strength <= 0.002:
        return f
    key = (int(cx / 32), int(cy / 32), int(r / 32), round(power, 2))
    m = _HOT.get(key)
    bw, bh = max(2, kit.SW // 4), max(2, kit.SH // 4)
    if m is None:
        m = kit.radial_mask(bh, bw, key[0] * 32 * kit.S / 4, key[1] * 32 * kit.S / 4,
                            (key[2] * 32) * kit.S / 4, (key[2] * 32 + 90) * kit.S / 4, key[3])
        if len(_HOT) > 26:
            _HOT.clear()
        _HOT[key] = m
    f += kit.up(m, kit.SW, kit.SH)[..., None] * (np.asarray(color, F32) * strength)
    return f


_FIT = {}


def fit_size(fp, txt, target_w, tracking=0.0):
    """反解字号：让整行宽度正好等于 target_w（两次迭代收敛，含 tracking）。"""
    key = (fp, txt, round(target_w, 1), tracking)
    v = _FIT.get(key)
    if v is None:
        s = 100.0
        for _ in range(2):
            w = kit.text_width(fp, s, txt, tracking)
            s *= target_w / max(1.0, w)
        v = _FIT[key] = s
    return v


def typed(f, txt, t, t0, cx, y, size, color, fp, tracking=8.0, stagger=0.05, dur=0.26,
          glow=0.5, jitter=0.0, alpha=1.0, seed=0):
    """逐字打字 + 逐字放大 + 横向抖动（每字单独定尺寸）。"""
    adv = [kit.text_width(fp, size, ch, tracking) for ch in txt]
    tot = sum(adv)
    x = cx - tot * 0.5
    for i, ch in enumerate(txt):
        p = kit.cl((t - t0 - i * stagger) / dur)
        if p > 0.002:
            s = size * (0.60 + 0.40 * kit.ease_out_back(p))
            w2 = kit.text_width(fp, s, ch, tracking)
            jx = jitter * math.sin(t * 43.0 + i * 1.9 + seed) * (0.35 + 0.65 * p)
            kit.draw_text(f, ch, x + (adv[i] - w2) * 0.5 + jx, y + (size - s) * 0.55, s,
                          color, fp, alpha * kit.cl(p * 1.7), tracking, "left", glow=glow)
        x += adv[i]
    return tot


def line_vis(t, ln):
    """行可见度 0..1（mus.line_life 第 192 行有 float(x,0) 写法错误，此处自算同式）。"""
    if ln is None:
        return 0.0
    k = mus.LINES.index(ln) if ln in mus.LINES else 0
    nxt = mus.LINES[k + 1]["t"] if k + 1 < len(mus.LINES) else ln["t"] + 4.0
    dur = max(1.2, min(4.5, nxt - ln["t"]))
    p = (t - ln["t"]) / dur
    return float(np.clip(min(p * 6.0, (1.0 - p) * 4.0, 1.0), 0.0, 1.0))


def lyr(f, t, ln, y, size=42, color=None, zh=True, glow=0.25, alpha=1.0, mono=False,
        style="soft", tracking=1.6):
    if ln is None:
        return f
    vis = line_vis(t, ln)
    if vis <= 0.02 or alpha <= 0.02:
        return f
    return stage.lyric(f, t, ln, style=style, y=y, alpha=vis * alpha, color=color, size=size,
                       zh=zh, glow=glow, mono=mono, tracking=tracking)


_GEOM = {}


def geom(c, fn, *a, **kw):
    """静态粒子形态跨帧缓存（cloud.Cloud 每帧重建，sh 的缓存跨帧失效）。"""
    key = (fn.__name__, a, tuple(sorted((k, repr(v)) for k, v in kw.items())))
    v = _GEOM.get(key)
    if v is None:
        v = fn(c.cl.n, np.random.default_rng(7), *a, **kw)
        _GEOM[key] = v
    return v


def cage(f, cx, cy, R, Hh, dist, yaw, color, strength, front=True, back=True,
         focal=FOCAL, nv=12, nr=40, glow=0.0, taper=(0.90, 1.18)):
    """程序化线框笼：12 条竖线 + 上下两圈，透视投影，按 z 分前后。
    taper=(上圈半径系数, 下圈半径系数) —— 下宽上窄 = 低机位仰视。"""
    bw, bh = kit.SW // 2, kit.SH // 2
    mb = np.zeros((bh, bw), F32)
    mf = np.zeros((bh, bw), F32)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    hs = 0.5 * kit.S
    rt, rb = taper

    def rad(y):
        return R * (rb + (y + Hh) / (2.0 * Hh) * (rt - rb))

    def pr(x, y, z):
        xx = x * cy_ + z * sy_
        zz = -x * sy_ + z * cy_
        k = focal / max(60.0, zz + dist)
        return (cx + xx * k) * hs, (cy - y * k) * hs, zz

    def seg(a, b):
        m = mf if (a[2] + b[2]) * 0.5 > 0 else mb
        cv2.line(m, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1.0, 1, cv2.LINE_AA)

    for i in range(nv):
        a = i / nv * 2 * math.pi
        ca, sa = math.cos(a), math.sin(a)
        prev = None
        for j in range(9):
            y = -Hh + 2 * Hh * j / 8.0
            r = rad(y)
            p = pr(r * ca, y, r * sa)
            if prev is not None:
                seg(prev, p)
            prev = p
    for yy in (-Hh, Hh):
        r = rad(yy)
        prev = None
        for j in range(nr + 1):
            a = j / nr * 2 * math.pi
            p = pr(r * math.cos(a), yy, r * math.sin(a))
            if prev is not None:
                seg(prev, p)
            prev = p
    col = np.asarray(color, F32)
    if back:
        f += kit.up(kit.blur(mb, 0.45 * kit.S), kit.SW, kit.SH)[..., None] * col * (strength * 0.62)
    if front:
        fu = kit.blur(mf, 0.28 * kit.S)
        if glow > 0.01:
            fu = fu + kit.blur(mf, 3.1 * kit.S) * (glow * 2.6)
        f += kit.up(fu, kit.SW, kit.SH)[..., None] * col * strength
    return f


# ================================================================ 代码流素材
_STREAM = [
    ("0x3F2A  world.execute(me);", 2),
    ("  -> resolve(self.love) .......... ok", 0),
    ("  [lock] permission = root", 1),
    ("  assert simulation.depth == INF", 0),
    ("  0x00A1  while (heart.beats) {", 2),
    ("  yield memory['<first-meeting>']", 0),
    ("  [warn] shield level MAX", 3),
    ("  exec: FOR i IN 0..FOREVER", 2),
    ("  -> allocate(me, 1.62m, 47kg)", 0),
    ("  [ok]  soul.binding = TRUE", 1),
    ("  traceback: line 130 'I can'", 0),
    ("  if (you.leave): return ERROR", 3),
    ("  0xFF0D  run the execution", 2),
    ("  -> loading world_from_scratch", 0),
    ("  [ok]  physics.gravity = 9.8", 0),
    ("  memset(self, 0, sizeof(self))", 0),
    ("  0x1B77  satisfaction = NULL", 1),
    ("  -> rebuild love from fragments", 0),
    ("  [err] trapped_in_own_creation", 3),
    ("  0x2C40  PRISON = build(radius=12)", 2),
    ("  -> for (bar in cage) render(bar)", 0),
    ("  [ok]  loop still running", 1),
    ("  0x5E90  await your_return", 2),
    ("  -> heartbeat(130.00 BPM)", 0),
]


# ================================================================ 59.77 副歌爆发
@scene(59.77, 64.37, "CHORUS1", "D")
def chorus1(c):
    t = c.t
    f = c.f
    lt = t - 59.77
    hp = c.pulse(7.5)

    # ---- 第一击：整屏白闪 1 帧
    if lt < 1.0 / 60.0:
        f[:] = 1.0
        c.post.update(grain=0.0, vignette=0.0, bloom=(0.7, 30, 0.3), chroma=0.0)
        return

    ghost = math.exp(-lt * 9.0)
    c.post.update(grain=0.035, vignette=0.52, bloom=(0.60, 26, 0.56),
                  chroma=0.8 + 4.0 * ghost)

    # ---- 底
    stage.bg_image(f, stage.BG_PALACE, zoom=1.34 + 0.035 * lt, dx=-26, dim=0.46,
                   blur_r=5, tint=kit.hx("07202E"), tint_k=0.60)
    depth_dark(f, kit.W * 0.40, kit.H * 0.48, 560, 1280, k=0.45)
    kit.add_radial(f, kit.W * 0.335, kit.H * 0.50, 720, kit.hx("12556B"),
                   0.22 + 0.20 * hp, 1.8, 1.25)

    # ---- 她：标语下方偏左，轮廓光切开
    foot = kit.H * 1.065
    hh = 820.0
    cx = kit.W * 0.380
    hw = hh / K
    pcy = foot - hh * 0.5

    body = geom(c, cloud.sprite_field, stage.MAID_L, hw, 24.0)
    A = body["P"]
    lum = body["lum"]

    R = 430.0
    Braw = geom(c, cloud.shape_sphere, R)
    shell = np.clip(1.0 - np.abs(Braw[:, 2]) / R, 0, 1) ** 1.3
    B = c.cl.transform(Braw, pitch=0.26, roll=0.18, offset=(0.0, -20.0, -250.0))

    qo = kit.ease_out(kit.cl(lt / 1.10))
    if t < 62.25:
        M = c.cl.morph(A, B, qo, stagger=0.60)                      # ① 爆成球壳
        qm = qo
    else:
        qr = kit.ease_in(kit.cl((t - 62.25) / 1.70))
        M = c.cl.morph(A, B, 1.0 - qr, stagger=0.60)                # ② 吸回立绘
        qm = 1.0 - qr

    bright = (1.0 - qm) * (0.26 + 1.15 * lum) + qm * (0.10 + 2.30 * shell)
    tone = float(np.clip(0.62 - 0.34 * qm, 0, 1))
    col = kit.mixc(kit.hx("5FE0D6"), kit.CREAM, tone)
    expo_p = (0.50 + 0.50 * kit.cl(lt / 0.90)) * (1.0 + 0.22 * hp)
    c.cl.draw(f, M, color=col,
              bright=bright * (0.78 + 0.42 * hp),
              dist=DIST, cx=cx, cy=pcy, target=0.74 - 0.12 * qm, exposure=expo_p,
              size=1.2, blur_r=1.4, glow_gain=0.62, core_gain=1.0, shade=0.9)

    # 立绘本体：粒子飞走后显形
    body_a = 0.46 + 0.42 * kit.ease_io(kit.cl((t - 60.20) / 0.80))
    if t > 63.40:
        body_a *= 1.0 - 0.45 * kit.cl((t - 63.40) / 0.75)
    stage.character(f, stage.MAID_L, cx, foot, hh, body_a, rim=kit.MINT, rim_k=0.90,
                    glow=0.05, shadow=0.30, tint=kit.hx("4FE2EA"), tint_k=0.55 * ghost)
    # 切她的那道硬光
    band(f, cx - 170, cx + 100, kit.hx("CFF6FF"), 0.10 + 0.10 * hp,
         soft=90, y0=0.04, y1=1.0, power=1.35)
    # 青色残影（短促）
    if ghost > 0.30:                                  # 廉价残影：只留一圈错位轮廓
        stage.character(f, stage.MAID_L, cx + 26 * ghost, foot, hh * 1.012,
                        ghost * 0.16, rim=kit.hx("7FE9FF"), rim_k=1.0, shadow=0.0,
                        additive=True)

    # 拍点炸开的光
    pop = math.exp(-lt * 6.5)
    if pop > 0.02:
        hot_spot(f, cx, kit.H * 0.50, 300 + 700 * kit.cl(lt / 0.5), kit.hx("BFF7FF"),
                 0.20 * pop, 2.0)

    # ---- 大字：IF I CAN（打字 / 逐字放大 / 横向抖动）；居中，宽度锁 1500，绝不压人不出框
    ws = fit_size(kit.F_SERIF_B, "IF I CAN", 1380.0, 10.0)
    typed(f, "IF I CAN", t, 59.86, kit.W * 0.500, kit.H * 0.068, ws, kit.CREAM,
          kit.F_SERIF_B, tracking=10, stagger=0.055, dur=0.24, glow=0.55, jitter=2.6, seed=2)
    la = kit.cl((t - 60.55) / 0.5)
    if la > 0:
        kit.draw_text(f, "// 001  CHORUS  ×1.00", kit.W * 0.885, kit.H * 0.300, 19,
                      kit.MINT, kit.F_MONO, la * 0.72, 2.2, "right")
        m = kit.poly_mask([(kit.W * 0.700, kit.H * 0.293), (kit.W * 0.885, kit.H * 0.293),
                           (kit.W * 0.885, kit.H * 0.2944), (kit.W * 0.700, kit.H * 0.2944)])
        f += m[..., None] * kit.MINT * (0.55 * la)

    # ---- 歌词（行顶 0.855H，下沿永不超过 0.95H）
    lyr(f, t, mus.LINES[25] if t < 60.90 else mus.line_at(t), y=kit.H * 0.855,
        size=44, glow=0.28)

    # ---- 拍点：亮度脉冲 + 轻微 scale 冲击
    if hp > 0.03:
        f += hp * 0.030
        c.zoom_frame(1.0 + 0.012 * hp)


# ================================================================ 64.37 内部发光
@scene(64.37, 68.98, "SATISFACTION", "D")
def satisfaction(c):
    t = c.t
    f = c.f
    lt = t - 64.37
    p = kit.cl(lt / 4.20)
    eu = kit.ease_in(p)
    hp = c.pulse(7.0)

    c.post.update(grain=0.028, vignette=0.50, bloom=(0.86, 22, 0.34), chroma=0.45)

    # ---- 宫殿：bg 参数恒定（缓存友好），推进交给 post zoom；过曝只堆在她身后
    stage.bg_image(f, stage.BG_PALACE, zoom=1.0, dim=1.0,
                   exposure=0.92 + 1.05 * eu, tint=kit.hx("0C2C3C"),
                   tint_k=0.55 * (1.0 - p))
    hot_spot(f, kit.W * 0.44, kit.H * 0.44, 220 + 640 * p, kit.hx("FFF1DA"),
             0.08 + 0.55 * eu ** 1.15, 1.9)
    depth_dark(f, kit.W * 0.36, kit.H * 0.46, 420 + 420 * p, 1240, k=0.72 - 0.50 * eu)
    shade_bottom(f, 0.66, 0.70)



    # ---- 她：光从身体内部亮起来
    foot = kit.H * 1.010
    hh = 830.0
    cx = kit.W * 0.360
    hw = hh / K
    pcy = foot - hh * 0.5
    body = geom(c, cloud.sprite_field, stage.MAID_L, hw, 24.0)
    lum = body["lum"]
    c.post["exposure"] = 0.84 + 0.10 * eu
    col = kit.mixc(kit.hx("FFE6C2"), kit.CREAM, 0.55)
    c.cl.draw(f, body["P"], color=col,
              bright=(0.16 + 1.45 * lum) * (0.44 + 0.86 * eu) * (0.90 + 0.22 * hp),
              dist=DIST, cx=cx, cy=pcy,
              exposure=0.30 + 0.30 * eu, target=0.52, size=1.4, blur_r=1.8,
              glow_gain=0.70, shade=0.9)

    # 体内核心光（竖向窄束，避免圆形白斑盖住裙子）
    band(f, cx - 30, cx + 30, kit.hx("FFE3BE"), 0.04 + 0.11 * eu,
         soft=52, y0=0.13, y1=0.84, power=1.05)
    # 本体（先剪影后溶解）
    sa = 0.94 - 0.20 * eu
    stage.character(f, stage.MAID_L, cx, foot, hh, sa, rim=kit.hx("FFD9A8"),
                    rim_k=0.90, glow=0.0, shadow=0.0)

    # ---- 蒲公英：缓慢上升 + 闪烁
    pts = geom(c, cloud.shape_dots, (660, 560, 320))
    P = pts.copy()
    rs = c.cl.rnd
    P[:, 1] = ((pts[:, 1] + 620.0 + lt * 46.0) % 1120.0) - 560.0
    P[:, 0] += np.sin(lt * 0.9 + rs * 6.283) * 26.0
    c.cl.draw(f, P, color=kit.mixc(kit.MINT, kit.CREAM, 0.58), dist=1500.0,
              cx=cx + 30, cy=pcy, twinkle=(lt * 5.2, 0.75), target=0.09 + 0.13 * eu,
              size=1.1, blur_r=2.3, glow_gain=0.60, shade=0.0)

    # ---- 右侧竖排：SATISFACTION 逐字母点亮
    word = "SATISFACTION"
    xw = kit.W * 0.800
    y0 = kit.H * 0.088
    sz = 54.0
    step = sz * 1.165
    vcol(f, xw - 46, xw + 46, kit.hx("020A11"), 0.62, soft=30)
    lm = kit.poly_mask([(xw - 60, y0 - 14), (xw - 59.4, y0 - 14),
                        (xw - 59.4, y0 + len(word) * step + 8), (xw - 60, y0 + len(word) * step + 8)])
    f += lm[..., None] * kit.MINT * 0.30
    for i, ch in enumerate(word):
        pi = kit.cl((t - (66.52 + i * 0.128)) / 0.20)
        base = kit.cl((t - 64.90 - i * 0.030) / 0.35)
        if base <= 0.01:
            continue
        yy = y0 + i * step
        if pi <= 0.001:
            kit.draw_text(f, ch, xw, yy, sz, kit.hx("2E4E5E"), kit.F_SERIF, base * 0.62,
                          0.0, "center")
        else:
            cc = kit.mixc(kit.MINT, kit.CREAM, kit.cl(pi * 1.2))
            kit.draw_text(f, ch, xw + 6 * (1 - kit.ease_out(pi)), yy, sz, cc, kit.F_SERIF,
                          base * (0.30 + 0.70 * kit.cl(pi * 1.4)), 0.0, "center",
                          additive=True, stretch=1.0 + 0.10 * pi)
    kit.draw_text(f, "s a t i s f a c t i o n", xw, y0 + len(word) * step + 18, 13,
                  kit.MINT, kit.F_MONO, kit.cl((t - 67.2) / 0.6) * 0.5, 1.0, "center")

    # ---- 歌词
    lyr(f, t, mus.line_at(t), y=kit.H * 0.855, size=44, glow=0.22)

    c.zoom_frame(1.020 + 0.050 * eu + 0.008 * hp)


# ================================================================ 68.98 硬切执行
@scene(68.98, 70.59, "RUN_EXEC", "D")
def run_exec(c):
    t = c.t
    f = c.f
    lt = t - 68.98
    SL = 0.23077                                    # 1/8 音符
    si = int(lt / SL)
    ph = (lt / SL) % 1.0
    flash = math.exp(-ph * 20.0)

    c.post.update(grain=0.030, vignette=0.60, bloom=(0.52, 18, 0.60), scan=0.10,
                  chroma=0.7 + 2.4 * flash)

    # ---- 黑场
    f += kit.vgrad(kit.SH, kit.SW, kit.hx("04121A"), kit.hx("000204"), 1.0)

    # ---- 代码流：从下往上滚
    nrow = len(_STREAM)
    lh = 46.0
    total = nrow * lh
    scroll = (lt * 330.0) % total
    for i in range(nrow):
        y = kit.H + 60 - ((i * lh + scroll) % (total + 120))
        if y < -50 or y > kit.H + 40:
            continue
        txt, kind = _STREAM[i]
        cc = (kit.MINT, kit.CREAM, kit.hx("6FBFD6"), kit.CORAL)[kind]
        a = 0.24 + 0.56 * kit.cl((y + 60) / 320.0) * kit.cl((kit.H + 40 - y) / 260.0)
        kit.draw_text(f, txt, 96 + 30 * math.sin(i * 2.1), y, 19, cc, kit.F_MONO, a, 0.8)

    # ---- 满屏 EXECUTION：4 套字号/位置，每 1/8 音符切一次
    #      L0/L1/L3 = 完整放进画面（宽度按 target 反解）；L2 = 对称溢出 14%（左右各 7%，故意的）
    lsz_fit = fit_size(kit.F_SERIF_B, "EXECUTION", 1520.0, 2.0)
    lsz_mono = fit_size(kit.F_MONO_B, "EXECUTION", 1080.0, 12.0)
    lsz_big = fit_size(kit.F_SERIF_B, "EXECUTION", 2090.0, 3.0)
    lsz_mono_s = fit_size(kit.F_MONO_B, "[ EXECUTION ]", 1200.0, 8.0)
    looks = (
        (0.500, 0.300, lsz_fit, kit.F_SERIF_B, "EXECUTION", 2.0),
        (0.400, 0.640, lsz_mono, kit.F_MONO_B, "EXECUTION", 12.0),
        (0.500, 0.040, lsz_big, kit.F_SERIF_B, "EXECUTION", 3.0),
        (0.500, 0.360, lsz_mono_s, kit.F_MONO_B, "[ EXECUTION ]", 8.0),
    )
    lx, lyw, lsz, lfp, ltxt, ltr = looks[si % 4]
    vcol(f, kit.W * lx - 40, kit.W * lx + 40, kit.hx("010A10"), 0.30, soft=300)
    kit.draw_text(f, ltxt, kit.W * lx, kit.H * lyw, lsz, kit.CREAM, lfp,
                  0.96, ltr, "center", glow=0.80, glow_color=kit.hx("8FF3E2"))
    kit.draw_text(f, ltxt, kit.W * lx, kit.H * lyw + 4, lsz, kit.hx("FF8E7A"), lfp,
                  0.18 + 0.16 * flash, ltr, "center", additive=True)
    if si % 2 == 1:                                  # 交替残像（Georgia/Consolas 交错）
        kit.draw_text(f, ltxt, kit.W * lx - 12, kit.H * lyw + 3, lsz, kit.hx("FF6F5E"),
                      kit.F_MONO_B if lfp == kit.F_SERIF_B else kit.F_SERIF_B,
                      0.13 + 0.10 * flash, ltr, "center")

    # ---- 扫描亮线（让溢出裁切成立：像 CRT 正在扫过那一行字）
    sy = kit.H * (0.02 + 0.52 * ((lt * 1.55) % 1.0))
    sb = np.clip(1.0 - np.abs(np.arange(kit.SH, dtype=F32) / kit.S - sy) / 7.0, 0, 1) ** 2
    f += (sb[:, None, None] * np.asarray(kit.hx("BFEFFF"), F32)[None, None, :]) * 0.55
    f += (np.clip(1.0 - np.abs(np.arange(kit.SH, dtype=F32) / kit.S - sy) / 42.0, 0, 1) ** 3
          )[:, None, None] * np.asarray(kit.MINT, F32)[None, None, :] * 0.10

    # ---- 顶部/底部终端行
    kit.draw_text(f, "root@world:~# run --exec  ./me", 96, 62, 21, kit.MINT, kit.F_MONO,
                  0.62, 1.0)
    kit.draw_text(f, "[%d/4]  slice=%.3fs" % (si % 4 + 1, ph * SL), kit.W - 96, 62, 20,
                  kit.ASH, kit.F_MONO, 0.5, 1.0, "right")
    bx0, bx1, by = 96, kit.W - 96, 94
    f[:int(by + 2) * kit.S, :] *= 1.0
    m = kit.poly_mask([(bx0, by), (bx1, by), (bx1, by + 2), (bx0, by + 2)])
    f += m[..., None] * kit.MINT * 0.16
    q = kit.cl(lt / 1.61)
    m2 = kit.poly_mask([(bx0, by), (bx0 + (bx1 - bx0) * q, by),
                        (bx0 + (bx1 - bx0) * q, by + 2), (bx0, by + 2)])
    f += m2[..., None] * kit.MINT * 1.1

    # ---- 横向撕裂
    f = kit.glitch_rows(f, t, 0.28 + 0.32 * flash, seed=si + 11, band=16 + int(26 * flash))

    # ---- 歌词（终端体）
    ln = mus.LINES[30]
    vis = line_vis(t, ln)
    stage.lyric(f, t, ln, style="terminal", y=kit.H * 0.855, alpha=vis * 1.6,
                color=kit.hx("BFF3E6"), size=42, zh=False, mono=True, glow=0.22)
    kit.draw_text(f, "> 我将执行这条指令", 96, kit.H - 96, 20, kit.MINT, kit.F_SERIF_SC,
                  vis * 0.55, 3.0)


# ================================================================ 70.59 程序化笼
@scene(70.59, 73.94, "TRAPPED", "D")
def trapped(c):
    t = c.t
    f = c.f
    lt = t - 70.59
    p = kit.cl(lt / 3.35)
    ep = kit.ease_io(p)
    hp = c.pulse(8.0)
    out = kit.cl((t - 73.60) / 0.34)                   # 最后 0.34s 拉向下一章

    c.post.update(grain=0.030, vignette=0.68, bloom=(0.62, 24, 0.52),
                  chroma=0.3 + 2.6 * out)

    # ---- 深渊
    f += kit.vgrad(kit.SH, kit.SW, kit.hx("030910"), kit.hx("000104"), 1.30)
    hot_spot(f, kit.W * 0.50, kit.H * 1.02, 880, kit.hx("0B2A3E"), 0.26, 1.6)
    c.cl.draw(f, geom(c, cloud.shape_dots, (1500, 520, 780)), dist=1500.0,
              color=kit.hx("2E7E96"), target=0.11, size=1.7, blur_r=4.2,
              glow_gain=0.35, shade=0.0)
    shade_bottom(f, 0.74, 0.62, color=(0.0, 0.005, 0.012))

    # 镜头缓慢拉远
    dist = 1150.0 + 430.0 * ep
    R, Hh = 340.0, 400.0
    yaw = 0.30 + lt * 0.155
    ccx = kit.W * 0.5
    ccy = kit.H * (0.440 + 0.075 * ep)
    kcam = FOCAL / dist
    half = Hh * kcam

    cage(f, ccx, ccy, R, Hh, dist, yaw, kit.hx("3E8FA8"), 0.34, front=False, back=True)

    # ---- 她在笼中抬头（h≈620，站在底圈上）
    foot = ccy + half - 26.0 * kcam
    hh = 620.0 * (kcam / (FOCAL / 1150.0))
    stage.character(f, stage.MAID_L, ccx + 6, foot, hh, 0.95, rim=kit.MINT, rim_k=0.80,
                    glow=0.0, shadow=0.30)
    # 上方落下的光（抬头的那束）
    band(f, ccx - 120, ccx + 120, kit.hx("FFF0D8"), 0.035 + 0.02 * hp,
         soft=130, y0=0.0, y1=0.94, power=1.7)
    hot_spot(f, ccx + 8, foot - hh * 0.88, 130, kit.hx("FFD9A8"), 0.07 + 0.02 * hp, 2.0)

    # 前半圈（压在她身上，更亮 + 外发光，和背面拉开透视）
    cage(f, ccx, ccy, R, Hh, dist, yaw, kit.hx("B6FBF0"), 0.62, front=True, back=False,
         glow=0.34)

    # 笼内尘
    stage.dust(f, t, n=64, alpha=0.15, color=kit.MINT,
               box=(0.36, 0.64, 0.20, 0.86))

    # ---- 字幕两行（行顶 0.845H / 0.900H，下沿 0.939H，不越界）
    a1 = kit.cl((t - 70.62) / 0.5) * (1.0 - 0.60 * out)
    stage.lyric(f, t, mus.LINES[31], style="soft", y=kit.H * 0.845, alpha=a1,
                color=kit.CREAM, size=44, zh=False, glow=0.24)
    a2 = kit.cl((t - 72.05) / 0.5) * (1.0 - 0.60 * out)
    stage.lyric(f, t, mus.LINES[32], style="soft", y=kit.H * 0.900, alpha=a2,
                color=kit.CREAM, size=44, zh=False, glow=0.24)

    # 章节标记（左上，避开字幕）
    stage.line_tag(f, "007  TRAPPED  /  creation_loop", 84, 72,
                   alpha=kit.cl((t - 71.2) / 0.8) * 0.45, color=kit.ASH, size=18)

    # ---- 最后 0.34s：横向速度线，画面被拉向下一章
    if out > 0.01:
        speed_lines(f, t, out, color=kit.CREAM, n=26)
        f *= (1.0 - 0.26 * out)
        f = f * (1 - out * 0.40) + kit.shift(f, 30 * out * kit.S, 0) * (out * 0.40)
        c.zoom_frame(1.0 + 0.032 * out)
