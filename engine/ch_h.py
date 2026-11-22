# -*- coding: utf-8 -*-
"""ch_h.py — 第三幕技术高潮：非法参数 → 空线程 → 14 次 EXECUTION → 多语言计数。
129.46 ─ 162.33

节奏全部从 mus 读：
  T_ILLEGAL = mus.find("You have made some illegal arguments")   129.460
  T_EXEC    = mus.find("Execution")                              148.358
  T_NUMS    = 最后一个 Execution 行                                158.311
  14 段硬切 = 12 个 Execution 起音（去掉落在边界上的那个）+ 把最宽的 3 段
             各自按最近的 mus.BEATS 拍点再劈一刀 = 14 段。
冷、硬、精确：黑 + 白 + 珊瑚红，青色只做「空线程」的心跳点缀。
"""
import math
import cv2
import numpy as np
import kit, mus, stage, cloud
from film2 import scene

F32 = np.float32
CORAL = kit.CORAL
CREAM = kit.CREAM
WHITE = kit.WHITE
BLACK = kit.BLACK
ASH = kit.ASH

# ---------------------------------------------------------------- 时间（只从 mus 取）
T_ILLEGAL = mus.find("You have made some illegal arguments")
T_EMPTY = 140.00
T_EXEC = mus.find("Execution")
_HIT = [ln["t"] for ln in mus.lines_between(T_EXEC, 158.6)]
T_NUMS = _HIT[-1]                                   # 158.311
T_END = 162.33

_ILL = mus.LINES[[i for i, l in enumerate(mus.LINES) if l["en"].startswith("You have made")][0]]

# 6 个数字的落点 = 两句多语言歌词的 6 个词
_WORD_T = []
for _ln in mus.lines_between(T_NUMS, T_END):
    if _ln["en"].startswith(("Ein", "Fem")):
        _WORD_T += [w[1] for w in _ln["words"]]
_WORD_T = _WORD_T[:6]


def _build_cuts():
    """14 段：11 个真实起音段 + 3 刀按拍点细分（取最宽的段）。"""
    bs = [t for t in _HIT if t < T_NUMS - 0.03]
    bs.append(T_NUMS)
    while len(bs) < 15:
        i = max(range(len(bs) - 1), key=lambda j: bs[j + 1] - bs[j])
        mid = 0.5 * (bs[i] + bs[i + 1])
        k = int(np.argmin(np.abs(mus.BEATS - mid)))
        cand = float(mus.BEATS[k])
        if not (bs[i] + 0.16 < cand < bs[i + 1] - 0.16):
            cand = mid
        bs.insert(i + 1, cand)
    return [(bs[i], bs[i + 1]) for i in range(len(bs) - 1)]


CUTS = _build_cuts()


# ---------------------------------------------------------------- 小工具
def _half():
    return np.zeros((kit.SH // 2, kit.SW // 2), F32)


def _ln(m, x0, y0, x1, y1, th=2.0):
    k = 0.5 * kit.S
    cv2.line(m, (int(x0 * k), int(y0 * k)), (int(x1 * k), int(y1 * k)),
             1.0, max(1, int(th * k)), cv2.LINE_AA)


def _flush(f, m, color, alpha=1.0, blur_r=1.2):
    if blur_r > 0:
        m = kit.blur(m, blur_r * kit.S * 0.5)
    m = kit.up(m, kit.SW, kit.SH)
    f += m[..., None] * np.asarray(color, F32) * alpha


def _ring_d():
    """半分辨率的到画面中心的距离场，给冲击波环用（比两次 circle_mask 便宜一半）。"""
    if "d" not in _RING:
        hh, ww = max(1, kit.SH // 2), max(1, kit.SW // 2)
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        _RING["d"] = np.hypot(xx * 2.0 + 1.0 - kit.W * 0.5, yy * 2.0 + 1.0 - kit.H * 0.5)
    return _RING["d"]


_RING = {}


def _mw(txt, fp, size, tracking=0.0):
    """字形掩码的真实宽度（逻辑像素）。kit 的 text_width 不含斜体侧边距。"""
    return max(1.0, kit.text_mask(fp, size, txt, tracking).shape[1] / kit.S)


def _fit(txt, fp, size, target):
    return target / _mw(txt, fp, size)


def _draw_st(f, txt, x, y, size, color, fp, alpha=1.0, tracking=0.0,
             stretch=1.0, align="center"):
    """draw_text 的 stretch 按未拉伸宽度居中且掩码更宽，这里补回偏移，保证拉伸后仍居中。"""
    if align == "center" and stretch != 1.0:
        wl = kit.text_width(fp, size, txt, tracking)
        x = x + (wl - _mw(txt, fp, size, tracking) * stretch) * 0.5
    return kit.draw_text(f, txt, x, y, size, color, fp, alpha, tracking, align, stretch=stretch)


def _squash(spr, w, h):
    h = max(2, int(h))
    return cv2.resize(spr, (max(1, int(w)), h), interpolation=cv2.INTER_AREA)


_CROP = {}


def _crop(path, box, w, h):
    """按归一化框裁一块并放大（缓存，避免每帧重建）。"""
    key = (path, box, w, h, kit.S)
    if key in _CROP:
        return _CROP[key]
    a = kit.trim(kit.load(path))
    H0, W0 = a.shape[:2]
    x0, y0, x1, y1 = box
    sub = a[int(y0 * H0):int(y1 * H0), int(x0 * W0):int(x1 * W0)]
    sub = kit.up(sub, int(w * kit.S), int(h * kit.S))
    _CROP[key] = sub
    return sub


_TH_CODE = stage.Theme("02060A", WHITE, kit.hx("7CFFB0"), CORAL)
_CODE_COL = {"kw": kit.hx("7CFFB0"), "str": kit.hx("E8D3B0"),
             "cm": kit.mixc(CORAL, BLACK, 0.28), "num": kit.hx("7FE3D4"),
             "pl": WHITE, "tx": WHITE}

# ================================================================ 1. ILLEGAL
_TRACE = [
    'Traceback (most recent call last):',
    '  File "world.py", line 4096, in <module>',
    '    love.execute(me)',
    '  File "core/human.py", line 88, in execute',
    '    if not self.is_valid(): raise',
    'ArgumentError: illegal arguments',
    '  [0x01] expected  Human.Love',
    '  [0x02] received  str "world.execute(me)"',
    '  [0x03] arity     1 given / 0 expected',
    '  [0x04] affinity  Infinity  (clamped)',
    '      at kernel/love.c:1204',
    '        0x7ffe  call  exec_love',
    '        0x7ff2  mov   eax, 0x0',
    '        0x7ff8  jmp   retry_forever',
    'retry 1 / inf ....................................',
    'STACK DEPTH 4096  HEAP 0x0  THREAD 0x00',
]


def _garble(s, c, salt):
    rs = c.rng(101 + salt)
    if rs.random() > 0.10 or len(s) < 4:
        return s
    t = list(s)
    for _ in range(int(rs.integers(1, 4))):
        i = int(rs.integers(0, len(t)))
        t[i] = "█▓▒░#@%&"[int(rs.integers(0, 8))]
    return "".join(t)


@scene(T_ILLEGAL, T_EMPTY, "ILLEGAL", "H")
def illegal(c):
    t = c.t
    f = c.f
    lt = t - T_ILLEGAL
    c.post.update(grain=0.052, vignette=0.72, bloom=(0.74, 18, 0.38), chroma=0.9, scan=0.10)

    f += kit.vgrad(kit.SH, kit.SW, kit.hx("0C0407"), kit.hx("020409"), 1.35)

    # ---- 满屏 illegal arguments（珊瑚单色，反向漂移）
    drift = lt * 30.0
    for i in range(9):
        sz = 74 + (i % 3) * 26
        txt = "illegal arguments   " * 4
        wrow = kit.text_width(kit.F_MONO_B, sz, txt)
        xx = -((drift * (1 + 0.35 * (i % 3)) + i * 370) % max(1.0, wrow - kit.W * 0.6))
        yy = -34 + i * (kit.H + 68) / 9.0
        a = 0.15 + 0.11 * ((i * 7) % 5)
        if i == 4:
            a = 0.78
        coin = 0.30 if (i % 2) else 0.0
        kit.draw_text(f, txt, xx, yy, sz, kit.mixc(CORAL, kit.BLACK, coin), kit.F_MONO_B, a, 1.0)

    # ---- 左侧：栈追踪（逐行出现、偶尔乱码）
    f[:, :int(kit.W * 0.44) * kit.S] *= 0.46
    idx = int(lt / 0.26) + 1
    shown = _TRACE[max(0, idx - 10):idx]
    for j, s in enumerate(shown):
        last = (j == len(shown) - 1)
        col = kit.mixc(CORAL, WHITE, 0.65) if last else CORAL
        kit.draw_text(f, _garble(s, c, j), 78, kit.H * 0.14 + j * 28, 20, col,
                      kit.F_MONO, 0.58 + 0.38 * last, 0.6)
    # 竖向分隔
    f[:, int(kit.W * 0.44) * kit.S:int(kit.W * 0.44) * kit.S + kit.S] += CORAL * 0.12

    # ---- 右侧：不断增长的计数器
    n = int(lt / 0.21) + 1
    kit.draw_text(f, "attempt #%04d" % n, kit.W - 84, kit.H * 0.17, 54, WHITE, kit.F_MONO_B,
                  0.94, 1.5, "right")
    kit.draw_text(f, "retry in %.2fs" % max(0.0, 0.30 - (lt % 0.30)), kit.W - 84,
                  kit.H * 0.17 + 74, 23, CORAL, kit.F_MONO, 0.62, 2.0, "right")
    kit.draw_text(f, "ERROR 0x0BAD", kit.W - 84, kit.H * 0.17 + 112, 23, CORAL, kit.F_MONO,
                  0.20 + 0.35 * (int(t * 3.0) % 2), 2.0, "right")
    kit.draw_text(f, "args %d / 0" % (int(lt * 9) + 1), kit.W - 84, kit.H * 0.17 + 150, 19,
                  kit.mixc(CORAL, BLACK, 0.15), kit.F_MONO, 0.5, 2.0, "right")

    # ---- 她的立绘被压成一缕（高度随拍点压缩）
    p = c.pulse()
    spr = kit.sprite(stage.MAID_R, 980)
    strand = _squash(spr, 24 + 10 * p, 980 * (1.0 - 0.46 * p))
    kit.add_radial(f, kit.W * 0.735, kit.H * 0.62, 240, CORAL, 0.08 + 0.14 * p, 2.2, 0.26)
    kit.place(f, strand, kit.W * 0.735 + 8 * math.sin(lt * 9.0), kit.H * 1.02, 1.0,
              0.80, tint=kit.mixc(CORAL, BLACK, 0.28), tint_k=0.9, additive=True)
    strand2 = _squash(spr, 6, 980 * (1.0 - 0.30 * p))
    kit.place(f, strand2, kit.W * 0.735 + 22, kit.H * 1.02, 1.0, 0.55,
              tint=WHITE, tint_k=1.0, additive=True)

    # ---- 歌词（终端行 + 注释）
    kit.draw_text(f, "> " + _ILL["en"], 78, kit.H * 0.845, 29,
                  kit.mixc(CORAL, WHITE, 0.25), kit.F_MONO, 0.92, 1.2)
    kit.draw_text(f, "# " + _ILL["zh"], 78, kit.H * 0.845 + 42, 17,
                  kit.mixc(CORAL, BLACK, 0.25), kit.F_SERIF_SC, 0.6, 3.0)

    # ---- 扫描亮带（一行行扫过去）
    yy = (lt * 0.82 % 1.0) * kit.H
    y0, y1 = int(yy * kit.S), int((yy + 3) * kit.S)
    f[y0:min(kit.SH, y1)] += CORAL * 0.10

    # ---- 撕裂 + 拍点抖动
    kit.glitch_rows(f, t, 0.20 + 0.50 * p, seed=13, band=16)
    if p > 0.82:
        c.shake(1.3, seed=5)

    # 结尾压暗，交给空线程
    if lt > 10.20:
        f *= kit.cl((T_EMPTY - t) / 0.34)


# ================================================================ 2. EMPTY
@scene(T_EMPTY, T_EXEC, "EMPTY", "H")
def empty(c):
    t = c.t
    f = c.f
    lt = t - T_EMPTY
    c.post.update(grain=0.016, vignette=0.80, bloom=(0.92, 26, 0.16), scan=0.0)

    f[:] = kit.hx("03070A")

    # 心跳：每 2 秒一次，暗青、半径很小
    ph = lt % 2.0
    k = math.exp(-ph * 5.2) * kit.cl(ph / 0.04)
    kit.add_radial(f, kit.W * 0.5, kit.H * 0.5, 74 + 18 * k, kit.hx("22585C"), 0.85 * k, 2.6)

    # 白色光标：画面中央
    stage.cursor(f, kit.W * 0.5, kit.H * 0.5, t, 56, WHITE, 1.0, "block", True, 0.9)

    # 左下角极小的计时器
    col = kit.mixc(WHITE, kit.hx("2A7C82"), 0.55)
    kit.draw_text(f, "thread 0x00 waiting...  00:%04.1f" % lt, 64, kit.H - 78, 19, col,
                  kit.F_MONO, 0.9, 1.0)


# ================================================================ 3. EXEC14 的 14 个切
def _d_a(c, lt, d):
    f = c.f
    p = c.pulse()
    rows = ((0.055, 236), (0.345, 300), (0.700, 200))
    for i, (yy, sz) in enumerate(rows):
        txt = "EXECUTION"
        st = _fit(txt, kit.F_SERIF_I, sz, kit.W * 0.93) * (1.0 + 0.015 * p)
        _draw_st(f, txt, kit.W / 2, kit.H * yy, sz, WHITE if i == 1 else kit.mixc(WHITE, ASH, 0.3),
                 kit.F_SERIF_I, 1.0 if i == 1 else 0.78, 0, st)
    # 细红线 + 角标
    yy = kit.H * 0.345 + 300 * 0.95
    f[int(yy * kit.S):int(yy * kit.S) + 3 * kit.S, int(kit.W * 0.06 * kit.S):int(kit.W * 0.94 * kit.S)] += CORAL
    kit.draw_text(f, "14 / 14", kit.W - 90, 54, 22, CORAL, kit.F_MONO, 0.55, 2.0, "right")


def _d_b(c, lt, d):
    f = c.f
    spr = _crop(stage.MAID_R2, (0.26, 0.165, 0.72, 0.305), 1920, 1110)
    q = lt / max(d, 1e-3)
    kit.place(f, spr, kit.W * 0.5 + 12 * math.sin(lt * 2.1), kit.H * 0.5 - 26 * q, 1.0, 1.0,
              tint=kit.mixc(CREAM, WHITE, 0.5), tint_k=0.14, anchor="center")
    # 扫描线切开
    for j in range(6):
        yy = (lt * 300 + j * 196) % (kit.H + 120) - 60
        h = 8 + 20 * ((j * 37) % 5) / 5.0
        y0 = max(0, int(yy * kit.S)); y1 = min(kit.SH, int((yy + h) * kit.S))
        if y1 > y0:
            f[y0:y1] *= 0.03
            f[y0:y0 + max(1, kit.S)] += WHITE * 0.5
    # 眼平线
    ye = int(kit.H * 0.47 * kit.S)
    f[ye:ye + 2 * kit.S, :] += CORAL * 0.35
    kit.glitch_rows(f, c.t, 0.14, seed=7, band=9)


def _d_c(c, lt, d):
    f = c.f
    q = lt / max(d, 1e-3)
    e = kit.ease_out(min(1.0, q * 0.95))
    # 火星：薄壳往外炸
    base = c.cl.sh(cloud.shape_sphere, 1.0, 0.14)
    P = base * (50.0 + 210.0 * e)
    c.cl.draw(f, P, dist=1250, blur_r=0.9, size=1.25, glow_gain=0.20,
              color=kit.mixc(WHITE, CORAL, 0.45), target=0.62,
              exposure=1.0 - 0.35 * q)
    # 放射速度线
    m = _half()
    cx, cy = kit.W * 0.5, kit.H * 0.5
    rs = np.random.default_rng(303)
    for j in range(56):
        a0 = rs.random() * math.tau
        r0 = (120.0 + 470.0 * e) * (0.55 + 0.5 * rs.random())
        ln = (90.0 + 320.0 * rs.random()) * (0.4 + 1.2 * e)
        _ln(m, cx + math.cos(a0) * r0, cy + math.sin(a0) * r0,
            cx + math.cos(a0) * (r0 + ln), cy + math.sin(a0) * (r0 + ln), 1.4 + 2.0 * rs.random())
    _flush(f, m, kit.mixc(WHITE, CORAL, 0.35), 0.55 * (1.0 - 0.5 * q), 0.9)
    # 冲击波环（半分辨率距离场，软边细环）
    rr = 70.0 + 460.0 * e
    ww = max(3.0, 12.0 * (1.0 - q))
    dm = _ring_d()
    ring = np.clip(1.0 - np.abs(dm - rr) / max(1.0, ww), 0.0, 1.0)
    f += cv2.resize(ring, (kit.SW, kit.SH), interpolation=cv2.INTER_LINEAR)[..., None] * \
        kit.mixc(WHITE, CORAL, 0.30) * (0.85 * (1.0 - q) ** 1.2)
    # 起手白闪（约 5 帧）
    kit.add_radial(f, cx, cy, 300, WHITE, 1.4 * max(0.0, 1.0 - q * 9.0), 1.6)
    # 起手白闪（约 5 帧）
    kit.add_radial(f, cx, cy, 300, WHITE, 1.4 * max(0.0, 1.0 - q * 9.0), 1.6)


def _d_d(c, lt, d):
    f = c.f
    x0, x1 = kit.W * 0.045, kit.W * 0.955
    y0, y1 = kit.H * 0.075, kit.H * 0.925
    # 面板边框
    f[int(y0 * kit.S):int(y0 * kit.S) + kit.S, int(x0 * kit.S):int(x1 * kit.S)] += CORAL * 0.6
    f[int(y1 * kit.S):int(y1 * kit.S) + kit.S, int(x0 * kit.S):int(x1 * kit.S)] += CORAL * 0.6
    f[int(y0 * kit.S):int(y1 * kit.S), int(x0 * kit.S):int(x0 * kit.S) + kit.S] += CORAL * 0.6
    f[int(y0 * kit.S):int(y1 * kit.S), int(x1 * kit.S):int(x1 * kit.S) + kit.S] += CORAL * 0.6
    # 标题行
    kit.draw_text(f, "execution.py", x0 + 24, y0 + 18, 25, WHITE, kit.F_MONO, 0.92, 2.0)
    kit.draw_text(f, "RUN 14/14   pid 0   thread 0x00", x1 - 40, y0 + 20, 20, CORAL,
                  kit.F_MONO, 0.70, 3.0, "right")
    kit.draw_text(f, "stdout / execution", x1 - 40, kit.H - 58, 20, CORAL, kit.F_MONO, 0.7, 2.0, "right")
    # 滚动代码 + 行号
    lead = 33.0
    i0 = int(lt * 26) % (len(_CODE_LONG) - 22)
    frac = (lt * 26) % 1.0
    ybase = y0 + 74 - frac * lead
    for i, (kind, txt) in enumerate(_CODE_LONG[i0:i0 + 22]):
        y = ybase + i * lead
        if y < y0 + 56 or y > y1 - 44:
            continue
        kit.draw_text(f, "%04d" % (i0 + i + 1), x0 + 96, y, 20,
                      kit.mixc(CORAL, BLACK, 0.32), kit.F_MONO, 0.55, 0, "right")
        kit.draw_text(f, txt, x0 + 122, y, 25, _CODE_COL.get(kind, WHITE), kit.F_MONO, 0.96, 0.8)
    # CRT 亮边
    yy = int(y1 * kit.S)
    f[yy - 3 * kit.S:yy - kit.S] += CORAL * 0.18
    # 右列：寄存器 + 内存 dump
    rx = kit.W * 0.690
    kit.draw_text(f, "registers", rx, y0 + 62, 18, CORAL, kit.F_MONO, 0.72, 4.0)
    regs = ("RIP", "RSP", "RAX", "RBX", "RCX", "R08", "R13", "FLG")
    tick = int(lt * 940)
    for i, rg in enumerate(regs):
        ry = y0 + 100 + i * 32
        kit.draw_text(f, rg, rx, ry, 20, kit.mixc(WHITE, ASH, 0.35), kit.F_MONO, 0.75, 1.5)
        kit.draw_text(f, "0x%08X" % (0x0BAD0000 + ((tick + i * 8191) & 0xFFFF)), rx + 76, ry,
                      20, CORAL if i % 3 == 0 else WHITE, kit.F_MONO, 0.88, 1.0)
    dy = y0 + 100 + 8 * 32 + 24
    kit.draw_text(f, "0x00BAD000", rx, dy, 18, kit.mixc(CORAL, BLACK, 0.18), kit.F_MONO, 0.72, 1.0)
    rsm = np.random.default_rng(5400 + (tick % 17))
    for i in range(7):
        bs = " ".join("%02X" % int(x) for x in rsm.integers(0, 256, 8))
        kit.draw_text(f, bs, rx, dy + 26 + i * 26, 20, kit.mixc(WHITE, CORAL, 0.40),
                      kit.F_MONO, 0.52, 1.4)
    # 右侧进度条
    bx = int(kit.W * 0.916 * kit.S)
    by0, by1 = int(y0 * kit.S), int(y1 * kit.S)
    f[by0:by1, bx:bx + 2 * kit.S] += kit.mixc(CORAL, BLACK, 0.55)
    fill = by0 + int((by1 - by0) * (lt / max(d, 1e-3)))
    f[by0:fill, bx:bx + 2 * kit.S] += CORAL * 0.9


def _d_e(c, lt, d):
    f = c.f
    f[:] = kit.hx("EFE6D6")
    p = c.pulse()
    for i, (yy, sz) in enumerate(((0.075, 210), (0.400, 280))):
        txt = "EXECUTION"
        st = _fit(txt, kit.F_SERIF_I, sz, kit.W * 0.92) * (1 + 0.01 * p)
        _draw_st(f, txt, kit.W / 2, kit.H * yy, sz, kit.hx("0B0B0D"), kit.F_SERIF_I,
                 0.96 if i == 1 else 0.72, 0, st)
    kit.draw_text(f, "execution", kit.W * 0.5, kit.H * 0.745, 62, kit.hx("B03A2E"),
                  kit.F_SERIF_I, 0.85, 10, "center")
    kit.draw_text(f, "INVERTED / 1 OF 14", 84, kit.H - 66, 20, kit.hx("2A2A2E"),
                  kit.F_MONO, 0.6, 3.0)


def _d_f(c, lt, d):
    f = c.f
    f += kit.vgrad(kit.SH, kit.SW, kit.hx("0A0507"), kit.hx("020306"), 1.2)
    m = _half()
    mw = _half()
    rs = np.random.default_rng(77)
    for j in range(72):
        yy = rs.random() * kit.H
        ln = 80 + (rs.random() ** 2) * 1150
        x0 = ((rs.random() * 1.5 - 0.35) * kit.W + lt * 2600 * (0.6 + rs.random())) % (kit.W * 1.5) - kit.W * 0.25
        th = 1.2 + 3.0 * rs.random()
        if j % 4 == 0:
            _ln(mw, x0, yy, x0 + ln * 0.6, yy, th)
        else:
            _ln(m, x0, yy, x0 + ln, yy, th)
    _flush(f, m, CORAL, 0.95, 0.7)
    _flush(f, mw, WHITE, 0.55, 0.7)
    stage.character(f, stage.MAID_L, kit.W * 0.5, kit.H * 1.03, 900, 0.96,
                    tint=BLACK, tint_k=0.97, rim=CORAL, rim_k=0.85)
    p = c.pulse()
    kit.draw_text(f, "EXECUTION", kit.W / 2, kit.H * 0.50, 150, WHITE, kit.F_MONO_B,
                  0.10 + 0.5 * p, 26, "center")


def _d_g(c, lt, d):
    f = c.f
    q = kit.ease_io(lt / max(d, 1e-3))
    yaw = 0.35 + math.radians(90.0) * q
    pitch = 0.18 + 0.10 * math.sin(lt * 2.0)
    sc = 560.0
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], F32) * sc
    cyw, syw = math.cos(yaw), math.sin(yaw)
    Ry = np.array([[cyw, 0, syw], [0, 1, 0], [-syw, 0, cyw]], F32)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], F32)
    P = corners @ Ry.T @ Rx.T
    z = P[:, 2] + 1500.0
    k = 1000.0 / np.maximum(z, 60.0)
    X = kit.W / 2 + P[:, 0] * k
    Y = kit.H * 0.50 - P[:, 1] * k
    m = _half()
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count("1") == 1:
                _ln(m, X[i], Y[i], X[j], Y[j], 2.4)
    # 内圈竖栏
    for a in range(-1, 2):
        for b in range(-1, 2):
            if a == 0 and b == 0:
                continue
            p0 = np.array([a * sc, -sc, b * sc], F32)
            p1 = np.array([a * sc, sc, b * sc], F32)
            q0 = p0 @ Ry.T @ Rx.T; q1 = p1 @ Ry.T @ Rx.T
            k0 = 1000.0 / max(60.0, q0[2] + 1500.0); k1 = 1000.0 / max(60.0, q1[2] + 1500.0)
            _ln(m, kit.W / 2 + q0[0] * k0, kit.H * 0.50 - q0[1] * k0,
                kit.W / 2 + q1[0] * k1, kit.H * 0.50 - q1[1] * k1, 1.4)
    _flush(f, m, WHITE, 0.95, 1.5)
    kit.draw_text(f, "90°", kit.W / 2, kit.H * 0.055, 64, CORAL, kit.F_SERIF_I, 0.7, 4, "center")


def _d_h(c, lt, d):
    f = c.f
    m = _half()
    ph = lt * 22.0
    specs = ((8.0, 230.0, 2.8, 0.0), (13.0, 150.0, 1.8, 1.7), (19.0, 86.0, 1.2, 3.4))
    for j, (k, A, th, off) in enumerate(specs):
        n = 64
        prev = None
        for i in range(n + 1):
            x = i / n * kit.W
            y = kit.H * 0.50 + A * math.sin(k * i / n * 6.283 + ph + off)
            if prev is not None:
                _ln(m, prev[0], prev[1], x, y, th)
            prev = (x, y)
    _flush(f, m, WHITE, 0.55, 0.9)
    _flush(f, m, CORAL, 0.28, 1.8)
    kit.draw_text(f, "sin(x)", 84, kit.H * 0.09, 26, CORAL, kit.F_MONO, 0.6, 2.0)


def _d_i(c, lt, d):
    f = c.f
    q = kit.cl(lt / max(d, 1e-3))
    rs = c.rng(31)
    cols, rows = 14, 7
    for r in range(rows):
        for cc in range(cols):
            v = int(rs.integers(0, 10))
            lock = (r * cols + cc) / float(rows * cols) < q
            a = 0.16 + 0.30 * float(rs.random()) + (0.55 if lock else 0.0)
            kit.draw_text(f, "1" if lock else str(v), 90 + cc * 130, 84 + r * 126, 48,
                          WHITE if lock else CORAL, kit.F_MONO, a, 2.0)
    big = "0" if q < 0.62 else "1"
    kit.draw_text(f, big, kit.W / 2, kit.H * 0.285, 420, WHITE, kit.F_MONO_B, 0.96, 0, "center")
    kit.draw_text(f, "%0.3f" % q, kit.W / 2, kit.H * 0.79, 60, CORAL, kit.F_MONO, 0.9, 8, "center")


def _d_j(c, lt, d):
    f = c.f
    q = kit.cl(lt / max(d, 0.2))
    P = c.cl.sh(cloud.shape_god, 900.0)
    P = c.cl.transform(P, yaw=0.10, offset=(0, -40, 0))
    c.cl.draw(f, P, dist=1250, blur_r=2.0, size=1.7, glow_gain=0.7,
              color=kit.mixc(WHITE, CORAL, 0.30), target=0.55 + 0.4 * q)
    kit.add_radial(f, kit.W * 0.5, kit.H * 0.26, 300, WHITE, 0.18, 2.0)
    kit.draw_text(f, "EXECUTION", kit.W / 2, kit.H * 0.80, 96, WHITE, kit.F_SERIF_I, 0.20, 26, "center")


def _d_k(c, lt, d):
    f = c.f
    # 背光板：把鲸剪影挖出来
    f += kit.vgrad(kit.SH, kit.SW, kit.hx("16242C"), kit.hx("08131A"), 1.15)
    kit.add_radial(f, kit.W * 0.5, kit.H * 0.58, 780, kit.mixc(WHITE, ASH, 0.45), 0.16, 1.5)
    stage.character(f, stage.WHALE["serious"], kit.W * 0.5, kit.H * 0.61, 760, 1.0,
                    tint=BLACK, tint_k=1.0, rim=WHITE, rim_k=0.55, anchor="center")
    # 大字压在上半屏
    st = _fit("EXECUTION", kit.F_SERIF_I, 210, kit.W * 0.93)
    _draw_st(f, "EXECUTION", kit.W / 2, kit.H * 0.015, 210, WHITE, kit.F_SERIF_I, 0.96, 0, st)
    yy = int(kit.H * 0.245 * kit.S)
    f[yy:yy + 3 * kit.S, int(kit.W * 0.07 * kit.S):int(kit.W * 0.93 * kit.S)] += CORAL
    kit.draw_text(f, "EXECUTION 12 / 14", kit.W - 84, kit.H * 0.945, 30, CORAL, kit.F_MONO,
                  0.62, 12, "right")


def _d_l(c, lt, d):
    f = c.f
    spr = kit.sprite(stage.MAID_R, 1040)
    hh = spr.shape[0] / 6.0
    p = c.pulse()
    top = kit.H * 0.02
    for i in range(6):
        y0 = i * hh / kit.S
        band = spr[int(i * hh):int((i + 1) * hh)]
        dx = (i - 2.5) * (30 + 46 * p) * (1 if i % 2 == 0 else -1)
        kit.place(f, band, kit.W * 0.5 + dx, top + y0, 1.0, 1.0,
                  tint=kit.mixc(CREAM, WHITE, 0.4), tint_k=0.10, anchor="top")
    for i in range(5):
        yy = int((top + (i + 1) * (kit.H * 0.96 / 6.0)) * kit.S)
        f[yy:yy + 2 * kit.S, :] += kit.mixc(WHITE, CORAL, 0.35) * 0.55


def _d_m(c, lt, d):
    f = c.f
    p = c.pulse()
    kit.draw_text(f, ";", kit.W / 2, kit.H * 0.5 - 380, 620, WHITE, kit.F_MONO_B,
                  0.97, 0, "center")
    stage.cursor(f, kit.W / 2 + 210, kit.H * 0.72, c.t, 30, WHITE, 0.85, "block", True, 0.4)
    kit.draw_text(f, "syntax error: expected '}'", kit.W / 2, kit.H * 0.88, 26, CORAL,
                  kit.F_MONO, 0.6 + 0.3 * p, 2.0, "center")


def _d_n(c, lt, d):
    f = c.f
    f *= 0.10
    x = kit.W * 0.60 + 6 * math.sin(lt * 0.9)
    y = kit.H * 0.44 + 4 * math.cos(lt * 0.7)
    kit.add_radial(f, x, y, 46, WHITE, 0.16, 2.4)
    kit.add_radial(f, x, y, 7, WHITE, 0.9, 1.2)


_DESIGN = [_d_a, _d_b, _d_c, _d_d, _d_e, _d_f, _d_g,
           _d_h, _d_i, _d_j, _d_k, _d_l, _d_m, _d_n]
_FLASH = ["w", "w", "b", "w", "b", "w", "w", "b", "w", "w", "b", "w", "w", None]


@scene(T_EXEC, T_NUMS, "EXEC14", "H")
def exec14(c):
    t = c.t
    f = c.f
    i = len(CUTS) - 1
    for k, (a, b) in enumerate(CUTS):
        if t < b:
            i = k
            break
    a, b = CUTS[i]
    lt = t - a
    d = b - a

    c.post.update(grain=0.036, vignette=0.56, bloom=(0.66, 22, 0.48), chroma=0.0, scan=0.05)
    f[:] = kit.hx("03060A")
    _DESIGN[i](c, lt, d)

    # 切点：1–2 帧纯白 / 纯黑
    fl = _FLASH[i]
    if fl and lt < 0.038:
        u = 1.0 - lt / 0.038
        f[:] = f * (1.0 - u) + (WHITE if fl == "w" else BLACK) * u

    p = c.pulse()
    if p > 0.80 and (i % 2 == 0):
        c.shake(1.5, seed=21 + i)


# ================================================================ 4. NUMS
_NUMS = [
    ("壹", kit.F_SERIF_SC, 196, WHITE, "zh", 0),
    ("弐", kit.F_HEI, 190, CREAM, "ja", 1),
    ("three", kit.F_SERIF_I, 124, CORAL, "en", 2),
    ("quatre", kit.F_SERIF, 118, WHITE, "fr", 3),
    ("cinco", kit.F_BAHN, 122, CORAL, "es", 0),
    ("六", kit.F_SANS_SC, 196, CREAM, "zh", 3),
]
_PACK = {}


def _pack():
    """把 6 个数字按字面宽度等间距排成一行，整体占 0.87W，基线对齐。"""
    if _PACK:
        return _PACK
    gap = kit.W * 0.050
    ws = [kit.text_width(fp, sz, txt) for (txt, fp, sz, col, lang, side) in _NUMS]
    tot = sum(ws) + gap * (len(_NUMS) - 1)
    k = min(1.42, max(0.55, kit.W * 0.87 / tot))
    sizes = [s[2] * k for s in _NUMS]
    ws = [w * k for w in ws]
    tot = sum(ws) + gap * (len(_NUMS) - 1)
    x = (kit.W - tot) / 2.0
    xs, ys, base = [], [], []
    for i, w in enumerate(ws):
        xs.append(x + w / 2.0)
        x += w + gap
        asc, desc = kit.text_metrics(_NUMS[i][1], sizes[i])
        ys.append(kit.H * 0.560 - asc)
        base.append(kit.H * 0.560)
    _PACK.update(x=xs, s=sizes, y=ys, base=base)
    return _PACK


def _fly(i, p, tx, ty):
    side = _NUMS[i][5]
    if side == 0:
        sx, sy = -460.0, ty
    elif side == 1:
        sx, sy = tx, -340.0
    elif side == 2:
        sx, sy = kit.W + 460.0, ty
    else:
        sx, sy = tx, kit.H + 340.0
    e = kit.ease_out(p)
    return sx + (tx - sx) * e, sy + (ty - sy) * e


@scene(T_NUMS, T_END, "NUMS", "H")
def nums(c):
    t = c.t
    f = c.f
    c.post.update(grain=0.030, vignette=0.56, bloom=(0.68, 20, 0.52), chroma=0.35, scan=0.04)
    f += kit.vgrad(kit.SH, kit.SW, kit.hx("05070C"), kit.hx("010204"), 1.4)

    # ---- 第一声之前：一条基线 + 读数（把 158.31–159.69 的空档填住）
    if t < _WORD_T[0] + 0.30:
        q0 = kit.cl((t - T_NUMS) / 0.45)
        bar = kit.W * 0.86 * q0
        yy = int(kit.H * 0.55 * kit.S)
        f[yy:yy + kit.S, int((kit.W - bar) / 2 * kit.S):int((kit.W + bar) / 2 * kit.S)] += \
            kit.mixc(WHITE, CORAL, 0.45) * 0.75
        n = sum(1 for ti in _WORD_T if t >= ti)
        kit.draw_text(f, "count %d / 6" % n, 84, kit.H * 0.105, 22,
                      kit.mixc(CREAM, CORAL, 0.40), kit.F_MONO, 0.66, 4.0)
        kit.draw_text(f, "world.execute(me);", kit.W - 84, kit.H * 0.105, 22,
                      kit.mixc(CREAM, BLACK, 0.35), kit.F_MONO, 0.5, 2.0, "right")
        kit.draw_text(f, "world.execute(me);", 84, kit.H * 0.86, 26,
                      kit.mixc(CREAM, BLACK, 0.35), kit.F_MONO, 0.55, 2.0)

    # ---- 多语言 1–6：从四面飞入 → 排成一行 → 缩成一个点
    if t < 161.768:
        shrink = kit.cl((t - _WORD_T[-1]) / 0.23)
        dst = f if shrink <= 0 else np.zeros_like(f)
        P = _pack()
        for i, (txt, fp, sz, col, lang, side) in enumerate(_NUMS):
            ti = _WORD_T[i]
            p = kit.cl((t - ti) / 0.34)
            if p <= 0:
                continue
            tx = P["x"][i]
            ty = P["y"][i]
            tsz = P["s"][i]
            x, y = _fly(i, p, tx, ty)
            st = 1.0 + 1.6 * (1.0 - p) if side in (0, 2) else 1.0
            jit = 2.5 * c.pulse() if p >= 1 else 0.0
            for g in range(3 if p < 0.85 else 1):
                ga = (0.20 if g else 1.0) * kit.cl(p * 3.0)
                gx = x - (0 if g == 0 else 16 * g * (1 - p) * (1 if side in (0, 2) else 0))
                gy = y + (0 if g == 0 else 12 * g * (1 - p) * (1 if side in (1, 3) else 0))
                _draw_st(dst, txt, gx + jit, gy, tsz, col, fp, ga, 1.0, st)
                if fp == kit.F_SANS_SC:
                    _draw_st(dst, txt, gx + jit, gy + 1.2, tsz, col, fp, ga * 0.6, 1.0, st)
                    _draw_st(dst, txt, gx + jit, gy - 1.2, tsz, col, fp, ga * 0.6, 1.0, st)
            if p >= 1.0:
                yy = int((P["base"][i] + 22) * kit.S)
                dst[yy:yy + max(2, kit.S + 1), int((tx - 34) * kit.S):int((tx + 34) * kit.S)] += CORAL * 0.55
                kit.draw_text(dst, lang, tx, P["base"][i] + 38, 17,
                              kit.mixc(CREAM, BLACK, 0.20), kit.F_MONO, 0.78, 4.0, "center")
        if shrink > 0:
            k = max(0.02, 1.0 - 0.985 * kit.ease_in(shrink))
            hh, ww2 = dst.shape[:2]
            nh, nw = max(1, int(hh * k)), max(1, int(ww2 * k))
            small = cv2.resize(dst, (nw, nh), interpolation=cv2.INTER_AREA)
            f[:] = 0
            y0s = (hh - nh) // 2
            x0s = (ww2 - nw) // 2
            f[y0s:y0s + nh, x0s:x0s + nw] += small
            if shrink > 0.55:
                kit.add_radial(f, kit.W / 2, kit.H * 0.5, 46, WHITE,
                               0.55 * (shrink - 0.55) / 0.45, 1.8)
        # 底部歌词行（机器读数）
        ln = mus.line_at(t)
        if ln is not None and ln["en"].startswith(("Ein", "Fem")):
            kit.draw_text(f, ln["en"], 84, kit.H * 0.86, 26, kit.mixc(CREAM, BLACK, 0.12),
                          kit.F_MONO, 0.72, 2.0)
        return

    # ---- 161.77 那一声 Execution：满屏白闪 + 大字收束
    tt = t - 161.768
    w = 1.0 - kit.cl(tt / 0.20)
    w = w ** 1.25
    f[:] = kit.hx("020306") * (1.0 - w) + WHITE * w
    a = kit.cl(tt / 0.025)
    if a > 0:
        sz = 206 + 130 * kit.ease_out(kit.cl((tt - 0.02) / 0.26))
        st = _fit("EXECUTION", kit.F_SERIF_I, sz, kit.W * 0.99) * a
        # 白闪期间用黑字，闪退之后用白字（真正的反相收束）
        if w > 0.42:
            _draw_st(f, "EXECUTION", kit.W / 2, kit.H * 0.30, sz, kit.hx("08080A"),
                     kit.F_SERIF_I, a, 0, st)
        else:
            _draw_st(f, "EXECUTION", kit.W / 2, kit.H * 0.30, sz, WHITE,
                     kit.F_SERIF_I, a * min(1.0, (1.0 - w) / 0.58), 0, st)
        b = kit.cl((tt - 0.14) / 0.16) * (1.0 - w)
        if b > 0:
            kit.draw_text(f, "world.execute(me);", kit.W / 2, kit.H * 0.74, 40,
                          kit.mixc(CREAM, CORAL, 0.25), kit.F_MONO, b * 0.75, 6.0, "center")
            yy = int(kit.H * 0.86 * kit.S)
            f[yy:yy + 2 * kit.S, int(kit.W * 0.30 * kit.S):int(kit.W * 0.70 * kit.S)] += CORAL * 0.5 * b


# ---------------------------------------------------------------- 代码墙
_CODE_BASE = [
    ("cm", "# world.execute(me) - run the execution"),
    ("kw", "def execute(me, *args, **kw):"),
    ("pl", "    assert me.is_human"),
    ("pl", "    if not me.is_valid():"),
    ("kw", "        raise ArgumentError(0x0BAD)"),
    ("cm", "        # YOU HAVE MADE SOME ILLEGAL ARGUMENTS"),
    ("str", '    me.role = "maid"   # switch my role'),
    ("pl", "    return EXECUTION"),
    ("kw", "for i in range(14):"),
    ("pl", "    print('Execution')"),
    ("cm", "# if I can give them all the execution"),
    ("kw", "while heart.beats():"),
    ("pl", "    love += 1"),
    ("pl", "    if love > ceiling: break"),
    ("cm", "# only satisfaction"),
    ("kw", "try:"),
    ("pl", "    world.execute(me)"),
    ("kw", "except Exception as e:"),
    ("str", '    log("illegal arguments")'),
    ("pl", "    raise"),
    ("cm", "# proof of my existence"),
    ("num", "0x7FE3D4  # 7FE3D4"),
    ("kw", "return None  # 0x0"),
    ("cm", "# thread 0x00 waiting..."),
]
_CODE_LONG = _CODE_BASE * 4
