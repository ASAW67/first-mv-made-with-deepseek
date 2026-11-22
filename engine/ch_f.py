# -*- coding: utf-8 -*-
"""ch_f.py — F 章：Switch my gender / do whatever / switch my role / enter the trance
89.53 ─ 103.84 （把「我」当成一个可以被 switch 的对象：换性别 / 六屏并置 / 选角 UI / 催眠隧道）

设计要点
  · GENDER  : 中央她 + 左右镜像分身，♂ ♀ 两个字形粒子化互换（每拍一次），配色 MINT→ROSE
  · WHATEVER: 3×2 六格，不同立绘/色调/取景，Consolas 的 self.mode = "..."，拍点重排
  · ROLE    : 五张 AI 立绘卡片（编号 / 名字 / 三条属性条），等距等基线，选中放大发光倾斜
  · TRANCE  : 粒子隧道 + 程序化同心环 + 1/8 音符频闪，她在尽头变小，结尾白闪加速线
  · 所有 UI 都在「逻辑坐标 + 整数像素」上落笔，卡片的编号/名字/属性条共用同一套网格
"""
import math, os
import numpy as np
import cv2
import kit, mus, stage, cloud
from film2 import scene

F32 = np.float32

# ---------------------------------------------------------------- 常量
T_GENDER = 89.53
T_WHATEVER = 92.55
T_ROLE = 96.44
T_TRANCE = 99.91
T_END = 103.84

FONT_M = kit.F_MONO
FONT_MB = kit.F_MONO_B
FONT_SYM = kit.F_SANS_SC      # Noto Sans SC：确实带 ♂ ♀ 字形（Consolas 也有，Georgia/Bahn 没有）

# stage.MAID_START_R 指向的文件名与实际素材不一致（引擎文件不能改），这里直接取真实文件
START_R = os.path.join(kit.ASSETS, "maid-atelier", "05-启动页-右女仆-马甲长裙.webp")

SEP = mus.find("switch my gender")       # 89.525
SEP2 = mus.find("And then do whatever")  # 92.552
SEP3 = mus.find("Oh switch my role")     # 96.441
SEP4 = mus.find("So we can enter")       # 99.913


# ================================================================ 帧空间基元
def _rect(f, x0, y0, x1, y1, color, a=1.0):
    """帧空间实心矩形（像素级对齐，直接切片，不用 mask）。"""
    S = kit.S
    xi0 = max(0, int(round(x0 * S))); yi0 = max(0, int(round(y0 * S)))
    xi1 = min(kit.SW, int(round(x1 * S))); yi1 = min(kit.SH, int(round(y1 * S)))
    if xi1 <= xi0 or yi1 <= yi0 or a <= 0.004:
        return
    c = np.asarray(color, F32)
    d = f[yi0:yi1, xi0:xi1]
    if a >= 0.999:
        d[:] = c
    else:
        d *= (1 - a); d += c * a


def _frame(f, x0, y0, x1, y1, th, color, a=1.0):
    _rect(f, x0, y0, x1, y0 + th, color, a)
    _rect(f, x0, y1 - th, x1, y1, color, a)
    _rect(f, x0, y0 + th, x0 + th, y1 - th, color, a)
    _rect(f, x1 - th, y0 + th, x1, y1 - th, color, a)


def _corner(f, x0, y0, x1, y1, ln, th, color, a=1.0):
    """四角 L 形卡口（选角 UI 的对焦框）。"""
    _rect(f, x0, y0, x0 + ln, y0 + th, color, a)
    _rect(f, x0, y0, x0 + th, y0 + ln, color, a)
    _rect(f, x1 - ln, y0, x1, y0 + th, color, a)
    _rect(f, x1 - th, y0, x1, y0 + ln, color, a)
    _rect(f, x0, y1 - th, x0 + ln, y1, color, a)
    _rect(f, x0, y1 - ln, x0 + th, y1, color, a)
    _rect(f, x1 - ln, y1 - th, x1, y1, color, a)
    _rect(f, x1 - th, y1 - ln, x1, y1, color, a)


def _scrim(f, y0=0.76, k=0.55, power=1.7):
    """底部压暗，保证歌词可读。"""
    h = kit.SH
    y = np.arange(h, dtype=F32) / h
    m = np.clip((y - y0) / max(1e-4, 1 - y0), 0, 1) ** power
    f *= (1 - (m * k)[:, None, None])


def _glowline(f, x0, y0, x1, y1, color, a=1.0, layers=3):
    """发光线：核心 1px + 向外扩张的几层低透明度，不耗 GaussianBlur。"""
    for i in range(layers):
        g = layers - i
        _rect(f, x0 - g * 0.6, y0 - g * 0.6, x1 + g * 0.6, y1 + g * 0.6, color, a * 0.13)
    _rect(f, x0, y0, x1, y1, color, a)


def _shadow(f, cx, cy, rx, ry, k=0.6):
    """廉价接触阴影：只在小缓冲里算，不碰全帧 mask。"""
    S = kit.S
    w = max(6, int(rx * 2 * S)); h = max(6, int(ry * 2 * S))
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    r = np.sqrt(((xx - w * 0.5) / (w * 0.5)) ** 2 + ((yy - h * 0.5) / (h * 0.5)) ** 2)
    m = np.clip(1.0 - r, 0, 1) ** 1.7
    xi = int(round(cx * S - w * 0.5)); yi = int(round(cy * S - h * 0.5))
    xs0 = max(0, xi); ys0 = max(0, yi)
    xe = min(kit.SW, xi + w); ye = min(kit.SH, yi + h)
    if xe <= xs0 or ye <= ys0:
        return
    f[ys0:ye, xs0:xe] *= (1 - (m[ys0 - yi:ye - yi, xs0 - xi:xe - xi] * k)[..., None])


def _shaft(f, x, w_top, w_bot, color, a=0.16, y0=0.06, y1=0.92, bands=30):
    """柔光柱：一串横向条带叠加（比 poly_mask+blur 便宜一个数量级）。"""
    c = np.asarray(color, F32)
    for i in range(bands):
        u = i / (bands - 1.0)
        y = kit.H * (y0 + (y1 - y0) * u)
        w = w_top + (w_bot - w_top) * u
        fall = max(0.0, 1 - abs(u - 0.42) / 0.58) ** 1.8
        _rect(f, x - w * 0.5, y, x + w * 0.5, y + kit.H * 0.032, c, a * fall)


def _splat(f, X, Y, rgb, wgt, soft=0.9, gain=1.0, div=3):
    """屏幕空间加法点云（字形粒子用）。X/Y 逻辑坐标数组，rgb (N,3)，wgt (N,)。
    在 1/div 分辨率上累加再放大：bincount 的代价随像素数线性下降，火花糊一点更耐看。"""
    if X.size < 3:
        return f
    S = kit.S
    bw, bh = kit.SW // div, kit.SH // div
    xs = X * (S / div); ys = Y * (S / div)
    keep = (xs > -6) & (xs < bw + 6) & (ys > -6) & (ys < bh + 6)
    if keep.sum() < 3:
        return f
    xs = xs[keep]; ys = ys[keep]; c3 = rgb[keep]; w = wgt[keep]
    ix = np.floor(xs).astype(np.int32); iy = np.floor(ys).astype(np.int32)
    fx = (xs - ix)[:, None]; fy = (ys - iy)[:, None]
    buf = np.zeros((bh, bw, 3), F32)
    idxs, wts = [], []
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xi = np.clip(ix + dx, 0, bw - 1); yi = np.clip(iy + dy, 0, bh - 1)
        ww = ((1 - fx) if dx == 0 else fx) * ((1 - fy) if dy == 0 else fy)
        idxs.append(yi * bw + xi); wts.append((ww[:, 0]) * w)
    idx = np.concatenate(idxs); ww = np.concatenate(wts)
    cc = np.concatenate([c3] * 4)
    npx = bw * bh
    for ch in range(3):
        buf[..., ch] = np.bincount(idx, weights=ww * cc[:, ch], minlength=npx).reshape(bh, bw)
    if soft > 0:
        buf = buf + kit.blur(buf, soft * S / div) * 0.75
    buf = kit.tonemap_soft(buf, 1.5) * gain
    f += kit.up(buf, kit.SW, kit.SH)
    return f


# ---------------------------------------------------------------- 面板（RGBA 局部画布）
def _panel(w, h, color=(0, 0, 0), a=0.0):
    b = np.zeros((max(1, int(round(h * kit.S))), max(1, int(round(w * kit.S))), 4), F32)
    b[..., :3] = np.asarray(color, F32)
    b[..., 3] = a
    return b


def _over(buf, s_rgb, s_a, xi, yi, g=1.0):
    h, w = buf.shape[:2]
    sh, sw = s_a.shape[:2]
    xs, ys = max(0, xi), max(0, yi)
    xe, ye = min(w, xi + sw), min(h, yi + sh)
    if xe <= xs or ye <= ys:
        return
    a = s_a[ys - yi:ye - yi, xs - xi:xe - xi][..., None] * g
    s = s_rgb[ys - yi:ye - yi, xs - xi:xe - xi]
    d = buf[ys:ye, xs:xe]
    if buf.shape[2] == 3:          # 直接贴到主帧
        d *= (1 - a); d += s * a
        return
    da = d[..., 3:4]
    oa = a + da * (1 - a)
    d[..., :3] = (s * a + d[..., :3] * da * (1 - a)) / np.maximum(oa, 1e-5)
    d[..., 3:4] = oa


def _prect(buf, x0, y0, x1, y1, color, a=1.0):
    S = kit.S
    h, w = buf.shape[:2]
    xi0 = max(0, int(round(x0 * S))); yi0 = max(0, int(round(y0 * S)))
    xi1 = min(w, int(round(x1 * S))); yi1 = min(h, int(round(y1 * S)))
    if xi1 <= xi0 or yi1 <= yi0 or a <= 0.004:
        return
    c = np.asarray(color, F32)
    d = buf[yi0:yi1, xi0:xi1]
    if a >= 0.999:
        d[..., :3] = c; d[..., 3] = 1.0
        return
    oa = d[..., 3:4] * (1 - a) + a
    d[..., :3] = (c * a + d[..., :3] * d[..., 3:4] * (1 - a)) / np.maximum(oa, 1e-5)
    d[..., 3:4] = oa


def _pframe(buf, x0, y0, x1, y1, th, color, a=1.0):
    _prect(buf, x0, y0, x1, y0 + th, color, a)
    _prect(buf, x0, y1 - th, x1, y1, color, a)
    _prect(buf, x0, y0 + th, x0 + th, y1 - th, color, a)
    _prect(buf, x1 - th, y0 + th, x1, y1 - th, color, a)


def _ptext(buf, txt, x, y, size, color, a=1.0, tracking=0.0, font=None, align="left"):
    if a <= 0.004 or not txt:
        return 0.0
    fp = font or FONT_M
    m = kit.text_mask(fp, size, txt, tracking)
    wl = m.shape[1] / kit.S
    if align == "right":
        x = x - wl
    elif align == "center":
        x = x - wl * 0.5
    _over(buf, np.broadcast_to(np.asarray(color, F32), (m.shape[0], m.shape[1], 3)),
          m, int(round(x * kit.S)), int(round(y * kit.S)), a)
    return wl


def _psprite(buf, spr, cx, cy, a=1.0, tint=None, tint_k=0.0, k=1.0, rot=0.0, flip=False,
             anchor="foot"):
    s = spr
    if abs(k - 1.0) > 1e-3:
        s = kit.up(s, max(1, int(round(s.shape[1] * k))), max(1, int(round(s.shape[0] * k))))
    if flip:
        s = s[:, ::-1]
    if rot:
        h0, w0 = s.shape[:2]
        big = int(math.hypot(h0, w0)) + 2
        pad = np.zeros((big, big, 4), F32)
        y0 = (big - h0) // 2; x0 = (big - w0) // 2
        pad[y0:y0 + h0, x0:x0 + w0] = s
        M = cv2.getRotationMatrix2D((big / 2, big / 2), -math.degrees(rot), 1.0)
        s = cv2.warpAffine(pad, M, (big, big), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    hh, ww = s.shape[:2]
    if anchor == "center":
        xi = int(round(cx * kit.S - ww / 2)); yi = int(round(cy * kit.S - hh / 2))
    elif anchor == "top":
        xi = int(round(cx * kit.S - ww / 2)); yi = int(round(cy * kit.S))
    else:
        xi = int(round(cx * kit.S - ww / 2)); yi = int(round(cy * kit.S - hh))
    rgb = s[..., :3]
    if tint is not None and tint_k > 0:
        rgb = rgb * (1 - tint_k) + np.asarray(tint, F32)[None, None, :] * tint_k
    _over(buf, rgb, s[..., 3], xi, yi, a)


def _pblit(f, buf, cx, cy, k=1.0, rot=0.0, a=1.0):
    """把 RGBA 面板贴回主帧，支持缩放/旋转。"""
    if a <= 0.004:
        return
    S = kit.S
    if abs(k - 1.0) < 1e-3 and abs(rot) < 1e-4:
        hh, ww = buf.shape[:2]
        _over(f, buf[..., :3], buf[..., 3], int(round(cx * S - ww / 2)),
              int(round(cy * S - hh / 2)), a)
        return
    hh, ww = buf.shape[:2]
    big = int(math.hypot(hh, ww) * max(k, 1.0)) + 4
    pad = np.zeros((big, big, 4), F32)
    y0 = (big - hh) // 2; x0 = (big - ww) // 2
    pad[y0:y0 + hh, x0:x0 + ww] = buf
    M = cv2.getRotationMatrix2D((big / 2, big / 2), -math.degrees(rot), k)
    wp = cv2.warpAffine(pad, M, (big, big), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    _over(f, wp[..., :3], wp[..., 3], int(round(cx * S - big / 2)), int(round(cy * S - big / 2)), a)


# ---------------------------------------------------------------- 字形（缓存）
_GLY = {}
_SYM_META = {}


def _glyph(ch, size, n=3000):
    key = (ch, int(size), n, kit.S)
    g = _GLY.get(key)
    if g is not None:
        return g
    m = kit.text_mask(FONT_SYM, size, ch)
    hh, ww = m.shape
    rs = np.random.default_rng({"\u2642": 31, "\u2640": 57}.get(ch, 11))
    p = m.ravel().astype(np.float64) + 1e-9
    cdf = np.cumsum(p); cdf /= cdf[-1]
    idx = np.searchsorted(cdf, rs.random(n))
    yy, xx = np.divmod(idx, ww)
    fx = ((xx + rs.random(n)) / ww).astype(F32)
    fy = ((yy + rs.random(n)) / hh).astype(F32)
    lum = np.clip(m[np.clip(yy, 0, hh - 1), np.clip(xx, 0, ww - 1)], 0.10, 1.0).astype(F32)
    g = {"fx": fx, "fy": fy, "w": ww / kit.S, "h": hh / kit.S, "n": n, "lum": lum,
         "lw": kit.text_width(FONT_SYM, size, ch)}
    _GLY[key] = g
    return g


def _sym_meta(ch, size):
    """每个字形在字行盒里的墨迹中心比例 (fx, fy)：两个符号各自按墨迹居中，
    否则 ♂ 的十字偏左上、♀ 的环偏右下，两个叠在同一个槽位时会错开。"""
    k = (ch, int(size), kit.S)
    m0 = _SYM_META.get(k)
    if m0 is not None:
        return m0
    m = kit.text_mask(FONT_SYM, size, ch)
    ys, xs = np.nonzero(m > 0.25)
    hh, ww = m.shape
    if len(xs) == 0:
        m0 = (0.5, 0.5, ww / kit.S, hh / kit.S)
    else:
        m0 = ((xs.min() + xs.max()) * 0.5 / ww, (ys.min() + ys.max()) * 0.5 / hh,
              ww / kit.S, hh / kit.S)
    _SYM_META[k] = m0
    return m0


def _sym_draw(f, ch, cx, cy, size, color, a=1.0, halo=0.5):
    """以「墨迹中心」定位画 ♂/♀，带一层加色光晕。"""
    if a <= 0.006:
        return
    fx, fy, mw, mh = _sym_meta(ch, size)
    xx = cx - (fx - 0.5) * mw
    yy = cy - fy * mh
    if halo > 0:
        art = size * 1.09
        fx2, fy2, mw2, mh2 = _sym_meta(ch, art)
        kit.draw_text(f, ch, cx - (fx2 - 0.5) * mw2, cy - fy2 * mh2, art,
                      np.asarray(color, F32) * 0.55, FONT_SYM, a * 0.26 * halo, 0.0,
                      "center", additive=True)
    kit.draw_text(f, ch, xx, yy, size, color, FONT_SYM, a, 0.0, "center")


# ================================================================ 1 · GENDER
@scene(89.53, 92.55, "GENDER", "F")
def gender(c):
    t = c.t
    f = c.f
    lt = t - T_GENDER
    k = kit.cl(lt / (T_WHATEVER - T_GENDER))
    wrm = kit.smoothstep(0.0, 1.0, k)                       # MINT → ROSE
    ACC = kit.mixc(kit.MINT, kit.ROSE, wrm)
    ACC2 = kit.mixc(kit.ROSE, kit.MINT, wrm)
    c.post.update(grain=0.017, vignette=0.50, bloom=(0.56, 18, 0.52), chroma=0.28)
    _scrim(f, 0.80, 0.48)

    # 拍点 / 交换相位
    ph = (t - SEP) / mus.BEAT
    bi = int(math.floor(ph))
    frac = ph - bi
    u = kit.ease_io(kit.cl(frac / 0.44))                    # 0..1 一次交换
    side = 1.0 if (bi % 2 == 0) else -1.0                   # ♂ 的基准边
    pulse = c.pulse(11.0)

    # ---------- 背景
    stage.bg_image(f, stage.BG_PALACE, zoom=1.22 + 0.06 * k, dy=-16 * k,
                   dim=0.62, exposure=1.15, blur_r=5.0,
                   tint=kit.mixc(kit.hx("07202E"), kit.hx("200A1C"), wrm * 0.7), tint_k=0.45)
    kit.add_radial(f, kit.W * 0.5, kit.H * 0.56, 780, ACC, 0.075 + 0.05 * pulse, 2.0, 1.5)
    _shaft(f, kit.W * 0.5, 150, 540, kit.mixc(kit.PALE, ACC, 0.45), 0.14 + 0.05 * pulse)
    stage.dust(f, t, n=120, color=kit.mixc(kit.PALE, ACC, 0.6), alpha=0.34,
               box=(0.06, 0.94, 0.10, 0.96))

    # ---------- 三个她
    flip = (bi % 2 == 0)
    hc = 880.0 + 6.0 * round(22.0 * kit.ease_out(k) / 6.0)
    _shadow(f, kit.W * 0.5 - kit.W * 0.345, kit.H * 0.985, 195, 26, 0.55)
    _shadow(f, kit.W * 0.5 + kit.W * 0.345, kit.H * 0.985, 195, 26, 0.55)
    _shadow(f, kit.W * 0.5, kit.H * 0.985, 250, 32, 0.62)
    for xk, al, hh, fl in ((0.5 - 0.345, 0.80, 690, flip), (0.5 + 0.345, 0.80, 690, flip)):
        stage.character(f, stage.MAID_L, kit.W * xk, kit.H * 1.035, hh, alpha=al,
                        tint=kit.mixc(ACC, kit.hx("0A2432"), 0.42), tint_k=0.42,
                        rim=ACC, rim_k=0.55, glow=0.0, flip=fl, shadow=0.0)
    stage.character(f, stage.MAID_L, kit.W * 0.5, kit.H * 1.035, hc, alpha=1.0,
                    tint=ACC, tint_k=0.10, rim=ACC, rim_k=0.45 + 0.2 * pulse,
                    glow=0.0, shadow=0.0)
    stage.floor_reflection(f, t, y=0.875, strength=0.14)

    # ---------- ♂ ♀：先铺粒子流（跨境交换），再压上两个槽位的常驻字形
    SPAN = kit.W * 0.172
    SY = kit.H * 0.500
    SIZE = 310.0
    slot = (-1.0, 1.0)
    pair = (("\u2642", ACC), ("\u2640", ACC2))
    for gi, (ch, cl0, sgn) in enumerate((("\u2642", ACC, 1.0), ("\u2640", ACC2, -1.0))):
        g = _glyph(ch, int(SIZE), 1600)
        gw = g["w"]; gh = g["h"]
        xa = kit.W * 0.5 - sgn * SPAN
        xb = kit.W * 0.5 + sgn * SPAN
        q = np.clip((u - (1 - g["lum"]) * 0.30) / 0.70, 0, 1)
        q = q * q * (3 - 2 * q)
        bow = 56.0
        xs = (xa + (xb - xa) * q) + (g["fx"] - 0.5) * gw
        ys = SY + (g["fy"] - 0.5) * gh + bow * np.sin(np.pi * q)
        burst = math.sin(math.pi * u) * 26.0
        jit = np.random.default_rng(7 + gi).standard_normal((g["n"], 2)).astype(F32)
        xs = xs + jit[:, 0] * burst
        ys = ys + jit[:, 1] * burst
        env = math.sin(math.pi * u) ** 0.7
        bright = (0.26 + 0.95 * g["lum"]) * (0.08 + 0.95 * env)
        col = np.asarray(cl0, F32)[None, :] * (1.0 + 0.35 * env)
        _splat(f, xs, ys, np.repeat(col, g["n"], 0), bright, soft=0.9, gain=1.0)

    # 交叉溶解：出场的一半走完再让新的一半上来，中间让给粒子（两个字形叠在一起会读成乱码）
    a_out = 1.0 - kit.smoothstep(0.0, 0.52, u)
    a_in = kit.smoothstep(0.48, 1.0, u)
    for s_i, sx in enumerate(slot):
        cx = kit.W * 0.5 + sx * SPAN
        first = pair[0 if (sx * side) < 0 else 1]
        second = pair[1 if (sx * side) < 0 else 0]
        _sym_draw(f, first[0], cx, SY, SIZE, first[1] * 1.15, a_out * 0.98, 0.9)
        _sym_draw(f, second[0], cx, SY, SIZE, second[1] * 1.15, a_in * 0.98, 0.9)

    # ---------- HUD：右上角开关
    bx, by, bw, bh = kit.W - 84 - 132, 54, 132, 40
    _rect(f, bx, by, bx + bw, by + bh, kit.hx("071018"), 0.55)
    _frame(f, bx, by, bx + bw, by + bh, 1.2, ACC, 0.75 + 0.2 * pulse)
    on_right = (bi % 2 == 0)
    knx = bx + (bw - 64) * (1.0 if on_right else 0.0)
    _rect(f, knx + 3, by + 3, knx + 66, by + bh - 3, ACC, 0.26)
    _rect(f, knx + 3, by + 3, knx + 5.0, by + bh - 3, ACC, 0.95)
    kit.draw_text(f, "F", bx + 20, by + 11, 16, kit.CREAM if not on_right else kit.ASH,
                  FONT_MB, 0.95 if not on_right else 0.45, 1.2, "left")
    kit.draw_text(f, "M", bx + bw - 32, by + 11, 16, kit.CREAM if on_right else kit.ASH,
                  FONT_MB, 0.95 if on_right else 0.45, 1.2, "left")
    kit.draw_text(f, "sex", bx - 14, by + 12, 13, kit.ASH, FONT_M, 0.70, 1.4, "right")

    kit.draw_text(f, '> switch(gender = "%s")' % ("m" if on_right else "f"), 84, 56, 19,
                  ACC, FONT_M, 0.85, 0.6)
    kit.draw_text(f, "  role  = NULL", 84, 84, 19, kit.ASH, FONT_M, 0.55, 0.6)
    _rect(f, 84, 112, 84 + 236, 113, ACC, 0.30)

    c.lyric(size=42, y=kit.H * 0.862, alpha=0.95, glow=0.22)
    stage.hud_corner(f, t, "SEQ 0x2A // gender", "beat %03d" % (bi + 128), 0.42, ACC,
                     y=kit.H - 42)


# ================================================================ 2 · WHATEVER
_CW, _CH, _GAP = 586.0, 358.0, 14.0
_CX0 = (1920.0 - (3 * _CW + 2 * _GAP)) / 2
_CY0 = 104.0

_CELLS = [
    dict(p=stage.MAID_START_L, mode="standing", col=kit.ICE, h=340, foot=352),
    dict(p=stage.MAID_L, mode="curtsy", col=kit.MINT, h=344, foot=356),
    dict(p=stage.MAID_Q, mode="faint", col=kit.GOLD, h=262, foot=290),
    dict(p=stage.MAID_R2, mode="monocle", col=kit.VIOLET, h=500, foot=536),
    dict(p=stage.MAID_R, mode="note_taking", col=kit.ROSE, h=330, foot=356),
    dict(p=START_R, mode="startup", col=kit.CREAM, h=320, foot=354),
]

_CBG = {}


def _cell_bg(w, h, col):
    key = (round(w, 1), round(h, 1), tuple(np.round(np.asarray(col, F32), 3)))
    b = _CBG.get(key)
    if b is not None:
        return b
    b = _panel(w, h)
    hh = b.shape[0]; ww = b.shape[1]
    ys = np.linspace(0, 1, hh, dtype=F32)[:, None, None]
    top = kit.mixc(kit.hx("08171F"), col, 0.14)
    bot = kit.hx("02060A")
    b[..., :3] = top * (1 - ys) + bot * ys
    yy, xx = np.mgrid[0:hh, 0:ww].astype(F32)
    rx, ry = ww * 0.66, hh * 1.00
    r = np.sqrt(((xx - ww * 0.5) / rx) ** 2 + ((yy - hh * 0.60) / ry) ** 2)
    g = np.clip(1 - r, 0, 1) ** 2.6
    b[..., :3] += (g * 0.16)[..., None] * np.asarray(col, F32)
    b[..., 3] = 1.0
    _CBG[key] = b
    return b


@scene(92.55, 96.44, "WHATEVER", "F")
def whatever(c):
    t = c.t
    f = c.f
    lt = t - T_WHATEVER
    c.post.update(grain=0.016, vignette=0.44, bloom=(0.52, 18, 0.54), chroma=0.18)
    _scrim(f, 0.80, 0.42)

    ph = (t - SEP2) / mus.BEAT
    bi = int(math.floor(ph))
    frac = ph - bi
    jolt = (1.0 if bi % 2 else -1.0) * 10.0 * math.exp(-frac * 13.0)
    flash = math.exp(-frac * 9.0)

    stage.bg_image(f, stage.BG_PALACE, zoom=1.5, dim=0.30, exposure=1.05, blur_r=16,
                   tint=kit.hx("061522"), tint_k=0.70)
    rs = np.random.default_rng(1789 + bi)
    perm = rs.permutation(6)

    for i in range(6):
        r, cc = divmod(i, 3)
        x0 = _CX0 + cc * (_CW + _GAP)
        y0 = _CY0 + r * (_CH + _GAP)
        item = _CELLS[perm[i]]
        col = item["col"]
        buf = _cell_bg(_CW, _CH, col).copy()
        spr = kit.sprite(item["p"], item["h"])
        cx = _CW * 0.5 + jolt * (1 if i % 2 else -1)
        _psprite(buf, spr, cx, item["foot"], 1.0, tint=col, tint_k=0.20, anchor="foot")
        buf[..., :3] *= 0.90
        _prect(buf, 0, 0, _CW, 22, kit.hx("02060A"), 0.45)
        # 底部压暗：Consolas 标签压在立绘上也读得清
        for j in range(0, 40, 2):
            aa = (j / 40.0) ** 1.5 * 0.62
            _prect(buf, 0, _CH - 40 + j, _CW, _CH - 40 + j + 2, kit.hx("02060A"), aa)
        _pblit(f, buf, x0 + _CW * 0.5, y0 + _CH * 0.5)

        # 发光线边框（拍点闪）
        _glowline(f, x0, y0, x0 + _CW, y0 + 1.0, col, 0.75 + 0.25 * flash, 3)
        _glowline(f, x0, y0 + _CH - 1.0, x0 + _CW, y0 + _CH, col, 0.55, 2)
        _glowline(f, x0, y0, x0 + 1.0, y0 + _CH, col, 0.55, 2)
        _glowline(f, x0 + _CW - 1.0, y0, x0 + _CW, y0 + _CH, col, 0.55, 2)
        # 左上角标
        kit.draw_text(f, "0%d" % (i + 1), x0 + 14, y0 + 9, 12, col, FONT_MB, 0.85, 1.2)
        kit.draw_text(f, "0x%02d" % i, x0 + _CW - 14, y0 + 9, 12, col, FONT_M, 0.62, 1.0, "right")
        # Consolas 标签
        lab = 'self.mode = "%s"' % item["mode"]
        kit.draw_text(f, lab, x0 + 14, y0 + _CH - 32, 17, col, FONT_M, 0.95, 0.4)
        if (t * 2.6) % 1.0 < 0.5:
            kit.draw_text(f, "_", x0 + 14 + kit.text_width(FONT_M, 17, lab, 0.4) + 3,
                          y0 + _CH - 32, 17, col, FONT_M, 0.85, 0.4)

    # 顶部仪表
    kit.draw_text(f, "MULTI-VIEW // 6 instances", 84, 56, 20, kit.CREAM, FONT_M, 0.72, 1.2)
    kit.draw_text(f, "seed %d" % (1789 + bi), kit.W - 84, 56, 20, kit.MINT, FONT_M, 0.62, 1.2,
                  "right")
    _rect(f, 84, 88, kit.W - 84, 89, kit.ASH, 0.22)

    c.lyric(size=42, y=kit.H * 0.845, alpha=0.95, glow=0.22)
    stage.hud_corner(f, t, "// do whatever", "beat %03d" % (bi + 134), 0.38, kit.MINT,
                     y=kit.H - 42)


# ================================================================ 3 · ROLE
_CARD_W, _CARD_H, _CARD_GAP = 330.0, 590.0, 24.0
_CARD_N = 5
_CARD_X0 = (1920.0 - (_CARD_N * _CARD_W + (_CARD_N - 1) * _CARD_GAP)) / 2
_CARD_Y0 = 196.0

_CARDS = [
    dict(k="claude", nm="CLAUDE", org="anthropic", col=kit.hx("E8A87C"),
         st=[("SYNTAX", 92), ("WARMTH", 88), ("CHAOS", 14)]),
    dict(k="gpt", nm="GPT", org="openai", col=kit.MINT,
         st=[("SYNTAX", 95), ("WARMTH", 81), ("CHAOS", 27)]),
    dict(k="gemini", nm="GEMINI", org="google", col=kit.VIOLET,
         st=[("SYNTAX", 90), ("WARMTH", 74), ("CHAOS", 36)]),
    dict(k="kimi", nm="KIMI", org="moonshot", col=kit.hx("9FD8FF"),
         st=[("SYNTAX", 86), ("WARMTH", 91), ("CHAOS", 19)]),
    dict(k="grok", nm="GROK", org="x.ai", col=kit.GOLD,
         st=[("SYNTAX", 88), ("WARMTH", 52), ("CHAOS", 97)]),
]
_SEQ = [3, 1, 4, 0, 2, 2, 2, 2, 2, 2]

_CARDC = {}


def _card_panel(idx, sel):
    """卡片静态图层（描边/编号/名字/属性条底），按 (S, idx, sel) 缓存。"""
    key = (kit.S, idx, sel)
    p = _CARDC.get(key)
    if p is not None:
        return p
    cd = _CARDS[idx]
    w, h = _CARD_W, _CARD_H
    p = _panel(w, h)
    hh, ww = p.shape[:2]
    ys = np.linspace(0, 1, hh, dtype=F32)[:, None, None]
    top = kit.mixc(kit.hx("081522"), cd["col"], 0.14 if sel else 0.09)
    bot = kit.hx("02070C")
    p[..., :3] = top * (1 - ys) + bot * ys
    p[..., 3] = 1.0
    pad = 16.0
    # ---- 立绘窗口
    wx0, wy0, wx1, wy1 = pad, 54.0, w - pad, 430.0
    _prect(p, wx0, wy0, wx1, wy1, kit.mixc(kit.hx("04101A"), cd["col"], 0.10))
    yy, xx = np.mgrid[0:hh, 0:ww].astype(F32)
    cx = (wx0 + wx1) * 0.5 * kit.S; cy = (wy0 + wy1) * 0.60 * kit.S
    r = np.sqrt(((xx - cx) / (ww * 0.44)) ** 2 + ((yy - cy) / (hh * 0.44)) ** 2)
    g = np.clip(1 - r, 0, 1) ** 2.3
    p[..., :3] += (g * 0.13)[..., None] * np.asarray(cd["col"], F32)
    spr = kit.sprite(stage.AI[cd["k"]], 376.0)
    _psprite(p, spr, (wx0 + wx1) * 0.5, wy1 + 6, 1.0, anchor="foot")
    # 窗口底部渐隐进卡片（逐条横带）
    for j in range(0, 96, 3):
        a = 1.0 - (1.0 - j / 96.0) ** 1.7
        _prect(p, wx0, wy1 - 96 + j, wx1, wy1 - 96 + j + 3, bot, a * 0.92)
    _pframe(p, wx0, wy0, wx1, wy1, 1.0, cd["col"], 0.34)

    # ---- 头部（同一基线：编号 14 / 组织 20 / 分隔线 46）
    _ptext(p, "%02d" % (idx + 1), pad, 14, 22, cd["col"], 1.0, 1.0, FONT_MB)
    _ptext(p, cd["org"], w - pad, 20, 11, kit.PALE, 0.55, 1.2, FONT_M, "right")
    _prect(p, pad, 46, w - pad, 47, cd["col"], 0.30)

    # ---- 名字（统一 y=442）
    _ptext(p, cd["nm"], pad, 442, 28, cd["col"] if sel else kit.CREAM, 1.0, 2.2, FONT_MB)
    _ptext(p, "// v1.0  ready", pad, 478, 11, kit.PALE, 0.40, 1.0, FONT_M)

    # ---- 三条属性（统一行距 26，条底 y+14..y+17）
    for j, (lab, val) in enumerate(cd["st"]):
        yy0 = 506 + j * 26
        _ptext(p, lab, pad, yy0, 11, kit.PALE, 0.62, 1.3, FONT_M)
        _ptext(p, "%3d" % val, w - pad, yy0 - 2, 12, cd["col"], 0.95, 0.8, FONT_M, "right")
        _prect(p, pad, yy0 + 14, w - pad, yy0 + 17, kit.ASH, 0.16)
        fw = (w - 2 * pad) * (val / 100.0)
        _prect(p, pad, yy0 + 14, pad + fw, yy0 + 17, cd["col"], 1.0)
        _prect(p, pad + fw - 1.2, yy0 + 13, pad + fw, yy0 + 18, kit.WHITE, 0.6)

    # ---- 边框
    if sel:
        _pframe(p, 0.5, 0.5, w - 0.5, h - 0.5, 2.0, kit.GOLD, 0.98)
        _pframe(p, 3.5, 3.5, w - 3.5, h - 3.5, 1.0, kit.GOLD, 0.32)
        ln, th = 24.0, 2.0
        for (ax, ay, sx, sy) in ((4, 4, 1, 1), (w - 4, 4, -1, 1), (4, h - 4, 1, -1),
                                 (w - 4, h - 4, -1, -1)):
            _prect(p, min(ax, ax + sx * ln), ay - (0 if sy > 0 else th),
                   max(ax, ax + sx * ln), ay + (th if sy > 0 else 0), kit.GOLD, 1.0)
            _prect(p, ax - (0 if sx > 0 else th), min(ay, ay + sy * ln),
                   ax + (th if sx > 0 else 0), max(ay, ay + sy * ln), kit.GOLD, 1.0)
    else:
        _pframe(p, 0.5, 0.5, w - 0.5, h - 0.5, 1.0, kit.mixc(cd["col"], kit.BLACK, 0.55), 0.55)
    _CARDC[key] = p
    return p


@scene(96.44, 99.91, "ROLE", "F")
def role(c):
    t = c.t
    f = c.f
    lt = t - T_ROLE
    c.post.update(grain=0.015, vignette=0.46, bloom=(0.56, 16, 0.50), chroma=0.12)
    _scrim(f, 0.76, 0.5)

    ph = (t - SEP3) / mus.BEAT
    bi = int(math.floor(ph))
    frac = ph - bi
    i = min(len(_SEQ) - 1, max(0, bi))
    cur = _SEQ[i]
    prev = _SEQ[min(len(_SEQ) - 1, max(0, bi - 1))]
    mv = kit.ease_out_quint(kit.cl(frac / 0.34))
    gsel = kit.ease_out_back(kit.cl(frac / 0.30)) if bi > 0 else 1.0
    pulse = c.pulse(10.0)

    # ---------- 背景：虚化宫殿
    stage.bg_image(f, stage.BG_PALACE, zoom=1.62 + 0.03 * math.sin(lt * 0.5), dy=-10,
                   dim=0.86, exposure=1.32, blur_r=16, tint=kit.hx("0A2036"), tint_k=0.42)
    f *= 0.86
    stage.caustics(f, t, color=kit.VIOLET, strength=0.055, scale=1.5, speed=0.22, y0=0.42)
    kit.add_radial(f, kit.W * 0.5, kit.H * 0.50, 1000, kit.hx("16294A"), 0.30, 1.7, 1.6)
    kit.draw_text(f, "character select", kit.W * 0.5, kit.H * 0.235, 132,
                  kit.mixc(kit.VIOLET, kit.BLACK, 0.42), kit.F_SERIF_I, 0.20, 8.0, "center")
    # 卡片脚下的地台光
    colsel = _CARDS[cur]["col"]
    for j in range(14):
        yy = _CARD_Y0 + _CARD_H + 4 + j * 5.0
        aa = (1 - j / 14.0) ** 1.8 * (0.16 + 0.07 * pulse)
        _rect(f, 96 + j * 7, yy, kit.W - 96 - j * 7, yy + 5, colsel, aa)

    # ---------- 标题
    kit.draw_text(f, "CHARACTER SELECT", 84, 92, 30, kit.CREAM, FONT_MB, 0.92, 9.0)
    kit.draw_text(f, "05 available  //  world.execute(me)", kit.W - 84, 100, 14, kit.MINT,
                  FONT_M, 0.58, 1.4, "right")
    _rect(f, 84, 146, kit.W - 84, 147.4, kit.ASH, 0.24)
    _rect(f, 84, 146, 84 + 168, 147.4, kit.GOLD, 0.9)

    # ---------- 五张卡（静态图层缓存）
    centers = []
    for i in range(_CARD_N):
        x0 = _CARD_X0 + i * (_CARD_W + _CARD_GAP)
        cx = x0 + _CARD_W * 0.5
        cy = _CARD_Y0 + _CARD_H * 0.5
        sel = (i == cur)
        kk = 1.0 + (0.055 * gsel if sel else 0.0)
        rot = (-0.030 * gsel) if sel else 0.0
        p = _card_panel(i, sel)
        _pblit(f, p, cx, cy, kk, rot, 1.0 if sel else 0.9)
        centers.append(cx)

    # ---------- 选中框 / 光标
    scx = centers[cur]
    hw = _CARD_W * 0.5 * 1.055
    _corner(f, scx - hw + 3, _CARD_Y0 + 40, scx + hw - 3, _CARD_Y0 + 442, 26, 2.0, kit.GOLD,
            0.55 + 0.4 * pulse)
    tw, th = 96.0, 22.0
    tx = scx - tw * 0.5
    ty = _CARD_Y0 - 32.0
    _rect(f, tx, ty, tx + tw, ty + th, kit.GOLD, 0.96)
    kit.draw_text(f, "SELECTED", tx + 11, ty + 5, 11, kit.hx("1A1206"), FONT_MB, 1.0, 1.6)

    px = centers[prev] + (scx - centers[prev]) * mv
    # 游标：一根竖直光条 + 上下小方块（选中卡右侧），比一个白点更像「指针」
    cxr = px + hw + 13
    _glowline(f, cxr, _CARD_Y0 + 30, cxr + 2.6, _CARD_Y0 + 30 + 78, kit.GOLD,
              0.85 + 0.15 * pulse, 3)
    _rect(f, cxr - 3, _CARD_Y0 + 24, cxr + 6, _CARD_Y0 + 28, kit.GOLD, 0.9)
    _rect(f, cxr - 3, _CARD_Y0 + 110, cxr + 6, _CARD_Y0 + 114, kit.GOLD, 0.9)
    if abs(px - scx) > 3:
        stage.cursor_trail(f, [(px + hw + 14, _CARD_Y0 + 70), (scx + hw + 14, _CARD_Y0 + 70)],
                           kit.GOLD, 0.3, 1.4)

    # ---------- 页脚
    _rect(f, 84, 862, kit.W - 84, 863, kit.ASH, 0.16)
    kit.draw_text(f, "ARROWS  MOVE      ENTER  CONFIRM      ESC  BACK", 84, 878, 14, kit.ASH,
                  FONT_M, 0.72, 1.4)
    kit.draw_text(f, "%d / 5" % (cur + 1), kit.W - 84, 878, 14, kit.GOLD, FONT_M, 0.85, 1.4,
                  "right")

    # ---------- 确认：白闪交接
    if t > T_TRANCE - 0.34:
        a = kit.cl((t - (T_TRANCE - 0.34)) / 0.30)
        f += kit.WHITE * (a ** 2) * 0.9

    c.lyric(size=40, y=kit.H * 0.905, alpha=0.92, glow=0.2, color=kit.CREAM)


# ================================================================ 4 · TRANCE
_TUN = {}
_TUNR = {}


def _tunnel_pts(n=62000):
    p = _TUN.get(n)
    if p is None:
        rs = np.random.default_rng(11)
        p = cloud.shape_tunnel(n, rs, 320.0, 1700.0)
        # 70% 贴在管壁上，30% 填进管内：轴线附近才不会空成一个洞
        rr = np.random.default_rng(23)
        fac = np.ones(n, F32)
        fill = rr.random(n) < 0.34
        fac[fill] = (0.16 + 0.68 * rr.random(int(fill.sum()))).astype(F32)
        _TUN[n] = p
        _TUNR[n] = fac
    return p, _TUNR[n]

def _rings(f, lt, amp, colA, colB, cx=None):
    """程序化隧道：沿 z 推进的同心环 + 缓慢旋转的辐条（半分辨率画完再放大）。
    环用透视幂律收在中心附近；辐条让「筒」这个形体立起来。"""
    bw, bh = kit.SW // 2, kit.SH // 2
    ccx, ccy = bw * 0.5, bh * 0.46
    if cx is not None:
        ccx = cx * bw / kit.W
    mA = np.zeros((bh, bw), F32)
    mB = np.zeros((bh, bw), F32)
    maxR = math.hypot(bw * 0.5, bh * 0.5) * 1.02
    # 辐条：从内到外的细线，整层随 lt 旋转
    for s in range(16):
        aa = lt * 0.85 + s * 2.0 * math.pi / 16.0
        a0 = 0.10 + 0.05 * (s % 2)
        cv2.line(mA, (int(ccx + math.cos(aa) * 26), int(ccy + math.sin(aa) * 26)),
                 (int(ccx + math.cos(aa) * maxR), int(ccy + math.sin(aa) * maxR)),
                 float(a0), 1, cv2.LINE_AA)
    # 环：16 个，半径按透视幂律，亮度随推进衰减
    for i in range(16):
        p = (lt * 0.52 + i / 16.0) % 1.0
        r = 6.0 + maxR * (p ** 2.7)
        a = (min(1.0, p * 7.0) ** 0.7) * (1.0 - p) ** 1.25
        if a <= 0.02:
            continue
        tgt = mB if (i % 3 == 0) else mA
        cv2.circle(tgt, (int(ccx), int(ccy)), int(r), float(a * 1.15),
                   max(1, int(1 + 3.2 * p)), cv2.LINE_AA)
    f += kit.up(mA, kit.SW, kit.SH)[..., None] * np.asarray(colA, F32) * (1.9 * amp)
    f += kit.up(mB, kit.SW, kit.SH)[..., None] * np.asarray(colB, F32) * (2.1 * amp)


def _speed(f, amt, col, n=120, seed=5):
    bw, bh = kit.SW // 2, kit.SH // 2
    m = np.zeros((bh, bw), F32)
    rs = np.random.default_rng(seed)
    cx, cy = bw * 0.5, bh * 0.46
    for i in range(n):
        a0 = rs.random() * 6.283
        r0 = (0.05 + 0.62 * rs.random() ** 2.2) * bw
        r1 = r0 + (0.05 + 0.5 * rs.random() ** 2) * bw
        cv2.line(m, (int(cx + math.cos(a0) * r0), int(cy + math.sin(a0) * r0)),
                 (int(cx + math.cos(a0) * r1), int(cy + math.sin(a0) * r1)),
                 float(0.35 + 0.65 * rs.random()), 1, cv2.LINE_AA)
    f += kit.up(m, kit.SW, kit.SH)[..., None] * np.asarray(col, F32) * amt


@scene(99.91, 103.84, "TRANCE", "F")
def trance(c):
    t = c.t
    f = c.f
    lt = t - T_TRANCE
    p = kit.cl(lt / (T_END - T_TRANCE))
    wf = kit.cl((t - (T_END - 0.42)) / 0.40)
    c.post.update(grain=0.022, vignette=0.58, bloom=(0.66, 22, 0.46),
                  chroma=0.8 * kit.cl(p * 1.5))

    # 1/8 音符频闪
    e8 = int(math.floor(lt / (mus.BEAT * 0.5)))
    st8 = float(e8 % 2)
    strobe = 1.0 + 0.26 * st8 + 0.22 * math.exp(-(lt % (mus.BEAT * 0.5)) * 16.0)

    f += kit.vgrad(kit.SH, kit.SW, kit.hx("0C0724"), kit.hx("01030A"), 1.4)
    cx0, cy0 = kit.W * 0.5, kit.H * 0.46
    kit.add_radial(f, cx0, cy0, 520, kit.VIOLET, 0.075 + 0.05 * st8, 2.2, 1.3)

    # ---------- 粒子隧道（密度向远端偏置 + 三成填进管内，避免中心塌成一个洞）
    P0, RF = _tunnel_pts()
    P = P0.copy()
    uu = np.mod((P0[:, 2] + 595.0) / 1700.0 + lt * (0.40 + 0.16 * p), 1.0)
    P[:, 2] = 1700.0 * (uu ** 0.55) - 520.0
    P[:, 0] *= RF
    P[:, 1] *= RF
    P = cloud.Cloud.transform(P, roll=lt * 0.60 + 0.30 * math.sin(lt * 0.8),
                              yaw=0.05 * math.sin(lt * 1.1))

    bw, bh = kit.SW // 2, kit.SH // 2
    bcx, bcy = bw * 0.5, bh * 0.46

    def cm(xs, ys, zz, kk):
        t01 = np.clip((kk - 1.1) / 4.2, 0, 1)
        col = np.asarray(kit.VIOLET, F32)[None, :] * (1 - t01)[:, None] + \
            np.asarray(kit.GOLD, F32)[None, :] * t01[:, None]
        # 螺旋带：绕视线的一条慢速旋臂，让粒子读出「筒」而不是「星野」
        ang = np.arctan2(ys - bcy, xs - bcx)
        rad = np.hypot(xs - bcx, ys - bcy) + 8.0
        band = 0.42 + 0.58 * (0.5 + 0.5 * np.cos(5.0 * ang + 7.0 * np.log(rad) - lt * 2.6))
        return col * band[:, None]

    c.cl.draw(f, P, dist=560.0, focal=1150.0, size=1.5, blur_r=1.1, glow_gain=0.4,
              color_map=cm, bright=(0.22 + 0.9 * c.cl.rnd), target=(0.30 + 0.10 * st8) * strobe,
              core_gain=1.0, tm=2.0, cutoff=2600.0, shade=0.9)

    # ---------- 隧道主体：推进的同心环 + 旋转辐条（在粒子之上）
    _rings(f, lt, strobe, kit.VIOLET, kit.GOLD)
    # 尽头的漩涡核
    kit.add_radial(f, cx0, cy0, 34 + 6 * math.sin(lt * 7.0), kit.GOLD,
                   0.30 + 0.18 * st8, 2.6, 1.0)

    # ---------- 她在尽头越来越小（高度量化，避免每帧新建精灵缓存）
    hs = 6.0 * round((260.0 * (1.0 - 0.58 * kit.ease_in(kit.cl(p * 1.05))) + 92.0) / 6.0)
    hy = cy0 + hs * 0.52
    stage.character(f, stage.MAID_L, cx0, hy, hs, alpha=0.94,
                    tint=kit.mixc(kit.VIOLET, kit.hx("1A1030"), 0.5), tint_k=0.42,
                    rim=kit.mixc(kit.GOLD, kit.CREAM, 0.35), rim_k=0.55, glow=0.0, shadow=0.0)

    # ---------- 速度线
    sp = kit.cl((p - 0.34) / 0.66)
    if sp > 0:
        _speed(f, sp * (0.45 + 0.35 * st8), kit.mixc(kit.CREAM, kit.GOLD, 0.35), n=110, seed=5)

    # ---------- 频闪
    f *= 0.94 + 0.16 * st8

    # ---------- 结尾白闪
    if wf > 0:
        f *= (1 - wf ** 1.7)
        f += kit.WHITE * (wf ** 2.1) * 1.1

    c.lyric(size=44, y=kit.H * 0.858, alpha=0.9 * (1 - 0.7 * wf), glow=0.25,
            color=kit.mixc(kit.CREAM, kit.GOLD, 0.35))
    stage.hud_corner(f, t, "// enter the trance", "F%05d" % int(round(t * 60)), 0.34,
                     kit.VIOLET, y=kit.H - 42)


# ---------------------------------------------------------------- 预热
# 卡片静态图层是 5 张 × 2 种状态；在导入时一次建好，
# 免得 ROLE 的第一帧为了画五张卡突然多背 0.7s。
_WARMED = set()


def _warm():
    s = kit.S
    if s in _WARMED:
        return
    for _i in range(_CARD_N):
        for _st in (False, True):
            _card_panel(_i, _st)
    for _k in list(stage.AI):
        kit.sprite(stage.AI[_k], 376.0)
    # 本章会用到的立绘高度（与场景里的常量保持一致）
    for _p, _h in ((stage.MAID_L, 690.0), (stage.MAID_L, 880.0),
                   (stage.MAID_START_L, 340.0), (stage.MAID_L, 344.0),
                   (stage.MAID_Q, 262.0), (stage.MAID_R2, 500.0),
                   (stage.MAID_R, 330.0), (START_R, 320.0)):
        kit.sprite(_p, _h)
    _WARMED.add(s)


try:
    _warm()
except Exception:
    pass

