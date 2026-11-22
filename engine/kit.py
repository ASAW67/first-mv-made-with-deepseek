# -*- coding: utf-8 -*-
"""kit.py — MV 引擎底座：分辨率无关的画布、调色板、缓动、精灵、发光、后期。
所有坐标都是「逻辑坐标」(1920×1080)，内部按 scale 放大。渲染函数必须是 t 的纯函数。
"""
import os, math, json
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw, ImageFilter

cv2.setNumThreads(1)

ROOT = os.path.dirname(os.path.abspath(__file__))


def _find_ffmpeg():
    """优先 MV_FFMPEG 环境变量，其次仓库内 .tools/，最后 PATH 上的 ffmpeg。"""
    env = os.environ.get("MV_FFMPEG")
    if env and os.path.exists(env):
        return env
    local = os.path.join(os.path.dirname(ROOT), ".tools")
    if os.path.isdir(local):
        for d, _, files in os.walk(local):
            for f in files:
                if f in ("ffmpeg.exe", "ffmpeg"):
                    return os.path.join(d, f)
    return "ffmpeg"


# 素材根目录：MV_ASSETS 指到你自己的 `角色/` 文件夹（默认与本仓库同级）
ASSETS = os.environ.get("MV_ASSETS",
                        os.path.join(os.path.dirname(ROOT), "角色"))
FFMPEG = _find_ffmpeg()

# ---------------------------------------------------------------- 全局画布状态
W, H, S = 1920, 1080, 1
SW, SH = W, H
F32 = np.float32


def init(scale=1, w=1920, h=1080):
    global W, H, S, SW, SH
    S = int(scale); W, H = w, h
    SW, SH = W * S, H * S
    return W, H, S


# ---------------------------------------------------------------- 颜色
def hx(s):
    s = s.lstrip("#")
    return np.array([int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], F32)


NAVY = hx("0A1622"); DEEP = hx("050B12"); INK = hx("111A24")
CORAL = hx("FF7A6B"); MINT = hx("7FE3D4"); BLUE = hx("4C7BE8")
PALE = hx("A8C4E8"); GOLD = hx("E8B27A"); VIOLET = hx("B79CFF"); ROSE = hx("FF6FA5")
WHITE = np.array([1, 1, 1], F32); BLACK = np.zeros(3, F32)
TEAL = hx("5FD0C6"); ICE = hx("DCEBF7"); SAND = hx("E7CFAE"); RUST = hx("C9503C")
CREAM = hx("F3E7D6"); PLUM = hx("6B3F63"); ASH = hx("8894A3")

# ---------------------------------------------------------------- 数值工具
def cl(x, a=0.0, b=1.0):
    return a if x < a else (b if x > b else x)


def lerp(a, b, t):
    return a + (b - a) * t


def mixc(c1, c2, t):
    c1 = np.asarray(c1, F32); c2 = np.asarray(c2, F32)
    tt = np.asarray(t, F32)
    if tt.ndim:
        tt = tt[..., None]
    return c1 + (c2 - c1) * tt


def remap(x, a, b, c=0.0, d=1.0):
    if b == a:
        return c
    return c + (d - c) * cl((x - a) / (b - a))


def smoothstep(a, b, x):
    t = cl((x - a) / (b - a if b != a else 1e-9))
    return t * t * (3 - 2 * t)


def env(t, t0, t1, fin=0.4, fout=0.4):
    """梯形包络：t0..t1 之间为 1，两端斜坡。"""
    if t <= t0 or t >= t1:
        return 0.0
    return min(1.0, (t - t0) / max(fin, 1e-6), (t1 - t) / max(fout, 1e-6))


def ease_io(t):
    t = cl(t); return t * t * (3 - 2 * t)


def ease_out(t):
    t = cl(t); return 1 - (1 - t) ** 3


def ease_in(t):
    t = cl(t); return t ** 3


def ease_out_back(t, s=1.9):
    t = cl(t); u = t - 1
    return 1 + u * u * ((s + 1) * u + s) * 0.5


def ease_expo_out(t):
    t = cl(t); return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def ease_out_quint(t):
    t = cl(t); return 1 - (1 - t) ** 5


def ease_in_out_expo(t):
    t = cl(t)
    return (2 ** (20 * t - 10) / 2) if t > 0.5 else (2 ** (20 * t - 10) if t > 0 else 0)


def bezier(p0, p1, p2, p3, t):
    u = 1 - t
    return u ** 3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3 * p3


def rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], F32)


# ---------------------------------------------------------------- 图像基础
def new_frame(color=BLACK):
    f = np.empty((SH, SW, 3), F32)
    f[:] = color
    return f


def canvas(color=BLACK):
    return new_frame(color)


def solid(color, h=H, w=W, scale=False):
    if scale:
        h, w = h * S, w * S
    f = np.empty((int(h), int(w), 3), F32); f[:] = color
    return f


def up(img, w, h):
    """放大/缩小到 (w,h)，float32 三/单通道。"""
    if img.shape[1] == w and img.shape[0] == h:
        return img
    it = cv2.INTER_AREA if (img.shape[1] > w) else cv2.INTER_LINEAR
    r = cv2.resize(img, (int(w), int(h)), interpolation=it)
    return r if r.ndim == img.ndim else r[..., None]


def blur(img, r):
    if r < 0.4:
        return img
    return cv2.GaussianBlur(img, (0, 0), r, borderType=cv2.BORDER_REFLECT)


def boxblur(img, r, passes=2):
    if r < 0.6:
        return img
    k = int(r * 2) | 1
    o = img
    for _ in range(passes):
        o = cv2.blur(o, (k, k), borderType=cv2.BORDER_REFLECT)
    return o


def shift(img, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def scale_about(img, cx, cy, k, angle=0.0):
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), -math.degrees(angle), float(k))
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def add(a, b, k=1.0):
    """加法合成（原地改 a）。"""
    if isinstance(b, (int, float)):
        a += b * k
        return a
    if b.ndim == 2:
        b = b[..., None]
    a += b * k
    return a


def over(dst, rgb, alpha, tint=None):
    """alpha 合成：dst = dst*(1-a) + rgb*a。rgb 可为常量或图。alpha 可为标量或 HxW。"""
    if np.isscalar(alpha):
        if alpha <= 0:
            return dst
        a = alpha
    else:
        a = alpha
        if a.ndim == 2:
            a = a[..., None]
    src = rgb if rgb.ndim == 3 else np.broadcast_to(rgb, dst.shape)
    if a is not None and not np.isscalar(a) and a.ndim == 3 and a.shape[2] == 1:
        dst *= (1 - a); dst += src * a
        return dst
    dst *= (1 - a); dst += src * a
    return dst


# ---------------------------------------------------------------- 渐变 / 光
def vgrad(h, w, c0, c1, power=1.0):
    t = (np.linspace(0, 1, h, dtype=F32) ** power)[:, None, None]
    return (c0[None, None, :] * (1 - t) + c1[None, None, :] * t).repeat(w, 1).astype(F32)


def radial_mask(h, w, cx, cy, rx, ry=None, power=2.0):
    ry = rx if ry is None else ry
    y = (np.arange(h, dtype=F32) - cy) / max(ry, 1e-3)
    x = (np.arange(w, dtype=F32) - cx) / max(rx, 1e-3)
    r = np.sqrt(x[None, :] ** 2 + y[:, None] ** 2)
    return np.clip(1.0 - r, 0, 1) ** power


def add_radial(f, cx, cy, radius, color, strength=1.0, power=2.2, ratio=1.0):
    cx, cy = cx * S, cy * S
    r = radius * S
    rx = max(int(r * ratio), 2); ry = max(int(r), 2)
    x0, x1 = int(cx - rx), int(cx + rx); y0, y1 = int(cy - ry), int(cy + ry)
    if x1 <= 0 or y1 <= 0 or x0 >= SW or y0 >= SH:
        return f
    xa, xb = max(0, x0), min(SW, x1); ya, yb = max(0, y0), min(SH, y1)
    m = radial_mask(yb - ya, xb - xa, cx - xa, cy - ya, rx, ry, power)
    f[ya:yb, xa:xb] += m[..., None] * (np.asarray(color, F32) * strength)
    return f


def light_shaft(f, x0, y0, x1, y1, width, color, strength, softness=0.5):
    """一束斜向光柱（用带软边的旋转矩形）。"""
    m = np.zeros((SH, SW), F32)
    pts = np.float32([[x0 * S, y0 * S], [x1 * S, y1 * S]])
    c = int(round(255 * cl(strength)))
    cv2.line(m, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), float(c),
             max(1, int(width * S)), cv2.LINE_AA)
    m = blur(m, width * S * softness / 2.0)
    f += m[..., None] * (np.asarray(color, F32)) * (strength / 255.0 * 3.0)
    return f


# ---------------------------------------------------------------- 图片素材
_IMG = {}


_MISSING = set()


def load(path):
    if path in _IMG:
        return _IMG[path]
    if path in _MISSING or not os.path.exists(path):
        # 素材缺失时不炸：返回 1×1 全透明，调用方照常画，只是「她」不在
        if path not in _MISSING:
            _MISSING.add(path)
            print("[kit] missing asset (rendered as empty): %s" % path)
        a = np.zeros((1, 1, 4), F32)
        _IMG[path] = a
        return a
    im = Image.open(path)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    a = np.asarray(im, dtype=F32) / 255.0
    _IMG[path] = a
    return a


def load_rgb(path):
    a = load(path)
    return a[..., :3], a[..., 3]


def trim(a):
    al = a[..., 3]
    ys, xs = np.where(al > 0.02)
    if len(xs) == 0:
        return a
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


_SPR = {}


def sprite(path, height=None, key=None):
    """按「物理高度(逻辑像素)」缩放好的 RGBA，命中缓存。"""
    k = (path, height, key)
    if k in _SPR:
        return _SPR[k]
    a = trim(load(path)) if key != "notrim" else load(path)
    if height:
        hh = int(round(height * S)); ww = max(1, int(round(a.shape[1] * hh / a.shape[0])))
        a = up(a, ww, hh)
    _SPR[k] = a
    return a


def place(f, spr, cx, cy, k=1.0, alpha=1.0, tint=None, tint_k=0.0, glow=0.0,
          additive=False, flip=False, rot=0.0, anchor="foot"):
    """把精灵放到逻辑坐标 (cx,cy)。anchor: foot=脚底中心, center, top。"""
    if spr is None or alpha <= 0.003:
        return f
    a = spr
    if abs(k - 1.0) > 1e-3:
        hh = max(1, int(round(a.shape[0] * k))); ww = max(1, int(round(a.shape[1] * k)))
        a = up(a, ww, hh)
    if flip:
        a = a[:, ::-1]
    if rot:
        h0, w0 = a.shape[:2]
        big = int(math.hypot(h0, w0)) + 2
        pad = np.zeros((big, big, 4), F32)
        y0 = (big - h0) // 2; x0 = (big - w0) // 2
        pad[y0:y0 + h0, x0:x0 + w0] = a
        M = cv2.getRotationMatrix2D((big / 2, big / 2), -math.degrees(rot), 1.0)
        a = cv2.warpAffine(pad, M, (big, big), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    hh, ww = a.shape[:2]
    if anchor == "center":
        x0 = int(round(cx * S - ww / 2)); y0 = int(round(cy * S - hh / 2))
    elif anchor == "top":
        x0 = int(round(cx * S - ww / 2)); y0 = int(round(cy * S))
    else:
        x0 = int(round(cx * S - ww / 2)); y0 = int(round(cy * S - hh))
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(SW, x0 + ww), min(SH, y0 + hh)
    if xe <= xs or ye <= ys:
        return f
    sub = a[ys - y0:ye - y0, xs - x0:xe - x0]
    rgb = sub[..., :3]; al = sub[..., 3] * alpha
    if tint is not None and tint_k > 0:
        rgb = rgb * (1 - tint_k) + np.asarray(tint, F32)[None, None, :] * tint_k
    if glow > 0:
        # 只让「她自己亮的地方」发光：按像素亮度加权，避免整身轮廓发白光斑
        lum = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114) * al
        g = np.zeros((SH, SW), F32)
        g[ys:ye, xs:xe] = lum
        sig = max(2.0, (6.0 + 8.0 * glow) * S)
        gb = blur(g, sig)[ys:ye, xs:xe]
        gtint = np.asarray(tint, F32) if tint is not None else np.array([0.75, 0.95, 1.0], F32)
        f[ys:ye, xs:xe] += gb[..., None] * gtint[None, None, :] * (glow * 1.1)
    if additive:
        f[ys:ye, xs:xe] += rgb * al[..., None]
    else:
        d = f[ys:ye, xs:xe]
        d *= (1 - al[..., None]); d += rgb * al[..., None]
    return f


# ---------------------------------------------------------------- 后期
def vignette(f, strength=0.55, power=2.4, tint=None):
    m = radial_mask(SH, SW, SW / 2, SH / 2, SW * 0.78, SH * 0.86, power)
    w = (1 - m) * strength
    if tint is not None:
        f *= (1 - w[..., None]); f += np.asarray(tint, F32)[None, None, :] * w[..., None]
    else:
        f *= (1 - w[..., None])
    return f


def grain(f, amt=0.035, seed=0):
    if amt <= 0:
        return f
    rs = np.random.default_rng(seed)
    g = rs.standard_normal((SH // 2 + 1, SW // 2 + 1)).astype(F32)
    g = up(g, SW, SH)
    f += (g * amt)[..., None]
    return f


def scanlines(f, amt=0.05, period=3 * 1):
    if amt <= 0:
        return f
    y = (np.arange(SH) % max(1, period * S)) < max(1, period * S / 2)
    m = np.where(y, 1.0 - amt, 1.0).astype(F32)
    f *= m[:, None, None]
    return f


def chroma(f, amount=2.0, angle=0.0):
    """RGB 分离（chromatic aberration）。amount 单位逻辑像素。"""
    if amount < 0.15:
        return f
    dx = math.cos(angle) * amount * S; dy = math.sin(angle) * amount * S
    r = shift(f[..., 0], dx, dy); b = shift(f[..., 2], -dx, -dy)
    o = f.copy(); o[..., 0] = r; o[..., 2] = b
    return o


def glitch_rows(f, t, amt=0.5, seed=0, band=14):
    """按行切块横向错位（确定性）。"""
    if amt <= 0.01:
        return f
    rs = np.random.default_rng(int(t * 997) + seed)
    o = f
    n = int(6 + 18 * amt)
    for _ in range(n):
        y = int(rs.integers(0, SH - 1))
        hh = int(rs.integers(2, band * S + 2))
        dx = int(rs.normal(0, 40 * amt * S))
        y2 = min(SH, y + hh)
        o[y:y2] = np.roll(o[y:y2], dx, axis=1)
    return o


def bloom(f, thresh=0.62, radius=26, strength=0.55, downscale=3):
    """亮部溢出。用半分辨率加速。"""
    small = up(f, SW // downscale, SH // downscale)
    lum = small.max(axis=2, keepdims=True)
    bright = np.clip((small - thresh) / (1 - thresh + 1e-6), 0, 2) * (lum > thresh * 0.6)
    b = blur(bright, radius * S / downscale / 2.0)
    b = up(b, SW, SH)
    return f + b * strength


def tone(f, exposure=1.0, contrast=1.0, sat=1.0, lift=0.0, gamma=1.0):
    o = f * exposure
    if contrast != 1.0:
        o = (o - 0.5) * contrast + 0.5
    if sat != 1.0:
        l = o.mean(axis=2, keepdims=True)
        o = l + (o - l) * sat
    if gamma != 1.0:
        o = np.clip(o, 0, None) ** gamma
    if lift:
        o = o + lift
    return o


def to_u8(f):
    a = np.clip(f, 0, 1)
    # 轻微胶片曲线：暗部不死黑
    a = a ** (1 / 1.02)
    return (a * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- 字体
_FONT = {}


def font(path, size):
    k = (path, size)
    if k not in _FONT:
        _FONT[k] = ImageFont.truetype(path, size)
    return _FONT[k]


# 字体：默认取 Windows 系统字体目录，可用 MV_FONTS 指向别的目录（Linux/macOS 上请自行放同名文件）
FONTS = os.environ.get("MV_FONTS", os.path.join("C:\\Windows\\Fonts"))


def _f(name):
    p = os.path.join(FONTS, name)
    return p if os.path.exists(p) else name


F_SERIF_I = _f("georgiai.ttf")
F_SERIF = _f("georgia.ttf")
F_SERIF_B = _f("georgiab.ttf")
F_MONO = _f("consola.ttf")
F_MONO_B = _f("consolab.ttf")
F_SANS_SC = _f("NotoSansSC-VF.ttf")
F_SERIF_SC = _f("NotoSerifSC-VF.ttf")
F_HEI = _f("simhei.ttf")
F_BAHN = _f("bahnschrift.ttf")


_GL = {}


def _glyph(fpath, size_px, ch):
    """渲染单字，返回 (alpha, dx, dy, advance)。dx/dy 是相对笔位的 ink 偏移（像素）。"""
    key = (fpath, size_px, ch)
    g = _GL.get(key)
    if g is not None:
        return g
    f = font(fpath, size_px)
    adv = f.getlength(ch)
    if not ch.strip():
        g = (np.zeros((1, 1), F32), 0.0, 0.0, adv)
        _GL[key] = g
        return g
    bw = int(size_px * 2.4) + 16
    bh = int(size_px * 2.6) + 16
    im = Image.new("L", (bw, bh), 0)
    d = ImageDraw.Draw(im)
    ox = oy = size_px * 0.75
    d.text((ox, oy), ch, font=f, fill=255)
    a = np.asarray(im, dtype=F32) / 255.0
    ys, xs = np.nonzero(a > 0.004)
    if len(xs) == 0:
        g = (np.zeros((1, 1), F32), 0.0, 0.0, adv)
        _GL[key] = g
        return g
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    g = (a[y0:y1, x0:x1], x0 - ox, y0 - oy, adv)
    _GL[key] = g
    return g


def _layout(fpath, size, txt, tracking=0.0):
    """返回 [(rel_x_px, dp), ...], 总宽(逻辑px), 行高(逻辑px)。"""
    size_px = max(4, int(round(size * S)))
    items = []
    x = 0.0
    for ch in txt:
        a, dx, dy, adv = _glyph(fpath, size_px, ch)
        items.append((x + dx, a, dy))
        x += adv + tracking * S
    f = font(fpath, size_px)
    asc, desc = f.getmetrics()
    return items, x / S, (asc + desc) / S, asc / S


def text_mask(fpath, size, txt, tracking=0.0):
    size_px = max(4, int(round(size * S)))
    items, wl, lh, asc = _layout(fpath, size, txt, tracking)
    wpx = max(2, int(math.ceil(wl * S)) + 2)
    hpx = max(2, int(math.ceil(lh * S)) + 4)
    m = np.zeros((hpx, wpx), F32)
    for dx, a, dy in items:
        xi = int(round(dx)); yi = int(round(dy)) + 2
        h0, w0 = a.shape
        x0 = max(0, xi); y0 = max(0, yi)
        x1 = min(wpx, xi + w0); y1 = min(hpx, yi + h0)
        if x1 <= x0 or y1 <= y0:
            continue
        m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], a[y0 - yi:y1 - yi, x0 - xi:x1 - xi])
    return m


def text_width(fpath, size, txt, tracking=0.0):
    return _layout(fpath, size, txt, tracking)[1]


def text_metrics(fpath, size):
    size_px = max(4, int(round(size * S)))
    asc, desc = font(fpath, size_px).getmetrics()
    return asc / S, desc / S


def draw_text(f, txt, x, y, size, color=WHITE, font_path=None, alpha=1.0, tracking=0.0,
              align="left", additive=False, glow=0.0, glow_color=None, clip=None,
              char_alpha=None, char_offset=None, stretch=1.0, shadow=0.0):
    """逻辑坐标绘制。y = 文本行顶部。char_alpha(txt索引, 相对笔位) 可做逐字动画。"""
    if alpha <= 0.004 or not txt:
        return 0
    fp = font_path or F_SERIF
    items, wl, lh, asc = _layout(fp, size, txt, tracking)
    if align == "center":
        x = x - wl / 2
    elif align == "right":
        x = x - wl
    if abs(stretch - 1.0) > 1e-3:
        wl *= stretch
    # 各字单独合成（支持逐字透明度/位移）
    fast = (char_alpha is None and char_offset is None)
    if fast:
        m = text_mask(fp, size, txt, tracking)
        if abs(stretch - 1.0) > 1e-3:
            m = up(m, max(1, int(round(m.shape[1] * stretch))), m.shape[0])
        return _blit_mask(f, m, x, y, color, alpha, glow, glow_color, additive, clip,
                          shadow=shadow, shadow_r=max(1.5, size * 0.075))
    for i, (dx, a, dy) in enumerate(items):
        aa = 1.0 if char_alpha is None else char_alpha(i, dx / S)
        if aa <= 0.004:
            continue
        ox, oy = (0.0, 0.0)
        if char_offset is not None:
            ox, oy = char_offset(i, dx / S)
        xi = int(round((x + dx * stretch / 1.0 + ox * S)))
        yi = int(round((y + dy + oy) * S))
        _blit_mask(f, a, (xi) / S, yi / S, color, alpha * aa, glow, glow_color, additive, clip, px=True,
                   shadow=shadow, shadow_r=max(1.5, size * 0.075))
    return wl


def _blit_mask(f, m, x, y, color, alpha, glow=0.0, glow_color=None, additive=False, clip=None, px=False,
               shadow=0.0, shadow_r=2.4):
    if px:
        xi, yi = int(round(x)), int(round(y))
    else:
        xi, yi = int(round(x * S)), int(round(y * S))
    hh, ww = m.shape
    xs, ys = max(0, xi), max(0, yi)
    xe, ye = min(SW, xi + ww), min(SH, yi + hh)
    if xe <= xs or ye <= ys:
        return 0
    sub = m[ys - yi:ye - yi, xs - xi:xe - xi] * alpha
    if clip is not None:
        c0, c1 = clip
        col = np.arange(xs, xe)
        sub = sub * ((col >= c0 * S) & (col <= c1 * S))[None, :]
    rgb = np.asarray(color, F32)
    d = f[ys:ye, xs:xe]
    if shadow > 0:
        # 文字底部垫一层柔和的暗晕（只作用在字形外侧，避免细笔画的字被自己的阴影吃掉）
        pad = int(max(3, shadow_r * S * 2.5))
        g = np.zeros((sub.shape[0] + pad * 2, sub.shape[1] + pad * 2), F32)
        g[pad:-pad, pad:-pad] = sub
        g = blur(g, max(1.5, shadow_r * S))[pad:-pad, pad:-pad]
        g = np.clip(g - sub, 0.0, 1.0)
        d *= (1.0 - np.clip(g * shadow * 1.6, 0, 0.9))[..., None]
    if glow > 0:
        g = np.zeros((SH, SW), F32); g[ys:ye, xs:xe] = sub
        gb = blur(g, max(2.0, 10 * S * glow))[ys:ye, xs:xe]
        d += gb[..., None] * np.asarray(glow_color if glow_color is not None else rgb, F32) * (2.0 * glow)
    if additive:
        d += sub[..., None] * rgb
    else:
        d *= (1 - sub[..., None]); d += rgb * sub[..., None]
    return ww / S


def text_block(f, lines, x, y, size, lead=None, color=WHITE, font_path=None, alpha=1.0,
               align="left", tracking=0.0, glow=0.0):
    lead = lead or size * 1.42
    for i, ln in enumerate(lines):
        draw_text(f, ln, x, y + i * lead, size, color, font_path, alpha, tracking, align, glow=glow)
    return y + len(lines) * lead


def tonemap_soft(buf, k=1.6):
    """亮度软压缩，保色相：密集粒子不再死白。"""
    lum = buf.mean(axis=2, keepdims=True)
    tl = 1.0 - np.exp(-lum * k)
    s = np.where(lum > 1e-5, tl / np.maximum(lum, 1e-5), 0.0)
    return buf * s


# ---------------------------------------------------------------- 形状 / 遮罩
def poly_mask(pts, value=1.0, close=False):
    m = np.zeros((SH, SW), F32)
    p = (np.asarray(pts, F32) * S).astype(np.int32)
    cv2.fillPoly(m, [p], float(value))
    return m


def line_mask(pts, thick=2.0, aa=True):
    m = np.zeros((SH, SW), F32)
    p = (np.asarray(pts, F32) * S).astype(np.int32)
    cv2.polylines(m, [p], False, 1.0, max(1, int(thick * S)), cv2.LINE_AA if aa else 8)
    return m


def circle_mask(cx, cy, r, soft=1.0):
    m = np.zeros((SH, SW), F32)
    cv2.circle(m, (int(cx * S), int(cy * S)), max(1, int(r * S)), 1.0, -1, cv2.LINE_AA)
    return blur(m, soft * S) if soft > 0.2 else m


def alpha_layer(m, color):
    o = np.zeros((SH, SW, 3), F32)
    o[:] = np.asarray(color, F32)
    return o, m


def mask_to_rgb(m, color):
    o = np.zeros((SH, SW, 3), F32)
    o[:] = np.asarray(color, F32)[None, None, :]
    return o, m
