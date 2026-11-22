# -*- coding: utf-8 -*-
"""stage.py — 舞台元件：宫殿背景 / 角色 / 光标「你」/ 终端 / 歌词排版 / 面板。
所有函数接收逻辑坐标与时间，向 frame 叠加。
"""
import math, os
import numpy as np
import cv2
import kit, mus, cloud

F32 = np.float32
A = kit.ASSETS
W, H, S = kit.W, kit.H, kit.S

BG_PALACE = os.path.join(A, "maid-atelier", "背景", "深海宫殿-暗.webp")
BG_PALACE_L = os.path.join(A, "maid-atelier", "背景", "深海宫殿-亮.webp")
MAID_L = os.path.join(A, "maid-atelier", "01-左女仆-提裙礼.webp")
MAID_R = os.path.join(A, "maid-atelier", "02-右女仆-记事本与笔.webp")
MAID_R2 = os.path.join(A, "maid-atelier", "03-右女仆-单片镜V4FlashVision版.webp")
MAID_START_L = os.path.join(A, "maid-atelier", "04-启动页-左女仆-马甲长裙.webp")
MAID_START_R = os.path.join(A, "maid-atelier", "05-启动页-右女仆.webp")
MAID_Q = os.path.join(A, "maid-atelier", "06-Q版侧栏-晕倒冒泡.webp")
PET = os.path.join(A, "galgame", "pet-dafeiyu.webp")
BG_STUDY = os.path.join(A, "galgame", "背景", "bg-deepseek-seaside-study.webp")
BG_OTHER = {
    "claude": os.path.join(A, "galgame", "背景", "bg-claude-writing-study.webp"),
    "gpt": os.path.join(A, "galgame", "背景", "bg-gpt-collaboration-workshop.webp"),
    "gemini": os.path.join(A, "galgame", "背景", "bg-gemini-twin-creative-studio.webp"),
    "kimi": os.path.join(A, "galgame", "背景", "bg-kimi-moonlit-reading-study.webp"),
    "grok": os.path.join(A, "galgame", "背景", "bg-grok-electronics-studio.webp"),
}
WHALE = {k: os.path.join(A, "galgame", f"whale-{k}.webp") for k in
         ("cheerful", "shy", "serious", "confused", "angry", "frightened", "exasperated", "starry")}
AI = {
    "claude": os.path.join(A, "galgame", "claude-zipzippipe.webp"),
    "gpt": os.path.join(A, "galgame", "gpt-zipzippipe.webp"),
    "grok": os.path.join(A, "galgame", "grok-zipzippipe.webp"),
    "gemini": os.path.join(A, "galgame", "gemini-dual-prism-translator-v4.webp"),
    "kimi": os.path.join(A, "galgame", "kimi-lunar-scroll-navigator-v5.webp"),
}

# ---------------------------------------------------------------- 主题
class Theme:
    def __init__(self, bg="03101C", fg=kit.CREAM, accent=kit.MINT, second=kit.CORAL,
                 deep=None, ink=None, fog=None):
        self.bg = np.asarray(kit.hx(bg) if isinstance(bg, str) else bg, F32)
        self.fg = fg; self.accent = accent; self.second = second
        self.deep = deep if deep is not None else mix(self.bg, kit.BLACK, 0.55)
        self.ink = ink if ink is not None else kit.INK
        self.fog = fog if fog is not None else mix(self.bg, kit.PALE, 0.22)


def mix(a, b, t):
    return kit.mixc(np.asarray(a, F32), np.asarray(b, F32), t)


TH_DEEP = Theme("08131F", kit.CREAM, kit.MINT, kit.CORAL)
TH_NIGHT = Theme("050A12", kit.ICE, kit.PALE, kit.ROSE)
TH_ELECTRIC = Theme("07131A", kit.CREAM, kit.hx("6EFFF0"), kit.hx("FF5FA2"))
TH_GOD = Theme("0B0716", kit.CREAM, kit.VIOLET, kit.GOLD)
TH_WARM = Theme("14100C", kit.CREAM, kit.GOLD, kit.CORAL)
TH_COLD = Theme("0A0E14", kit.hx("C8D4E0"), kit.hx("9FB6CC"), kit.hx("C9503C"))
TH_ALARM = Theme("120406", kit.hx("FFD9D0"), kit.hx("FF4A3C"), kit.hx("FF9B5A"))
TH_WHITE = Theme("F2EDE6", kit.hx("141414"), kit.hx("B03A2E"), kit.hx("2E5C8A"))

# ---------------------------------------------------------------- 背景
_BG = {}


def _bg_scaled(path):
    k = (path, kit.SW, kit.SH)
    if k in _BG:
        return _BG[k]
    a = kit.load(path)
    rgb = a[..., :3]
    # 背景原图只有 1586×992：先放大到目标分辨率再缓存
    big = kit.up(rgb, kit.SW, kit.SH)
    _BG[k] = big
    return big


def bg_image(f, path=BG_PALACE, zoom=1.0, dx=0.0, dy=0.0, tint=None, tint_k=0.0,
             dim=1.0, blur_r=0.0, exposure=1.0, sat=1.0, rot=0.0):
    """铺满画面的背景（缩放/平移/调色）。"""
    img = _bg_scaled(path)
    if blur_r > 0.2:
        img = kit.blur(img, blur_r * S)
    if abs(zoom - 1.0) > 1e-3 or dx or dy or rot:
        cx, cy = kit.SW / 2, kit.SH / 2
        M = cv2.getRotationMatrix2D((cx, cy), -math.degrees(rot), zoom)
        M[0, 2] += dx * S; M[1, 2] += dy * S
        img = cv2.warpAffine(img, M, (kit.SW, kit.SH), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    if tint is not None and tint_k > 0:
        img = img * (1 - tint_k) + np.asarray(tint, F32)[None, None, :] * tint_k
    if sat != 1.0:
        l = img.mean(axis=2, keepdims=True)
        img = l + (img - l) * sat
    f += img * dim * exposure
    return f


def caustics(f, t, color=None, strength=0.10, scale=1.0, speed=0.35, y0=0.35):
    """水面/内存里的光斑：两层正弦干涉网格（半分辨率计算再放大）。"""
    color = kit.MINT if color is None else color
    bw, bh = kit.SW // 4, kit.SH // 4
    yy, xx = np.mgrid[0:bh, 0:bw].astype(F32)
    x = xx / bw * 6.0 * scale; y = yy / bh * 3.4 * scale
    a = np.sin(x * 2.1 + t * speed * 2.4) * np.cos(y * 2.7 - t * speed * 1.7)
    b = np.sin((x + y) * 3.3 - t * speed * 2.9) * np.cos((x - y) * 1.9 + t * speed * 1.3)
    m = np.clip((a * 0.6 + b * 0.4) * 0.5 + 0.5, 0, 1) ** 3.2
    m *= np.clip(1.0 - np.abs(yy / bh - y0) * 1.4, 0, 1)
    m = kit.up(m, kit.SW, kit.SH)
    f += m[..., None] * np.asarray(color, F32) * strength
    return f


def lightbeam(f, t, x=0.5, width=0.22, color=None, strength=0.22, sway=0.05, tilt=0.32):
    color = kit.PALE if color is None else color
    xx = x + math.sin(t * 0.31) * sway
    m = np.zeros((kit.SH, kit.SW), F32)
    x0 = int((xx - width / 2) * kit.SW); x1 = int((xx + width / 2) * kit.SW)
    top = int(-0.1 * kit.SH); bot = int(1.05 * kit.SH)
    dx = int(tilt * kit.SW)
    p = np.array([[x0, top], [x1, top], [x1 + dx, bot], [x0 + dx, bot]], np.int32)
    cv2.fillPoly(m, [p], 1.0)
    m = kit.blur(m, 40 * S)
    v = np.clip((np.arange(kit.SH, dtype=F32) / kit.SH) * 1.4, 0, 1)[:, None]
    m = m * (0.25 + 0.75 * v)
    f += m[..., None] * np.asarray(color, F32) * strength
    return f


def dust(f, t, n=260, color=None, alpha=0.5, box=(0.0, 1.0, 0.0, 1.0)):
    """漂浮尘埃：确定性随机轨迹。"""
    color = kit.PALE if color is None else color
    rs = np.random.default_rng(4242)
    px = rs.random(n).astype(F32); py = rs.random(n).astype(F32)
    sp = (0.2 + rs.random(n) * 0.8).astype(F32)
    ph = (rs.random(n) * 6.283).astype(F32)
    x0, x1, y0, y1 = box
    X = (x0 + px * (x1 - x0)) * kit.SW + np.sin(t * 0.35 * sp + ph) * 22 * S
    Y = ((y0 + py * (y1 - y0)) * kit.SH - (t * 6 * sp * S)) % kit.SH
    ix = np.clip(X.astype(np.int32), 0, kit.SW - 1)
    iy = np.clip(Y.astype(np.int32), 0, kit.SH - 1)
    m = np.zeros((kit.SH, kit.SW), F32)
    np.add.at(m, (iy, ix), (0.4 + 0.6 * rs.random(n)).astype(F32))
    m = kit.blur(m, 1.1 * S)
    f += m[..., None] * np.asarray(color, F32) * alpha
    return f


def floor_reflection(f, t, y=0.78, strength=0.20, color=None):
    """把画面下半部做水面倒影 + 涟漪。"""
    img = f.copy()
    yc = int(y * kit.SH)
    hh = kit.SH - yc
    if hh <= 2:
        return f
    sub = img[yc:kit.SH][::-1].copy()
    sub = kit.blur(sub, 1.4 * S)
    sub *= 0.55
    rs = np.random.default_rng(9)
    rows = np.arange(hh, dtype=F32)
    dxs = (np.sin(rows * 0.06 + t * 1.7) * 5 * S + rs.normal(0, 0.7, hh) * S).astype(F32)
    M = np.stack([np.ones(hh), np.zeros(hh), dxs], 1).astype(F32)
    sub = cv2.remap(sub, np.tile(np.arange(kit.SW, dtype=F32), (hh, 1)),
                    np.tile(rows[:, None], (1, kit.SW)), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    fade = (1 - rows / hh) ** 2.2
    f[yc:kit.SH] = f[yc:kit.SH] * 0.65 + sub * (fade[:, None, None] * strength)
    return f


# ---------------------------------------------------------------- 角色
def rim_light(spr, color, k=1.0, dx=3.0, dy=-2.0):
    """把 alpha 边缘移到外侧作为轮廓光。"""
    al = spr[..., 3]
    m = kit.shift(al, dx * S, dy * S) - al
    m = np.clip(m, 0, 1)
    out = spr.copy()
    out[..., :3] = out[..., :3] * (1 - 0.0) + np.asarray(color, F32)[None, None, :] * m[..., None] * k
    out[..., 3] = np.clip(al + m * k, 0, 1)
    return out


def character(f, path, cx, cy, h, alpha=1.0, k=1.0, tint=None, tint_k=0.0, glow=0.0,
              flip=False, rot=0.0, rim=None, rim_k=0.0, shadow=0.0, additive=False, anchor="foot"):
    spr = kit.sprite(path, h)
    if rim is not None and rim_k > 0:
        spr = rim_light(spr, rim, rim_k)
    if shadow > 0:
        gy = cy
        m = kit.circle_mask(cx, gy, h * 0.20, h * 0.028)
        m = kit.blur(m, h * 0.02)
        f *= (1 - (m * shadow)[..., None] * 0.85)
    kit.place(f, spr, cx, cy, k, alpha, tint, tint_k, glow, additive, flip, rot, anchor)
    return f


def chars_duo(f, t, h=800, gap=250, alpha=1.0, dy=0.0, tint=None, tint_k=0.0, rim=None,
              rim_k=0.5, glow=0.0, paths=None, xs=None):
    L, R = paths or (MAID_L, MAID_R)
    xl = xs[0] if xs else kit.W / 2 - gap
    xr = xs[1] if xs else kit.W / 2 + gap
    character(f, L, xl, kit.H * 0.98 + dy, h, alpha, tint=tint, tint_k=tint_k, rim=rim, rim_k=rim_k, glow=glow)
    character(f, R, xr, kit.H * 0.98 + dy, h, alpha, tint=tint, tint_k=tint_k, rim=rim, rim_k=rim_k, glow=glow)
    return f


_SHEET = {}


def sheet_cell(path, idx, h, cols=6, rows=7):
    """图集切片（pet-dafeiyu.webp 6×7）。"""
    key = (path, idx, h)
    if key in _SHEET:
        return _SHEET[key]
    a = kit.load(path)
    hh, ww = a.shape[:2]
    ch, cw = hh // rows, ww // cols
    r, c = divmod(idx, cols)
    cell = a[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
    cell = kit.trim(cell)
    kk = h / cell.shape[0]
    out = kit.up(cell, max(1, int(cell.shape[1] * kk)), int(h * S))
    _SHEET[key] = out
    return out


# ---------------------------------------------------------------- 光标（「你」）
def cursor(f, x, y, t, size=26, color=None, alpha=1.0, style="beam", blink=True, glow=0.7):
    color = kit.WHITE if color is None else color
    if blink:
        # 每 0.53s 闪一次，避开整拍
        alpha *= 0.35 + 0.65 * (0.5 + 0.5 * math.cos(2 * math.pi * t / 1.06))
    if alpha <= 0.01:
        return f
    if style == "beam":
        w = max(1.4, size * 0.09)
        m = np.zeros((kit.SH, kit.SW), F32)
        p0 = (int((x - size * 0.18) * kit.S), int((y - size * 0.5) * kit.S))
        p1 = (int((x - size * 0.18) * kit.S), int((y + size * 0.5) * kit.S))
        cv2.line(m, p0, p1, 1.0, max(1, int(w * kit.S)), cv2.LINE_AA)
        for yy, ww in ((-0.5, 0.6), (0.5, 0.6)):
            cv2.line(m, (int((x - size * 0.42) * kit.S), int((y + yy * size) * kit.S)),
                     (int((x + size * 0.06) * kit.S), int((y + yy * size) * kit.S)),
                     1.0, max(1, int(w * kit.S)), cv2.LINE_AA)
    elif style == "block":
        w = size * 0.56; hh = size * 0.92
        m = kit.poly_mask([(x, y - hh / 2), (x + w, y - hh / 2), (x + w, y + hh / 2), (x, y + hh / 2)])
    else:  # ring
        m = kit.circle_mask(x, y, size * 0.5, size * 0.02)
        m = np.clip(m, 0, 1)
    if glow > 0:
        g = kit.blur(m, 6 * S) * glow
        f += g[..., None] * np.asarray(color, F32) * 2.4 * alpha
    f += (m * alpha)[..., None] * np.asarray(color, F32)
    return f


def cursor_trail(f, pts, color=None, alpha=0.5, width=1.6):
    if len(pts) < 2:
        return f
    color = kit.WHITE if color is None else color
    m = kit.line_mask(pts, width)
    m = kit.blur(m, 1.2 * S)
    f += m[..., None] * np.asarray(color, F32) * alpha
    return f


# ---------------------------------------------------------------- 文字排版
def _scrim(f, yy, size, k=0.30, span=2.1):
    """在字幕下方垫一层柔和暗场，保证不论背景多亮字幕都读得清。"""
    y0 = max(0, int((yy - size * 0.75) * kit.S))
    y1 = min(kit.SH, int((yy + size * span) * kit.S))
    if y1 - y0 < 4:
        return
    h = y1 - y0
    v = np.linspace(0.0, 1.0, h, dtype=np.float32)
    prof = np.clip(np.sin(np.pi * v), 0.0, 1.0) ** 0.85
    f[y0:y1] *= (1.0 - (k * prof)[:, None, None])


def lyric(f, t, ln=None, style="soft", y=None, alpha=1.0, color=None, size=46, accent=None,
          align="center", x=None, tracking=1.6, zh=True, words=True, glow=0.25, mono=False,
          scrim=0.36, shadow=0.85):
    """主歌词排版：英文主行 + 中文小行。style: soft/hard/terminal/white。"""
    if ln is None:
        ln = mus.line_at(t)
    if ln is None or alpha <= 0.01:
        return f
    n, wp, prog = mus.word_progress(ln, t)
    appear = kit.cl(prog * 8)
    col = color if color is not None else (kit.CREAM if style != "terminal" else kit.MINT)
    fp = kit.F_MONO if (mono or style == "terminal") else kit.F_SERIF_I
    yy = kit.H * 0.855 if y is None else y
    xx = kit.W / 2 if x is None else x
    a = alpha * appear
    if scrim > 0 and align == "center":
        _scrim(f, yy, size, scrim * a)
    if style == "hard":
        # 整行硬显示 + 逐字推入
        def ca(i, rel):
            return 1.0 if (i < n or (i == n and wp > 0.4)) else 0.0
        def co(i, rel):
            return (0, (1 - kit.cl(wp * 2)) * 4) if i == n else (0, 0)
        kit.draw_text(f, ln["en"], xx, yy, size, col, fp, a, tracking, align, glow=glow,
                      shadow=shadow)
    else:
        # 柔和逐字显影：已到词亮、当前词渐入
        def ca(i, rel):
            p = kit.cl((prog - i * 0.006) * 6)
            return p
        kit.draw_text(f, ln["en"], xx, yy, size, col, fp, a, tracking, align, glow=glow,
                      shadow=shadow)
    if zh and ln.get("zh"):
        kit.draw_text(f, ln["zh"], xx, yy + size * 1.26, size * 0.54, col, kit.F_SERIF_SC,
                      a * 0.66, tracking * 2.2, align, shadow=shadow)
    return f


def line_tag(f, txt, x=None, y=None, alpha=1.0, color=None, size=20, prefix="// "):
    color = kit.ASH if color is None else color
    if alpha <= 0.01:
        return f
    xx = 84 if x is None else x
    yy = kit.H - 96 if y is None else y
    kit.draw_text(f, prefix + txt, xx, yy, size, color, kit.F_MONO, alpha * 0.8, 0.6)


def terminal(f, lines, x=90, y=120, size=22, alpha=1.0, color=None, lead=1.55,
             cursor_at=None, t=0.0, dim=0.55):
    color = kit.MINT if color is None else color
    for i, (txt, c) in enumerate(lines):
        cc = c if c is not None else color
        kit.draw_text(f, txt, x, y + i * size * lead, size, cc, kit.F_MONO, alpha, 0.4)
    if cursor_at is not None:
        yy = y + cursor_at * size * lead
        xx = x + sum(kit.text_width(kit.F_MONO, size, s[0], 0.4) for s in lines[:cursor_at])
        cursor(f, xx + 4, yy + size * 0.55, t, size * 0.9, color, alpha, "beam", False, 0.5)
    return y + len(lines) * size * lead


def hud_corner(f, t, left="", right="", alpha=0.5, color=None, y=None):
    color = kit.ASH if color is None else color
    yy = kit.H - 54 if y is None else y
    if left:
        kit.draw_text(f, left, 84, yy, 20, color, kit.F_MONO, alpha, 1.2)
    if right:
        kit.draw_text(f, right, kit.W - 84, yy, 20, color, kit.F_MONO, alpha, 1.2, "right")


def title_card(f, main, sub="", t=0.0, alpha=1.0, color=None, sub_color=None, size=92,
               y=None, tracking=8.0, mono=False, glow=0.4, zh=None):
    color = kit.CREAM if color is None else color
    sub_color = kit.MINT if sub_color is None else sub_color
    yy = kit.H * 0.44 if y is None else y
    fp = kit.F_MONO if mono else kit.F_SERIF_I
    kit.draw_text(f, main, kit.W / 2, yy, size, color, fp, alpha, tracking, "center", glow=glow)
    if sub:
        kit.draw_text(f, sub, kit.W / 2, yy + size * 1.15, size * 0.26, sub_color, kit.F_MONO,
                      alpha * 0.75, tracking * 1.6, "center")
    if zh:
        kit.draw_text(f, zh, kit.W / 2, yy + size * 1.55, size * 0.30, color, kit.F_SERIF_SC,
                      alpha * 0.55, 3.0, "center")


def code_panel(f, t, rows, x=None, y=None, w=None, alpha=1.0, size=19, theme=None,
               typed=1.0, caret=True, bg=True, title="world.py"):
    """仿代码编辑器面板：语法高亮 + 打字机 + 行号 + 光标。rows=[(kind, text)]"""
    theme = theme or TH_DEEP
    x = kit.W * 0.10 if x is None else x
    y = kit.H * 0.12 if y is None else y
    w = kit.W * 0.62 if w is None else w
    lead = size * 1.62
    total = len(rows)
    vis = int(np.clip(typed, 0, 1) * total * 1.35)
    if bg:
        m = kit.poly_mask([(x - 26, y - 30), (x + w, y - 30), (x + w, y + total * lead + 34), (x - 26, y + total * lead + 34)])
        m = kit.blur(m, 1.0)
        f *= (1 - (m * 0.55 * alpha)[..., None])
        f += (m * alpha)[..., None] * theme.bg * 0.9
    colors = {"kw": theme.accent, "fn": theme.second, "str": mix(theme.fg, theme.accent, 0.35),
              "num": kit.GOLD, "cm": mix(theme.fog, kit.BLACK, 0.25), "pl": theme.fg,
              "tx": mix(theme.fg, kit.BLACK, 0.15)}
    for i, (kind, txt) in enumerate(rows):
        if i >= vis:
            break
        part = txt
        if i == vis - 1:
            frac = vis - int(vis)
            part = txt[:max(0, int(len(txt) * frac))] if frac > 0 else txt
        yy = y + i * lead
        kit.draw_text(f, "%02d" % (i + 1), x, yy, size * 0.82, mix(theme.fog, kit.BLACK, 0.35),
                      kit.F_MONO, alpha * 0.5, 0.5)
        kit.draw_text(f, part, x + size * 2.4, yy, size, colors.get(kind, theme.fg), kit.F_MONO, alpha, 0.3)
    if caret:
        i = min(total - 1, int(vis))
        yy = y + i * lead
        xx = x + size * 2.4 + kit.text_width(kit.F_MONO, size, rows[i][1][:6], 0.3)
        cursor(f, xx, yy + size * 0.6, t, size, theme.accent, alpha, "beam", True, 0.4)
    return f


def param_list(f, t, items, x=None, y=None, alpha=1.0, size=24, theme=None, typed=1.0,
               label_w=280, gap=None):
    """参数表：左标签右数值，逐条点亮。"""
    theme = theme or TH_DEEP
    x = kit.W * 0.14 if x is None else x
    y = kit.H * 0.20 if y is None else y
    gap = gap or size * 2.6
    for i, (k, v, ok) in enumerate(items):
        p = kit.cl((typed * len(items) - i) * 1.6)
        if p <= 0:
            continue
        a = alpha * p
        kit.draw_text(f, k, x, y + i * gap, size, theme.fog, kit.F_MONO, a * 0.55, 2.0)
        kit.draw_text(f, "." * max(2, int((label_w - kit.text_width(kit.F_MONO, size, k, 2.0)) / (size * 0.55))),
                      x, y + i * gap, size, mix(theme.fog, kit.BLACK, 0.4), kit.F_MONO, a * 0.28, 2.0)
        col = theme.accent if ok else theme.second
        kit.draw_text(f, v, x + label_w, y + i * gap, size, col, kit.F_MONO, a, 1.4)
    return y + len(items) * gap


def big_word(f, word, t, alpha=1.0, color=None, size=250, y=None, tracking=14, mono=False,
             stretch=1.0, glow=0.5):
    color = kit.CREAM if color is None else color
    yy = (kit.H - size * 0.72) / 2 if y is None else y
    fp = kit.F_MONO_B if mono else kit.F_SERIF_B
    kit.draw_text(f, word, kit.W / 2, yy, size, color, fp, alpha, tracking, "center", glow=glow, stretch=stretch)


def letterbox(f, k=0.12):
    hh = int(kit.H * k * kit.S)
    f[:hh] *= 0
    f[kit.SH - hh:] *= 0
    return f


def frame_number(f, t, color=None, alpha=0.35):
    color = kit.ASH if color is None else color
    fr = int(round(t * 60))
    kit.draw_text(f, "F%05d  %06.2fs" % (fr, t), kit.W - 44, 34, 17, color, kit.F_MONO, alpha, 1.0, "right")


def split_lines(f, x, y, items, size=26, lead=1.9, alpha=1.0, color=None, accent=None):
    color = kit.CREAM if color is None else color
    accent = kit.MINT if accent is None else accent
    for i, (a, b) in enumerate(items):
        kit.draw_text(f, a, x, y + i * size * lead, size, accent, kit.F_MONO, alpha * 0.9, 1.0)
        kit.draw_text(f, b, x + kit.text_width(kit.F_MONO, size, a, 1.0) + 18, y + i * size * lead,
                      size, color, kit.F_MONO, alpha, 1.0)
