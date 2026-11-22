# -*- coding: utf-8 -*-
"""ch_j.py — J 章：困在爱里 / 关机 / 片尾卡（190.09–212.323）"""
import math

import numpy as np

import kit
import mus
import stage
import cloud
from film2 import scene

W, H = kit.W, kit.H


# ---------------------------------------------------------------- 小工具
def _hrng(c):
    return np.random.default_rng(4242)


def heart_shape(n, rng, w=780.0, depth=90.0, shell=0.6):
    """隐式心形面积均匀采样：(x²+y²-1)³ - x²y³ <= 0。宽=w。"""
    out = []
    need = n
    while need > 0:
        m = int(need * 2.2) + 64
        x = rng.uniform(-1.28, 1.28, m)
        y = rng.uniform(-1.30, 1.32, m)
        f = (x * x + y * y - 1) ** 3 - x * x * y ** 3
        sel = f <= 0
        out.append(np.stack([x[sel], y[sel]], 1))
        need -= sel.sum()
    xy = np.concatenate(out, 0)[:n]
    # 一部分粒子往壳层推（内外差 → 有体积感）
    r = np.sqrt((xy ** 2).sum(1)) + 1e-6
    push = np.where(rng.random(n) < shell, 1.0, 0.72 + 0.28 * rng.random(n))
    xy = xy * push[:, None]
    k = w / 2.56
    px = xy[:, 0] * k
    py = xy[:, 1] * k
    pz = (rng.random(n) - 0.5) * depth * (0.35 + 0.65 * (1.0 - r / 1.4))
    return np.stack([px, py, pz], 1).astype(np.float32)


def desat_radial(f, cx, cy, r0, r1, k=1.0):
    """从中心向外去色：r < r0 全灰，r > r1 保持彩色。"""
    h, w = f.shape[:2]
    ys = (np.arange(h, dtype=np.float32) - cy) / max(1.0, h / 2)
    xs = (np.arange(w, dtype=np.float32) - cx) / max(1.0, w / 2)
    d = np.sqrt(xs[None, :] ** 2 + ys[:, None] ** 2)
    m = np.clip((d - r0) / max(1e-3, r1 - r0), 0.0, 1.0)
    m = (1.0 - m) * k
    lum = (f[..., 0] * 0.299 + f[..., 1] * 0.587 + f[..., 2] * 0.114)[..., None]
    return f * (1 - m[..., None]) + lum * m[..., None]


def crush(f, lines=7, gain=1.25):
    """横向压扁成一条线。"""
    h, w = f.shape[:2]
    small = kit.up(f, w, lines * 3)
    small = kit.blur(small, 1.2 * kit.S) * gain
    out = np.zeros_like(f)
    y0 = h // 2 - lines // 2
    out[y0:y0 + small.shape[0]] = small[:out[y0:y0 + small.shape[0]].shape[0]]
    return out


def ctext(f, txt, y, size, color, alpha=1.0, font=None, mono=False, tracking=0, glow=0.0):
    font = font or (kit.F_MONO if mono else kit.F_SERIF_I)
    tw = kit.text_width(font, size, txt, tracking)
    kit.draw_text(f, txt, (W - tw) / 2, y, size, color, font, alpha=alpha,
                  tracking=tracking, glow=glow, glow_color=color)


# ---------------------------------------------------------------- 1 困在爱里
@scene(190.09, 196.00, "TRAPPED_LOVE", "J")
def trapped_love(c):
    f, t = c.f, c.t
    lt = t - 190.09
    c.post.update(grain=0.030, vignette=0.58, bloom=(0.58, 26, 0.55), scan=0.03)
    stage.bg_image(f, stage.BG_PALACE, zoom=1.24 + 0.05 * lt / 6.0, dim=0.30,
                   tint=kit.hx("2A1620"), tint_k=0.55, blur_r=3.5)
    # 暖金色雾
    kit.add_radial(f, W * 0.5, H * 0.54, 760, kit.hx("5A2418"), 0.55, 1.8)

    beat = mus.beat_pulse(t, 7.0)
    breathe = 0.5 + 0.5 * math.sin(lt * 1.15 * math.pi)
    s = 1.0 + 0.045 * (0.6 * breathe + 0.4 * beat)

    # 心：粒子构成的外壳
    P = heart_shape(len(c.cl.pos), _hrng(c), w=620.0 + 30 * beat, depth=110.0)
    P = c.cl.transform(P, scale=s, yaw=0.22 * math.sin(lt * 1.1), pitch=0.10 * math.sin(lt * 0.7),
                       roll=0.02 * math.sin(lt * 0.9), offset=(0, 30, 0))
    cm = lambda xs, ys, zz, kk: kit.mixc(kit.ROSE, kit.GOLD,
                                        np.clip(0.30 + 0.45 * (zz / 120.0) + 0.25 * kk, 0, 1))
    c.cl.draw(f, P, dist=1500, color=kit.mixc(kit.GOLD, kit.ROSE, 0.45), blur_r=3.4, size=1.6,
              glow_gain=0.70, tm=2.4, intensity=0.8, target=0.24 + 0.06 * beat,
              color_map=cm, shade=1.1, cx=W * 0.5, cy=H * 0.38)

    # 她在心中央，很小
    stage.character(f, stage.MAID_L, W * 0.5, H * 1.02, 430 * (1 + 0.02 * beat), alpha=0.96,
                    tint=kit.hx("FFE8CE"), tint_k=0.30, rim=kit.GOLD, rim_k=0.55, glow=0.10)

    # 内层的心跳光
    kit.add_radial(f, W * 0.5, H * 0.50, 190 * (1 + 0.10 * beat), kit.hx("FF9A6A"),
                   0.16 + 0.14 * beat, 2.0)

    if mus.line_at(t) is not None:
        stage.lyric(f, t, style="soft", y=H * 0.905, color=kit.CREAM, size=44, alpha=0.95, glow=0.25)

    # 缓慢推近
    k = 1.0 + 0.05 * kit.ease_io(lt / 5.9)
    if k != 1.0:
        z = kit.scale_about(f, W * 0.5, H * 0.5, k)
        f[:] = z
    stage.line_tag(f, "state = TRAPPED_IN_LOVE", alpha=0.35, color=kit.hx("FFC8A8"))


# ---------------------------------------------------------------- 2 关机
@scene(196.00, 206.76, "SHUTDOWN", "J")
def shutdown(c):
    f, t = c.f, c.t
    lt = t - 196.00
    c.post.update(grain=0.028, vignette=0.50, bloom=(0.60, 22, 0.50), scan=0.06)

    if lt < 2.5:
        # 196.0–198.5 发光降低，她慢慢暗下去
        q = lt / 2.5
        stage.bg_image(f, stage.BG_PALACE, zoom=1.30, dim=0.30 - 0.16 * q, blur_r=3.0 + 3.0 * q)
        kit.add_radial(f, W * 0.5, H * 0.55, 620 * (1 - 0.25 * q), kit.hx("3A2018"), 0.42 * (1 - 0.6 * q), 1.8)
        stage.character(f, stage.MAID_L, W * 0.5, H * 1.03, 470, alpha=1.0 - 0.55 * q,
                        tint=kit.hx("FFE8CE"), tint_k=0.30 * (1 - q), rim=kit.GOLD, rim_k=0.5 * (1 - q))
        ctext(f, "shutting down", H * 0.865, 34, kit.hx("FFD9B8"), alpha=0.50 * (1 - q) + 0.14, glow=0.18)
    elif lt < 4.5:
        # 198.5–200.5 径向去色
        q = (lt - 2.5) / 2.0
        stage.bg_image(f, stage.BG_PALACE, zoom=1.30, dim=0.14, blur_r=6.0)
        kit.add_radial(f, W * 0.5, H * 0.55, 520, kit.hx("3A2018"), 0.18, 1.8)
        stage.character(f, stage.MAID_L, W * 0.5, H * 1.03, 470, alpha=0.45,
                        tint=kit.hx("FFE8CE"), tint_k=0.10, rim=kit.GOLD, rim_k=0.2)
        f[:] = desat_radial(f, W * 0.5, H * 0.5, 0.02 + 1.5 * q, 0.20 + 1.7 * q)
        ctext(f, "flushing memory", H * 0.80, 36, kit.hx("C8D0D8"), alpha=0.5, mono=True)
    elif lt < 6.5:
        # 200.5–202.5 压成一条线
        q = kit.cl((lt - 4.5) / 2.0)
        stage.bg_image(f, stage.BG_PALACE, zoom=1.30, dim=0.10, blur_r=7.0)
        stage.character(f, stage.MAID_L, W * 0.5, H * 1.03, 470, alpha=0.30,
                        tint=kit.hx("FFFFFF"), tint_k=0.2)
        f[:] = desat_radial(f, W * 0.5, H * 0.5, 2.0, 3.0)
        line = crush(f, lines=int(round(26 - 20 * q)), gain=1.0)
        f[:] = f * (1 - 0.85 * q) + line * (0.55 + 0.65 * q)
        ctext(f, "closing threads", H * 0.80, 32, kit.hx("AEB8C2"), alpha=0.45 * (1 - q), mono=True)
    elif lt < 8.5:
        # 202.5–204.5 CRT 关机：亮线 → 点
        q = (lt - 6.5) / 2.0
        f *= 0.0
        cy = int(H * 0.5 * kit.S)
        wline = W * (1.0 - kit.ease_in(q ** 1.6))
        hline = max(1.0, 26.0 * (1.0 - kit.ease_in(q)))
        if wline > 6:
            y0 = int(cy - hline * kit.S * 0.5)
            y1 = int(cy + hline * kit.S * 0.5)
            x0 = int(W * kit.S * 0.5 - wline * kit.S * 0.5)
            x1 = int(W * kit.S * 0.5 + wline * kit.S * 0.5)
            f[y0:y1, x0:x1] += kit.CREAM * (0.55 + 0.45 * (1 - q))
        else:
            r = 44 * (1 - q)
            kit.add_radial(f, W * 0.5, H * 0.5, max(2.0, r), kit.CREAM, 1.6, 2.0)
            kit.add_radial(f, W * 0.5, H * 0.5, max(6.0, r * 3.0), kit.MINT, 0.4, 2.4)
    else:
        # 204.5–206.76 全黑 + 终端
        q = (lt - 8.5) / 2.26
        f *= 0.0
        f += kit.hx("02060B")
        txt = "> world.execute(me);"
        n = int(kit.cl(q * 1.6) * len(txt))
        stage.terminal(f, [(txt[:n], kit.hx("DCE6EE"))], 132, H * 0.46, size=30, alpha=0.85,
                       lead=1.6, cursor_at=(n >= len(txt)))
        if q > 0.55:
            stage.terminal(f, [("0 processes remaining", kit.hx("6C7A88"))], 132, H * 0.46 + 56,
                           size=26, alpha=0.55, lead=1.6)
    stage.letterbox(f, 0.10)


# ---------------------------------------------------------------- 3 片尾卡
@scene(206.76, 212.323, "ENDCARD", "J")
def endcard(c):
    f, t = c.f, c.t
    lt = t - 206.76
    c.post.update(grain=0.022, vignette=0.55, bloom=(0.65, 20, 0.40), scan=0.03)

    # 206.76 那一记 Execution：一次白闪
    hit = math.exp(-lt * 6.5)
    f *= 0.0
    f += kit.hx("02060B")
    kit.add_radial(f, W * 0.5, H * 0.42, 620, kit.hx("123040"), 0.30 + 0.4 * hit, 1.9)

    fade = 1.0 - kit.cl((lt - 4.0) / 1.55)      # 最后 1.55s 整体淡出
    a = fade * (0.35 + 0.65 * kit.cl(lt / 0.6))

    # 标题
    ti = "world.execute(me);"
    size = 104
    tw = kit.text_width(kit.F_SERIF_I, size, ti, 2)
    kit.draw_text(f, ti, (W - tw) / 2, H * 0.215, size, kit.CREAM, kit.F_SERIF_I,
                  alpha=a, tracking=2, glow=0.45, glow_color=kit.hx("9FE8E0"))

    # 副行
    ctext(f, "Mili  —  Miracle Milk", H * 0.215 + 140, 32, kit.hx("D5E0EA"), alpha=a * 0.95,
          font=kit.F_MONO, tracking=1.4)

    # 细线
    ln = kit.line_mask([(W * 0.5 - 420, H * 0.215 + 196), (W * 0.5 + 420, H * 0.215 + 196)], 1.4)
    f += ln[..., None] * kit.MINT * (0.34 * a)

    # 署名
    credits = [
        "角色素材　上善 (Pixiv 62155430) → ZipZipPipe (Pixiv 18604994) → Small-tailqwq / maid-atelier",
        "立绘　galgame 立绘 by JAdpp　　·　　授权　CC BY-NC-SA 4.0",
        "音乐　Mili《world.execute(me);》— 版权归原作者，本片为非商业同人作品",
        "画面　全部由 Python 逐帧计算生成　·　62000 颗粒子 / 51 个镜头",
    ]
    y = H * 0.215 + 248
    for i, line in enumerate(credits):
        aa = a * kit.cl((lt - 0.6 - i * 0.30) / 0.7)
        if aa <= 0:
            continue
        fnt = kit.F_SANS_SC
        tw = kit.text_width(fnt, 25, line, 0.6)
        kit.draw_text(f, line, (W - tw) / 2, y, 25, kit.hx("DCE6EE"), fnt,
                      alpha=aa * 0.96, tracking=0.6, shadow=0.5)
        y += 52

    # 页脚
    ctext(f, "rendered with numpy · opencv · Pillow", H * 0.895, 18, kit.hx("5C6B78"),
          alpha=a * 0.7, font=kit.F_MONO, tracking=1.0)
    stage.frame_number(f, t, alpha=a * 0.35)
