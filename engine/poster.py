# -*- coding: utf-8 -*-
"""poster.py — 从成片时间点渲染一张 3:4 海报（4K 静帧整幅 + 排版，不裁切画面）。

python poster.py --t 190.6 --out out/poster.jpg
python poster.py --t 61.2 --out out/poster_ifcan.jpg
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import kit
import film2

PW, PH = 2160, 2880


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=190.6)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "poster.jpg"))
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--mx", type=int, default=140, help="左右留白")
    ap.add_argument("--my", type=int, default=180, help="上留白")
    a = ap.parse_args()

    kit.init(a.scale)
    film2.load_chapters()
    f, _ = film2.render(a.t, scale=a.scale)
    src = Image.fromarray((np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8))

    iw = PW - 2 * a.mx
    ih = int(round(iw * src.size[1] / src.size[0]))
    still = src.resize((iw, ih), Image.LANCZOS)

    frame = np.zeros((PH, PW, 3), np.float32)
    frame[:] = np.asarray(kit.hx("050B12"), np.float32) * 0.85
    x0, y0 = a.mx, a.my
    frame[y0:y0 + ih, x0:x0 + iw] = np.asarray(still, np.float32) / 255.0
    kit.init(1, PW, PH)

    def tx(txt, x, y, size, color, font=kit.F_SERIF_I, alpha=1.0, tracking=0.0, glow=0.0,
           align="left", shadow=0.0):
        w = kit.text_width(font, size, txt, tracking)
        xx = x if align == "left" else (x - w / 2 if align == "center" else x - w)
        kit.draw_text(frame, txt, xx, y, size, color, font, alpha=alpha, tracking=tracking,
                      glow=glow, glow_color=color, shadow=shadow)

    def hline(x0_, x1_, y, color, alpha=0.5, thick=2):
        yy = int(y)
        xs, xe = int(x0_), int(x1_)
        frame[yy:yy + thick, xs:xe] = (frame[yy:yy + thick, xs:xe] * (1 - alpha)
                                       + np.asarray(kit.hx(color), np.float32) * alpha)

    # 剧照细框
    hline(a.mx, PW - a.mx, y0 - 3, "26404E", 0.85, 1)
    hline(a.mx, PW - a.mx, y0 + ih + 2, "26404E", 0.85, 1)

    yb = y0 + ih + 96
    tx("world.execute(me);", 140, yb, 186, kit.CREAM, kit.F_SERIF_I, 1.0, 3, 0.38, shadow=0.55)
    yh = yb + 262
    hline(146, 1290, yh, "7FE3D4", 0.5, 2)
    tx("一次 execute 请求的一生", 148, yh + 34, 62, kit.hx("8FE3DA"), kit.F_SANS_SC, 0.94, 5)
    tx("Mili — Miracle Milk       130 BPM       A minor       3:32", 148, yh + 128, 42,
       kit.hx("9FB2C0"), kit.F_MONO, 0.86, 2)

    # 镜头目录（两列）
    hl = ["BOOT_1", "WORLD", "POINTS", "SINE", "INFINITY", "TRAVEL", "CHORUS1", "EGGPLANT",
          "GOD", "ROLE", "TRANCE", "CHORUS2", "CHALLENGE", "EXEC14", "ONLY_EXEC", "ALGEBRA",
          "TRAPPED_LOVE", "ENDCARD"]
    rows = [(s[0], s[2]) for s in film2.timeline() if s[2] in hl][:18]
    yidx = yh + 250
    tx("SCENE INDEX", 148, yidx, 30, kit.hx("7FE3D4"), kit.F_MONO, 0.7, 4)
    hline(148, 2010, yidx + 40, "35505F", 0.9, 1)
    for i, (tt, nm) in enumerate(rows[:9]):
        yy = yidx + 74 + i * 42
        m, s = divmod(int(round(tt)), 60)
        tx("%02d:%02d" % (m, s), 148, yy, 29, kit.hx("4E6472"), kit.F_MONO, 0.8, 2)
        tx(nm.replace("_", " "), 298, yy, 29, kit.hx("9FB2C0"), kit.F_MONO, 0.72, 2)
    for i, (tt, nm) in enumerate(rows[9:]):
        yy = yidx + 74 + i * 42
        m, s = divmod(int(round(tt)), 60)
        tx("%02d:%02d" % (m, s), 1130, yy, 29, kit.hx("4E6472"), kit.F_MONO, 0.8, 2)
        tx(nm.replace("_", " "), 1280, yy, 29, kit.hx("9FB2C0"), kit.F_MONO, 0.72, 2)

    # 技术说明
    yc = yidx + 74 + 9 * 42 + 56
    tx("全部由 Python 逐帧计算生成：62000 颗粒子 · 51 个镜头 · 没有一张素材被重绘",
       148, yc, 38, kit.hx("C9D6DE"), kit.F_SANS_SC, 0.9, 1)
    tx("3840 × 2160 · 60 fps · 12 739 frames · H.264 4:2:0 · 130.00 BPM",
       148, yc + 56, 33, kit.hx("6E828F"), kit.F_MONO, 0.75, 1.2)

    tx("角色素材 CC BY-NC-SA 4.0　上善 → ZipZipPipe → Small-tailqwq（maid-atelier）",
       148, yc + 156, 31, kit.hx("5A6B77"), kit.F_SANS_SC, 0.72, 1.5)
    tx("galgame 立绘 by JAdpp　·　音乐 Mili《world.execute(me);》　·　非商业同人作品",
       148, yc + 200, 31, kit.hx("5A6B77"), kit.F_SANS_SC, 0.65, 1.5)

    out = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Image.fromarray(out).save(a.out, quality=94, subsampling=1)
    print("poster ->", a.out, out.shape)


if __name__ == "__main__":
    main()
