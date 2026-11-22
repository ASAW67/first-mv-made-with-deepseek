# -*- coding: utf-8 -*-
"""qc.py — 预览：按时间点渲染静帧，拼成联系表（contact sheet）。
用法：
  python qc.py 0,2,5,9 ...        # 时间点
  python qc.py --range 15,30,0.8  # 起,止,步长
可选 --scale 1 --cols 3 --out qc.png --w 640
"""
import sys, time, argparse
import numpy as np
from PIL import Image
import kit, mus


def parse(a):
    p = argparse.ArgumentParser()
    p.add_argument("times", nargs="*", default=[])
    p.add_argument("--range", dest="rng", default=None)
    p.add_argument("--scale", type=int, default=1)
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--w", type=int, default=640)
    p.add_argument("--out", default="qc.png")
    p.add_argument("--ch", default=None)
    return p.parse_args(a)


def main():
    a = parse(sys.argv[1:])
    if a.rng:
        t0, t1, st = [float(x) for x in a.rng.split(",")]
        ts = list(np.arange(t0, t1 + 1e-6, st))
    else:
        ts = [float(x) for x in ",".join(a.times).split(",") if x.strip()]
    kit.init(a.scale)
    import film2
    film2.load_chapters(a.ch.split(",") if a.ch else None)
    print("scenes", len(film2.SCENES))
    tiles = []
    t_start = time.time()
    for t in ts:
        t0 = time.time()
        f, c = film2.render(t, a.scale)
        ms = (time.time() - t0) * 1000
        img = kit.to_u8(f)
        img = kit.up(img.astype(np.float32) / 255, a.w, int(a.w * 9 / 16))
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(a.out.replace(".png", "_%06.2f.png" % t))
        tiles.append((t, img, ms, c.scene))
        print("  %7.2f  %-14s %6.0f ms" % (t, c.scene, ms))
    cols = a.cols
    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0][1].shape[:2]
    sheet = np.zeros((rows * (th + 22), cols * tw, 3), np.uint8)
    for i, (t, img, ms, nm) in enumerate(tiles):
        r, cc = divmod(i, cols)
        y = r * (th + 22)
        sheet[y:y + th, cc * tw:(cc + 1) * tw] = img
    Image.fromarray(sheet).save(a.out)
    print("total %.1fs -> %s" % (time.time() - t_start, a.out))


if __name__ == "__main__":
    main()
