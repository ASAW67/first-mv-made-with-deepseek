# -*- coding: utf-8 -*-
"""render2.py — 多进程分段渲染 + 拼接 + 混音。

python render2.py --scale 1 --workers 10 --out out/preview_1080p.mp4
python render2.py --scale 2 --workers 14 --out out/world.execute(me)_4K60.mp4 --crf 15
python render2.py --range 30,60 --scale 1 --out out/cut_30_60.mp4     # 局部试渲染（不混音）
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FFMPEG = r"D:\AI_SS\Ying_Shi\.tools\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe"
FFPROBE = r"D:\AI_SS\Ying_Shi\.tools\ffmpeg-9.0.2-essentials_build\bin\ffprobe.exe"
AUDIO = os.environ.get("MV_AUDIO", os.path.join(kit.ASSETS, "Mili - world.execute(me);.m4a"))
FPS = 60
DUR = 212.323265


def worker(job):
    idx, frames, scale, tmp, crf = job
    import numpy as np
    import kit
    import film2

    kit.init(scale)
    film2.load_chapters()
    if not film2.SCENES:
        raise RuntimeError("no scenes loaded")
    w, h = kit.W * scale, kit.H * scale
    path = os.path.join(tmp, "seg_%04d.mp4" % idx)
    cmd = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (w, h), "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-g", "120", "-threads", "4",
           "-x264-params", "aq-mode=3:psy-rd=1.0,0.15", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in frames:
        t = i / FPS
        f, _ = film2.render(t, scale=scale)
        buf = (np.clip(f, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        p.stdin.write(buf.tobytes())
    p.stdin.close()
    p.wait()
    return idx, path, time.time() - t0, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--range", default=None, help="a,b 秒")
    ap.add_argument("--chunk", type=int, default=180)
    ap.add_argument("--audio", type=int, default=1)
    ap.add_argument("--resume", type=int, default=0, help="跳过已完成的段")
    a = ap.parse_args()

    if a.range:
        t0, t1 = [float(x) for x in a.range.split(",")]
    else:
        t0, t1 = 0.0, DUR
    f0, f1 = int(round(t0 * FPS)), min(int(round(t1 * FPS)), int(round(DUR * FPS)))
    allf = list(range(f0, f1))
    chunks = [allf[i:i + a.chunk] for i in range(0, len(allf), a.chunk)]
    tmp = os.path.join(HERE, "_render_tmp")
    os.makedirs(tmp, exist_ok=True)
    out = a.out or os.path.join(HERE, "out", "mv_%dp.mp4" % (2160 if a.scale == 2 else 1080))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    print("frames %d (%d chunks) scale %d workers %d" % (len(allf), len(chunks), a.scale, a.workers))
    jobs = [(i, ch, a.scale, tmp, a.crf) for i, ch in enumerate(chunks)]
    if a.resume:
        have = 0
        rest = []
        for j in jobs:
            p = os.path.join(tmp, "seg_%04d.mp4" % j[0])
            ok = os.path.exists(p) and os.path.getsize(p) > 1 << 20
            if ok:
                # 被杀掉的 ffmpeg 会留下没有 moov 的残段：用帧数确认
                r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                                    "-count_frames", "-show_entries", "stream=nb_read_frames",
                                    "-of", "csv=p=0", p], capture_output=True, text=True)
                got = r.stdout.strip()
                ok = got.isdigit() and int(got) == len(j[1])
                if not ok:
                    print("  drop broken seg %04d (%s frames)" % (j[0], got or "?"))
            if ok:
                have += 1
            else:
                rest.append(j)
        print("resume: %d chunks already done, %d to go" % (have, len(rest)))
        jobs = rest
        done0 = have * a.chunk
    else:
        done0 = 0
    t0c = time.time()
    done = done0
    with mp.Pool(a.workers) as pool:
        for idx, path, dt, n in pool.imap_unordered(worker, jobs):
            done += n
            el = time.time() - t0c
            print("  seg %3d  %4d frames  %6.1fs  |  %d/%d  eta %.0fs"
                  % (idx, n, dt, done, len(allf), el / max(1, done) * (len(allf) - done)), flush=True)
    pool.join()

    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        for i in range(len(chunks)):
            fh.write("file '%s'\n" % os.path.join(tmp, "seg_%04d.mp4" % i).replace("\\", "/"))
    silent = os.path.join(tmp, "video.mp4")
    print("concat ->", silent)
    subprocess.run([FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", silent], check=True)
    if a.audio and not a.range:
        print("mux audio ->", out)
        subprocess.run([FFMPEG, "-y", "-v", "error", "-i", silent, "-i", AUDIO,
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                        "-c:a", "aac", "-b:a", "320k", "-shortest", "-movflags", "+faststart",
                        out], check=True)
    else:
        shutil.copyfile(silent, out)
    print("done %.1fs -> %s" % (time.time() - t0c, out))
    print("size %.1f MB" % (os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
