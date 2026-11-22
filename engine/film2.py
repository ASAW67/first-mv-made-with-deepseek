# -*- coding: utf-8 -*-
"""film2.py — 总装：时间线、场景注册、逐帧渲染、全局后期。
「一次 execute 请求的一生」。
"""
import importlib, math, os, sys, time
import numpy as np
import kit, mus, stage, cloud

F32 = np.float32
SCENES = []


def scene(t0, t1, name, chapter=""):
    def deco(fn):
        SCENES.append({"t0": float(t0), "t1": float(t1), "fn": fn, "name": name, "ch": chapter})
        return fn
    return deco


def load_chapters(names=None):
    """导入章节模块以注册场景。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for fn in sorted(os.listdir(here)):
        if not (fn.startswith("ch_") and fn.endswith(".py")):
            continue
        mod = fn[:-3]
        if names and mod not in names:
            continue
        if mod in sys.modules:
            continue
        importlib.import_module(mod)
    SCENES.sort(key=lambda s: s["t0"])


class Ctx:
    """单帧上下文。所有随机性都由 seed 派生，保证可重复。"""

    def __init__(self, t, scale=1, cloud_n=62000):
        self.t = t
        self.f = kit.new_frame(kit.BLACK)
        self.cl = cloud.Cloud(cloud_n, seed=7)
        self.frame_i = int(round(t * 60))
        self.post = {"grain": 0.020, "vignette": 0.42, "bloom": (0.60, 22, 0.45),
                     "chroma": 0.0, "exposure": 1.0, "scan": 0.0, "letterbox": 0.0,
                     "sat": 1.0, "contrast": 1.0}
        self.scene = None
        self.local = 0.0
        self.span = 1.0
        self.theme = stage.TH_DEEP
        self._extra = {}

    # ---------- 便捷访问
    def rng(self, salt=0):
        return np.random.default_rng(self.frame_i * 7919 + salt * 104729 + 17)

    def band(self, n, win=0.3):
        return mus.env(n, self.t, win)

    def beat(self):
        return mus.beat_phase(self.t)

    def pulse(self, decay=9.0):
        return mus.beat_pulse(self.t, decay)

    def kick(self):
        return mus.kick_near(self.t)

    def rms(self):
        return mus.rms_env(self.t)

    def lv(self, a, b):
        """local 进度（场景内 0..1，带裁剪）。"""
        return kit.cl((self.t - a) / max(1e-4, b - a))

    def line(self):
        return mus.line_at(self.t)

    def lyric(self, **kw):
        ln = kw.pop("ln", None) or self.line()
        kw.setdefault("y", kit.H * 0.855)
        return stage.lyric(self.f, self.t, ln, **kw)

    def text(self, *a, **kw):
        return kit.draw_text(self.f, *a, **kw)

    def fade(self, a, b, mode="in"):
        """场景内淡入淡出系数。"""
        p = self.lv(a, b)
        return p if mode == "in" else 1 - p

    def shake(self, amt, seed=3):
        """整帧位移（拍驱动的微抖）。"""
        if amt <= 0.01:
            return
        rs = self.rng(seed)
        dx = float(rs.normal(0, amt * kit.S)); dy = float(rs.normal(0, amt * kit.S))
        self.f = np.roll(np.roll(self.f, int(dy), axis=0), int(dx), axis=1)

    def roll(self, ang):
        """整帧旋转（用于眩晕）。"""
        if abs(ang) < 1e-4:
            return
        self.f = kit.scale_about(self.f, kit.SW / 2, kit.SH / 2, 1.06, ang)

    def zoom_frame(self, k, ang=0.0):
        self.f = kit.scale_about(self.f, kit.SW / 2, kit.SH / 2, k, ang)


def scene_at(t):
    best = None
    for s in SCENES:
        if s["t0"] <= t < s["t1"]:
            return s
    return SCENES[-1] if SCENES and t >= SCENES[-1]["t1"] else None


def render(t, scale=1, cloud_n=62000, post=True):
    s = scene_at(t)
    c = Ctx(t, scale, cloud_n)
    c.scene = s["name"] if s else "?"
    if s:
        c.local = t - s["t0"]; c.span = s["t1"] - s["t0"]
        s["fn"](c)
    f = c.f
    if post:
        po = c.post
        if po.get("bloom"):
            th, r, st = po["bloom"]
            f = kit.bloom(f, th, r * kit.S / 1.0, st)
        if po.get("chroma"):
            f = kit.chroma(f, po["chroma"], 0.15)
        if po.get("scan"):
            kit.scanlines(f, po["scan"])
        if po.get("sat", 1) != 1 or po.get("contrast", 1) != 1:
            f = kit.tone(f, 1.0, po.get("contrast", 1.0), po.get("sat", 1.0))
        if po.get("letterbox"):
            stage.letterbox(f, po["letterbox"])
        if po.get("vignette"):
            kit.vignette(f, po["vignette"], 2.3)
        if po.get("grain"):
            kit.grain(f, po["grain"], c.frame_i % 997)
    return f, c


def timeline():
    return [(s["t0"], s["t1"], s["name"], s["ch"]) for s in SCENES]


def report():
    kit.init(1)
    load_chapters()
    print("scenes:", len(SCENES))
    prev = 0.0
    for s in SCENES:
        flag = "" if abs(s["t0"] - prev) < 0.02 else "  <-- gap %.2f" % (s["t0"] - prev)
        print("%7.2f  %7.2f  %-22s %s%s" % (s["t0"], s["t1"], s["name"], s["ch"], flag))
        prev = s["t1"]
    print("end %.2f / dur %.2f" % (prev, mus.DUR))


if __name__ == "__main__":
    sys.modules["film2"] = sys.modules["__main__"]
    report()
