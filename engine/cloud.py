# -*- coding: utf-8 -*-
"""cloud.py — 粒子云引擎：形状采样 / 形变 / 透视投影 / 加法 splat / 景深。
坐标系：世界为右手系，x 右, y 上, z 朝观众；相机在 z=-dist 处朝 +z 看。
形状函数统一返回 (N,3) float32 世界坐标，尺度约 ±320 单位。
"""
import math
import numpy as np
import cv2
from PIL import Image
import kit
from kit import W, H, S, SW, SH

F32 = np.float32
GOLDEN = 2.399963229728653


# ---------------------------------------------------------------- 采样工具
def fib_sphere(n, rng=None):
    i = np.arange(n, dtype=F32) + 0.5
    y = 1.0 - 2.0 * i / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    th = i * GOLDEN
    return np.stack([r * np.cos(th), y, r * np.sin(th)], 1).astype(F32)


def sample_mask(mask, n, width_world, depth_world=6.0, rng=None, center=(0.5, 0.5),
                curvature=0.0, z_bias=0.0):
    """在 2D mask 内按 alpha 概率采样 n 个点，映射到世界平面 (宽 width_world)。"""
    h, w = mask.shape
    m = mask.astype(np.float64)
    p = m.ravel() + 1e-6
    c = np.cumsum(p); c /= c[-1]
    rs = rng or np.random.default_rng(12345)
    idx = np.searchsorted(c, rs.random(n))
    yy, xx = np.divmod(idx, w)
    fx = (xx + rs.random(n)) / w
    fy = (yy + rs.random(n)) / h
    x = (fx - center[0]) * width_world
    y = (center[1] - fy) * width_world * (h / w)
    z = rs.normal(0, depth_world, n) + z_bias
    if curvature:
        z += curvature * (x * x / (width_world ** 2)) * width_world
    return np.stack([x, y, z], 1).astype(F32)


# ---------------------------------------------------------------- 形状库
def shape_dots(n, rng, spread=(520, 300, 260)):
    return (rng.random((n, 3)) - 0.5) * np.array(spread, F32)


def shape_sphere(n, rng, R=300.0, thick=0.0):
    p = fib_sphere(n, rng) * R
    if thick:
        p *= (1 + rng.normal(0, thick, (n, 1))).astype(F32)
    return p


def shape_ball(n, rng, R=300.0):
    """实心球（体积采样）。"""
    p = fib_sphere(n, rng)
    rr = (rng.random(n).astype(F32) ** (1 / 3.0))[:, None]
    return p * rr * R


def shape_ring(n, rng, R=300.0, tube=6.0, plane="xy"):
    th = rng.random(n) * 2 * np.pi
    rr = R + rng.normal(0, tube, n)
    x = rr * np.cos(th); y = rr * np.sin(th); z = rng.normal(0, tube * 1.4, n)
    if plane == "xz":
        return np.stack([x, z, y], 1).astype(F32)
    if plane == "yz":
        return np.stack([z, y, x], 1).astype(F32)
    return np.stack([x, y, z], 1).astype(F32)


def shape_circle_filled(n, rng, R=300.0):
    r = R * np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * np.pi
    return np.stack([r * np.cos(th), r * np.sin(th), rng.normal(0, 3, n)], 1).astype(F32)


def shape_box_wire(n, rng, E=300.0):
    e = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [1, 2], [2, 0], [2, 1], [2, 0], [2, 1], [2, 2], [0, 2]], F32)
    corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], F32) * E
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    pi = rng.integers(0, len(pairs), n)
    tt = rng.random(n).astype(F32)[:, None]
    a = corners[pi, :]; b = corners[[pairs[i][1] for i in pi], :]
    return (a + (b - a) * tt).astype(F32)


def shape_box_fill(n, rng, E=300.0):
    p = (rng.random((n, 3)) * 2 - 1) * E
    # 只保留表面附近
    return p


def shape_sine(n, rng, L=560.0, A=110.0, k=2.0, phase=0.0, tube=4.0):
    x = (rng.random(n) - 0.5) * L
    y = A * np.sin(k * x / (L / 2) * np.pi + phase)
    z = rng.normal(0, tube, n)
    return np.stack([x, y, z], 1).astype(F32)


def shape_lemniscate(n, rng, A=300.0, tube=5.0):
    t = rng.random(n) * 2 * np.pi
    d = 1 + np.sin(t) ** 2
    x = A * np.cos(t) / d
    y = A * np.sin(t) * np.cos(t) / d
    z = rng.normal(0, tube, n)
    return np.stack([x, y, z] * 1, 1).astype(F32)


def shape_heart(n, rng, A=15.0):
    t = rng.random(n) * 2 * np.pi
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    fill = np.sqrt(rng.random(n))
    x = x * fill; y = y * fill
    z = rng.normal(0, 0.9, n) * (1 - fill)
    return (np.stack([x, y, z], 1) * (A * 1.0)).astype(F32)


def shape_eggplant(n, rng, Len=250.0, R=92.0, grow=0.55):
    u = fib_sphere(n, rng)
    yy = u[:, 1]
    taper = 1.0 - grow * np.clip(yy * 0.5 + 0.5, 0, 1) ** 2
    x = u[:, 0] * R * taper
    z = u[:, 2] * R * taper
    y = yy * Len
    # 蒂
    stem = rng.random(n) < 0.06
    y = np.where(stem, Len + 26 + rng.normal(0, 10, n), y)
    x = np.where(stem, rng.normal(0, 5, n), x)
    z = np.where(stem, rng.normal(0, 5, n), z)
    return np.stack([x, y * 1.0 - 40, z], 1).astype(F32)


def shape_tomato(n, rng, R=175.0, lobes=5):
    u = fib_sphere(n, rng)
    th = np.arctan2(u[:, 2], u[:, 0])
    wob = 1.0 + 0.055 * np.cos(lobes * th) * np.clip(u[:, 1] + 0.6, 0, 1)
    p = u * (R * wob)[:, None]
    p[:, 1] = p[:, 1] * 0.82 + 4
    # 萼片
    cal = rng.random(n) < 0.10
    if cal.any():
        k = cal.sum()
        ang = rng.random(k) * 2 * np.pi
        rr = rng.random(k) * 0.55
        p[cal, 0] = np.cos(ang) * rr * R * 0.8
        p[cal, 2] = np.sin(ang) * rr * R * 0.8
        p[cal, 1] = R * 0.76 + rng.normal(0, 6, k)
    return p.astype(F32)


def _mask_from_draw(w, h, fn):
    m = np.zeros((h, w), F32)
    fn(m)
    return m


def shape_cat(n, rng, size=520.0):
    """虎斑猫剪影（躯干+头+尾+耳），用 mask 采样。"""
    w, h = 256, 256
    def draw(m):
        cv2.ellipse(m, (110, 150), (62, 44), 0, 0, 360, 1.0, -1)          # 躯干
        cv2.circle(m, (176, 116), 34, 1.0, -1)                            # 头
        cv2.fillPoly(m, [np.int32([[152, 100], [163, 62], [180, 96]])], 1.0)   # 左耳
        cv2.fillPoly(m, [np.int32([[176, 96], [192, 58], [204, 100]])], 1.0)   # 右耳
        cv2.line(m, (56, 168), (24, 208), 1.0, 7)                          # 尾
        cv2.line(m, (24, 208), (16, 182), 1.0, 6)
        cv2.line(m, (10, 182), (26, 172), 1.0, 5)
        cv2.line(m, (92, 190), (86, 224), 1.0, 7)                          # 腿
        cv2.line(m, (128, 190), (134, 224), 1.0, 7)
        cv2.line(m, (60, 190), (56, 222), 1.0, 6)
    m = _mask_from_draw(w, h, draw)
    m = cv2.GaussianBlur(m, (3, 3), 0.8)
    return sample_mask(m, n, size, depth_world=8, rng=rng, center=(0.5, 0.52))


def shape_god(n, rng, size=760.0):
    """神：长袍剪影 + 头颅 + 缓缓抬起的手 + 光环。"""
    w, h = 200, 320
    def draw(m):
        # 长袍：从肩到地的 A 字
        cv2.fillPoly(m, [np.int32([[100, 96], [76, 104], [58, 150], [34, 260], [26, 316],
                                   [174, 316], [166, 260], [142, 150], [124, 104]])], 1.0)
        # 颈
        cv2.rectangle(m, (94, 80), (106, 100), 1.0, -1)
        # 头（椭球）
        cv2.ellipse(m, (100, 56), (17, 22), 0, 0, 360, 1.0, -1)
        # 手臂：一手垂、一手上举
        cv2.line(m, (74, 118), (44, 196), 1.0, 11)
        cv2.line(m, (126, 118), (156, 88), 1.0, 11)
        cv2.circle(m, (158, 84), 8, 1.0, -1)
    m = _mask_from_draw(w, h, draw)
    m = cv2.GaussianBlur(m, (5, 5), 1.3)
    p = sample_mask(m, n, size, depth_world=12, rng=rng, center=(0.5, 0.5), curvature=0.22)
    # 光环
    halo = rng.random(n) < 0.20
    if halo.any():
        k = int(halo.sum())
        ang = rng.random(k) * 2 * np.pi
        rr = 0.205 * size + rng.normal(0, 3.5, k)
        p[halo, 0] = np.cos(ang) * rr
        p[halo, 1] = size * 0.335 + np.sin(ang) * rr * 0.22
        p[halo, 2] = rng.normal(0, 5, k)
    return p


def shape_text(n, rng, txt, size=300.0, fp=None, tracking=6.0, depth=9.0, vertical=False):
    """size = 逻辑像素字高（世界单位）。"""
    m = up_mask(kit.text_mask(fp or kit.F_SERIF_B, size, txt, tracking))
    h, w = m.shape
    width_world = size * (w / h) * 1.02
    p = sample_mask(m, n, width_world, depth_world=depth, rng=rng)
    if vertical:
        p = p[:, [1, 0, 2]]
        p[:, 0] *= -1
    return p.astype(F32)


def up_mask(m, maxw=1400):
    h, w = m.shape
    if w <= maxw:
        return m
    return kit.up(m, maxw, int(h * maxw / w))


def sample_mask_idx(mask, n, rng=None):
    h, w = mask.shape
    m = mask.astype(np.float64)
    p = m.ravel() + 1e-6
    c = np.cumsum(p); c /= c[-1]
    rs = rng or np.random.default_rng(12345)
    idx = np.searchsorted(c, rs.random(n))
    yy, xx = np.divmod(idx, w)
    fx = (xx + rs.random(n)) / w
    fy = (yy + rs.random(n)) / h
    return fx, fy, yy, xx


def sprite_field(n, rng, path, height_world=620.0, depth=34.0, maxw=460, alpha_key=0.28,
                 edge=1.4, soft=1.2):
    """把立绘同时变成「位置 + 自带颜色/亮度」的粒子场：角色由自己的光影点亮。"""
    a = kit.trim(kit.load(path))
    rgb = a[..., :3].copy(); m0 = a[..., 3]
    h, w = m0.shape
    if w > maxw:
        nw = maxw; nh = int(h * maxw / w)
        m = kit.up(m0, nw, nh)
        rgb = kit.up(rgb, nw, nh)
    else:
        m = m0
    m = np.clip((m - alpha_key) / (1 - alpha_key), 0, 1)
    if soft:
        m = kit.blur(m, soft)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    e = np.sqrt(gx * gx + gy * gy); e /= (e.max() + 1e-6)
    wgt = m * (0.6 + edge * e)
    fx, fy, yy, xx = sample_mask_idx(wgt, n, rng)
    H, Wd = m.shape
    # 双线性取色
    xs = np.clip((fx * Wd).astype(np.int32), 0, Wd - 1)
    ys = np.clip((fy * H).astype(np.int32), 0, H - 1)
    col = rgb[ys, xs]
    lum = (0.299 * col[:, 0] + 0.587 * col[:, 1] + 0.114 * col[:, 2]).astype(F32)
    lum = np.clip((lum - 0.04) / 0.8, 0, 1) ** 0.85
    P = np.stack([(fx - 0.5) * height_world * (Wd / H),
                  (0.5 - fy) * height_world,
                  rng.normal(0, depth, n)], 1).astype(F32)
    return {"P": P, "rgb": col.astype(F32), "lum": lum.astype(F32),
            "edge": e[ys, xs].astype(F32)}


def shape_fragments(n, rng, spread=420.0, clusters=26):
    c = (rng.random((clusters, 3)) - 0.5) * spread * 1.6
    ci = rng.integers(0, clusters, n)
    off = rng.normal(0, 9, (n, 3))
    return (c[ci] + off).astype(F32)


def shape_spiral(n, rng, R=320.0, arms=3, twist=3.4):
    t = rng.random(n) ** 0.7 * 1.0
    arm = rng.integers(0, arms, n)
    ang = t * twist * 2 * np.pi + arm * (2 * np.pi / arms)
    r = R * t
    y = rng.normal(0, 8, n) * (1 - t * 0.6)
    return np.stack([r * np.cos(ang), y, r * np.sin(ang)], 1).astype(F32)


def shape_tunnel(n, rng, R=300.0, L=1800.0):
    th = rng.random(n) * 2 * np.pi
    r = R * (0.75 + 0.5 * rng.random(n))
    z = rng.random(n) * L - L * 0.35
    return np.stack([r * np.cos(th), r * np.sin(th), z], 1).astype(F32)


def shape_grid(n, rng, size=760.0, y=-220.0, jit=4.0):
    side = int(np.sqrt(n)) + 1
    gx, gz = np.meshgrid(np.linspace(-0.5, 0.5, side), np.linspace(-0.5, 0.5, side))
    p = np.stack([gx.ravel(), np.zeros(side * side), gz.ravel()], 1)[:n] * np.array([size, 1, size], F32)
    p[:, 1] = y + rng.normal(0, jit, len(p))
    return p.astype(F32)


def shape_wave_field(n, rng, size=760.0, y=0.0, A=26.0, k=2.2):
    side = int(np.sqrt(n)) + 1
    gx, gz = np.meshgrid(np.linspace(-0.5, 0.5, side), np.linspace(-0.5, 0.5, side))
    x = gx.ravel()[:n] * size; z = gz.ravel()[:n] * size
    yy = y + A * np.sin(k * x / size * 6.283) * np.cos(k * z / size * 6.283)
    return np.stack([x, yy, z], 1).astype(F32)


def shape_lattice(n, rng, size=900.0, y=-260.0, depth=0.0, jit=6.0, skip=0.12):
    """稀疏平面点阵（落到地面上的「物件」）。"""
    side = int(np.sqrt(n / max(1e-6, 1 - skip))) + 1
    gx, gz = np.meshgrid(np.linspace(-0.5, 0.5, side), np.linspace(-0.5, 0.5, side))
    p = np.stack([gx.ravel(), np.zeros(side * side), gz.ravel()], 1)
    p = p[rng.random(len(p)) > skip][:n]
    p = p * np.array([size, 1, size], F32)
    p[:, 1] = y + rng.normal(0, jit, len(p))
    p[:, 2] += rng.normal(0, depth, len(p))
    return p.astype(F32)


def shape_column(n, rng, R=95.0, Hh=520.0):
    th = rng.random(n) * 2 * np.pi
    r = R * (0.7 + 0.5 * rng.random(n))
    return np.stack([r * np.cos(th), (rng.random(n) - 0.5) * Hh, r * np.sin(th)], 1).astype(F32)


def shape_figure(n, rng, h=560.0):
    """抽象人形（头/躯干/四肢），用于「程序本体」的极简形态。"""
    w, hh = 120, 300
    def draw(m):
        cv2.circle(m, (60, 40), 17, 1.0, -1)
        cv2.fillPoly(m, [np.int32([[60, 56], [46, 70], [42, 140], [50, 200], [70, 200], [78, 140], [74, 70]])], 1.0)
        cv2.line(m, (48, 78), (22, 150), 1.0, 7)
        cv2.line(m, (72, 78), (98, 150), 1.0, 7)
        cv2.line(m, (52, 198), (44, 292), 1.0, 9)
        cv2.line(m, (68, 198), (78, 292), 1.0, 9)
    m = _mask_from_draw(w, hh, draw)
    m = cv2.GaussianBlur(m, (5, 5), 1.1)
    return sample_mask(m, n, h * (w / hh) * 1.9, depth_world=8, rng=rng, center=(0.5, 0.5), curvature=0.2)


# ---------------------------------------------------------------- 云
class Cloud:
    def __init__(self, n=170000, seed=7):
        self.n = n
        rs = np.random.default_rng(seed)
        self.rng = rs
        self.u = rs.random(n).astype(F32)
        self.v = rs.random(n).astype(F32)
        self.rnd = rs.random(n).astype(F32)
        self.seed = seed
        self.pos = shape_dots(n, rs)
        self.prev = self.pos.copy()

    # ---- 形状缓存（同参只算一次）
    def sh(self, fn, *a, **kw):
        key = (getattr(fn, "__name__", str(fn)), a, tuple(sorted((k, repr(v)) for k, v in kw.items())))
        cache = getattr(self, "_c", None)
        if cache is None:
            cache = self._c = {}
        if key in cache:
            return cache[key]
        p = fn(self.n, self.rng, *a, **kw)
        cache[key] = p
        return p

    # ---- 逐粒子错峰插值
    def morph(self, A, B, p, stagger=0.55, ease="io", curve=None):
        """p: 0..1 全局进度；stagger: 每粒子延迟比例（0=同时）。"""
        if p <= 0:
            return A
        if p >= 1 and stagger <= 0:
            return B
        u = self.u
        if stagger > 0:
            d = (u * stagger)
            q = np.clip((p - d) / max(1e-4, 1 - stagger), 0, 1)
        else:
            q = np.full(self.n, cl01(p), F32)
        if curve is not None:
            q = curve(q)
        elif ease == "io":
            q = q * q * (3 - 2 * q)
        elif ease == "out":
            q = 1 - (1 - q) ** 3
        elif ease == "in":
            q = q ** 3
        elif ease == "back":
            u2 = q - 1
            q = 1 + u2 * u2 * (2.7 * u2 + 1.7) * 0.5
        elif ease == "quint":
            q = 1 - (1 - q) ** 5
        q = q.astype(F32)[:, None]
        d = B - A
        # 弧线：加一点点绕轴旋转的旁路，避免直线穿插
        return A + d * q

    def morph_arc(self, A, B, p, stagger=0.55, bow=0.0, axis=(0, 1, 0)):
        base = self.morph(A, B, p, stagger, "io")
        if bow <= 0:
            return base
        q = np.clip(p, 0, 1)
        s = math.sin(q * math.pi) * bow
        n = np.cross(B - A, np.array(axis, F32))
        nl = np.linalg.norm(n, axis=1, keepdims=True) + 1e-6
        return base + (n / nl) * s

    # ---- 世界变换
    @staticmethod
    def transform(P, yaw=0.0, pitch=0.0, roll=0.0, scale=1.0, offset=(0, 0, 0)):
        if yaw:
            c, s = math.cos(yaw), math.sin(yaw)
            R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], F32)
            P = P @ R.T
        if pitch:
            c, s = math.cos(pitch), math.sin(pitch)
            R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], F32)
            P = P @ R.T
        if roll:
            c, s = math.cos(roll), math.sin(roll)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], F32)
            P = P @ R.T
        if scale != 1.0:
            P = P * scale
        o = np.asarray(offset, F32)
        if o.any():
            P = P + o
        return P

    # ---- 投影
    @staticmethod
    def project(P, dist=1400.0, focal=1150.0, cx=None, cy=None, oy=0.0):
        cx = W / 2 if cx is None else cx
        cy = H / 2 if cy is None else cy
        z = P[:, 2] + dist
        z = np.maximum(z, 40.0)
        k = focal * (S / 1.0) * (S / S) / z
        k = focal / z
        sx = cx + P[:, 0] * k
        sy = cy + oy - P[:, 1] * k
        return sx, sy, z, k

    # ---- splat 渲染
    def draw(self, f, P, color=1.0, size=1.6, dist=1400.0, focal=1150.0, cx=None, cy=None,
             oy=0.0, intensity=0.24, blur_r=2.6, half=True, additive=True, dof=0.0,
             color_map=None, cutoff=None, shade=0.0, bright=None, twinkle=None, tm=2.0,
             core_gain=1.0, glow_gain=1.0, auto=True, target=0.9, exposure=1.0):
        sx, sy, z, k = self.project(P, dist, focal, cx, cy, oy)
        _SW, _SH = kit.W * kit.S, kit.H * kit.S
        bw, bh = (_SW // 2, _SH // 2) if half else (_SW, _SH)
        hs = 2.0 if half else 1.0
        xs = sx * (bw / kit.W)
        ys = sy * (bh / kit.H)
        keep = (xs > -30) & (xs < bw + 30) & (ys > -30) & (ys < bh + 30)
        if cutoff is not None:
            keep &= z < cutoff
        m = int(keep.sum())
        if m < 4:
            return f
        xs = xs[keep]; ys = ys[keep]; zz = z[keep]; kk = k[keep]
        rnd = self.rnd[keep]
        if color_map is not None:
            col = np.asarray(color_map(xs, ys, zz, kk), F32)
            if col.ndim == 1:
                col = np.repeat(col[None, :], m, 0)
        else:
            cv = np.asarray(color, F32)
            if cv.ndim == 0 or cv.size == 1:
                cv = np.repeat(cv.reshape(1, 1), 3, 1)
            col = np.tile(cv.reshape(1, 3), (m, 1))
        b = 0.30 + 0.70 * (rnd ** 1.6)
        if bright is not None:
            b = b * (bright if np.ndim(bright) == 0 else np.asarray(bright, F32)[keep])
        if twinkle:
            ph, amt = twinkle
            b = b * (1 + amt * np.sin(ph + rnd * 6.283))
        if shade:
            kmean = focal / dist
            b = b * np.clip(1 + shade * (kk / kmean - 1.0), 0.15, 2.6)
        sp = 1.0 if size <= 1.4 else min(3.0, size * 0.55)
        ix = np.floor(xs).astype(np.int32); iy = np.floor(ys).astype(np.int32)
        fx = xs - ix; fy = ys - iy
        idxs, wgts = [], []
        offs = ((0, 0), (1, 0), (0, 1), (1, 1))
        if sp > 1.3:
            offs = offs + ((2, 0), (0, 2), (2, 2), (1, 2), (2, 1))
        for dx, dy in offs:
            xi = np.clip(ix + dx, 0, bw - 1); yi = np.clip(iy + dy, 0, bh - 1)
            idxs.append(yi * bw + xi)
            if dx < 2 and dy < 2:
                w = ((1 - fx) if dx == 0 else fx) * ((1 - fy) if dy == 0 else fy)
            else:
                w = 0.20 * np.clip(1.0 - np.minimum(np.abs(dx - fx), np.abs(dy - fy)), 0, 1)
            wgts.append(np.maximum(w, 0) * b)
        idx = np.concatenate(idxs)
        wgt = np.concatenate(wgts) * intensity
        c3 = np.concatenate([col] * len(offs))
        size_px = bw * bh
        buf = np.empty((bh, bw, 3), F32)
        for c in range(3):
            buf[..., c] = np.bincount(idx, weights=wgt * c3[:, c], minlength=size_px).astype(F32).reshape(bh, bw)
        # 曝光自动标定：按自身亮度分位数归一，避免密集处死白、稀疏处发灰
        if auto:
            nz = buf[buf > 1e-4]
            if nz.size > 16:
                ref = float(np.percentile(nz, 96))
                buf *= float(np.clip(target * exposure / max(ref, 1e-5), 0.05, 12.0))
        core = buf
        out = core * core_gain
        if blur_r > 0.3 and glow_gain > 0:
            out = out + kit.blur(core, blur_r * hs * 0.5) * glow_gain
        out = kit.tonemap_soft(out, tm)
        out = kit.up(out, _SW, _SH)
        if additive:
            f += out
        else:
            a = np.clip(out.max(axis=2, keepdims=True), 0, 1)
            f *= (1 - a); f += out
        return f


def cl01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)
