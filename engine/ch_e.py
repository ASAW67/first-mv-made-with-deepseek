# -*- coding: utf-8 -*-
"""ch_e.py — 第三幕 A：她把自己变成物件、变成神，然后索要一个存在证明。
73.94 ─ 89.53

EGGPLANT 73.94  竖在宫殿地板上的一颗粒子茄子，她去摸它
TOMATO   78.11  珊瑚红的番茄，表面在流动
CAT      81.21  虎斑猫走过地板，她蹲下来看
GOD      85.16  唯一的神，仰视，紫+金，袍子上粒子往上飘
PROOF    87.41  九条程序化细线，最后一条接到她心口

形状说明：茄子/番茄/猫/神都在本文件内重新实现（不用 cloud.shape_*）。
做法是 2D 剪影 mask → 距离变换「吹鼓」成有厚度的点云（_puff），
剪影就是画出来的那张图，所以一眼可读；法线由平滑后的 z-map 求梯度，
再按视空间光向算 Lambert + 边缘光，旋转时明暗跟着转。
全屏级元件（背景/光柱/尘埃/辉光/径向光）全部走半分辨率再放大：
1080p 单帧压在 900ms 以内。
"""
import math
import numpy as np
import cv2
import kit, mus, stage, cloud
from film2 import scene

F32 = np.float32
FOC = 1150.0
DIST = 1500.0
K = FOC / DIST           # 世界单位 → 屏幕像素


# ================================================================ 2D → 3D 点云
def _mask(w, h, fn, blur=1.1):
    m = np.zeros((h, w), F32)
    fn(m)
    return cv2.GaussianBlur(m, (0, 0), blur) if blur > 0 else m


def _poly(m, pts, v=1.0):
    cv2.fillPoly(m, [np.int32(np.asarray(pts, F32))], float(v))


def _puff(mask, n, u, v, q, r, height_world, depth=40.0, edge=1.05, dref=0.72,
          zpow=0.55, blur=1.0, nz=0.95, zjit=0.30):
    """2D 剪影 → (n,3) 点云：剪影内按边缘加权取样，z 由距离变换鼓成厚度。"""
    ys, xs = np.nonzero(mask > 0.22)
    if len(xs):
        mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = mask.shape
    img = cv2.GaussianBlur(mask, (0, 0), blur) if blur > 0 else mask
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    e = np.sqrt(gx * gx + gy * gy)
    e /= (e.max() + 1e-9)
    wgt = img * (0.45 + edge * e)
    cdf = np.cumsum(wgt.ravel().astype(np.float64)) + 1e-9
    cdf /= cdf[-1]
    idx = np.searchsorted(cdf, u)
    py, px = np.divmod(idx, w)
    pxf = np.clip(px + v, 0, w - 1)
    pyf = np.clip(py + q, 0, h - 1)
    ix = np.clip(px, 0, w - 1)
    iy = np.clip(py, 0, h - 1)

    dt = cv2.distanceTransform((mask > 0.40).astype(np.uint8), cv2.DIST_L2, 5)
    s = np.clip(dt / (dref * (dt.max() + 1e-6)), 0.0, 1.0)
    zmap = (depth * s ** zpow).astype(F32)
    zb = cv2.GaussianBlur(zmap, (0, 0), 2.2)          # 平滑法线，别让边缘过黑
    zx = cv2.Sobel(zb, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    zy = cv2.Sobel(zb, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    sc = height_world / h
    nx = np.clip(-zx[iy, ix] / sc * nz, -1.4, 1.4)
    ny = np.clip(zy[iy, ix] / sc * nz, -1.4, 1.4)
    nzv = np.ones_like(nx)
    ln = np.sqrt(nx * nx + ny * ny + nzv * nzv) + 1e-9
    N = np.stack([nx / ln, ny / ln, nzv / ln], 1).astype(F32)

    X = (pxf - 0.5 * w) * sc
    Y = (0.5 * h - pyf) * sc
    Z = zmap[iy, ix] * (1.0 - zjit + 2.0 * zjit * r)
    P = np.stack([X, Y, Z], 1).astype(F32)
    P[:, 1] += 0.5 * h * sc
    return {"P": P, "N": N, "mask": mask, "sc": sc}


_SH = {}


def _obj(name, seed, builder, n):
    key = (name, n)
    if key not in _SH:
        rs = np.random.default_rng(seed)
        a = rs.random(n); b = rs.random(n); cc = rs.random(n); dd = rs.random(n)
        _SH[key] = builder(n, a, b, cc, dd)
    return _SH[key]


def _shade(N, yaw=0.0, key=1.0, amb=0.44, rim=0.40, dirv=(-0.46, 0.48, 0.74)):
    """视空间光：法线随物体一起转，明暗关系保持正确。"""
    N = cloud.Cloud.transform(N, yaw=yaw)
    L = np.asarray(dirv, F32)
    L = L / (np.linalg.norm(L) + 1e-9)
    lam = np.clip(N @ L, 0.0, 1.0)
    ed = np.clip(1.0 - N[:, 2], 0.0, 1.0)
    return (amb + key * lam ** 0.80 + rim * ed ** 1.6).astype(F32)


def _cmap(stops, cols, y0=None, y1=None):
    """按屏幕 y 的分段渐变 color_map；y0/y1 用逻辑像素给出（内部换算 buffer 坐标）。"""
    stops = np.asarray(stops, F32)
    cols = [np.asarray(c, F32) for c in cols]

    def f(xs, ys, zz, kk):
        if y0 is not None:                       # color_map 收到的是半分辨率 buffer 坐标
            ys = ys * (kit.W / max(1.0, kit.SW * 0.5))
        a = ys.min() if y0 is None else y0
        b = ys.max() if y1 is None else y1
        q = np.clip((ys - a) / max(1e-6, (b - a)), 0, 1)
        out = np.repeat(cols[0][None, :], len(q), 0)
        for i in range(len(stops) - 1):
            m = q >= stops[i]
            u = np.clip((q - stops[i]) / max(1e-6, stops[i + 1] - stops[i]), 0, 1)
            out = np.where(m[:, None], kit.mixc(cols[i], cols[i + 1], u), out)
        return out
    return f


# ---------------------------------------------------------------- 形状：茄子
def _build_eggplant(n, u, v, w, r):
    W_, H_ = 200, 350
    cx = W_ * 0.5

    def draw(m):
        yb, yt = 338.0, 96.0
        R = 74.0
        ys = np.linspace(0.0, 1.0, 96)
        # 最宽在 0.30 处，向上收成细颈，向下收圆底
        prof = (np.clip(ys / 0.085, 0, 1) ** 0.42
                * np.clip(1.0 - ((ys - 0.30) / 0.74) ** 2, 0, 1) ** 0.55)
        rr = prof * R
        yy = yb + (yt - yb) * ys
        bend = 9.0 * np.sin(ys * 1.7)
        right = np.stack([cx + rr + bend * ys, yy], 1)
        left = np.stack([cx - rr + bend * ys, yy], 1)[::-1]
        _poly(m, np.concatenate([right, left]))
        # 蒂：五枚窄尖叶伏在细颈上（「一眼认茄子」的关键）
        hy = yt + 10
        for k in range(5):
            a = -math.pi / 2 + k * 2 * math.pi / 5
            _poly(m, [[cx, hy + 4],
                      [cx + math.cos(a - 0.72) * 20, hy + math.sin(a - 0.72) * 13],
                      [cx + math.cos(a) * 50, hy + math.sin(a) * 27],
                      [cx + math.cos(a + 0.72) * 20, hy + math.sin(a + 0.72) * 13]])
        cv2.polylines(m, [np.int32([(cx + 1, hy - 2), (cx + 7, hy - 20),
                                    (cx + 2, hy - 36), (cx - 5, hy - 48)])],
                      False, 1.0, 11, cv2.LINE_AA)
    return _puff(_mask(W_, H_, draw), n, u, v, w, r, height_world=790.0,
                 depth=58.0, edge=1.25, dref=0.70, zpow=0.55)


# ---------------------------------------------------------------- 形状：番茄
def _build_tomato(n, u, v, w, r):
    W_, H_ = 268, 218
    cx, cy, RX, RY = 134.0, 109.0, 128.0, 104.0      # 1.23:1 扁球

    def draw(m):
        cv2.ellipse(m, (int(cx), int(cy)), (int(RX), int(RY)), 0, 0, 360, 1.0, -1)
        for k in range(5):                            # 肩部微鼓（5 个浅瓣）
            a = -math.pi / 2 + k * 2 * math.pi / 5
            cv2.circle(m, (int(cx + math.cos(a) * RX * 0.60),
                           int(cy + math.sin(a) * RY * 0.60)), int(RX * 0.30), 1.0, -1)
    d = _puff(_mask(W_, H_, draw), n, u, v, w, r, height_world=490.0,
              depth=255.0, edge=0.34, dref=1.05, zpow=0.55, zjit=0.10)
    # 中间一条凹槽：暗肩 + 亮芯（跟着球一起转）
    lx = d["P"][:, 0] - 0.34 * d["P"][:, 2]
    dd = lx - 22.0
    d["groove"] = ((1.0 - 0.26 * np.exp(-(dd ** 2) / (2 * 34.0 ** 2)))
                   * (1.0 + 0.58 * np.exp(-(dd ** 2) / (2 * 8.0 ** 2)))).astype(F32)
    d["P"][:, 2] -= (20.0 * np.exp(-(dd ** 2) / (2 * 34.0 ** 2))).astype(F32)
    return d


# ---------------------------------------------------------------- 形状：虎斑猫
def _build_cat(n, u, v, w, r, swing=0.0):
    W_, H_ = 430, 260

    def draw(m):
        cv2.ellipse(m, (180, 150), (100, 48), 0, 0, 360, 1.0, -1)      # 躯干（横椭圆）
        cv2.circle(m, (300, 100), 50, 1.0, -1)                          # 头
        cv2.ellipse(m, (262, 152), (48, 40), 0, 0, 360, 1.0, -1)        # 前胸
        _poly(m, [[258, 62], [268, 8], [300, 56]])                      # 耳
        _poly(m, [[312, 54], [340, 4], [354, 62]])
        # 四腿（两前两后）：按 swing 换相位
        for i, (lx, ly) in enumerate(((116, 178), (162, 184), (226, 180), (262, 172))):
            sgn = 1.0 if (i % 2 == 0) else -1.0
            off = float(13.0 * sgn * swing)
            cv2.line(m, (int(lx), int(ly)), (int(lx + off), 232), 1.0, 15, cv2.LINE_AA)
        # 尾巴：一条上扬的曲线
        p0 = np.array([94, 134], F32)
        p1 = np.array([22, 98 + 34 * swing], F32)
        p2 = np.array([10 + 12 * swing, 44 - 26 * swing], F32)
        p3 = np.array([46 + 30 * swing, 18 - 8 * swing], F32)
        tt = np.linspace(0, 1, 16)[:, None]
        pts = ((1 - tt) ** 3 * p0 + 3 * (1 - tt) ** 2 * tt * p1
               + 3 * (1 - tt) * tt * tt * p2 + tt ** 3 * p3)
        cv2.polylines(m, [np.int32(pts)], False, 1.0, 15, cv2.LINE_AA)
    d = _puff(_mask(W_, H_, draw), n, u, v, w, r, height_world=430.0,
              depth=54.0, edge=1.05, dref=0.62, zpow=0.58)
    x = d["P"][:, 0]
    d["stripe"] = (1.0 - 0.40 * (0.5 + 0.5 * np.sin(x * 0.105)) ** 1.6).astype(F32)
    return d

# ---------------------------------------------------------------- 形状：神
def _build_god(n, u, v, w, r):
    W_, H_ = 340, 480
    cx = 170.0

    def draw(m):
        # 长袍：A 字，下摆 6 个波浪折
        hem = []
        for i in range(7):
            x = cx - 148 + i * 49.3
            hem.append((x, 452 + (11 if i % 2 else -3)))
        body = [[cx - 56, 168], [cx + 56, 168], [cx + 106, 312], [cx + 150, 440]]
        body += hem[::-1]
        body += [[cx - 106, 312]]
        _poly(m, body)
        # 胸 + 圆肩（别做成尖角披风）
        cv2.ellipse(m, (int(cx), 198), (44, 58), 0, 0, 360, 1.0, -1)
        cv2.ellipse(m, (int(cx - 58), 172), (33, 26), 0, 0, 360, 1.0, -1)
        cv2.ellipse(m, (int(cx + 58), 172), (33, 26), 0, 0, 360, 1.0, -1)
        # 颈 + 头（颈细，头明显宽出来）
        cv2.rectangle(m, (int(cx - 13), 108), (int(cx + 13), 150), 1.0, -1)
        cv2.ellipse(m, (int(cx), 76), (34, 39), 0, 0, 360, 1.0, -1)
        # 冠：箍 + 五根短尖
        cv2.rectangle(m, (int(cx - 42), 34), (int(cx + 42), 48), 1.0, -1)
        for k in range(5):
            x0 = cx - 34 + k * 17.0
            _poly(m, [[x0 - 6, 42], [x0, 14], [x0 + 6, 42]])
        # 双臂上举：肩 → 肘 → 手，逐渐变细 + 掌心
        for s in (-1.0, 1.0):
            cv2.line(m, (int(cx + 58 * s), 166), (int(cx + 104 * s), 120), 1.0, 27, cv2.LINE_AA)
            cv2.line(m, (int(cx + 104 * s), 120), (int(cx + 130 * s), 74), 1.0, 20, cv2.LINE_AA)
            cv2.circle(m, (int(cx + 136 * s), 64), 18, 1.0, -1)
    d = _puff(_mask(W_, H_, draw), n, u, v, w, r, height_world=1370.0,
              depth=110.0, edge=0.80, dref=0.85, zpow=0.55)
    th = np.arctan2(d["P"][:, 0], d["P"][:, 2] + 1.0)
    d["fold"] = (0.74 + 0.26 * (0.5 + 0.5 * np.cos(th * 8.0))).astype(F32)
    # 8.5% 粒子：袍子上方往上飘的光尘
    rs = np.random.default_rng(919)
    mote = rs.random(n) < 0.085
    k = int(mote.sum())
    hw = 470.0 * d["sc"]
    d["P"][mote, 0] = (rs.random(k) - 0.5) * 300.0
    d["P"][mote, 1] = hw * (0.30 + rs.random(k) * 0.85)
    d["P"][mote, 2] = (rs.random(k) - 0.5) * 120.0
    d["mote"] = mote
    return d


# ================================================================ 廉价全屏工具
def _glow(f, fn, color, alpha=1.0, res=0.5, sig=1.2, mode="add"):
    """半分辨率画 mask → 模糊 → 放大叠加：比全分辨率少 4 倍工作量。"""
    sw, sh = int(kit.SW * res), int(kit.SH * res)
    sc = kit.S * res
    m = np.zeros((sh, sw), F32)
    fn(m, sc)
    if sig > 0:
        m = cv2.GaussianBlur(m, (0, 0), max(0.6, sig * sc))
    m = kit.up(m, kit.SW, kit.SH)
    if mode == "add":
        f += m[..., None] * np.asarray(color, F32) * alpha
    elif mode == "over":
        a = (m * alpha)[..., None]
        f *= (1 - a)
        f += a * np.asarray(color, F32)
    else:
        f *= (1 - (m * alpha)[..., None])


def _rad(f, cx, cy, radius, color, strength=1.0, power=2.2, ratio=1.0, res=0.25):
    """径向光：低分辨率算衰减，再放大叠加。"""
    bw, bh = max(8, int(kit.SW * res)), max(8, int(kit.SH * res))
    yy, xx = np.mgrid[0:bh, 0:bw].astype(F32)
    dx = (xx / bw * kit.W - cx) / max(1.0, radius)
    dy = (yy / bh * kit.H - cy) / max(1.0, radius * ratio)
    m = np.clip(1.0 - np.sqrt(dx * dx + dy * dy), 0, 1) ** power
    m = kit.up(m, kit.SW, kit.SH)
    f += m[..., None] * np.asarray(color, F32) * strength


_BG = {}


def _bg(f, path, zoom=1.0, dx=0.0, dy=0.0, dim=1.0, tint=None, tint_k=0.0, blur_r=0.0):
    """半分辨率铺底（舞台背景本来就是虚的）。"""
    key = (path, round(zoom, 2), int(dx), int(dy), int(blur_r * 2), kit.SW)
    ho, wo = kit.SH // 2, kit.SW // 2
    img = _BG.get(key)
    if img is None:
        base = _BG.get(("raw", path, kit.SW))
        if base is None:
            base = kit.up(kit.load(path)[..., :3], kit.SW, kit.SH)
            _BG[("raw", path, kit.SW)] = base
        small = cv2.resize(base, (wo, ho), interpolation=cv2.INTER_AREA)
        M = cv2.getRotationMatrix2D((wo * 0.5, ho * 0.5), 0.0, zoom)
        M[0, 2] += dx * kit.S * 0.5
        M[1, 2] += dy * kit.S * 0.5
        img = cv2.warpAffine(small, M, (wo, ho), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
        if blur_r > 0.2:
            img = cv2.GaussianBlur(img, (0, 0), blur_r * kit.S * 0.5)
        if len(_BG) > 40:
            _BG.clear()
            _BG[("raw", path, kit.SW)] = base
        _BG[key] = img
    up = kit.up(img, kit.SW, kit.SH)
    if tint is not None and tint_k > 0:
        up = up * (1.0 - tint_k) + np.asarray(tint, F32)[None, None, :] * tint_k
    f += up * dim
    return f


def _beam(f, t, x=0.5, width=0.22, strength=0.15, tilt=0.12, color=None, sway=0.05,
          res=0.25, sig=11.0):
    """光柱：低分辨率多边形 + 大模糊，替掉 stage.lightbeam 的 120ms。"""
    color = kit.PALE if color is None else color
    xx = x + math.sin(t * 0.31) * sway

    def draw(m, sc):
        x0 = (xx - width * 0.5) * kit.W * sc
        x1 = (xx + width * 0.5) * kit.W * sc
        top = -0.10 * kit.H * sc
        bot = 1.05 * kit.H * sc
        dx = tilt * kit.W * sc
        cv2.fillPoly(m, [np.int32([[x0, top], [x1, top], [x1 + dx, bot], [x0 + dx, bot]])], 1.0)
    _glow(f, draw, color, strength, res=res, sig=sig)


def _dust(f, t, n=150, alpha=0.26, color=None, res=0.4, box=(0.0, 1.0, 0.0, 1.0)):
    color = kit.PALE if color is None else color
    rs = np.random.default_rng(4242)
    px = rs.random(n); py = rs.random(n); sp = 0.2 + rs.random(n) * 0.8
    ph = rs.random(n) * 6.283; br = 0.3 + 0.7 * rs.random(n)
    x0, x1, y0, y1 = box
    X = (x0 + px * (x1 - x0)) * kit.W + np.sin(t * 0.35 * sp + ph) * 22
    Y = ((y0 + py * (y1 - y0)) * kit.H - t * 6 * sp) % kit.H

    def draw(m, sc):
        for i in range(n):
            cv2.circle(m, (int(X[i] * sc), int(Y[i] * sc)), max(1, int(1.3 * sc)),
                       float(br[i]), -1, cv2.LINE_AA)
    _glow(f, draw, color, alpha, res=res, sig=0.9)


def _caustic(f, t, strength=0.07, color=None, res=0.25, y0=0.55):
    """水面光斑：低分辨率正弦干涉，再放大。"""
    color = kit.MINT if color is None else color
    bw, bh = max(16, int(kit.SW * res)), max(8, int(kit.SH * res))
    yy, xx = np.mgrid[0:bh, 0:bw].astype(F32)
    x = xx / bw * 6.0
    y = yy / bh * 3.4
    a = np.sin(x * 2.1 + t * 0.84) * np.cos(y * 2.7 - t * 0.60)
    b = np.sin((x + y) * 3.3 - t * 1.02) * np.cos((x - y) * 1.9 + t * 0.46)
    m = np.clip((a * 0.6 + b * 0.4) * 0.5 + 0.5, 0, 1) ** 3.4
    m *= np.clip(1.0 - np.abs(yy / bh - y0) * 1.5, 0, 1)
    m = kit.up(m, kit.SW, kit.SH)
    f += m[..., None] * np.asarray(color, F32) * strength


def _blit(dst, src, x, y):
    dh, dw = dst.shape[:2]
    sh, sw = src.shape[:2]
    x0, y0 = int(round(x)), int(round(y))
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    dx0 = max(0, x0); dy0 = max(0, y0)
    w = min(dw - dx0, sw - sx0); h = min(dh - dy0, sh - sy0)
    if w <= 0 or h <= 0:
        return
    dst[dy0:dy0 + h, dx0:dx0 + w] += src[sy0:sy0 + h, sx0:sx0 + w]


def _layers(f, items, res=0.5):
    """把多条叠加光一次性合成：低分辨率累加 → 只上采样/相加一次（省 3/4 全屏开销）。"""
    sw, sh = int(kit.SW * res), int(kit.SH * res)
    sc = kit.S * res
    buf = np.zeros((sh, sw, 3), F32)
    for it in items:
        fn, color, alpha = it[0], it[1], it[2]
        sig = it[3] if len(it) > 3 else 1.2
        m = np.zeros((sh, sw), F32)
        fn(m, sc)
        if sig > 0:
            m = cv2.GaussianBlur(m, (0, 0), max(0.6, sig * sc))
        buf += m[..., None] * np.asarray(color, F32)[None, None, :] * alpha
    f += kit.up(buf, kit.SW, kit.SH)


def _dust_fn(t, n=150, box=(0.0, 1.0, 0.0, 1.0), seed=4242):
    """返回一个往 mask 里点点尘的闭包，便于和别的光一起合成。"""
    rs = np.random.default_rng(seed)
    px = rs.random(n); py = rs.random(n); sp = 0.2 + rs.random(n) * 0.8
    ph = rs.random(n) * 6.283; br = 0.3 + 0.7 * rs.random(n)
    x0, x1, y0, y1 = box
    X = (x0 + px * (x1 - x0)) * kit.W + np.sin(t * 0.35 * sp + ph) * 22
    Y = ((y0 + py * (y1 - y0)) * kit.H - t * 6 * sp) % kit.H

    def draw(m, sc):
        for i in range(n):
            cv2.circle(m, (int(X[i] * sc), int(Y[i] * sc)), max(1, int(1.3 * sc)),
                       float(br[i]), -1, cv2.LINE_AA)
    return draw


def _ghost(f, O, cx, cy, k=1.0, color=None, alpha=0.28, sig=7.0):
    """把剪影本身当柔光垫在粒子后面：轮廓因此一眼可读（纯代码的「本体阴影」）。"""
    color = kit.VIOLET if color is None else color
    m = O["mask"]
    h, w = m.shape
    sw = max(2, int(w * O["sc"] * K * k * 0.5))
    sh = max(2, int(h * O["sc"] * K * k * 0.5))
    sm = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA)
    buf = np.zeros((max(2, kit.SH // 2), max(2, kit.SW // 2)), F32)
    _blit(buf, sm, cx * 0.5 - sw * 0.5, cy * 0.5 - sh)
    buf = cv2.GaussianBlur(buf, (0, 0), max(0.8, sig * 0.5))
    buf = kit.up(buf, kit.SW, kit.SH)
    f += buf[..., None] * np.asarray(color, F32) * alpha


def _shadow(f, cx, cy, rx, ry, k=0.5, sig=None):
    s = sig if sig is not None else max(0.6, ry * 0.55)
    _glow(f, lambda m, sc: cv2.ellipse(m, (int(cx * sc), int(cy * sc)),
                                      (max(1, int(rx * sc)), max(1, int(ry * sc))),
                                      0, 0, 360, 1.0, -1),
          kit.BLACK, k, res=0.4, sig=s, mode="mul")


def _rings(f, cx, cy, base_r, t, color, n=3, alpha=1.0, speed=0.8, thick=1.6):
    def draw(m, sc):
        for i in range(n):
            ph = (t * speed + i / n) % 1.0
            a = (1.0 - ph) ** 2
            if a < 0.04:
                continue
            r = base_r * (0.28 + 0.95 * ph)
            cv2.circle(m, (int(cx * sc), int(cy * sc)), max(1, int(r * sc)), float(a),
                       max(1, int(thick * sc)), cv2.LINE_AA)
    _glow(f, draw, color, alpha, res=0.5, sig=1.0)


def _word_grow(f, txt, x, y, size, p, t, color=kit.CREAM, tracking=4.0, glow=0.35,
               start=0.62, jitter=2.6, font=None):
    """单词逐字放大落位（每个字母独立放大 + 手写抖动）。"""
    fp = font or kit.F_SERIF_I
    xs = x
    for i, ch in enumerate(txt):
        wch = kit.text_width(fp, size, ch, tracking)
        if ch == " ":
            xs += wch
            continue
        q = kit.cl((p * (len(txt) + 1.6) - i) / 1.25)
        if q > 0.002:
            e = kit.ease_out_back(q)
            sc = start + (1.0 - start) * e
            a = kit.cl(q * 2.4)
            jy = math.sin(i * 1.93 + t * 2.35) * jitter * kit.cl(p * 2.5)
            kit.draw_text(f, ch, xs, y + (1 - e) * 22 + jy, size * sc, color, fp, a,
                          tracking, "left", glow=glow)
        xs += wch
    return xs


def _word(f, txt, y, p, t, size, color=kit.CREAM, tracking=6.0, glow=0.30, font=None):
    """居中大字：按 kit.text_width 算宽度，左右各留 ≥100px。"""
    fp = font or kit.F_SERIF_I
    w = kit.text_width(fp, size, txt, tracking)
    x = max(100.0, (kit.W - w) * 0.5)
    if w > kit.W - 200:
        size *= (kit.W - 200) / w
        w = kit.text_width(fp, size, txt, tracking)
        x = (kit.W - w) * 0.5
    _word_grow(f, txt, x, y, size, p, t, color, tracking, glow, font=fp)
    return x, w, size


def _curve(p0, p1, bow=0.0, n=26):
    p0 = np.asarray(p0, F32); p1 = np.asarray(p1, F32)
    d = p1 - p0
    L = math.hypot(d[0], d[1]) + 1e-6
    C = (p0 + p1) * 0.5 + np.array([-d[1] / L * bow, d[0] / L * bow], F32)
    u = np.linspace(0, 1, n)[:, None].astype(F32)
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * C + 3 * (1 - u) * u * u * C + u ** 3 * p1


def _her(f, cx, foot, h, alpha=1.0, tint=None, tint_k=0.0, rim=kit.MINT, rim_k=0.42,
         glow=0.10, shadow=0.45, flip=False, squash=1.0, rot=0.0):
    spr = kit.sprite(stage.MAID_L, h)
    if squash != 1.0:
        spr = kit.up(spr, max(1, int(spr.shape[1] / squash ** 0.55)),
                     max(1, int(spr.shape[0] * squash)))
    if rim is not None and rim_k > 0:
        spr = stage.rim_light(spr, rim, rim_k)
    if shadow > 0:
        _shadow(f, cx, foot, h * 0.30, h * 0.045, shadow)
    kit.place(f, spr, cx, foot, 1.0, alpha, tint, tint_k, glow, False, flip, rot, "foot")
    return spr


# ================================================================ 73.94 EGGPLANT
@scene(73.94, 78.11, "EGGPLANT", "E")
def eggplant(c):
    t, f = c.t, c.f
    lt = t - 73.94
    sp = 4.17
    Wc, Hc = kit.W, kit.H
    c.post.update(grain=0.030, vignette=0.55, bloom=(0.62, 24, 0.45))
    p_in = kit.ease_out(kit.cl(lt / 0.45))
    floor = Hc * 0.845

    _bg(f, stage.BG_PALACE, zoom=1.24 + 0.09 * (lt / sp), dx=14 * math.sin(lt * 0.42),
        dy=6, dim=(0.30 + 0.20 * p_in), tint=kit.hx("0A2233"), tint_k=0.46)
    _caustic(f, t, 0.075 * p_in)
    _beam(f, t, x=0.76, width=0.30, strength=0.13 * p_in, tilt=0.12)
    _dust(f, t, n=150, alpha=0.26 * p_in)

    ox = Wc * 0.60
    O = _obj("eggplant", 4101, _build_eggplant, c.cl.n)
    ew_px = (O["P"][:, 0].max() - O["P"][:, 0].min()) * K
    eh_px = 790.0 * K

    _shadow(f, ox, floor + 12, ew_px * 0.58, 22 + 3 * c.pulse(), 0.50)

    yaw = 0.34 * math.sin(lt * 0.50) + 0.30 * lt
    bass = c.band("sub", 0.20)
    sy = 1.0 + 0.030 * math.sin(2 * math.pi * lt * 0.85) + 0.014 * bass
    sxz = 1.0 - (sy - 1.0) * 0.45
    P = cloud.Cloud.transform(O["P"], yaw=yaw)
    P[:, 1] *= sy
    P[:, 0] *= sxz
    P[:, 2] *= sxz
    b = _shade(O["N"], yaw=yaw, key=1.02, amb=0.44, rim=0.44)
    b *= 1.0 + 0.10 * c.pulse(7)
    cm = _cmap([0.0, 0.075, 0.175, 0.30, 0.70, 1.0],
               [kit.mixc(kit.TEAL, kit.MINT, 0.45), kit.TEAL,
                kit.mixc(kit.VIOLET, kit.TEAL, 0.30), kit.VIOLET,
                kit.mixc(kit.VIOLET, kit.PLUM, 0.55), kit.mixc(kit.PLUM, kit.BLACK, 0.30)],
               y0=floor - eh_px, y1=floor)
    c.cl.draw(f, P, color_map=cm, dist=DIST, focal=FOC, cx=ox, cy=floor,
              size=2.1, blur_r=2.3, glow_gain=0.62, bright=b, shade=0.92,
              target=0.80, tm=2.0)

    # 她：小立绘，伸手去碰
    hh = 400.0
    surf = ox - ew_px * 0.5 + 6
    hx = surf - 0.278 * hh - 26
    _her(f, hx, floor + 4, hh, alpha=0.96, rim=kit.MINT, rim_k=0.44, glow=0.10, shadow=0.5)
    hand = (hx + 0.168 * hh, floor - 0.765 * hh)
    touch = (surf + 4, floor - eh_px * 0.53)
    _glow(f, lambda m, sc: cv2.line(m, (int(hand[0] * sc), int(hand[1] * sc)),
                                    (int(touch[0] * sc), int(touch[1] * sc)), 1.0,
                                    max(1, int(1.5 * sc)), cv2.LINE_AA),
          kit.CREAM, 0.42, res=0.5, sig=1.2)
    sp1 = 0.55 + 0.45 * math.sin(t * 9.0)
    _rad(f, touch[0], touch[1], 54, kit.MINT, 0.34 * sp1, 2.0, res=0.15)
    _rings(f, touch[0], touch[1], 56, t, kit.mixc(kit.MINT, kit.CREAM, 0.45),
           n=3, alpha=0.55, speed=0.72, thick=1.5)

    ca = kit.cl((lt - 0.45) / 0.85)
    stage.terminal(f, [("obj.eggplant = True", kit.MINT),
                       ("skin = \"violet\"", kit.VIOLET),
                       ("nutrients = 0.93", kit.GOLD),
                       ("give(you, me)", kit.ASH)],
                   x=92, y=Hc * 0.585, size=21, alpha=ca * 0.95, lead=1.72)

    _word(f, "eggplant", Hc * 0.095, kit.cl((lt - 1.45) / 1.45), t, 168,
          kit.CREAM, tracking=6.0, glow=0.30)
    la = kit.cl((lt - 1.9) / 0.6)
    if la > 0:
        kit.draw_text(f, "// one object, still edible", Wc * 0.5, Hc * 0.322, 20,
                      kit.MINT, kit.F_MONO, la * 0.62, 2.2, "center")
    c.lyric(size=38, y=Hc * 0.885, alpha=0.95)
    stage.hud_corner(f, t, "OBJ 04 / EGGPLANT", "73.94", 0.32 * p_in)


# ================================================================ 78.11 TOMATO
@scene(78.11, 81.21, "TOMATO", "E")
def tomato(c):
    t, f = c.t, c.f
    lt = t - 78.11
    sp = 3.10
    Wc, Hc = kit.W, kit.H
    c.post.update(grain=0.028, vignette=0.48, bloom=(0.58, 26, 0.44), sat=1.05)
    p_in = kit.ease_out(kit.cl(lt / 0.40))

    _bg(f, stage.BG_PALACE, zoom=1.20 + 0.10 * (lt / sp), dy=8,
        dim=0.10 + 0.22 * p_in, tint=kit.hx("0B2231"), tint_k=0.50)
    _rad(f, Wc * 0.58, Hc * 0.62, 520, kit.hx("C4482F"), 0.28 * p_in, 1.9)
    _caustic(f, t, 0.045 * p_in, kit.MINT, y0=0.74)

    T = _obj("tomato", 4102, _build_tomato, c.cl.n)
    yaw = 0.55 * lt + 0.40
    P = cloud.Cloud.transform(T["P"], yaw=yaw, pitch=0.08 * math.sin(lt * 0.9))
    P[:, 2] += 6.0 * np.sin(c.cl.u * 6.283 + t * 2.1)
    b = _shade(T["N"], yaw=yaw, key=0.86, amb=0.62, rim=0.44) * T["groove"]
    b *= 0.94 + 0.12 * (0.5 + 0.5 * np.sin(c.cl.u * 12.0 + t * 2.6))
    b *= 1.0 + 0.08 * c.pulse(7)
    base_y = Hc * 0.815
    _ghost(f, T, Wc * 0.58, base_y, color=kit.CORAL, alpha=0.30 * p_in, sig=9.0)
    cm = _cmap([0.0, 0.055, 0.125, 0.34, 0.68, 1.0],
               [kit.mixc(kit.TEAL, kit.BLACK, 0.30), kit.mixc(kit.TEAL, kit.MINT, 0.35),
                kit.mixc(kit.CORAL, kit.CREAM, 0.55), kit.CORAL,
                kit.mixc(kit.CORAL, kit.hx("8A2E22"), 0.55), kit.hx("8A2E22")],
               y0=base_y - 490.0 * K, y1=base_y)
    c.cl.draw(f, P, color_map=cm, dist=DIST, focal=FOC, cx=Wc * 0.58, cy=base_y,
              size=2.0, blur_r=2.0, glow_gain=0.34, bright=b, shade=0.78,
              target=0.36, exposure=0.95, tm=2.0)
    _shadow(f, Wc * 0.58, Hc * 0.848, 210, 18, 0.46)

    # 萼片：5 片细长叶 + 短柄（2D 压在最上面，绕竖轴自转时它就是正确的）
    scx = Wc * 0.58
    scy = base_y - 490.0 * K * 0.925
    crot = yaw * 0.40

    def calyx(m, sc):
        for k in range(5):
            a = -math.pi / 2 + k * 2 * math.pi / 5 + crot
            cv2.ellipse(m, (int((scx + math.cos(a) * 52) * sc),
                            int((scy + math.sin(a) * 52 * 0.66) * sc)),
                        (int(48 * sc), int(15 * sc)), math.degrees(a),
                        0, 360, 0.90, -1, cv2.LINE_AA)
        cv2.circle(m, (int(scx * sc), int(scy * sc)), int(17 * sc), 1.0, -1, cv2.LINE_AA)
        cv2.line(m, (int(scx * sc), int(scy * sc)),
                 (int((scx + 8) * sc), int((scy - 34) * sc)), 1.0,
                 max(2, int(10 * sc)), cv2.LINE_AA)
    _glow(f, calyx, kit.hx("2F6B5E"), 0.88 * p_in, res=0.5, sig=0.9, mode="over")
    _glow(f, calyx, kit.mixc(kit.MINT, kit.TEAL, 0.25), 0.10 * p_in, res=0.5, sig=3.2)

    # 表面流动的亮丝 + 尘 + 萼片微光：合成一次
    fa = kit.cl((lt - 0.45) / 0.6)

    def flow(m, sc):
        for i in range(5):
            ang = t * 0.55 + i * 1.2566
            pts = [(Wc * 0.58 + math.cos(ang + k * 0.12) * 200 * (1 - 0.05 * k),
                    base_y - 190 + math.sin(ang * 1.3 + k * 0.15) * 186 * 0.92)
                   for k in range(16)]
            cv2.polylines(m, [np.int32(np.asarray(pts, F32) * sc)], False,
                          0.55, max(1, int(1.1 * sc)), cv2.LINE_AA)
    _layers(f, [(flow, kit.mixc(kit.CORAL, kit.CREAM, 0.55), 0.20 * fa, 1.4),
                (_dust_fn(t, 120, seed=5150), kit.mixc(kit.CREAM, kit.CORAL, 0.4),
                 0.40 * p_in, 0.9)], res=0.5)

    _her(f, Wc * 0.262, Hc * 0.905, 340, alpha=0.80, rim=kit.GOLD, rim_k=0.40,
         glow=0.08, shadow=0.4)

    ca = kit.cl((lt - 0.35) / 0.8)
    stage.terminal(f, [("obj.tomato = True", kit.CORAL),
                       ("CORAL -> 8A2E22", kit.GOLD),
                       ("antioxidants 1.0", kit.MINT)],
                   x=92, y=Hc * 0.585, size=20, alpha=ca * 0.95, lead=1.72)
    _word(f, "tomato", Hc * 0.095, kit.cl((lt - 0.28) / 1.20), t, 178,
          kit.CREAM, tracking=6.0, glow=0.25)
    la = kit.cl((lt - 0.72) / 0.5)
    if la > 0:
        kit.draw_text(f, "// still whole, still yours", Wc * 0.5, Hc * 0.326, 20,
                      kit.mixc(kit.CORAL, kit.CREAM, 0.40), kit.F_MONO, la * 0.80, 2.2, "center")
    c.lyric(size=38, y=Hc * 0.885, alpha=0.95, color=kit.CREAM)
    stage.hud_corner(f, t, "OBJ 05 / TOMATO", "78.11", 0.32 * p_in)


# ================================================================ 81.21 CAT
@scene(81.21, 85.16, "CAT", "E")
def cat(c):
    t, f = c.t, c.f
    lt = t - 81.21
    sp = 3.95
    Wc, Hc = kit.W, kit.H
    c.post.update(grain=0.030, vignette=0.52, bloom=(0.62, 22, 0.42))
    p_in = kit.ease_out(kit.cl(lt / 0.40))
    floor = Hc * 0.885

    _bg(f, stage.BG_PALACE, zoom=1.52 + 0.10 * (lt / sp), dy=-24,
        dim=0.32, tint=kit.hx("0B2030"), tint_k=0.42)
    _caustic(f, t, 0.06 * p_in, y0=0.72)
    _beam(f, t, x=0.30, width=0.22, strength=0.12 * p_in, tilt=0.22)
    _dust(f, t, n=130, alpha=0.24 * p_in, box=(0, 1, 0.3, 1))

    q = kit.ease_io(kit.cl((lt - 0.12) / 3.20))
    cxp = (0.25 + 0.47 * q) * Wc
    bar = 4.0 * mus.BEAT
    hop = 0.5 - 0.5 * math.cos(2 * math.pi * lt / bar)
    foot = floor - hop * 11.0

    A = _obj("cat", 4103, _build_cat, c.cl.n)
    # 单一形状 + 区域形变（两套取样混用会把猫糊掉），每拍换一次腿相位、尾巴跟着摆
    sw = math.sin(2 * math.pi * lt / (2 * mus.BEAT))
    P = A["P"].copy()
    xw = P[:, 0]
    yw = P[:, 1]
    for x0, sg in ((-130.0, 1.0), (-43.0, -1.0), (77.0, 1.0), (145.0, -1.0)):
        mm = (np.abs(xw - x0) < 32.0) & (yw < 96.0)
        P[mm, 0] += 11.0 * sg * sw * np.clip((96.0 - yw[mm]) / 96.0, 0, 1)
    td = np.clip((-232.0 - xw) / 118.0, 0.0, 1.0)
    P[:, 1] += td * td * 64.0 * sw
    P[:, 0] -= td * td * 26.0 * sw
    b = _shade(A["N"], key=0.95, amb=0.52, rim=0.42) * A["stripe"]
    b *= 1.0 + 0.10 * c.pulse(8)
    _shadow(f, cxp, floor + 4, 250, 16, 0.52)
    _ghost(f, A, cxp, foot, color=kit.GOLD, alpha=0.26 * p_in, sig=7.0)
    cm = _cmap([0.0, 0.20, 0.60, 1.0],
               [kit.mixc(kit.SAND, kit.CREAM, 0.55), kit.SAND,
                kit.GOLD, kit.mixc(kit.GOLD, kit.RUST, 0.55)],
               y0=foot - 430.0 * (FOC / 1350.0), y1=foot)
    c.cl.draw(f, P, color_map=cm, dist=1350, focal=FOC, cx=cxp, cy=foot,
              size=2.1, blur_r=1.5, glow_gain=0.34, bright=b, shade=0.80,
              target=0.56, tm=2.0)

    # 她：蹲下看（压扁 + 微倾）
    _her(f, Wc * 0.895, floor + 14, 360, alpha=0.92, rim=kit.MINT, rim_k=0.42,
         glow=0.09, shadow=0.42, squash=0.72, rot=-0.05)

    g = np.clip((np.arange(kit.SH, dtype=F32) / kit.SH - 0.74) * 3.4, 0, 1)[:, None]
    f *= (1 - g * 0.30)[..., None]

    ca = kit.cl((lt - 0.30) / 0.8)
    stage.terminal(f, [("obj.tabby_cat = True", kit.GOLD),
                       ("pattern = \"striped\"", kit.MINT),
                       ("purr = for_your_enjoyment", kit.ASH)],
                   x=92, y=Hc * 0.625, size=21, alpha=ca * 0.95, lead=1.72)
    _word(f, "tabby cat", Hc * 0.09, kit.cl((lt - 0.95) / 0.95), t, 152,
          kit.CREAM, tracking=5.0, glow=0.30)
    c.lyric(size=38, y=Hc * 0.90, alpha=0.95)
    stage.hud_corner(f, t, "OBJ 06 / TABBY CAT", "81.21", 0.32 * p_in)


# ================================================================ 85.16 GOD
@scene(85.16, 87.41, "GOD", "E")
def god(c):
    t, f = c.t, c.f
    lt = t - 85.16
    sp = 2.25
    Wc, Hc = kit.W, kit.H
    c.post.update(grain=0.026, vignette=0.62, bloom=(0.62, 30, 0.58))
    p_in = kit.ease_out(kit.cl(lt / 0.55))

    _bg(f, stage.BG_PALACE, zoom=1.40 + 0.06 * (lt / sp), dy=-30, dim=0.14,
        blur_r=6, tint=kit.hx("150B26"), tint_k=0.64)
    _rad(f, Wc * 0.5, Hc * 0.55, 980, kit.hx("2A1B52"), 0.36 * p_in, 1.5)
    _beam(f, t, x=0.535, width=0.26, strength=0.13 * p_in, tilt=-0.03, color=kit.VIOLET)
    _dust(f, t, n=170, color=kit.GOLD, alpha=0.24 * p_in)

    G = _obj("god", 4105, _build_god, c.cl.n)
    gh_px = 1370.0 * K                       # ≈1050 屏幕像素：整尊（含冠与光环）进画
    foot = Hc * 1.100 - lt * 2.0
    gx = Wc * 0.50
    _ghost(f, G, gx, foot, color=kit.VIOLET, alpha=0.30 * p_in, sig=9.0)

    rise = ((t * 0.40 + c.cl.u) % 1.0).astype(F32)
    P = G["P"].copy()
    mo = G["mote"]
    P[mo, 1] += rise[mo] * 250.0
    b = _shade(G["N"], key=1.05, amb=0.42, rim=0.50) * G["fold"]
    b[mo] *= (1.0 - rise[mo]) ** 0.9 * 1.25
    b *= 1.0 + 0.10 * c.pulse(6)
    cm = _cmap([0.0, 0.050, 0.082, 0.125, 0.22, 0.70, 0.885, 1.0],
               [kit.mixc(kit.CREAM, kit.GOLD, 0.35), kit.GOLD,
                kit.mixc(kit.CREAM, kit.GOLD, 0.60), kit.mixc(kit.GOLD, kit.VIOLET, 0.55),
                kit.VIOLET, kit.mixc(kit.VIOLET, kit.PLUM, 0.50), kit.GOLD,
                kit.mixc(kit.GOLD, kit.PLUM, 0.55)],
               y0=foot - gh_px, y1=foot)
    c.cl.draw(f, P, color_map=cm, dist=DIST, focal=FOC, cx=gx, cy=foot,
              size=2.2, blur_r=2.5, glow_gain=0.70, bright=b, shade=1.0,
              target=0.50, tm=2.1)

    # 光环：头后上方，扁椭圆，宽约头宽的 2.2 倍
    hy = foot - gh_px * 0.855 - 30
    hrx = 1.20 * 68.0 * G["sc"] * K
    rot = lt * 0.16

    def halo(m, sc):
        cv2.ellipse(m, (int(gx * sc), int(hy * sc)), (int(hrx * sc), int(hrx * 0.28 * sc)),
                    0, 0, 360, 1.0, max(1, int(2.4 * sc)), cv2.LINE_AA)
        for i in range(26):
            a = rot + i * 2 * math.pi / 26
            r0 = 1.03 if i % 6 else 1.10
            r1 = 1.08 if i % 6 else 1.22
            cv2.line(m, (int((gx + math.cos(a) * hrx * r0) * sc),
                         int((hy + math.sin(a) * hrx * 0.28 * r0) * sc)),
                     (int((gx + math.cos(a) * hrx * r1) * sc),
                      int((hy + math.sin(a) * hrx * 0.28 * r1) * sc)),
                     0.9, max(1, int(1.6 * sc)), cv2.LINE_AA)
    # 面容：一对杏仁形的眼（古雕像式的细线，不画嘴）
    ey = foot - gh_px * 0.855 + 6

    def eyes(m, sc):
        for s in (-1, 1):
            cv2.ellipse(m, (int((gx + s * 24) * sc), int(ey * sc)),
                        (int(13 * sc), int(5 * sc)), 0, 0, 360, 0.95,
                        max(1, int(3.2 * sc)), cv2.LINE_AA)
    _layers(f, [(halo, kit.GOLD, 0.85 * p_in, 1.3),
                (eyes, kit.CREAM, 0.80 * p_in, 1.0)], res=0.5)
    _rad(f, gx, hy, 300, kit.GOLD, 0.16 * p_in, 2.4, ratio=0.58)

    # 她：画面下缘极小的剪影
    _her(f, gx - 8, Hc * 1.005, 126, alpha=0.55,
         tint=kit.mixc(kit.VIOLET, kit.BLACK, 0.30), tint_k=0.55,
         rim=kit.GOLD, rim_k=0.55, glow=0.0, shadow=0)

    wq = kit.cl((lt - 0.90) / 1.00)
    if wq > 0:
        e = kit.ease_out_quint(wq)
        sz = 64 + 24 * e
        a = kit.cl(wq * 1.6) * 0.95
        tx = 118 + (1 - e) * 40
        kit.draw_text(f, "ONLY GOD", tx, Hc * 0.075, sz, kit.GOLD,
                      kit.F_SERIF_I, a, 14 + 12 * e, "left", glow=0.40)
        kit.draw_text(f, "the only one you may call", tx + 4, Hc * 0.075 + sz * 1.02, 18,
                      kit.mixc(kit.CREAM, kit.GOLD, 0.35), kit.F_MONO, a * 0.80, 3.0, "left")
    c.lyric(size=34, y=Hc * 0.905, alpha=0.60, color=kit.CREAM)
    stage.hud_corner(f, t, "DEUS / 00", "85.16", 0.26 * p_in)


# ================================================================ 87.41 PROOF
@scene(87.41, 89.53, "PROOF", "E")
def proof(c):
    t, f = c.t, c.f
    lt = t - 87.41
    Wc, Hc = kit.W, kit.H
    c.post.update(grain=0.026, vignette=0.60, bloom=(0.62, 28, 0.55))
    p_in = kit.ease_out(kit.cl(lt / 0.32))

    _bg(f, stage.BG_PALACE, zoom=1.46, dy=-18, dim=0.10, blur_r=8,
        tint=kit.hx("120A22"), tint_k=0.66)
    _rad(f, Wc * 0.5, Hc * 0.60, 860, kit.hx("241650"), 0.28 * p_in, 1.5)

    G = _obj("god", 4105, _build_god, c.cl.n)
    gk = 0.76
    gh_px = 1370.0 * K * gk
    foot = Hc * 1.045
    gx = Wc * 0.50
    spx = G["sc"] * K * gk
    rise = ((t * 0.40 + c.cl.u) % 1.0).astype(F32)
    P = G["P"].copy()
    mo = G["mote"]
    P[mo, 1] += rise[mo] * 250.0
    b = _shade(G["N"], key=1.0, amb=0.34, rim=0.46) * G["fold"] * 0.66
    b[mo] *= (1.0 - rise[mo]) ** 0.9 * 0.85
    cm = _cmap([0.0, 0.055, 0.10, 0.20, 0.72, 1.0],
               [kit.mixc(kit.CREAM, kit.GOLD, 0.4), kit.GOLD,
                kit.mixc(kit.GOLD, kit.CREAM, 0.5), kit.VIOLET,
                kit.mixc(kit.VIOLET, kit.PLUM, 0.55), kit.mixc(kit.GOLD, kit.PLUM, 0.4)],
               y0=foot - gh_px, y1=foot)
    _ghost(f, G, gx, foot, k=gk, color=kit.VIOLET, alpha=0.20 * p_in, sig=10.0)
    c.cl.draw(f, P, color_map=cm, dist=DIST / gk, focal=FOC, cx=gx, cy=foot,
              size=2.1, blur_r=2.5, glow_gain=0.75, bright=b, shade=1.0,
              target=0.44, tm=2.1)
    hy = foot - gh_px * 0.855 - 16
    hrrx = 1.20 * 68.0 * spx

    def phalo(m, sc):
        cv2.ellipse(m, (int(gx * sc), int(hy * sc)), (int(hrrx * sc), int(hrrx * 0.30 * sc)),
                    0, 0, 360, 0.9, max(1, int(2.0 * sc)), cv2.LINE_AA)
    _glow(f, phalo, kit.GOLD, 0.55 * p_in, res=0.5, sig=1.3)
    _rad(f, gx, hy, 240, kit.GOLD, 0.14 * p_in, 2.4, ratio=0.62)

    def MX(mx):
        return gx + (mx - 170.0) * spx

    def MY(my):
        return foot - (464.0 - my) * spx

    # 她：神像脚下，抬头
    her_h = 300.0
    hx = gx - 40
    hfoot = Hc * 0.995
    _her(f, hx, hfoot, her_h, alpha=0.96, rim=kit.MINT, rim_k=0.48, glow=0.11, shadow=0.42)
    heart = (hx + 2, hfoot - her_h * 0.640)

    kq = kit.cl((lt - 0.28) / 1.30)
    nline = int(kq * 9.0 + 0.001)
    lit = kit.cl((lt - 1.56) / 0.36)
    anchors = [(MX(34), MY(64)), (MX(306), MY(64)), (MX(104), MY(168)),
               (MX(236), MY(168)), (MX(170), MY(226)), (MX(84), MY(330)),
               (MX(258), MY(330)), (MX(170), MY(18)), (MX(170), MY(206))]
    ends = [(hx - 104, hfoot - 12), (hx + 104, hfoot - 12), (hx - 42, hfoot - 118),
            (hx + 48, hfoot - 128), (hx + 2, hfoot - 186), (hx - 62, hfoot - 4),
            (hx + 64, hfoot - 6), (hx - 8, hfoot - 300), heart]
    curves = {}
    for i in range(9):
        bow = (26.0 if i % 2 == 0 else -30.0) * (1.0 + 0.08 * i)
        curves[i] = _curve(anchors[i], ends[i], bow, n=30)

    # 大字先画，细线压在上面（免得线被字盖住）
    tp = kit.cl((lt - 0.92) / 1.00)
    if tp > 0:
        e = kit.ease_out_quint(tp)
        a = kit.cl(tp * 1.8)
        sz = 58 + 152 * e
        band = np.exp(-((np.arange(kit.SH, dtype=F32) / kit.S - Hc * 0.50 - sz * 0.40)
                        / (sz * 0.62)) ** 2) * (0.38 * a)
        f *= (1 - band)[:, None, None]
        kit.draw_text(f, "PROOF", gx, Hc * 0.50 - sz * 0.34, sz, kit.CREAM,
                      kit.F_SERIF_I, a * 0.96, 10 + 18 * e, "center", glow=0.48)
        a2 = kit.cl((lt - 1.22) / 0.56)
        if a2 > 0:
            kit.draw_text(f, "OF MY EXISTENCE", gx, Hc * 0.452, 22 + 4 * e, kit.GOLD,
                          kit.F_MONO, a2 * 0.90, 18 + 8 * e, "center", glow=0.30)

    if nline > 0:
        def lines(m, sc):
            for i in range(min(9, nline)):
                if i == 8:
                    continue
                cv2.polylines(m, [np.int32(curves[i] * sc)], False, 0.95,
                              max(1, int(2.0 * sc)), cv2.LINE_AA)

        def flow(m, sc):
            for i in range(min(9, nline)):
                pts = curves[i]
                L = len(pts)
                for j in range(7):
                    s = ((j / 7.0) + t * 0.55 + i * 0.13) % 1.0
                    x, y = pts[int(s * (L - 1))]
                    cv2.circle(m, (int(x * sc), int(y * sc)), max(1, int(2.6 * sc)),
                               float(0.35 + 0.65 * (1 - s)), -1, cv2.LINE_AA)
        items = [(lines, kit.mixc(kit.MINT, kit.CREAM, 0.30), 0.30, 5.0),
                 (lines, kit.mixc(kit.MINT, kit.CREAM, 0.30), 0.70, 1.1),
                 (flow, kit.CREAM, 0.85, 0.9)]
        if lit > 0:
            def hot(m, sc):
                cv2.polylines(m, [np.int32(curves[8] * sc)], False, 1.0,
                              max(1, int(2.2 * sc)), cv2.LINE_AA)
            items.append((hot, kit.CREAM, 0.95 * lit, 1.3))
        _layers(f, items, res=0.5)

    if lit > 0:
        pul = kit.ease_out(kit.cl((lt - 1.66) / 0.36))
        _rad(f, heart[0], heart[1], 110, kit.CREAM, 0.60 * pul, 2.0, res=0.15)
        _rings(f, heart[0], heart[1], 48, t, kit.mixc(kit.MINT, kit.CREAM, 0.6),
               n=2, alpha=0.55 * lit, speed=1.1, thick=1.5)

    c.lyric(size=34, y=Hc * 0.935, alpha=0.68 * kit.cl((lt - 0.85) / 0.5))

    if t > 89.38:
        a = kit.cl((t - 89.38) / 0.15)
        f *= (1 - a)
        f += kit.WHITE * a


# ================================================================ 自检
if __name__ == "__main__":
    kit.init(1)
    import film2
    film2.load_chapters(["ch_e"])
    seen = set()
    for s in film2.SCENES:
        if s["ch"] == "E" and s["name"] not in seen:
            seen.add(s["name"])
            print("%7.2f %7.2f  %s" % (s["t0"], s["t1"], s["name"]))
