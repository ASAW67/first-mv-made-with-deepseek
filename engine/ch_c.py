# -*- coding: utf-8 -*-
"""ch_c.py — 第三幕：从眩晕里冲出去 / 穿过光 / 与鲸合一。
51.93 ─ 59.77  （速度把人拉成线，最后落进那只眼睛里）
"""
import math
import numpy as np
import cv2
import kit, mus, stage, cloud
from film2 import scene

F32 = np.float32

L_T = 2600.0          # 隧道长度（世界单位）
R_T = 320.0           # 隧道半径
SPEED = 4.2           # 每秒穿过几个隧道长度
RING = 760.0          # 光环间距
BUSTW = 1500.0        # 鲸的母版宽度（逻辑像素）
EYE_FX, EYE_FY = 0.531, 0.286   # 她右眼在胸像里的相对位置（实测）

_C = {}
_PREV = {"i": -99999, "buf": None}
MINT = None


# ---------------------------------------------------------------- 小工具
def _blit(f, spr, x0, y0, alpha=1.0, tint=None, tint_k=0.0, additive=False):
    if alpha <= 0.004 or spr is None:
        return
    sh, sw = spr.shape[:2]
    x0 = int(x0); y0 = int(y0)
    xs = max(0, x0); ys = max(0, y0)
    xe = min(f.shape[1], x0 + sw); ye = min(f.shape[0], y0 + sh)
    if xe <= xs or ye <= ys:
        return
    sub = spr[ys - y0:ye - y0, xs - x0:xe - x0]
    al = sub[..., 3] * alpha
    rgb = sub[..., :3]
    if tint is not None and tint_k > 0:
        rgb = rgb * (1 - tint_k) + np.asarray(tint, F32)[None, None, :] * tint_k
    d = f[ys:ye, xs:xe]
    if additive:
        d += rgb * al[..., None]
    else:
        d *= (1 - al[..., None]); d += rgb * al[..., None]


def _ring(f, cx, cy, r, color, a, thick=2.2, soft=2.0):
    if a <= 0.004 or r <= 1:
        return
    bw, bh = kit.SW // 2, kit.SH // 2
    m = np.zeros((bh, bw), F32)
    cv2.circle(m, (int(cx * kit.S * 0.5), int(cy * kit.S * 0.5)),
               max(1, int(r * kit.S * 0.5)), 1.0,
               max(1, int(thick * kit.S * 0.5)), cv2.LINE_AA)
    m = kit.blur(m, max(0.6, soft * kit.S * 0.5))
    f += kit.up(m, kit.SW, kit.SH)[..., None] * np.asarray(color, F32) * a


def _iris(f, cx, cy, r, soft, color, a, inner=0.0, div=4):
    """光圈（低分辨率算，肉眼看不出来）。"""
    if a <= 0.004:
        return
    qw, qh = max(8, kit.SW // div), max(8, kit.SH // div)
    kq = kit.S / div
    yy, xx = np.mgrid[0:qh, 0:qw].astype(F32)
    d = np.hypot(xx - cx * kq, yy - cy * kq)
    rr = max(1.0, r * kq)
    ss = max(1.0, soft * kq)
    m = np.clip(1.0 - np.abs(d - rr) / ss, 0, 1) ** 1.6
    if inner > 0:
        m += np.clip(1.0 - d / rr, 0, 1) ** 2.6 * inner
    f += kit.up(m, kit.SW, kit.SH)[..., None] * np.asarray(color, F32) * a


def _speed_lines(f, cx, cy, t, color, alpha, n=9, seed=77, rmin=0.30,
                 speed=1.0, length=1.0, jit=0.0):
    """6–10 条纵向速度线，长度随高频能量抖动。"""
    if alpha <= 0.004:
        return
    bw, bh = kit.SW // 2, kit.SH // 2
    m = np.zeros((bh, bw), F32)
    rs = np.random.default_rng(seed)
    base = min(bw, bh)
    ang = rs.random(n) * 6.28318 + 0.2
    off = rs.random(n) * 2.0 * base
    ln0 = (0.16 + 0.34 * rs.random(n)) * base * (0.7 + 1.2 * speed) * length
    th = (1 + 1.6 * rs.random(n)).astype(np.int32)
    cxp, cyp = int(cx * kit.S * 0.5), int(cy * kit.S * 0.5)
    span = base * 2.0
    for i in range(n):
        ca, sa = math.cos(ang[i]), math.sin(ang[i])
        ln = ln0[i] * (0.55 + 0.9 * jit * (0.6 + 0.4 * math.sin(t * 9.0 + i)))
        r1 = (rmin * base + off[i] + t * 900 * speed) % span + 30
        p1 = (int(cxp + ca * r1), int(cyp + sa * r1))
        p2 = (int(cxp + ca * (r1 + ln)), int(cyp + sa * (r1 + ln)))
        cv2.line(m, p1, p2, 1.0, int(th[i]), cv2.LINE_AA)
    m = kit.blur(m, 1.0 * kit.S)
    f += kit.up(m, kit.SW, kit.SH)[..., None] * np.asarray(color, F32) * alpha


def _vstreaks(f, t, color, alpha, speed=1.0, seed=31, n=26):
    """程序化竖线：设计稿里的「星星状」残影（只做底纹）。"""
    if alpha <= 0.004:
        return
    bw, bh = kit.SW // 2, kit.SH // 2
    m = np.zeros((bh, bw), F32)
    rs = np.random.default_rng(seed)
    xs = rs.random(n) * bw
    ys = rs.random(n) * bh
    ln = (0.04 + 0.14 * rs.random(n) ** 2.0) * bh * (0.6 + 1.3 * speed)
    th = (1 + 1.4 * rs.random(n)).astype(np.int32)
    ys = (ys + t * 700 * speed) % (bh * 1.2) - bh * 0.1
    for i in range(n):
        cv2.line(m, (int(xs[i]), int(ys[i])), (int(xs[i]), int(ys[i] + ln[i])),
                 1.0, int(th[i]), cv2.LINE_AA)
    m = kit.blur(m, 0.9 * kit.S)
    f += kit.up(m, kit.SW, kit.SH)[..., None] * np.asarray(color, F32) * alpha


def _punch(f, t, amt=0.021):
    """每 1/8 拍一次的径向冲击：镜头永远在动。"""
    ph = (t / (mus.BEAT / 8.0)) % 1.0
    d = amt * (1.0 - ph) ** 2.6
    if d < 0.0012:                 # 已经在 1.000 附近，省掉一次全幅 warp
        return
    f[:] = kit.scale_about(f, kit.SW * 0.5, kit.SH * 0.5, 1.0 + d, 0)


def _scrim(f, y0=0.70, k=0.62):
    y = np.arange(kit.SH, dtype=F32) / kit.SH
    m = np.clip((y - y0) / max(1e-3, 1 - y0), 0, 1) ** 1.35
    f *= (1 - (m * k)[:, None, None])


def _cam(t):
    """全章共用的相机：保证 56.07 处两个场景严丝合缝。"""
    p = kit.cl((t - 51.93) / (56.07 - 51.93))
    dist = 940.0 - 120.0 * p + 22.0 * math.sin(t * 1.63)
    focal = 1470.0 + 70.0 * p
    cx = kit.W * 0.5 + 34.0 * math.sin(t * 0.71)
    cy = kit.H * 0.5 + 20.0 * math.sin(t * 1.13 + 1.0)
    bank = 0.055 * math.sin(t * 0.62) + 0.16 * (1.0 - p) ** 2 * math.sin(t * 9.0)
    return dist, focal, cx, cy, bank


def _cmap(xs, ys, zz, kk):
    q = np.clip((2650.0 - zz) / 2500.0, 0, 1)
    col = kit.mixc(kit.hx("2A5480"), kit.MINT, np.clip(q * 1.5, 0, 1) ** 0.8)
    col = kit.mixc(col, kit.CREAM, np.clip((q - 0.74) / 0.26, 0, 1) ** 1.3 * 0.95)
    return col


def _prev_get(i):
    if _PREV["buf"] is not None and i - _PREV["i"] == 1:
        return _PREV["buf"]
    return None


_TAPS = ((1.05, 0.92), (1.13, 0.72), (1.24, 0.54), (1.38, 0.38),
         (1.56, 0.26), (1.80, 0.16), (2.10, 0.09))
_TAPS5 = ((1.08, 0.90), (1.20, 0.72), (1.36, 0.52), (1.58, 0.32), (1.90, 0.16))


def _zoom_blur(small, cx, cy, taps=_TAPS):
    """半分辨率径向缩放模糊 = 一帧真正的运动拖尾。"""
    acc = small.copy()
    for k, w in taps:
        acc += kit.scale_about(small, cx * kit.S * 0.5, cy * kit.S * 0.5, k, 0) * w
    return acc


# ---------------------------------------------------------------- 隧道
def _tunnel(c, t, dist, focal, cx, cy, bank, rf, near_target=1.10,
            mid_target=0.85, far_target=0.55, near_gain=1.0, trail=1.0,
            use_frac=0.92, rib=0.70, bands=3, taps=_TAPS):
    cl = c.cl
    A = cl.sh(cloud.shape_tunnel, R_T, L_T)

    ph = (t * SPEED) % 1.0
    u = (A[:, 2] + 0.35 * L_T) / L_T
    u = (u - ph) % 1.0
    z = (u * L_T - 0.35 * L_T).astype(F32)
    th = np.arctan2(A[:, 1], A[:, 0]) + 0.85 * (z / L_T) + t * 0.30
    # 6 条螺旋光肋 + 一串向镜头飞来的光环
    ribs = (1.0 - rib) + rib * (0.5 + 0.5 * np.cos(6.0 * th)) ** 1.4
    rings = 0.82 + 0.18 * (0.5 + 0.5 * np.cos(6.28318 * z / RING)) ** 2.0
    mod = ribs * rings
    r = np.hypot(A[:, 0], A[:, 1]) * rf
    P = np.stack([r * np.cos(th), r * np.sin(th), z], 1).astype(F32)
    P = cl.transform(P, roll=bank)

    dz = z + dist
    pl = np.zeros((kit.SH, kit.SW, 3), F32)
    use = cl.u < use_frac

    def _band(mask, size, blur_r, btgt, bval, shade, glow, tm):
        mask = mask & use
        if int(mask.sum()) < 24:
            return
        Q = P.copy()
        Q[~mask, 0] = 1.0e6
        Q[~mask, 1] = 1.0e6
        bb = np.where(mask, bval, 0.0).astype(F32)
        cl.draw(pl, Q, dist=dist, focal=focal, cx=cx, cy=cy, size=size,
                blur_r=blur_r, color_map=_cmap, bright=bb, target=btgt,
                shade=shade, glow_gain=glow, core_gain=2.0, tm=tm)

    # 近处大而亮、远处小而暗
    _band((dz > 240.0) & (dz < 700.0), 6.6, 6.0, near_target,
          np.clip(2.10 - dz / 520.0, 0.4, 2.2) * near_gain * mod, 1.5, 0.75, 2.1)
    if bands >= 3:
        _band((dz >= 700.0) & (dz < 1500.0), 3.0, 3.0, mid_target,
              np.clip(1.25 - dz / 2600.0, 0.05, 1.4) * mod, 1.3, 0.45, 1.8)
    _band(dz >= (700.0 if bands < 3 else 1500.0), 1.35, 1.6, far_target,
          np.clip(0.52 - dz / 5200.0, 0.03, 0.6) * mod, 1.2, 0.30, 1.6)

    bw, bh = kit.SW // 2, kit.SH // 2
    small = kit.up(pl, bw, bh)
    wsum = 1.0 + sum(w for _, w in taps)
    acc = _zoom_blur(small, cx, cy, taps) * (1.0 / wsum)   # 归一化：亮度不随帧数漂移
    g = 0.62 * trail
    if g > 0.01:
        prev = _prev_get(c.frame_i)
        if prev is not None:                       # 上一帧的模糊副本，低权重混合 = 真实残影
            acc *= 0.78
            acc += kit.blur(prev, 2.2) * 0.22
        _PREV["i"] = c.frame_i
        _PREV["buf"] = acc
    return kit.up(acc, kit.SW, kit.SH) * (wsum * g)


# ---------------------------------------------------------------- 壁上的她
def _maid_src():
    s = _C.get("maid")
    if s is None:
        s = kit.trim(kit.load(stage.MAID_L))
        _C["maid"] = s
    return s


_SLOTS = ((0.06, math.pi * 1.03, 340.0),
          (0.28, math.pi * 0.60, 300.0),
          (0.48, math.pi * 1.42, 370.0),
          (0.70, 0.05, 320.0))


def _wall_her(f, t, dist, focal, cx, cy, pw, alpha_k, band_n=6):
    """贴在隧道壁上的她：横切成条、越远越小越暗，加法叠加（光里的残影）。"""
    src = _maid_src()
    sh_, sw_ = src.shape[:2]
    tint = kit.hx("9CC4E8")
    for i, (u0, th0, hw) in enumerate(_SLOTS):
        u = (u0 - pw) % 1.0
        dz = (u * L_T - 0.35 * L_T) + dist
        if dz < 210.0 or dz > 1500.0:
            continue
        k = focal / dz
        th = th0 + 0.10 * math.sin(t * 0.5 + i)
        sx = cx + R_T * 1.06 * math.cos(th) * k
        sy = cy - R_T * 1.06 * math.sin(th) * k
        hpx = min(1250.0, hw * k)
        if hpx < 80.0:
            continue
        a = alpha_k * kit.cl((dz - 300.0) / 460.0) * (1.0 - 0.85 * kit.cl((dz - 620.0) / 900.0))
        a = min(a, 0.62)
        if a <= 0.01:
            continue
        a *= 0.84 + 0.16 * math.sin(t * 5.3 + i * 2.1)
        wpx = max(8, int(hpx * sw_ / sh_))
        spr = cv2.resize(src, (wpx, int(hpx)),
                         interpolation=cv2.INTER_AREA if hpx < sh_ else cv2.INTER_LINEAR)
        if hpx < 380.0:
            spr = kit.blur(spr, 1.2)
        hh, ww = spr.shape[:2]
        x0 = sx * kit.S - ww * 0.5
        y0 = sy * kit.S - hh * 0.5
        bh = max(2, hh // band_n)
        for b in range(band_n):
            ys = b * bh
            ye = hh if b == band_n - 1 else (b + 1) * bh
            off = (math.sin(t * 3.4 + i * 1.7 + b * 2.3) * 34.0
                   + (b - 2.5) * 7.0) * kit.S
            _blit(f, spr[ys:ye], x0 + off, y0 + ys,
                  a * (0.62 + 0.38 * math.cos(b * 0.9)), tint=tint, tint_k=0.52,
                  additive=True)
        my = int(y0 + hh * (0.5 + 0.33 * math.sin(t * 1.6 + i)))
        ya, yb = max(0, my), min(f.shape[0], my + max(1, int(2.0 * kit.S)))
        xa, xb = max(0, int(x0)), min(f.shape[1], int(x0 + ww))
        if yb > ya and xb > xa:
            f[ya:yb, xa:xb] += kit.MINT * (a * 0.55)

    # 两张整身的残影（stage.character，不同 scale/alpha）
    for j, i in enumerate((1, 2)):
        u0, th0, hw = _SLOTS[i]
        u = (u0 - pw) % 1.0
        dz = (u * L_T - 0.35 * L_T) + dist
        if dz < 280.0 or dz > 1500.0:
            continue
        k = focal / dz
        sx = cx + R_T * 1.00 * math.cos(th0) * k
        sy = cy - R_T * 1.00 * math.sin(th0) * k
        hpx = min(680.0, hw * k * 1.2)
        if hpx < 150.0:
            continue
        hq = max(160, int(round(hpx / 160.0) * 160))
        stage.character(f, stage.MAID_L, sx, sy, hq,
                        alpha=alpha_k * 0.30 * kit.cl((dz - 250.0) / 700.0)
                        * (1.0 - 0.8 * kit.cl((dz - 700.0) / 800.0)),
                        tint=kit.hx("A8CBE8"), tint_k=0.35,
                        flip=(j == 0), rot=0.03 * math.sin(t + i),
                        additive=True)


# ---------------------------------------------------------------- 鲸
def _bust():
    """胸像母版：裁到 47%，底边融进光里，暗色本体 + 青色描边（烘焙进去）。"""
    b = _C.get("bust")
    if b is not None:
        return b
    a = kit.trim(kit.load(stage.WHALE["serious"]))
    h, w = a.shape[:2]
    cut = int(h * 0.47)
    b = a[:cut].copy()
    yy = np.arange(cut, dtype=F32) / max(1, cut - 1)
    ramp = np.clip((0.84 - yy) / 0.34, 0, 1) ** 1.15
    b[..., 3] *= ramp[:, None]

    al = b[..., 3]
    edge = np.clip(al - cv2.erode(al, np.ones((7, 7), np.uint8)), 0, 1)
    edge = cv2.GaussianBlur(edge, (0, 0), 2.0)[..., None]
    ink = np.array([0.012, 0.036, 0.060], F32)
    body = b[..., :3] * 0.16 + ink * 0.84
    rim = np.array([0.40, 0.85, 0.80], F32)
    b[..., :3] = body * (1 - edge) + rim * edge

    b = kit.up(b, int(BUSTW * kit.S), max(2, int(cut * BUSTW * kit.S / w)))
    _C["bust"] = b
    return b


def _whale_geo(cx, cy, wlog):
    bust = _bust()
    hlog = wlog * bust.shape[0] / bust.shape[1]
    return cx + (EYE_FX - 0.5) * wlog, cy + (EYE_FY - 0.5) * hlog, hlog


def _bust_at(wpx):
    """按宽度缓存缩放后的胸像（每 8px 一档，避免每帧重采样）。"""
    q = max(64, int(round(wpx / 8.0)) * 8)
    key = ("bw", q)
    b = _C.get(key)
    if b is None:
        b = kit.up(_bust(), q, max(2, int(round(q * _bust().shape[0] / _bust().shape[1]))))
        _C[key] = b
        if len(_C) > 26:                       # 别让缓存无限长
            for k in [k for k in _C if isinstance(k, tuple)][:-12]:
                _C.pop(k, None)
    return b


def _whale(f, t, cx, cy, wlog, alpha, glow_eye=0.0, lit=0.0):
    """鲸：暗剪影（描边已烘焙）+ 眼睛一点光。"""
    spr = _bust_at(wlog * kit.S)
    sw_, sh_ = spr.shape[1], spr.shape[0]
    x0 = cx * kit.S - sw_ * 0.5
    y0 = cy * kit.S - sh_ * 0.5
    _blit(f, spr, x0, y0, alpha)
    if lit > 0.02:
        _blit(f, spr, x0, y0, alpha * 0.6 * lit, tint=kit.CREAM, tint_k=0.82,
              additive=True)
    ex, ey, hlog = _whale_geo(cx, cy, wlog)
    if glow_eye > 0:
        ec = kit.MINT
        kit.add_radial(f, ex, ey, 20 + 14 * glow_eye, ec, 0.62 * glow_eye, 2.2)
        kit.add_radial(f, ex, ey, 150 + 90 * glow_eye, ec, 0.13 * glow_eye, 2.6)
        f += kit.circle_mask(ex, ey, 3.6, 1.1)[..., None] * ec * (2.2 * glow_eye)
    return ex, ey, hlog


# ================================================================ 51.93 冲进光里
@scene(51.93, 56.07, "TRAVEL", "C")
def travel(c):
    t = c.t
    f = c.f
    W, H = kit.W, kit.H
    p = kit.cl((t - 51.93) / (56.07 - 51.93))
    pulse = c.pulse()
    jit = min(1.0, (c.band("pres") or 0.0) * 2.2 + 0.4)
    c.post.update(grain=0.028, vignette=0.60, bloom=(1.05, 30, 0.60),
                  chroma=2.4 * (1 - p) ** 2.2)

    f += kit.vgrad(kit.SH, kit.SW, kit.hx("061625"), kit.hx("010306"), 1.6)
    dist, focal, cx, cy, bank = _cam(t)

    rf = 1.0 + 0.135 * pulse + 0.055 * math.sin(t * 2.35) + 0.10 * (1 - p) ** 2
    pl = _tunnel(c, t, dist, focal, cx, cy, bank, rf,
                 near_target=1.05 + 0.15 * pulse, mid_target=0.85,
                 far_target=0.55, near_gain=1.0, trail=1.0, taps=_TAPS5)
    f += pl * (0.95 + 0.18 * pulse)

    kit.add_radial(f, cx, cy, 62 + 26 * pulse, kit.CREAM, 0.95 + 0.30 * pulse, 2.0)
    kit.add_radial(f, cx, cy, 240 + 120 * pulse, kit.MINT, 0.15 + 0.06 * pulse, 2.4)

    _wall_her(f, t, dist, focal, cx, cy, (t * 0.55) % 1.0, 0.78)

    _speed_lines(f, cx, cy, t, kit.mixc(kit.MINT, kit.CREAM, 0.45),
                 0.52 + 0.26 * pulse, n=11, seed=77, rmin=0.24,
                 speed=1.0 + 0.8 * p, length=1.9, jit=jit)
    _vstreaks(f, t, kit.MINT, 0.09 + 0.05 * pulse, speed=1.0)

    _punch(f, t, 0.020)

    bp = c.beat()
    _ring(f, cx, cy, 90 + 1020 * bp ** 0.85, kit.MINT, 0.16 * (1 - bp) ** 2.2,
          thick=2.2, soft=6.0)
    if mus.is_down(t, 0.05):
        _ring(f, cx, cy, 60, kit.CREAM, 0.15, thick=4.0, soft=7.0)
        f += kit.CREAM * 0.05

    if p < 0.14:                      # 冲出眩晕的第一口白
        kq = 1.0 - p / 0.14
        f += (kit.CREAM * 0.15 + kit.MINT * 0.06) * (kq ** 2.2)
        _ring(f, cx, cy, 60 + 1700 * (1 - kq), kit.CREAM, 0.22 * kq, thick=6.0, soft=12.0)

    q = kit.cl((t - 55.15) / 0.90)     # 尽头越来越亮（UNITE 的引子）
    if q > 0:
        kit.add_radial(f, cx, cy, 180 + 380 * q, kit.CREAM, 0.30 * q, 2.1)

    _scrim(f, 0.74, 0.62)
    c.lyric(y=H * 0.815, size=42, color=kit.CREAM, glow=0.45)
    stage.line_tag(f, "travel through the light   ·   %0.2fc" % (1.0 + 0.42 * p),
                   84, H - 96, alpha=0.55 + 0.25 * pulse, color=kit.MINT, size=20)
    kit.draw_text(f, "SPEED OF LIGHT", W - 84, 44, 18, kit.MINT, kit.F_MONO,
                  0.30 + 0.22 * pulse, 5.0, "right")
    c.post["vignette"] = 0.60 - 0.10 * p


# ================================================================ 56.07 合一
@scene(56.07, 59.77, "UNITE", "C")
def unite(c):
    t = c.t
    f = c.f
    W, H = kit.W, kit.H
    p = kit.cl((t - 56.07) / (59.77 - 56.07))
    pulse = c.pulse()
    jit = min(1.0, (c.band("pres") or 0.0) * 2.2 + 0.4)
    c.post.update(grain=0.026, vignette=0.56, bloom=(1.00, 30, 0.72),
                  chroma=0.9 * (1 - p))
    f += kit.vgrad(kit.SH, kit.SW, kit.hx("071B2A"), kit.hx("010407"), 1.55)

    dist0, focal0, cx, cy, bank = _cam(t)
    q = kit.ease_out(kit.cl(p / 0.70))
    dist = dist0 + 1500.0 * q
    focal = focal0 - 300.0 * q
    rf = (1.0 + 0.13 * pulse) * (1.0 + 3.2 * q)

    pl = _tunnel(c, t, dist, focal, cx, cy, bank * 0.4, rf,
                 near_target=1.05, mid_target=0.85, far_target=0.52,
                 near_gain=1.0 - 0.65 * q, trail=1.0 - 0.45 * q,
                 use_frac=0.92 * (1.0 - 0.35 * q), rib=0.70 * (1.0 - 0.5 * q),
                 bands=2)
    f += pl * (0.95 + 0.16 * pulse + 1.35 * q)
    sl = (0.30 + 0.22 * pulse) * (1.0 - 0.8 * q)
    if sl > 0.07:
        _speed_lines(f, cx, cy, t, kit.MINT, sl,
                     n=9, rmin=0.28, speed=1.0, length=1.2, jit=jit)
    _punch(f, t, 0.020 * (1.0 - 0.6 * q))

    wf = kit.cl((t - 58.88) / 0.62) ** 0.75      # 最后一拍的白场权重
    # 隧道尽头张开的那张嘴
    _iris(f, cx, cy, 150 + 520 * q, 90 + 150 * q,
          kit.mixc(kit.MINT, kit.CREAM, q), (0.26 + 0.30 * q) * (1.0 - 0.85 * wf),
          inner=0.40 + 0.45 * q)
    kit.add_radial(f, cx, cy, 240 + 700 * q, kit.CREAM, 0.16 + 0.26 * q, 2.2)

    _wall_her(f, t, dist0, focal0, cx, cy, (t * 0.55) % 1.0, 0.95 * (1 - q) ** 1.4)

    # ---------- 鲸：暗剪影，眼睛一点光（构图 0.55W / 0.50H，宽 1180→1360）
    wa = kit.cl((t - 56.18) / 1.00) * (1.0 - 0.92 * wf)
    gw = kit.cl((t - 56.20) / 2.90)
    wlog = 1180.0 + 180.0 * kit.ease_out(gw)
    wcx, wcy = W * 0.55, H * 0.50
    ex0, ey0, _ = _whale_geo(wcx, wcy, wlog)
    if wa > 0.012:
        kit.add_radial(f, ex0, ey0, 460, kit.hx("1F6B78"), 0.22 * wa, 2.0)
        ex, ey, hlog = _whale(f, t, wcx, wcy, wlog, wa * 0.97,
                              glow_eye=wa * (0.55 + 0.45 * kit.cl((t - 58.2) / 1.2)),
                              lit=kit.cl((t - 58.6) / 1.1) * (1.0 - 0.9 * wf))
    else:
        ex, ey = ex0, ey0

    # ---------- 她溶解成粒子，被吞进那只眼睛
    if "body" not in _C:
        bd = c.cl.sh(cloud.sprite_field, stage.MAID_L, 470.0, 30.0)
        _C["bodyP"] = bd["P"].copy()
        _C["bodyL"] = (0.34 + 1.5 * bd["lum"]).astype(F32)
        sp = c.cl.sh(cloud.shape_spiral, 330.0, 3, 3.4)
        _C["spirP"] = c.cl.transform(sp, pitch=1.0).astype(F32)
    P0 = c.cl.transform(_C["bodyP"], offset=(-350.0, 130.0, -60.0))
    spP = _C["spirP"] + np.array([-350.0, 130.0, -60.0], F32)
    q1 = kit.cl((t - 56.55) / 1.05)
    q2 = kit.cl((t - 57.62) / 1.15)
    P = c.cl.morph(P0, spP, q1, stagger=0.72, ease="io")
    tgt = np.array([(ex - W * 0.5) * 300.0 / 1400.0,
                    (H * 0.5 - ey) * 300.0 / 1400.0, 250.0], F32)
    P = c.cl.morph(P, np.tile(tgt, (c.cl.n, 1)), q2, stagger=0.55, ease="in")
    if q1 > 0:
        c.cl.draw(f, P, dist=1400, focal=1400, cx=W * 0.5, cy=H * 0.5, size=2.2,
                  blur_r=1.9, color=kit.mixc(kit.PALE, kit.CREAM, 0.6),
                  bright=_C["bodyL"] * (1.0 - 0.55 * q2),
                  target=0.60, glow_gain=0.55, shade=1.0, tm=1.7, core_gain=1.5)

    # ---------- 最后一拍：整屏奶油白 + 青
    if wf >= 0.86:                              # 已经是一片奶油白，直接铺满（省掉两次全幅运算）
        f[:] = kit.CREAM * 0.97 + kit.MINT * 0.10
    elif wf > 0:
        f *= (1.0 - 0.88 * wf)
        f += (kit.CREAM * 1.04 + kit.MINT * 0.15) * wf
        if wf < 0.62:
            _speed_lines(f, ex, ey, t, kit.CREAM, 0.18 * wf, n=12, seed=13,
                         rmin=0.02, speed=0.6, length=2.4, jit=0.0)

    if t > 59.735:                    # 留白 2 帧，再切副歌
        f[:] = kit.CREAM * 1.0 + kit.MINT * 0.12

    if wf < 0.97:
        _scrim(f, 0.77, 0.45 * (1 - wf))
    wq = kit.cl(wf * 1.15)
    la = 1.0 - kit.cl((t - 59.28) / 0.22)
    if la > 0.02:
        c.lyric(y=H * 0.815, size=42, color=kit.mixc(kit.CREAM, kit.hx("0A1E2C"), wq),
                glow=0.45 * (1 - wq), alpha=la)
    if wf > 0.05:
        kit.draw_text(f, "unite → 1", W - 84, 44, 18, kit.hx("0B2231"), kit.F_MONO,
                      wf * 0.55, 5.0, "right")
    else:
        kit.draw_text(f, "unite into one", W - 84, 44, 18, kit.MINT, kit.F_MONO,
                      0.30 + 0.22 * pulse, 5.0, "right")
    c.post["vignette"] = 0.56 - 0.45 * wf
