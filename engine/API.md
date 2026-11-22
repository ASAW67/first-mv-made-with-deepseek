# mv2 渲染 API（给章节作者）

所有坐标是 **1920×1080 逻辑像素**，`kit.S` 是超采样倍率（1=1080p 预览，2=4K）。字幕安全区：y 860–960 之间不要压重要画面。

## 0. 一个章节长什么样

```python
# ch_b.py
import math
import numpy as np
import kit, mus, stage, cloud
from film2 import scene

@scene(30.00, 33.92, "POINTS", "B")
def points(c):
    f = c.f                      # 当前帧（float32 HxWx3，已乘好 S 倍采样）
    t = c.t                      # 绝对时间（秒）
    c.post.update(grain=0.02, vignette=0.4, bloom=(0.7, 24, 0.5))
    stage.bg_image(f, stage.BG_PALACE, zoom=1.3, dim=0.4)
    ...
```

- 场景时间必须 **首尾相接**覆盖章节区间，不留缝、不重叠。跑 `python film2.py` 会打印缺口。
- 每个场景函数只负责画 `c.f`；**不要**做 bloom/vignette/grain，那是 `c.post` 的事。
- 随机数一律用 `c.rng(salt)`，保证可复现。
- 想加私有工具函数就写在自己章节文件里，**不要改** kit/stage/cloud/mus/film2。

## 1. kit（2D 底座）
```python
kit.init(scale)                     # 由 qc/render 调用，章节里不要调
kit.cl(x) / kit.lerp / kit.smoothstep / kit.remap(a,b,c,d,x)
kit.ease_io/ease_out/ease_in/ease_out_back/ease_expo_out/ease_out_quint
kit.mixc(c1, c2, t)                 # t 可以是标量或 (N,) 数组 → 返回 (3,) 或 (N,3)
kit.hx("FF7A6B")                    # 十六进制 → float32 颜色
kit.new_frame()/solid(c)/vgrad(H,W,c0,c1,power)
kit.add(dst,src)/over(...)/blur(img,r)/boxblur/shift(img,dx,dy)/scale_about(img,cx,cy,k,angle)
kit.up(img,w,h)                     # 缩放（缩小用 INTER_AREA）
kit.add_radial(f,cx,cy,radius,color,strength,power=2.2,ratio=1)
kit.circle_mask(cx,cy,r,thick)/poly_mask(pts)/line_mask(pts,width)
kit.sprite(path,height)             # 按逻辑高度缩放好的 RGBA，有缓存
kit.place(f,spr,cx,cy,k=1,alpha=1,tint=None,tint_k=0,glow=0,additive=False,flip=False,rot=0,anchor="foot")
kit.bloom(f,thresh,radius,strength)/chroma(f,amount)/scanlines(f,amt)/glitch_rows(f,t,amt,seed,band)
kit.tone(f,exposure,contrast,sat)/tonemap_soft(buf,k)
kit.text_mask/text_width/draw_text(f,txt,x,y,size,color,font,alpha=1,tracking=0,align="left",
                                   additive=False,glow=0,glow_color=None,clip=None,
                                   char_alpha=None,char_offset=None,stretch=1.0)
```
字体常量：`kit.F_SERIF_I`（Georgia 斜体，歌词首选）、`F_SERIF`、`F_SERIF_B`、`F_MONO`（Consolas）、`F_MONO_B`、`F_SANS_SC`、`F_SERIF_SC`（思源宋体）、`F_HEI`、`F_BAHN`。
`draw_text` 的 `y` 是**行顶部**。`align` = left/center/right。`glow` 是柔光，不要超过 0.5。

调色板：`NAVY DEEP INK CORAL MINT BLUE PALE GOLD VIOLET ROSE TEAL ICE SAND RUST CREAM PLUM ASH`，另有 `kit.WHITE / BLACK`。

## 2. mus（音乐）
```python
mus.DUR / mus.BPM(130.0) / mus.BEAT(0.46155) / mus.DOWN / mus.KEY / mus.LINES
mus.band(name, t)      # name: sub bass lowmid mid pres brill air
mus.env(name, t, win=0.35)   # 窗口内峰值，驱动动作更稳
mus.rms(t) / mus.rms_env(t) / mus.flux(t) / mus.hfc(t) / mus.centroid(t) / mus.stereo(t)
mus.novelty(t, win=1.2)
mus.beat_phase(t)  # 0..1
mus.beat_pulse(t, decay=9)   # 拍点冲击，拍上=1
mus.bar_phase(t) / mus.is_down(t) / mus.kick_near(t)
mus.line_at(t) / mus.line_index(t) / mus.find("If I'm a set of points") / mus.lines_between(t0,t1)
mus.word_progress(line, t) -> (已到词数, 当前词进度, 整行进度)
```
歌曲结构（可用时间点）：30.00 points / 33.92 circle / 37.60 sine / 41.30 infinity / 45.01 switch my current / 48.67 blind my vision / 50.52 so dizzy / 51.93 travel / 56.07 unite / 59.77 chorus「If I can」/ 64.37 only satisfaction / 68.98 run the execution / 70.59 trapped / 73.94 eggplant / 78.11 tomato / 81.21 tabby cat / 85.16 only god / 87.41 proof of my existence / 89.53 switch my gender / 92.55 do whatever / 96.44 switch my role / 99.91 enter the trance / 103.84 chorus 2 / 105.00 vibrations / 107.52 completion / 111.24 though you have left / 116.65 isolation / 119.98 erase fragments / 125.99 challenging your god / 129.46 illegal arguments / 148.36–158.31 Execution×14 / 159.27 Ein dos trois ne / 161.09 Fem liu / 161.77 Execution / 162.33 final chorus / 169.68 have you back / 172.38 run the execution / 174.62 trapped / 178.84 properly love / 181.43 question me / 185.29 algebraic expression of love / 188.52 though you are free / 190.09 trapped in love / 206.76 Execution(尾)。

## 3. stage（舞台元件）
```python
stage.bg_image(f, path=BG_PALACE, zoom=1, dx=0, dy=0, tint=None, tint_k=0, dim=1, blur_r=0, exposure=1, sat=1, rot=0)
stage.caustics(f, t, color=MINT, strength=0.1, scale=1, speed=0.35, y0=0.35)
stage.lightbeam(f, t, x=0.5, width=0.22, color=PALE, strength=0.22, sway=0.05, tilt=0.32)
stage.dust(f, t, n=260, color=PALE, alpha=0.5, box=(x0,x1,y0,y1))
stage.floor_reflection(f, t, y=0.78, strength=0.2)
stage.character(f, path, cx, cy, h, alpha=1, k=1, tint=None, tint_k=0, glow=0,
                flip=False, rot=0, rim=None, rim_k=0, shadow=0, additive=False, anchor="foot")
stage.chars_duo(f, t, h=800, gap=250, alpha=1, dy=0, tint=None, tint_k=0, rim=None, rim_k=0.5,
                glow=0, paths=None, xs=None)        # 左右两位女仆
stage.sheet_cell(path, idx, h, cols=6, rows=7)      # pet 图集单格
stage.cursor(f, x, y, t, size=26, color=WHITE, alpha=1, style="beam|block|ring", blink=True, glow=0.7)
stage.cursor_trail(f, pts, color, alpha, width)
stage.lyric(f, t, ln=None, style="soft|hard|terminal", y=0.855H, alpha=1, color=None, size=44,
            align="center", x=None, tracking=1.6, zh=True, words=True, glow=0.25, mono=False)
stage.line_tag(f, txt, x=84, y=H-96, alpha=1, color=ASH, size=20, prefix="// ")
stage.terminal(f, [(txt,color),...], x, y, size=22, alpha=1, color=MINT, lead=1.55, cursor_at=None)
stage.hud_corner(f, t, left="", right="", alpha=0.5, color=ASH, y=None)
stage.title_card(f, main, sub="", t=0, alpha=1, color=CREAM, sub_color=MINT, size=92, y=0.44H,
                 tracking=8, mono=False, glow=0.4, zh=None)
stage.code_panel(f, t, rows, x, y, w, alpha=1, size=19, theme=None, typed=1, caret=True, bg=True)
stage.param_list(f, t, [(label,value,ok),...], x, y, alpha=1, size=24, theme=None, typed=1, label_w=280)
stage.big_word(f, word, t, alpha=1, color=CREAM, size=250, y=None, tracking=14, mono=False, stretch=1, glow=0.5)
stage.letterbox(f, k=0.12) / stage.frame_number(f, t, ...) / stage.split_lines(f, x, y, [(a,b),...], ...)
```
素材：`BG_PALACE`(暗) `BG_PALACE_L`(亮) `MAID_L` `MAID_R` `MAID_R2`(带单片镜) `MAID_START_L` `MAID_START_R` `MAID_Q`(Q版) `PET`(6×7 图集) `BG_STUDY` `BG_OTHER["claude"|"gpt"|"gemini"|"kimi"|"grok"]` `WHALE["cheerful"|"shy"|"serious"|"confused"|"angry"|"frightened"|"exasperated"|"starry"]` `AI["claude"|"gpt"|"gemini"|"kimi"|"grok"]`。
`stage.TH_*` 是一组主题色，用 `stage.Theme(...)` 自定义。

## 4. cloud（3D 粒子）
```python
cl = c.cl                       # 62000 个粒子的实例，直接复用
P = cl.sh(cloud.shape_sine, 620.0, 120.0, 1.6, 0.0, 3.0)   # 会在实例内缓存
P = cl.transform(P, yaw=0.2, pitch=0.1, roll=0.0, scale=1.0, offset=(0,0,0))
M = cl.morph(A, B, p, stagger=0.6, ease="io", curve=None)  # 逐粒子错峰
cl.draw(f, P, color=1.0, size=1.6, dist=1500, focal=1150, cx=None, cy=None, oy=0,
        intensity=1.0, blur_r=1.6, half=True, additive=True, dof=0,
        color_map=lambda xs,ys,zz,kk: (N,3) 颜色, cutoff=None, shade=0.9,
        bright=标量或(N,), twinkle=(phase,amt), tm=2.0,
        core_gain=1.0, glow_gain=0.6, auto=True, target=0.9, exposure=1.0)
```
- **auto 曝光**：按自身 96 分位归一，`target=0.9` 是正常亮度，`target=0.3` 很暗，`1.2` 很亮。想让粒子层当背景就调 `target`。
- `bright` 数组式亮度（例如立绘的 lum）能把身体内部结构显出来。
- 形状：`shape_dots(n,rng,spread=(x,y,z))` `shape_sphere(R)` `shape_ball(R)` `shape_ring(R,tube,plane="xy"|"xz"|"yz")` `shape_sine(L,A,k,phase,tube)` `shape_lemniscate(A,tube)` `shape_heart(A)` `shape_eggplant(Len,R,grow)` `shape_tomato(R,lobes)` `shape_cat(size)` `shape_god(size)` `shape_text(txt,size,fp,tracking,depth)` `shape_fragments(spread,clusters)` `shape_spiral(R,arms,twist)` `shape_tunnel(R,L)` `shape_lattice(size,y,depth,jit,skip)` `shape_wave_field(size,y,A,k)` `shape_column(R,Hh)` `shape_figure(h)` `shape_circle_filled(R)`。
- `cloud.sprite_field(n, rng, path, height_world, depth)` → `{"P","rgb","lum","edge"}`：把立绘变成粒子，并带回立绘自身颜色/亮度。用法见 ch_a 的 PARAMS。

## 5. 渲染与检查
```bash
python qc.py 30,33.9,37.6 --scale 1 --cols 3 --w 640 --out qcB.png
python qc.py --range 30,52,1.8 --scale 1 --cols 4 --w 560 --out qcB.png
```
`qc.py` 会同时打印每帧耗时（1080p 应 < 900ms）。

## 6. 视觉宪法（必须遵守）
1. **底色永远是深海蓝黑**（#02060B–#0B2231），发光只有青绿 `MINT 7FE3D4`、珊瑚 `CORAL FF7A6B`、紫 `VIOLET B79CFF`、金 `GOLD E8B27A`、奶油 `CREAM FFF4E2`。
2. **文字三种**：Georgia 斜体（歌词/诗）、Consolas（终端/代码/数字）、思源宋体（中文）。行距 > 1.2，字距 1–3。大标语可用 `stage.big_word` 或 100px 以上。
3. **镜头永远在动**：缓慢推/拉/横移，用 `kit.ease_*`；不要静止超过 1.5 秒。
4. **拍点要有事发生**：每个强拍至少一个视觉事件（闪光、切割、粒子跳动、字跳动）。用 `c.pulse()`。
5. **别糊**：不要叠加多层半透明大光斑；`bloom` 阈值 0.6–0.75；暗角 0.3–0.5。
6. 角色立绘只在必要时满屏；一般做成舞台的一部分（远、小、被光切开）。
7. 中文只在歌词行、终端注释、标题副行出现，不滥用。
8. 每个场景至少一个「只有代码能做到」的东西：粒子形变、逐字排版、程序化线框、音频驱动的位移。
