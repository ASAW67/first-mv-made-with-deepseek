# 引擎拆解

9 825 行 Python（引擎 3 375 + 章节 6 450），没有任何渲染框架。下面按「数据怎么流」讲，不讲 API 手册（那是 `engine/API.md`）。

---

## 0. 数据流

```
m4a ──audio2.py──► analysis2.json ──lyralign.py──► analysis3.json
                                                      │
                                    ┌─────────────────┘
                                    ▼
              mus.py  ──  拍/频带/歌词/结构的纯函数查询层
                                    │
   ch_a.py … ch_j.py  ──►  @scene 注册到 film2.SCENES
                                    │
                         film2.Ctx（每帧新建）
                                    │
              kit.py（2D 底座） + cloud.py（粒子场） + stage.py（舞台元件）
                                    │
                         film2.render(t) → float32 (2160,3840,3)
                                    │
                       render2.py：分段 → ffmpeg → concat → 混音
```

**关键约束：渲染必须是 `t` 的纯函数。** 没有跨帧状态、没有随机漂移（随机数全部来自 `c.rng(salt)` 的固定种子），
所以 51 个场景可以被切成 106 段扔给 6 个进程并行算，拼回来一帧不差。

---

## 1. 音乐 → 时间轴

### 1.1 自己写 STFT

没有 librosa。`audio2.py` 里 numpy 手写：

- STFT：`NFFT=2048 / HOP=512 / Hann`，44.1 kHz → 每帧 **23.2 ms**；
- 频带能量：sub / bass / lowmid / mid / presence / brilliance / air 七段；
- 谱质心、谱平坦度、高频含量（HFC）、谱通量、立体声宽度、RMS。

### 1.2 BPM 不是估的，是搜出来的

先用低频 kick 峰做粗估，再用**相位离散最小化**（phase dispersion minimization）在 100–160 BPM 上扫：

对每个候选 BPM，把每个 kick 时间折叠到拍相位上，算相位的圆统计离散度；离散度最小的那个就是答案。
结果是 **130.00 BPM，beat = 0.46155 s，相位 ≈ 0.2524 s**。

分段核验残差：0–53 s / 53–106 s / 106–159 s / 159–212 s 分别是 0.003 / 0.005 / 0.012 / −0.011 拍——
**全曲没有速度漂移**，所以一张拍网格能管到底。

拍→低频起音命中率 60.0%（offset 半拍时掉到 8.0%），说明相位正确。

### 1.3 歌词吸附（这支片子最「较真」的一步）

歌词文本给的时间是「大致在这附近」。要让每个词都落在人声起音上，`lyralign.py` 做了：

1. 再算一遍 **hop = 128（2.90 ms）** 的窄带 STFT；
2. 取**中央声道** `mid − 0.6 × side`，只在 **1400–6200 Hz** 求能量——这是人声辅音/元音的起音带；
3. 用动态规划把 96 行歌词整体吸附：搜索半径 0.85 s、偏离惩罚 λ=1.6、最小间隔 0.10 s；
4. 行内逐词再吸附一次（±0.28 s 窗口、单调、最小 0.02 s 间隔）。

**结果：行锚点处的人声起音能量 0.076 → 0.264（3.5 倍），行位移中位数 4 ms，最大 0.22 s。**

没有这一步，中文字幕会整体早/晚半拍——观众说不清哪里不对，但会觉得「字幕在飘」。

---

## 2. 分辨率无关

```python
kit.init(scale=2)      # 逻辑 1920×1080，物理 3840×2160
```

所有章节代码写的是 1920×1080 里的坐标。`kit.S` 负责放大，`kit.up()` 负责降采样，
`draw_text` 用 `size × S` 写字、用真实 ink 包围盒做字形缓存。

于是 `--scale 1` 是 1080p 预览（单帧 0.3–0.9 s），`--scale 2` 是 4K 母版（单帧 1.0–3.3 s），**同一份代码**。

---

## 3. 粒子场（cloud.py）

### 3.1 形状 = 采样函数

每个形状函数都返回 `(n, 3)` 的坐标：

`sphere / ball / ring / circle_filled / box_wire / box_fill / sine / lemniscate / heart / eggplant /
tomato / cat / god / text / fragments / spiral / tunnel / grid / wave_field / lattice / column / figure`

「文字」也是形状：`shape_text("EXECUTION", ...)` 用 PIL 把字画成 mask 再采样——
所以歌词里的 `EXECUTION` 真的可以是由 62 000 个点组成的字。

### 3.2 角色 = 立绘的采样

```python
f = cloud.sprite_field(n, rng, path, height_world=620)
# → {"P": 位置, "rgb": 该点的颜色, "lum": 该点的亮度}
```

粒子不是白色小点，而是**带着她围裙的白、裙子的蓝、尾巴的暗紫**。
`draw(..., bright=0.2 + 1.6*lum)` 让亮的地方更亮——她是由**她自己的光**点亮的。

### 3.3 形变

`morph(A, B, p)`：`p` 按粒子序号错峰推进（`stagger`），
`morph_arc(A, B, p, bow=110)` 让路径鼓出来一段弧——不然所有点走直线，像溶解，不像运动。

### 3.4 为什么必须自动曝光

62 000 个加性混合的点，密度天然不均：正面朝向相机的球面积累得比边缘多十倍。
固定亮度 → 正面死白、边缘发灰。

```python
ref = np.percentile(buf[buf > 1e-4], 96)
buf *= np.clip(target * exposure / ref, 0.05, 12)
```

取自身亮度的 96 分位当基准反推曝光，再 `1-exp(-lum·k)` 做保色相的软压缩。
**这一条是整支片子从「白团」变成「雾」的分界线。**

### 3.5 性能

- 用 `np.bincount` 做 splat，不用 `np.add.at`（慢 30 倍）；
- 双线性 4 邻域累积（`size > 1.4` 时 9 邻域）；
- 半分辨率累加 + 模糊上采样做辉光；
- 4K 单帧最重场景（番茄，79.4 s）2.95 s，其余 1.0–2.2 s。

---

## 4. 文字

自己写逐字排版，因为要**逐字动画**（打字机、逐字显影、逐字抖动）：

- `_glyph()` 把单个字画到 2.4×2.6 倍的画布上，再用 **alpha > 0.004 的真实非零包围盒**裁剪。
  用 `font.getbbox()` 会切掉斜体的降部，字形会「错字」——这是踩过的坑。
- `_layout()` 按 `font.getlength(ch)` 累计笔位，与 PIL 整行渲染逐像素一致（479 px == 479 px）。
- 阴影只作用在字形**外侧**：`g = clip(blur(sub) − sub, 0, 1)`。
  直接乘会让 44 px 的细衬线字被自己的阴影吃掉，变成暗橄榄色。

---

## 5. 每帧的总装

```python
f = kit.new_frame()          # float32 逻辑画布
scene_fn(c)                  # 章节画内容（只画 c.f）
# film2.render 里的 post：
bloom   (thresh=0.72, radius=26S, strength=0.62, downscale=3)
chroma  (0.6 px 色散)
scanlines / grain (0.018)
tone    (exposure 1.02, contrast 1.06, sat 1.05)
vignette(0.34)
```

后期是**整片统一**的，所以 51 个镜头的颗粒感和光晕是同一套——这是「看起来像一支片子」而不是 51 段拼接的原因。

---

## 6. 渲染

```bash
python render2.py --scale 2 --workers 6 --crf 21 --chunk 120 --resume 1
```

- 12 739 帧切成 106 段（每段 120 帧 = 2 s）；
- 每段一个 ffmpeg 子进程，吃 rawvideo rgb24，`libx264 / yuv420p / gop 120 / aq-mode 3`；
- `--resume 1` 用 **ffprobe 逐段核对 `nb_read_frames == 120`**——
  只查文件大小会漏掉被中断的残段（没有 moov atom，concat 出来只有 8 秒）；
- 全部完成后 `concat -c copy` 再混音。

**并行度实测（32 核 / 16 GB）：4 worker 0.71 段/分，6 worker 0.63 段/分，10 worker 0.19 段/分。**
4K 是内存带宽瓶颈，每进程约 1.3 GB——开太多反而慢 3 倍。

---

## 7. 为什么 4K 文件有 3.9 GB

全片每一帧都有**真实的胶片颗粒和粒子噪声**（不是后期贴的噪点，是 62 000 个点的随机分布本身）。
x264 在 crf 21 下必须保留这些高频细节，砍掉就变成塑料。
155 Mbps 看着夸张，对这类内容是这个量级的正常值。
