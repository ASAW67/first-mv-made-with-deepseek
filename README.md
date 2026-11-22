# 用 DeepSeek 做的第一个 MV

**《world.execute(me);》— 一支每一帧都由 Python 算出来的 MV**

没有剪辑软件，没有 AE 工程，没有一张现成的分镜图。
画面是 numpy 数组，角色是立绘采样出来的粒子场，时间是音乐分析出来的拍网格。

[![preview](stills/01_chorus1_if_i_can.jpg)](preview/highlight_silent.mp4)

> 上面这张是成片 61.2 秒的原始帧。点开看 **42 秒静音高光**（`preview/highlight_silent.mp4`，14 个镜头各 3 秒）。
> 完整成片不随仓库分发（原因见 [NOTICE](NOTICE.md)）：4K60 / 12 739 帧 / 3 分 32 秒 / 3.94 GB。

---

## 这是什么

2026 年 11 月，我用 DeepSeek 写了一个近 1 万行的渲染引擎，把 Mili 的《world.execute(me);》做成一支 MV。

整支片子的读法是**一次 `execute` 请求的一生**：

一颗由立绘采样出来的粒子身体，在 51 个镜头里不断按照歌词把自己重组成
**点 → 球面 → 圆 → 正弦 → 无穷 → 电流 → 隧道 → 茄子 → 番茄 → 虎斑猫 → 神 → 性别 → 角色 → 振动 → 碎片 → 心**；
白色的光标是「你」，深海宫殿是内存，而她最后被困在自己写的那行代码里。

| | |
|---|---|
| 时长 | 212.32 s（3:32），**130.00 BPM** 实测（相位 0.25 s） |
| 画幅 | 3840 × 2160 / 60 fps / 12 739 帧 |
| 镜头 | 51 个场景，10 个章节文件，首尾严丝合缝（0 → 212.32 s 无 gap） |
| 粒子 | 每帧 62 000 个，带立绘自身颜色与亮度 |
| 代码 | 引擎 3 375 行 + 章节 6 450 行 = 9 825 行，全部 Python（numpy / OpenCV / Pillow） |
| 依赖 | 没有 AE、没有 Blender、没有 ComfyUI、没有任何生成式模型 |
| 渲染 | 4K 单帧 1.0–3.3 s，6 进程约 95 分钟 |

---

## 怎么做到的（四个关键决定）

**1. 音乐不是配乐，是时间轴本身。**
自己写 STFT（`audio2.py`，2048 点 / hop 512），用手写「相位离散最小化」把 BPM 定到 130.00、把拍网格相位定到 0.25 s；
再用 hop=128（2.9 ms）的窄带 STFT 在 **1400–6200 Hz 中央声道**上算人声起音，
用动态规划把 96 行歌词逐行吸附到起音上——**行锚点处的起音能量从 0.076 提到 0.264（3.5 倍）**，
中位偏移 4 ms。之后每一个强拍、每一句歌词都直接驱动运镜和事件。

**2. 角色不是贴图，是 62 000 个粒子。**
`sprite_field()` 把立绘当作**采样源**：随机取点、读取该点像素的颜色和亮度，
于是每颗粒子自带她衣服的蓝、围裙的白、尾巴的暗紫。粒子聚合时是她，散开时是星云——
不是把图淡出，是**同一批点换了一个形状函数**。全程没有重绘过任何一张原画。

**3. 加性粒子必然糊成一团白，所以要「自动曝光」。**
62 000 个加性混合的点在球面上密度天然不均，定死亮度要么死白要么发灰。
解法是每帧取自身亮度的 **96 分位**做基准反推曝光（`cloud.draw(auto=True, target=…)`），
再走 `1-exp(-lum·k)` 的保色相软压缩。这一条把画面从「白团」救成了「雾」。

**4. 4K 不是另一套代码。**
引擎内部永远在逻辑 1920×1080 上构图，`kit.S` 做超采样。所以
`--scale 1` 是 1080p 预览、`--scale 2` 就是 4K 母版，**代码里没有一个像素写死**。

---

## 快速开始

```bash
pip install numpy opencv-python Pillow

# 1) 把音乐放好（不随仓库分发，见 NOTICE）
export MV_AUDIO="/path/to/Mili - world.execute(me);.m4a"

# 2) 分析 → 对齐
python engine/audio2.py        # 频谱 / 拍网格 / 调性 → analysis2.json
python engine/lyralign.py      # 把歌词吸附到人声起音 → analysis3.json

# 3) 看时间线（51 个场景是否首尾相接）
cd engine && python film2.py

# 4) 抽帧看效果
python qc.py 61.2 --scale 1 --cols 1 --w 960 --ch ch_d --out shot.png

# 5) 渲染
python render2.py --scale 1 --workers 8 --out out/preview_1080p.mp4
python render2.py --scale 2 --workers 6 --crf 21 --chunk 120 --resume 1 \
                  --out "out/world.execute(me)_4K60.mp4"
```

没有素材也能跑：`MV_ASSETS` 指不到立绘时，粒子场会退化成纯几何形状（球、正弦、猫的剪影照旧成立）。
`engine/sunset.py` 是另一支独立的 5 秒日落短片（无素材依赖，`python sunset.py --video` 直接出片），
当作「这套引擎还能干什么」的样例。

---

## 目录

```
engine/            渲染引擎（纯 Python）
  audio2.py        手写 STFT → 频谱 / 拍 / 调性
  lyralign.py      DP 歌词–人声起音对齐
  analysis3.json   分析结果（含词级时间；歌词正文因版权移除）
  kit.py           画布 / 调色板 / 缓动 / 精灵 / 辉光 / 后期
  cloud.py         3D 粒子场：形变、相机、splat、景深
  stage.py         舞台元件：背景 / 立绘 / 光标 / 终端 / 歌词排版
  mus.py           音乐数据的运行时查询层
  film2.py         场景注册 + 逐帧总装（bloom / 暗角 / 颗粒）
  ch_a … ch_j.py   十个章节，每个文件 = 一段镜头的代码
  render2.py       多进程分段渲染 + 拼接 + 混音（断点续跑）
  qc.py            联系表 / 单帧检查
  sunset.py        5 秒日落（独立样例）
  API.md           章节作者契约（引擎签名 + 视觉宪法）
docs/ENGINE.md     引擎拆解：每个数字是怎么来的
docs/SCENES.md     51 个镜头表
docs/SUNSET.md     日落那支的逐块拆解
stills/            联系表与关键帧
preview/           42 秒静音高光
assets/README.md   素材与音乐怎么自备（不随仓库分发）
NOTICE.md          版权与署名（**必读**）
```

---

## 版权

- **代码**：MIT（见 [LICENSE](LICENSE)）。
- **角色素材**：CC BY-NC-SA 4.0，署名链 上善（Pixiv 62155430）→ ZipZipPipe（Pixiv 18604994）→ Small-tailqwq / maid-atelier；galgame 立绘 by JAdpp。**不随仓库分发**。
- **音乐**：Mili《world.execute(me);》，版权归原作者 / 厂牌。本片是**非商业同人作品**，音频与含音轨的成片都不随仓库分发。
- **字体**：Windows 系统字体，不随仓库分发。

完整条款、署名格式、下架联系方式见 **[NOTICE.md](NOTICE.md)**。

---

## English

A music video for Mili's *world.execute(me);*, rendered frame by frame in pure Python —
no editing software, no generative models. 9 800 lines of Python, 51 shots, 62 000 particles per frame,
3840×2160 @ 60 fps. The song's own beat grid (130.00 BPM) and its vocal onsets drive every cut.
Code is MIT; the character art (CC BY-NC-SA 4.0) and the music are **not** redistributed here —
see [NOTICE.md](NOTICE.md).
