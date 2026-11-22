# 素材与音乐：怎么自备

这个仓库**只放代码**。角色立绘、背景、音乐、字体都不随仓库分发，原因见 [../NOTICE.md](../NOTICE.md)。

程序对「缺素材」是**容错**的：找不到立绘时粒子场退化成纯几何形状，片子照样渲染得出，
只是「她」不在了。想把原片复原，按下面准备。

---

## 1. 目录约定

默认素材根目录 = 仓库同级的 `角色/`，也可以用环境变量指到任何地方：

```bash
export MV_ASSETS="/path/to/角色"     # 素材根
export MV_AUDIO="/path/to/Mili - world.execute(me);.m4a"
export MV_FONTS="/path/to/fonts"     # 可选，默认 C:\Windows\Fonts
export MV_FFMPEG="/usr/bin/ffmpeg"   # 可选，默认找仓库内 .tools/ 再找 PATH
```

`MV_ASSETS` 指向的目录需要长这样：

```
角色/
  maid-atelier/
    01-左女仆-提裙礼.webp
    02-右女仆-记事本与笔.webp
    03-右女仆-单片镜V4FlashVision版.webp
    04-启动页-左女仆-马甲长裙.webp
    05-启动页-右女仆.webp
    06-Q版侧栏-晕倒冒泡.webp
    背景/
      深海宫殿-亮.webp
      深海宫殿-暗.webp
  galgame/
    pet-dafeiyu.webp
    whale-cheerful.webp … whale-starry.webp        （8 张表情差分）
    claude-zipzippipe.webp  gpt-zipzippipe.webp
    gemini-dual-prism-translator-v4.webp
    kimi-lunar-scroll-navigator-v5.webp
    grok-zipzippipe.webp
    背景/
      bg-deepseek-seaside-study.webp … bg-grok-electronics-studio.webp
```

文件名不对会读不到——`engine/stage.py` 第 14–33 行的常量就是这张表。

---

## 2. 立绘从哪来

- **maid-atelier 女仆立绘**：上善（Pixiv 62155430）→ ZipZipPipe（Pixiv 18604994）→ Small-tailqwq / maid-atelier，
  **CC BY-NC-SA 4.0**。去原作者的发布页下载，并遵守署名 / 非商业 / 相同方式共享。
- **galgame 立绘**：by JAdpp，去作者发布页获取。
- **背景**：随上述素材包提供；作者的再分发许可不明确，所以本仓库不放。

> 注意：CC BY-NC-SA 4.0 里 **NC 同样约束成片**。你要是拿这套代码做商单，请换掉角色素材——代码是 MIT，可以商用。

---

## 3. 音乐

**Mili《world.execute(me);》**（专辑 *Miracle Milk*）。买一份、或从你合法拥有的副本里导出，
用 `MV_AUDIO` 指过去即可。仓库里不提供、也不要提交到 issue 里。

`audio2.py` 会自己解码（调 `ffmpeg`），不需要额外库。

---

## 4. 歌词

`engine/lyrics.lrc` 是 `[mm:ss.xx]English|中文` 格式，**只用于字幕同步**。
`analysis3.json` 里只保留时间戳，正文靠这个文件合并回来（`mus._lrc()`）。
不想要歌词就把这个文件删掉，程序会跑一支「无字幕版」。

---

## 5. 完全不要素材也能跑

```bash
cd engine
python sunset.py --video      # 5 秒日落，无任何素材依赖
python qc.py 61.2 --scale 1 --cols 1 --w 960 --ch ch_d --out shot.png   # 看粒子球
```
