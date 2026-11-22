# 怎么发布到 GitHub

这个目录已经是**可以直接 push 的仓库**：README / LICENSE / NOTICE / .gitignore 都在该在的位置，
`角色/`、音乐、字体这些不能公开的东西**没有**被包含进来（16 MB，43 个文件）。

下面两条路，选一条。

---

## 路 A — 这台机器没有 git，用脚本直接传（推荐）

只需要一个 **Personal Access Token**：

1. 打开 <https://github.com/settings/tokens> → *Generate new token (classic)*
2. 勾 **repo** 权限（如果要建在组织下，再勾 `read:org`）
3. 复制 `ghp_...`，然后在本目录执行：

```powershell
cd D:\AI_SS\Ying_Shi\publish\mv-first-deepseek
$env:GITHUB_TOKEN = "ghp_你的token"
python tools\upload_to_github.py
```

脚本会：建仓库 → 传 43 个文件 → 提交 → 推 `main` → 写好 description 和 topics。
产物地址会打印在最后，形如 `https://github.com/你的名字/first-mv-made-with-deepseek`。

先看清楚要传什么，可以只试跑：

```powershell
python tools\upload_to_github.py --dry
```

换仓库名 / 改成私有：

```powershell
python tools\upload_to_github.py --repo my-mv --private
```

**token 用完记得去 GitHub 撤销。** 脚本不写任何配置文件，token 只从环境变量读。

---

## 路 B — 装了 git 的话（任何机器）

先在这台机器装 Git for Windows（<https://git-scm.com/download/win>），然后：

```bash
cd publish/mv-first-deepseek
git init -b main
git config user.name "ASAW"
git config user.email "acom202302@163.com"
git add .
git commit -m "用 DeepSeek 做的第一个 MV — world.execute(me);"
git remote add origin https://github.com/<你的用户名>/first-mv-made-with-deepseek.git
git push -u origin main
```

仓库要先在网页上建好（**不要**勾 Add a README，否则要先 pull）。

---

## 建好之后手动补三件小事

1. **仓库 About（右上角齿轮）**
   - Description：`用 DeepSeek 做的第一个 MV — world.execute(me);`
   - Website：你的 B 站 / YouTube 链接（如果有）
   - Topics：`mv` `python` `numpy` `opencv` `procedural-graphics` `particle-system` `music-visualizer`
2. **Release**（可选）：把成片压成分卷或传网盘，Release 里放链接。
   **不要**把带 Mili 音轨的 mp4 直接传到仓库——见 [NOTICE.md](NOTICE.md)。
3. **核对署名**：`NOTICE.md` 第 2 节的署名链必须和原作者发布页一致。

---

## 如果你想顺带发成片

成片带 Mili 的音乐，**不适合放进代码仓库**。常见做法：

- 传 B 站 / YouTube，标题写清楚是**非商业同人作品**，简介里贴 `NOTICE.md` 的署名段；
- 仓库里只放静音高光（`preview/highlight_silent.mp4`，已经有了）和静帧；
- 想给别人复现，就写「自备音乐副本 + `MV_AUDIO`」，README 里已经这么写了。

---

## 目录里哪些东西是故意没有的

| 没有 | 为什么 |
|---|---|
| `角色/`（6 张立绘 + 8 张表情 + 6 张背景） | CC BY-NC-SA 4.0，别人该去原作者发布页拿 |
| `Mili - world.execute(me);.m4a` | 音乐版权，不能随代码分发 |
| `world.execute(me)_4K60.mp4` / `_1080p60.mp4` | 含音轨 |
| `*.ttf` | Windows 系统字体，不许再分发 |
| `assets.py` | 从本机插件包里抠素材的脚本，只对这台机器有意义 |
