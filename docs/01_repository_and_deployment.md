# 01. ACE-Ego Project Page Repository And Deployment Notes

本文档记录 `ace_ego_page` 仓库当前状态、文件组织、GitHub Pages 部署方式，以及后续更新和推送的推荐流程。

## 1. 仓库用途

该仓库用于发布 ACE-Ego 论文项目主页：

**ACE-Ego: Unifying Egocentric Human and Robotic Data for VLA Pretraining**

线上地址：

```text
https://1223haohao.github.io/ace_ego_page/
```

GitHub 仓库：

```text
https://github.com/1223haohao/ace_ego_page
```

当前主页是纯静态站点，不依赖 Node、Vite、React 或后端服务。GitHub Pages 会直接部署仓库根目录中的静态文件。

## 2. 主要文件结构

```text
.
├── index.html
├── styles.css
├── script.js
├── README.md
├── .gitignore
├── .github/workflows/pages.yml
├── assets/
│   ├── ACE_Logo.png
│   ├── figures/
│   ├── posters/
│   └── videos/
└── docs/
    └── 01_repository_and_deployment.md
```

关键文件说明：

- `index.html`：主页主体内容，包括标题、作者、摘要、方法、视频、结果表格和 citation。
- `styles.css`：页面排版和响应式样式。
- `script.js`：移动端导航和 BibTeX 复制按钮逻辑。
- `.github/workflows/pages.yml`：GitHub Pages 静态站点部署 workflow。
- `assets/ACE_Logo.png`：主页 hero 中使用的 ACE Robotics logo。
- `assets/figures/`：从 paper 工程复制来的 PNG 图，包括 teaser、method、data pipeline、dataset statics 和实验结果图。
- `assets/posters/`：真实机器人视频的封面帧。
- `assets/videos/`：网页专用压缩版 MP4 视频。

## 3. 原始素材来源

论文工程路径：

```text
/data/lh/projects/paper
```

真实机器人原始视频路径：

```text
/data/lh/projects/ace_ego_page/整理视频
```

当前网页使用的是压缩后的 MP4，不直接使用原始视频。原因是原始视频较大，其中 `整理鞋盒.mp4` 超过 GitHub 单文件 100MB 限制。

压缩后的网页视频放在：

```text
assets/videos/
```

对应关系：

```text
倒咖啡豆.mp4      -> assets/videos/scoop-coffee.mp4
分装饮料零食.mp4  -> assets/videos/category-sorting.mp4
收纳饮料.mp4      -> assets/videos/arrange-drinks.mp4
整理碗.mp4        -> assets/videos/stack-bowls.mp4
整理积木.mp4      -> assets/videos/sweep-cubes.mp4
整理鞋盒.mp4      -> assets/videos/pack-shoes.mp4
```

## 4. 被忽略的大文件

`.gitignore` 中忽略了：

```text
整理视频/
整理视频.zip
```

目的：

- 避免提交原始大视频。
- 避免 GitHub 100MB 单文件限制。
- 保持 GitHub Pages 加载速度。

当前工作区还出现过未跟踪文件：

```text
_CoRL2026_ACE_Ego_0.zip
```

该文件不是主页运行必需文件，除非明确需要发布或备份，否则不建议提交到 GitHub Pages 仓库。

## 5. Git 状态说明

当前目录中的 `.git` 在 Codex 沙箱环境里表现为只读占位目录，不能作为普通 Git 仓库使用。因此本次操作使用了临时 bare Git 目录：

```text
/tmp/ace_ego_page.git
```

配合工作区路径：

```text
/data/lh/projects/ace_ego_page
```

常用命令形式：

```bash
git --git-dir=/tmp/ace_ego_page.git --work-tree=/data/lh/projects/ace_ego_page status --short
```

历史提交：

```text
f11681d Create project homepage
247ee93 Add GitHub Pages workflow
2c314c5 Redesign homepage for ACE-Ego paper
```

## 6. GitHub 认证

机器上没有 `gh` CLI，也没有可用 SSH key。推送使用 HTTPS token credential。

曾使用的 credential 配置方式：

```bash
git config --global credential.helper store
git credential approve
```

然后在终端输入：

```text
protocol=https
host=github.com
username=1223haohao
password=YOUR_TOKEN_HERE

```

注意：

- 不要把 GitHub token 写入聊天、文档或代码。
- `credential.helper store` 会把 token 存到当前用户的本地 credential 文件中，使用方便但安全性一般。
- fine-grained token 至少需要对 `1223haohao/ace_ego_page` 有 `Contents: Read and write` 权限。

## 7. 推送方式

由于当前环境访问 GitHub HTTPS 偶尔会出现 TLS 中断，推荐推送时显式使用 `HTTP/1.1`：

```bash
GIT_TERMINAL_PROMPT=0 git -c http.version=HTTP/1.1 \
  --git-dir=/tmp/ace_ego_page.git \
  --work-tree=/data/lh/projects/ace_ego_page \
  push
```

查看远端分支：

```bash
GIT_TERMINAL_PROMPT=0 git -c http.version=HTTP/1.1 \
  --git-dir=/tmp/ace_ego_page.git \
  --work-tree=/data/lh/projects/ace_ego_page \
  ls-remote --heads origin main
```

如果是首次设置远端：

```bash
git --git-dir=/tmp/ace_ego_page.git \
  --work-tree=/data/lh/projects/ace_ego_page \
  remote add origin https://github.com/1223haohao/ace_ego_page.git
```

## 8. 更新页面的推荐流程

修改页面内容后，按以下流程提交：

```bash
git --git-dir=/tmp/ace_ego_page.git \
  --work-tree=/data/lh/projects/ace_ego_page \
  status --short
```

暂存文件：

```bash
git --git-dir=/tmp/ace_ego_page.git \
  --work-tree=/data/lh/projects/ace_ego_page \
  add index.html styles.css script.js README.md docs assets .github .gitignore
```

提交：

```bash
git --git-dir=/tmp/ace_ego_page.git \
  --work-tree=/data/lh/projects/ace_ego_page \
  commit -m "Update project homepage"
```

推送：

```bash
GIT_TERMINAL_PROMPT=0 git -c http.version=HTTP/1.1 \
  --git-dir=/tmp/ace_ego_page.git \
  --work-tree=/data/lh/projects/ace_ego_page \
  push
```

## 9. GitHub Pages 部署方式

仓库通过 GitHub Actions 部署静态 HTML：

```text
.github/workflows/pages.yml
```

workflow 名称：

```text
Deploy static site to Pages
```

GitHub Pages 设置中应使用：

```text
Source: GitHub Actions
```

每次 push 到 `main` 后，GitHub Actions 会自动重新部署。可在 GitHub 仓库顶部的 `Actions` 页面查看部署状态。

成功后访问：

```text
https://1223haohao.github.io/ace_ego_page/
```

如果刚推送后仍显示旧页面或 404，通常是 Pages 构建或 CDN 缓存延迟。等待 1-5 分钟后强制刷新即可。

## 10. 本地检查命令

检查 JS 语法：

```bash
node --check script.js
```

检查关键 PNG 图是否存在：

```bash
test -f assets/figures/teaser.png
test -f assets/figures/method.png
test -f assets/figures/data-pipeline.png
```

检查 HTML 中引用的 assets 是否存在：

```bash
rg -o 'assets/[^" ]+' index.html | sort -u | while IFS= read -r ref; do
  test -f "$ref" || echo "missing $ref"
done
```

查看网页资产大小：

```bash
find assets -type f -printf '%p\t%k KB\n' | sort
```

## 11. 视频压缩命令参考

如需从原始视频重新生成网页版视频，可参考：

```bash
ffmpeg -y -i "整理视频/原始视频.mp4" \
  -vf "scale='min(1280,iw)':-2,fps=20" \
  -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 31 \
  -movflags +faststart -an \
  "assets/videos/output.mp4"
```

生成 poster：

```bash
ffmpeg -y -ss 1 -i "整理视频/原始视频.mp4" \
  -frames:v 1 -vf "scale=1280:-2" -q:v 4 \
  "assets/posters/output.jpg"
```

## 12. 后续待完善事项

- 将 `index.html` 中 Paper、Code、Data 按钮替换成真实链接。
- 如果论文正式上传 arXiv，更新 BibTeX 中的 `journal` 或 `eprint` 字段。
- 当前网页先使用 paper 工程导出的 PNG 图；如果后续安装 PDF 转 SVG 工具，可把正式 PDF 图转成 SVG 后替换对应 PNG。
- 如果需要更接近 F1-VLA 风格，可继续补充更多 benchmark video、method figure 和 appendix link。
