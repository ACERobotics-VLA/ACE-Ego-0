# 02. GitHub Pages Fix And Generation Notes

本文档记录 `ACERobotics-VLA/ACE-Ego` 仓库如何生成 GitHub Pages，以及本次从 404 修复到可访问页面的具体改动。

## 1. 当前页面地址

GitHub Pages 地址：

```text
https://acerobotics-vla.github.io/ACE-Ego/
```

当前默认仓库：

```text
https://github.com/ACERobotics-VLA/ACE-Ego
```

本地当前使用的 Git 配置：

```text
origin   -> https://github.com/ACERobotics-VLA/ACE-Ego.git
```

后续只推送 `origin`。

## 2. 页面是如何生成的

该项目主页是纯静态站点：

```text
index.html
styles.css
script.js
assets/
```

没有使用 React、Vite、Node build、Jekyll 或后端服务。GitHub Pages workflow 直接把仓库根目录作为静态站点 artifact 上传并部署。

部署 workflow 位于：

```text
.github/workflows/pages.yml
```

关键步骤：

```yaml
- name: Checkout
  uses: actions/checkout@v4

- name: Setup Pages
  uses: actions/configure-pages@v5
  with:
    enablement: true

- name: Upload artifact
  uses: actions/upload-pages-artifact@v3
  with:
    path: "."

- name: Deploy to GitHub Pages
  id: deployment
  uses: actions/deploy-pages@v4
```

含义：

- `actions/checkout@v4`：拉取 `main` 分支代码。
- `actions/configure-pages@v5`：配置 GitHub Pages 环境。
- `enablement: true`：如果 Pages site 没有被 workflow 正确识别或启用，让 action 自动启用/校准 Pages。
- `actions/upload-pages-artifact@v3`：把仓库根目录上传为 Pages artifact。
- `actions/deploy-pages@v4`：把 artifact 发布到 `github-pages` environment。

## 3. 本次 404 的原因

最初访问：

```text
https://acerobotics-vla.github.io/ACE-Ego/
```

返回：

```text
HTTP 404
Site not found · GitHub Pages
```

仓库中已经有 `index.html` 和 Pages workflow，但 GitHub Actions 最近几次运行失败。

通过 GitHub Actions API 查看最近 workflow：

```text
084a31d completed failure
da73008 completed failure
f2a34d3 completed failure
```

失败 job 是：

```text
deploy -> Setup Pages
```

日志中的核心错误：

```text
Get Pages site failed. Please verify that the repository has Pages enabled and configured to build using GitHub Actions, or consider exploring the `enablement` parameter for this action. Error: Not Found
```

结论：仓库 Pages 配置和 workflow 部署模式没有被 `actions/configure-pages@v5` 正确识别，导致 artifact 上传和 deploy 步骤被跳过，最终公开页面仍然是 404。

## 4. 本次修复

修复 commit：

```text
b032c31 Enable Pages setup in workflow
```

改动文件：

```text
.github/workflows/pages.yml
```

具体改动：

```diff
 - name: Setup Pages
   uses: actions/configure-pages@v5
+  with:
+    enablement: true
```

推送后，新的 workflow 成功运行：

```text
27552436548 b032c31 completed success
```

随后访问 Pages URL 返回：

```text
HTTP_STATUS:200
FINAL_URL:https://acerobotics-vla.github.io/ACE-Ego/
```

页面 HTML 中确认包含：

```text
ACE-Ego-0
```

## 5. 后续更新页面的流程

由于当前目录的 `.git` 在 Codex 沙箱里不可直接作为普通 Git 仓库使用，后续仍使用临时 bare Git 目录：

```text
/tmp/ace_ego_page.git
```

配合 work tree：

```text
/data/lh/projects/ace_ego_page
```

检查状态：

```bash
git --git-dir=/tmp/ace_ego_page.git \
  --work-tree=/data/lh/projects/ace_ego_page \
  status --short --branch
```

只暂存需要发布的文件，例如：

```bash
git --git-dir=/tmp/ace_ego_page.git \
  --work-tree=/data/lh/projects/ace_ego_page \
  add index.html styles.css script.js assets docs .github
```

提交：

```bash
git --git-dir=/tmp/ace_ego_page.git \
  --work-tree=/data/lh/projects/ace_ego_page \
  commit -m "Update project page"
```

开启代理并推送：

```bash
source /etc/profile.d/clash.sh
proxy_on

GIT_TERMINAL_PROMPT=0 git -c http.version=HTTP/1.1 \
  --git-dir=/tmp/ace_ego_page.git \
  --work-tree=/data/lh/projects/ace_ego_page \
  push
```

push 到 `main` 后，`.github/workflows/pages.yml` 会自动触发部署。

## 6. 验证命令

检查 workflow 最近运行状态：

```bash
curl -sS \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/ACERobotics-VLA/ACE-Ego/actions/workflows/pages.yml/runs?per_page=1"
```

检查页面是否可访问：

```bash
curl -sS -L -o /tmp/ace_site.html \
  -w 'HTTP_STATUS:%{http_code}\nFINAL_URL:%{url_effective}\n' \
  https://acerobotics-vla.github.io/ACE-Ego/
```

检查页面内容是否包含当前项目名：

```bash
rg "ACE-Ego-0" /tmp/ace_site.html
```

预期结果：

```text
HTTP_STATUS:200
FINAL_URL:https://acerobotics-vla.github.io/ACE-Ego/
```

## 7. 注意事项

- GitHub Pages URL 仍是 `https://acerobotics-vla.github.io/ACE-Ego/`，这是仓库名决定的路径；网页展示名可以是 `ACE-Ego-0`。
- 不要把 GitHub token 写入仓库、文档或聊天记录。
- 当前仓库中原始视频和论文源文件体积较大，推送前要确认 `.gitignore` 和 staged 文件，避免误提交大文件。
- 如果 Pages 再次 404，优先检查 Actions 页面中 `Deploy static site to Pages` workflow 的最新 run 是否成功。
