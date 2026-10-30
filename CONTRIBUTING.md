# 内容维护约定

1. `docs/roadmap.md` 保存稳定路线，不记录每日零散问题。
2. 每日学习结果写入 `notes/weekXX/`，不覆盖历史答案。
3. 一篇论文建立一个独立 Markdown 文件。
4. 每次实验必须记录模型、框架、代码、硬件和 workload 版本。
5. 未知值写 `null` 或“未知”，不要写 `0`。
6. 修改前后使用 `git diff` 检查。
7. 每完成一天提交一次，例如：

```bash
git add .
git commit -m "Complete Week 01 Day 01 tokenizer study"
git push
```
