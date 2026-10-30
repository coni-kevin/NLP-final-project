# 本轮只改：回答前逐项核对问题条件

研究问题：模型是否把部分条件成立误当作整个问题成立，或忽略对象、来源、时间和否定条件？

与原版基线相比，仅在原提示词中追加核对要求：逐项核对条件；对于要求多个条件同时成立的判断题，全部有支持才答 Yes，必要条件被证据反驳则答 No，无法确定必要条件则拒答。事件时间不能被文章发布日期替代，不用其他来源、实体或背景知识补齐缺失关系。内部核对后仍只输出原格式的短答案。

不要求展示推理，不要求 JSON，不更换解析器，不修改模型权重或解码参数。完整英文改动位于 `prompt_policy.py`。

## 运行

把 `condition_check_experiment.zip` 上传到 `/root/autodl-tmp/multihop_rag/`，执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rag
cd /root/autodl-tmp/multihop_rag
python -m zipfile -e condition_check_experiment.zip tools/condition_check_experiment
python -u tools/condition_check_experiment/run_pilot.py
```

不下载新模型、不安装新依赖。沿用 `models/MiniCPM4.1-8B/` 和 rag 环境。

必须保留原版 `outputs/minicpm_bm25_baseline/` 中的 `run_config.json` 和 `execution_records.jsonl`。

脚本直接读取原版日志的前 200 题，验证片段来自原 corpus，复用实际送入过模型的 8 个片段及原有顺序，只取问题和片段，不把旧答案输入模型。核对数据、原代码、模型配置与上下文来源后才继续。

加载 MiniCPM 前，先用已有分词器检查新提示词加原片段是否都在 6000-token 上限内；如果有任何题超限，会停止，不通过删片段来继续。运行结束会逐题检查实际上下文完全一致。

模型仍为 BF16、SDPA、关闭思考、贪心解码、输出上限 128 tokens；原版切块、每篇上限和答案提取均保留。

## 结果

只运行相同前 200 题，与原版 **163/200，81.50%** 比较，不自动跑全部 2556 题。

看到 `ALL DONE` 后，下载项目目录中的 **`condition_check_pilot_results.zip`**。包含预测、上下文、原始输出、配置、长度检查、分类得分、改对/改错题目及耗时。

输出目录：`outputs/minicpm_condition_check_pilot/`。重复相同命令可以恢复中断；修改代码或设置需新输出目录；失败记录不会自动重新回答。没有自动训练或关机。

可选后台命令：

```bash
nohup python -u tools/condition_check_experiment/run_pilot.py > outputs/condition_check_pilot.log 2>&1 < /dev/null &
tail -n 20 outputs/condition_check_pilot.log
```

## 检查与解释范围

代码保留未改的原版基线源码用于验证。已检查提示词追加、原答案解析和上下文打包函数保持一致、旧答案不进入模型输入、超长时停止、缺失答案计错，以及完整真实 200 题的上下文复用。CPU 检查不能代替 GPU 实测，本轮真实效果仍需在 AutoDL 运行确认。

这 200 题已用于开发，不是独立留出测试集；提示词可能改善部分错误，也可能增加拒答，需同时检查改对和改错。分数为本地复现课程规则，正式成绩仍需老师固定脚本复核。
