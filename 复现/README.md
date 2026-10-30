# MultiHop-RAG 多跳问答项目

本项目使用 **MiniCPM4.1-8B + BM25 检索 + 条件核对提示词**完成多跳问答。模型权重保持冻结，未进行微调。数据为 MultiHop-RAG 官方的 609 篇英文文档和 2,556 道问题。

| 评测范围 | 原始基线 | 条件核对方法 |
| --- | ---: | ---: |
| 前 200 题开发子集 | 81.50% | 83.00% |
| 全部 2,556 题 | 76.17% | 77.23% |

83.00% 对应前 200 题开发子集；最终全量 official-script Accuracy 为 77.23%（1,974/2,556）。前 200 题包含在全量数据中，不是独立测试集。

## 项目目录

下面展示文件的放置结构。原复现包中的其他文件也应保留，尤其是 `FILES_SHA256.json` 列出的文件。标为“补入”或“运行生成”的内容，需要由相应的实际结果补齐。

```text
MultiHop-RAG-Project/
├── README.md                              # 项目说明及复现命令
├── reproduce.py                           # 一键重新推理入口
├── run_condition_check_full.py            # 全量实验入口
├── score_official.py                      # 转换预测格式并调用官方评测
├── verify.py                              # 校验原复现包及归档实验
├── FILES_SHA256.json                       # 原复现包文件校验清单
├── export_complete.py                     # 导出权重和环境记录
│
├── configs/
│   ├── expected_run.json                   # 原200题实验配置及预期成绩
│   └── actual_model_files_sha256.json      # 完整导出包中提供（若已生成）
│
├── environment/
│   ├── requirements.txt                   # Python依赖
│   ├── install.sh                         # 依赖安装脚本
│   ├── recorded_environment.json          # 原环境记录
│   ├── server_environment.json            # 完整导出包中提供（若已生成）
│   └── server_python_packages.txt         # 完整导出包中提供（若已生成）
│
├── tools/
│   └── condition_check_experiment/         # 原文件夹整体保留
│       ├── baseline_core.py                # 切块、BM25及生成基础实现
│       ├── fixed_contexts.py               # 恢复原始检索上下文
│       ├── prompt_policy.py                # 条件核对提示词
│       ├── run_condition_check.py          # 模型推理与执行记录
│       ├── evaluate_answers.py             # 本地分析及配对比较
│       ├── run_pilot.py
│       ├── test_experiment.py
│       └── README.md
│
├── data/
│   └── raw/
│       ├── corpus.json                     # 609篇英文文档
│       └── MultiHopRAG.json                 # 2556道问题，保留官方顺序
│
├── models/
│   └── MiniCPM4.1-8B/                       # 下载或恢复后的完整模型文件
│
├── outputs/
│   ├── minicpm_bm25_baseline/               # 原基线目录整体保留
│   │   ├── run_config.json
│   │   ├── execution_records.jsonl          # 恢复相同检索片段所必需
│   │   ├── predictions.json
│   │   ├── summary.json
│   │   ├── hardware.json
│   │   └── 其他原有文件
│   └── minicpm_condition_check_full/        # 补入最终全量结果；复现也输出到此
│       ├── predictions.json                # 最终2556题短答案
│       ├── run_config.json                 # 实际全量运行配置
│       ├── execution_records.jsonl          # 逐题执行记录
│       ├── summary.json
│       ├── hardware.json
│       └── 其他运行生成的文件
│
├── reference_runs/
│   └── condition_check_200/                 # 原200题归档目录整体保留
│
├── evaluation/
│   ├── qa_evaluate.py                       # 固定版本官方评测脚本
│   ├── pilot_200/                           # 原200题官方评分材料
│   ├── full_2556/                           # 全量官方评分后生成或补入
│   │   ├── qa_submission.json
│   │   ├── eval_output.txt
│   │   └── eval_progress.txt
│   └── 其他原有评测文件
│
├── tests/                                  # 原测试目录
└── report/
    ├── GroupXX_project_report.docx
    ├── GroupXX_project_report.pdf
    └── MultiHop_RAG_experiment_report.md
```

## 文件准备

1. 以原 `rag83_repro` 文件夹为基础，改名为 `MultiHop-RAG-Project`，保留内部目录和文件名。
2. 将已完成的全量实验结果补入 `outputs/minicpm_condition_check_full/`。不要用 `reference_runs/condition_check_200/predictions.json` 代替全量预测。
3. 将报告放入 `report/`，将官方全量评分输出放入 `evaluation/full_2556/`。
4. 恢复完整的 MiniCPM4.1-8B 模型，或在运行时通过 `--model-dir` 指定外部模型目录。

当前代码依赖 `outputs/minicpm_bm25_baseline/run_config.json` 和 `execution_records.jsonl` 恢复相同的8个检索片段及其顺序；不能删去这两项后直接运行。基线旧答案不作为新生成的输入。

若模型和大执行记录未直接保存在仓库中，请在下表补入实际可访问的下载地址；下载后恢复到对应位置。

| 材料 | 下载地址 | 恢复位置 |
| --- | --- | --- |
| 实验使用的完整模型文件 | 待填写 | `models/MiniCPM4.1-8B/`，或外部模型目录 |
| 原基线执行记录 | 待填写 | `outputs/minicpm_bm25_baseline/execution_records.jsonl` |
| 最终全量执行记录（若单独提供） | 待填写 | `outputs/minicpm_condition_check_full/execution_records.jsonl` |

模型来源：[openbmb/MiniCPM4.1-8B](https://huggingface.co/openbmb/MiniCPM4.1-8B)。原环境完整导出的模型文件及其哈希用于约束实际复现版本。

## 环境与配置

原实验环境为 Python 3.11.17、PyTorch 2.6.0+cu124、Transformers 4.56.2、Accelerate 1.10.1，使用 NVIDIA A100-PCIE-40GB。完整导出包若包含服务器环境记录，以实际记录为准。

| 配置项 | 设置 |
| --- | --- |
| 文档切块 | 250词，重叠50词，共5480个片段 |
| 检索 | BM25，k1=1.5、b=0.75 |
| 输入证据 | 原始前8个片段，每篇文档最多2个 |
| 模型 | MiniCPM4.1-8B，BF16、SDPA，无量化、无微调 |
| 解码 | 关闭思考，贪心解码，随机种子42 |
| 长度上限 | 输入6000 tokens，输出128 tokens |
| 最终改动 | 固定实际检索上下文，追加条件核对指令 |

## 复现命令

以下命令在 Linux、Python 3.11 环境中运行。先进入项目根目录：

```bash
cd MultiHop-RAG-Project
bash environment/install.sh
```

### 1. 校验归档材料

```bash
python verify.py
```

该命令只检查文件、原始上下文和归档的166/200结果，不调用模型。

### 2. 重新生成全量预测

若已有完整模型目录，使用下面的命令，并把模型路径换成自己的实际路径：

```bash
mkdir -p models
python -u reproduce.py --full --model-dir /实际路径/MiniCPM4.1-8B
```

若模型已放在项目内的 `models/MiniCPM4.1-8B/`，则运行：

```bash
python -u reproduce.py --full
```

答案输出到 `outputs/minicpm_condition_check_full/predictions.json`。如已放入归档的全量结果，应先把该目录改名备份，再开始独立复现；同一输出目录同时只运行一个任务。

### 3. 对新预测运行官方评分

```bash
python score_official.py \
  --predictions outputs/minicpm_condition_check_full/predictions.json \
  --limit 2556
```

评分程序完成题号映射后，调用固定的官方脚本。结果保存到 `evaluation/full_2556/eval_output.txt`。

## 评测说明

- 问题顺序固定，题号从 `q0000` 到 `q2555`。
- 预测只保存 `question_id` 和 `model_answer`，失败题也保留，答案为空字符串。
- 标准答案和参考证据标注只用于评测分析，不用于生成答案。
- 官方指标为小写化后按空白切词的词重合判定，不是严格精确匹配或语义评分。
- 77.23%为原归档全量运行的结果；不同GPU或软件环境可能产生输出差异，新运行应按实际预测重新评分。

## 更新本README后的校验清单

原复现包的 `FILES_SHA256.json` 包含 `README.md` 的哈希。用本文替换原README后，需要在项目根目录执行以下命令，仅更新README这一项；其他代码、数据和归档结果的校验值保持不变：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

path = Path("FILES_SHA256.json")
manifest = json.loads(path.read_text(encoding="utf-8"))
manifest["README.md"] = hashlib.sha256(Path("README.md").read_bytes()).hexdigest()
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
```

## 数据与评测来源

- 数据：[MultiHop-RAG官方数据](https://huggingface.co/datasets/yixuantt/MultiHopRAG/tree/71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82)，作者Yixuan Tang、Yi Yang，许可ODC-BY。
- 数据固定revision：`71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`。
- [固定版本官方评测代码](https://github.com/yixuantt/MultiHop-RAG/blob/c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8/qa_evaluate.py)。
- 评测固定commit：`c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8`。

模型许可见原MiniCPM仓库及完整模型文件中的许可说明。

