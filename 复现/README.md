# MultiHop-RAG 多跳问答项目

本项目使用 **MiniCPM4.1-8B + BM25 + 条件核对提示词**构建RAG系统，未进行模型微调。完整benchmark包含609篇英文文档和2556道问题。

| 评测范围 | 基线准确率 | 条件核对准确率 |
| --- | ---: | ---: |
| 前200题开发子集 | 81.50% | 83.00% |
| 全量2556题 | 76.17% | 77.23% |

成绩为固定官方脚本的Accuracy。前200题用于开发，包含在全量评测中，不是独立测试集。

## 1. GitHub仓库目录

按照任务书第5节，**GitHub存放源码、配置、依赖和README**。模型权重、benchmark数据、大执行记录、报告及结果文件均不放进此仓库。

```text
MultiHop-RAG-Project/
├── README.md                           # 项目说明、资源准备和运行命令
├── .gitignore                          # 排除模型、数据及运行产物
├── .gitattributes                      # 保持文本文件为LF换行
├── run_condition_check_full.py         # 全部2556题的预测入口
├── score_official.py                   # 格式转换及官方评分入口
├── verify.py                           # 原工具模块，评分入口导入ROOT和sha256
│
├── configs/
│   ├── final_run.json                  # 最终全量实验的实际配置记录
│   └── expected_run.json               # 原模型、数据哈希及200题配置记录
│
├── environment/
│   ├── requirements.txt                # Python依赖
│   ├── install.sh                      # 安装PyTorch及其他依赖
│   └── recorded_environment.json       # 原实验环境记录
│
├── tools/
│   └── condition_check_experiment/
│       ├── baseline_core.py            # 切块、BM25和基础生成实现
│       ├── fixed_contexts.py           # 恢复原基线实际检索片段
│       ├── prompt_policy.py            # 条件核对提示词
│       ├── run_condition_check.py      # 批量推理和执行记录
│       ├── evaluate_answers.py         # 本地评分与分析
│       └── test_experiment.py          # 已有一致性测试
│
└── evaluation/
    └── qa_evaluate.py                  # 固定版本官方评测代码
```

`configs/final_run.json`是归档配置记录；本版核心参数固定在源码中，不通过编辑此JSON改变运行行为。

## 2. 课程提交包

报告、预测、评分输出和执行记录按任务书单独提交，不混入源码仓库：

```text
GroupXX_NLP_RAG.zip
├── GroupXX_project_report.docx
├── GroupXX_project_report.pdf
├── predictions.json                    # 全部2556题最终短答案
├── scores/
│   └── eval_output.txt                 # 官方全量和各题型评分输出
└── execution_records/
    ├── final/                          # 最终全量运行的实际记录
    │   ├── execution_records.jsonl
    │   ├── run_config.json
    │   ├── summary.json
    │   ├── hardware.json
    │   ├── fixed_context_check.json
    │   └── 其他实际记录文件
    └── baseline/                       # 恢复固定检索上下文所需的原基线记录
        ├── execution_records.jsonl
        └── run_config.json
```

报告填写实际组号、全体成员信息和分工，以及可访问的GitHub完整仓库地址。上述提交包中不放源码、模型或benchmark。

## 3. 复现前准备

### 3.1 环境

在Linux、已安装conda的终端中执行：

```bash
conda create -n multihop-rag python=3.11 -y
conda activate multihop-rag
cd MultiHop-RAG-Project
bash environment/install.sh
```

原实测：Python 3.11.17、PyTorch 2.6.0+cu124、Transformers 4.56.2、Accelerate 1.10.1，GPU为A100-PCIE-40GB。生成需要CUDA GPU；CPU可用于上下文检查和评分。安装依赖需要网络。

### 3.2 模型与benchmark（仅放本地，不提交GitHub）

| 资源 | 获取方式 | 本地放置位置 |
| --- | --- | --- |
| MiniCPM4.1-8B完整模型 | [官方模型](https://huggingface.co/openbmb/MiniCPM4.1-8B)，或实验原完整导出包 | `models/MiniCPM4.1-8B/` |
| 官方609篇文档 | [固定版本benchmark](https://huggingface.co/datasets/yixuantt/MultiHopRAG/tree/71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82)中的`corpus.json` | `data/raw/corpus.json` |
| 官方2556道问题 | 同一固定版本中的`MultiHopRAG.json` | `data/raw/MultiHopRAG.json` |

```bash
mkdir -p models data/raw outputs/minicpm_bm25_baseline
```

将实际下载的文件放入上述位置。模型目录须包含完整权重、分词器、配置、自定义Python代码及相关许可文件，不能只放权重分片。优先使用原完整导出包中的模型，确保实际版本一致；原模型代码和配置哈希记录在`configs/expected_run.json`。原始数据保持字节和顺序不变，不编辑、重排。

### 3.3 恢复基线检索记录（来自课程提交包）

当前条件核对实验固定使用原基线的实际检索片段及顺序。因此，老师取得课程提交包并解压后，执行下面的命令，将路径换成实际解压位置：

```bash
cp /实际解压路径/execution_records/baseline/run_config.json \
  outputs/minicpm_bm25_baseline/run_config.json
cp /实际解压路径/execution_records/baseline/execution_records.jsonl \
  outputs/minicpm_bm25_baseline/execution_records.jsonl
```

这两个文件不上传GitHub，但运行时必须存在。程序只使用记录中的问题与真实文档片段，旧模型答案不进入新生成输入；标准答案和参考证据标注仅用于评测分析。

## 4. 预测与评分命令

以下命令均在仓库根目录执行。此精简源码仓库使用已有全量入口，不使用旧大复现包的`reproduce.py`或`python verify.py`入口；无需恢复200题归档或旧`FILES_SHA256.json`。

### 4.1 检查全部题目的固定检索上下文

```bash
python tools/condition_check_experiment/run_condition_check.py \
  --project-root . --limit 0 --dry-run
```

应显示609篇文档、5480个片段、2556道问题，以及`CPU retrieval check passed`。该步骤不调用模型，仅打印前三题示例；分词后长度检查由正式推理中的preflight完成。

### 4.2 生成全量预测

新复现时保持`outputs/minicpm_condition_check_full/`不存在；若已有结果，先改名备份。保留`outputs/minicpm_bm25_baseline/`，同一输出目录同时只运行一个任务。

```bash
python -u run_condition_check_full.py --project-root .
```

生成文件包括：

- `outputs/minicpm_condition_check_full/predictions.json`：最终短答案。
- 同目录的`execution_records.jsonl`、`run_config.json`、`summary.json`、`hardware.json`等：实际执行记录。
- `condition_check_full_results.zip`：本次全量运行结果压缩包。

完成后，`summary.json`中的`processed`、`successful`应为2556，`failed`为0，`complete`和`full_benchmark`为true；`fixed_context_check.json`中题数为2556且`all_original_contexts_fit`为true。

### 4.3 对新预测运行官方评分

```bash
python score_official.py \
  --predictions outputs/minicpm_condition_check_full/predictions.json \
  --limit 2556
```

官方输出保存到`evaluation/full_2556/eval_output.txt`。使用其中`Overall Metrics`下的`Accuracy`，归档结果为0.7723。生成脚本打印的本地分析结果不能替代官方评分。

新生成结果按实际预测评分，不保证跨GPU或软件环境逐字一致。这里的复现是重新调用模型生成答案，不是只对已有预测再次评分。

## 5. 方法与来源

- 模型：冻结MiniCPM4.1-8B，8,185,253,888个参数，BF16、SDPA，无量化。
- 切块：250词，重叠50词；BM25使用k1=1.5、b=0.75。
- 输入：原基线实际8个片段，每篇最多2个，顺序保持不变。
- 生成：贪心解码、关闭思考、种子42，输入上限6000 tokens、输出上限128 tokens。
- 最终改动：只追加条件核对指令，保持模型、实际上下文和答案提取方式不变。
- [MultiHop-RAG官方仓库](https://github.com/yixuantt/MultiHop-RAG)，作者Yixuan Tang、Yi Yang；数据许可ODC-BY。
- 数据固定revision：`71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`。
- [官方评测固定代码](https://github.com/yixuantt/MultiHop-RAG/blob/c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8/qa_evaluate.py)，commit为`c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8`。
- 模型许可与使用说明见原MiniCPM模型仓库。

官方Accuracy按小写化、空白切词后的词重合计算，不等同于严格精确匹配或语义正确率。系统保存最终短答案，不拼接检索全文或候选答案。
