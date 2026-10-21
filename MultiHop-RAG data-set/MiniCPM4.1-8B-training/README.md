# MiniCPM RAG 完整包复现说明

## 1. 需要哪些文件

必须提供：`rag83_complete_with_weights.tar`，约 16GB，包含模型权重、数据、代码、配置和归档实验记录。

建议同时提供：

- `condition_check_full_results.zip`：本次全量预测与执行记录，供核对结果。
- `RAG_experiment_results.md`：实验改动和成绩汇总。
- 本说明文件。

注意：约 24MB 的 `rag83_repro.zip` 不包含模型权重，单独拿到这个小包仍需另外准备原始模型。

## 2. 复现目标与环境

这套方案使用 **MiniCPM4.1-8B + BM25 检索的 8 个片段 + 条件核对提示词**，没有微调模型权重。

| 评测范围 | 原实验成绩 |
| --- | --- |
| 官方原顺序前 200 题 | 166/200，83.00% |
| 全部 2,556 题 | 1,974/2,556，77.23% |

报告中的全量成绩应写 **77.23%**。83% 仅是前 200 题的成绩。

原运行环境：AutoDL、Linux、A100 40GB、Python 3.11.17、PyTorch 2.6.0+cu124、Transformers 4.56.2、Accelerate 1.10.1。优先使用相同环境。原全量运行峰值显存约 18.12GiB，运行约 51 分钟；其他机器的耗时和输出可能不同。

需要预留大包及解压后文件的空间，并留出运行余量。首次安装依赖需要联网；完整大包本身已经包含模型权重，无需重新下载模型。导出时的实际环境信息位于包内 `environment/server_environment.json` 和 `environment/server_python_packages.txt`。

以下命令在 **AutoDL 的 Linux 终端**执行。

## 3. 上传并解压完整大包

将 `rag83_complete_with_weights.tar` 上传到服务器。在大包所在目录执行：

```bash
tar -xf rag83_complete_with_weights.tar
cd rag83_repro
```

建议解压到新目录，避免与旧复现结果混用。

## 4. 创建环境并安装依赖

新服务器执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -n rag83 python=3.11 -y
conda activate rag83
bash environment/install.sh
```

如果 conda 安装在其他位置，将第一行改为实际的 `conda.sh` 路径。

原服务器已有配置正确的 `rag` 环境时，可以直接激活原环境，无需重新安装依赖：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rag
```

后续命令都在解压出来的 `rag83_repro` 目录、同一个 Python 环境中执行。

## 5. 检查包内文件

```bash
python verify.py
```

出现 `VERIFIED` 表示数据、代码、原始上下文和归档的前 200 题结果通过检查。**这一步只是检查已有材料，没有重新运行模型。**

后续运行 `reproduce.py` 时还会核对完整包中模型文件的哈希与核心依赖版本。首次校验大模型文件需要读取全部权重，可能耗时一段时间。

## 6. 重新生成全部 2,556 题答案

```bash
nohup python -u reproduce.py --full > reproduce_full.log 2>&1 < /dev/null &
```

这会真正调用模型生成答案，使用固定的原始 BM25 片段和条件核对提示词。

查看进度：

```bash
tail -n 20 reproduce_full.log
```

日志出现 `ALL DONE` 表示全量运行及本地结果打包完成。如果出现 `Traceback`，应先检查报错，不能将终端不再输出当作成功。

后台运行期间，关闭本地电脑或断开终端不会停止这个 `nohup` 任务，但服务器实例需要保持运行。

全量结果位于：

```text
outputs/minicpm_condition_check_full/
  predictions.json
  execution_records.jsonl
  run_config.json
  summary.json
  scores_local.json
```

项目目录中还会生成 `condition_check_full_results.zip`。

## 7. 使用固定官方脚本评分

确认全量生成成功后，执行：

```bash
python score_official.py --predictions outputs/minicpm_condition_check_full/predictions.json --limit 2556
```

该命令会检查题号、转换输入格式，并调用包内未修改的固定官方评测脚本。

原实验的 `Overall Metrics` 中，Accuracy 显示为 **0.7723**，对应 **77.23%**。固定评测代码版本为 `c1c1287aa60a94acf9c4d20c891c9cd611a0f6e8`。

官方评测输出保存在：

```text
evaluation/full_2556/eval_output.txt
```

这里的指标是 **official-script Accuracy**，不是严格 Exact Match。不同 GPU、驱动或软件版本可能造成少量输出差异，不保证所有机器逐字一致。

## 8. 可选：先复现前 200 题

如果只想先验证流程，可以先执行：

```bash
python -u reproduce.py
```

它会重新生成前 200 题，结果保存在 `outputs/reproduced_condition_check_200/`，原实验目标为 **166/200，83.00%**。

再运行官方评分：

```bash
python score_official.py --predictions outputs/reproduced_condition_check_200/predictions.json --limit 200
```

前 200 题和全量运行使用不同输出目录，可以先跑前 200 题再跑全量。

如果前 200 题的新输出目录已经存在，复现入口会拒绝把旧结果当成一次新实验。重新独立运行前，请先将旧目录改名备份。全量入口可能恢复已有运行；需要独立复现时，使用新解压的目录。

## 9. 完成后保存什么

- 完整模型大包：保留模型、代码和运行配置。
- 新生成的 `predictions.json`：全量最终短答案。
- 新生成的 `execution_records.jsonl`：检索上下文、模型输入输出及执行记录。
- `run_config.json` 和 `summary.json`：配置与完成状态。
- 官方评测输出：记录本次实际准确率。

归档成绩与重新运行的成绩应分别标注。不能只对包内已有预测重新评分，就声称已经重新运行了模型。
