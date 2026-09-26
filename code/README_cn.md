# MP1 代码 — 安装与使用

作业要求、评分、截止日期与同行评审请参阅[项目指南](../GUIDE_zh.md)。本 README 仅包含运行说明与技术规则。代码包内仅包含这两份文档。

下列所有命令均在 **code/** 目录下执行。数据与分词器已包含在内；无需 API 密钥、预训练权重或额外下载数据集；安装依赖后，训练与评估均可离线完成。

## 1. 安装

请使用 **Python 3.12**。在解压后的代码包目录下：

```bash
cd code
python -m venv .venv
source .venv/bin/activate
```

在 Windows PowerShell 中，请改用 `.venv\Scripts\Activate.ps1` 激活环境。

根据所选设备安装 PyTorch（**任选其一**）：

```bash
# Linux/Windows CPU（推荐），无需显卡
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

若使用装有兼容驱动的 NVIDIA GPU，请改用以下命令：

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

在 macOS 上，请从默认 PyPI 源安装 `torch==2.7.1` 并在 CPU 上运行。安装 PyTorch 后，安装其余依赖并检查模型：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Linux CPU 命令已在 Python 3.12 与 PyTorch 2.7.1+cpu 下验证。Windows/macOS 上的耗时尚未测量。

## 2. 训练与评估

**快速安装检查** — 训练 10 步后进行完整测试集评估：

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/checkpoint.pt --split test
```

该命令用于验证流程是否正常运行；其分数**并非**完整基线分数。每次训练运行都需要一个新的输出目录。

**完整基线** — 1,200 次训练更新后进行评估：

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

基线模型包含 4 个 GPT 模块，宽度为 128，4 个注意力头，参数总数为 **1,088,256**，可达到约 **2.10 测试 BPB**。在参考的四线程 Xeon Platinum 8457C 上，测得训练耗时约 **311 秒**，评分耗时 **5.92 秒**，不含安装与加载时间。以上为参考测量，并非笔记本上的保证值或固定时间上限。

**你自己的模型** — 编辑 `student.py` 及相关文件后，运行：

```bash
python train.py --implementation student --seed 17 --eval-every 300 --run-dir runs/my-model
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split validation
# 在测试前冻结最终方法：
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split test
```

训练会写入 `checkpoint.pt` 与 `metrics.json`。评估会写入 `test_cpu_fp32.json`（或对应的设备/划分名称）以及每个窗口的损失。请提交完整测试 JSON 中的 **bpb** 值，而非词元困惑度或验证集 BPB。默认评估使用 FP32。添加 `--device cuda` 可在 GPU 上运行；训练可以使用 BF16，但排名评估必须使用 FP32，并在 CPU 上保持可复现。提供的 CUDA 运行器将 PyTorch 显存上限设为 20 GB，驱动开销另计。

## 3. 文件与模型接口

| 文件 | 用途 |
|---|---|
| `model.py`、`configs/baseline.json` | 可运行的基线模型；请保留用于对比。 |
| `student.py`、`train.py` | 你的模型工厂与训练配方；可按需添加辅助代码。 |
| `common.py`、`evaluate.py` | 固定的数据校验、窗口划分与评分器；请勿修改。 |
| `data/` | 已提供的划分、分词器与数据集哈希；请勿修改。 |
| `tests/test_contract.py` | 检查模型因果性、归一化、独立性与梯度。 |
| `RUN_LOG_TEMPLATE.csv` | 可选的实验记录模板。 |
| `PACKAGE_MANIFEST.json` | 发布哈希；路径相对于包含 code/ 与 guide/ 的代码包根目录。 |

- `build_model(config)` 返回一个 `context=256` 的 PyTorch 模型。
- 训练器调用 `forward(ids)` 获取未归一化 logits；评分器调用 `predict_log_probs(ids)` 获取有限、归一化的自然对数概率。两者的输出形状均为 `[batch, time, 2048]`。
- 位置 t 处的预测只能使用截至 t 的已观测前缀。在独立的窗口、样本与评分过程之间必须重置临时状态。训练阶段产生的紧凑型资源可在窗口间复用，但评估阶段的上下文状态不可复用。
- 检查点会记录实现模块与配置。请同时提供该模块及全部所需资源，以便评分器重建所提交的预测器。直接评估无需保存优化器状态。
- 在指南约束范围内，训练长度、架构、正则化、自训练权重平均与集成方法均可调整。请记录所有随机种子、已处理的训练目标数、检查点血缘关系与搜索成本；复用已有检查点并不会抹消其原本的训练成本。不强制使用特定种子或必须取得特定分数提升。

## 4. 基准测试与资源测量

**固定评分。** 协议 `7506-mp1-wt2-v2`：WikiText-2 原始文本，基于训练集拟合的 BPE-2048，独立的 256 目标窗口，包含末尾的短窗口。每个划分中除首词元外的所有目标均被评分一次。输入窗口共享边界词元，但不携带状态。BPB 为下一个词元负对数（以 2 为底）概率之和除以该划分全部原始 UTF-8 字节长度（含首词元的字节）。

| 划分 | 被评分目标数 | UTF-8 字节数 |
|---|---:|---:|
| 验证集 | 376,599 | 1,148,007 |
| 测试集 | 428,405 | 1,292,013 |

所有开发与检查点/混合权重选择均应使用验证集。权重、统计量与检索条目只能来源于训练文本。公开的测试文本用于复现，不得用于调参。方法冻结后，同一预测器可重复评估以测量耗时或进行复现。词元困惑度与已发表的词级困惑度不可直接比较。

请针对同一冻结预测器测量以下三项限制：

- **CPU 时间 ≤ 基线的 5 倍：**
- **峰值内存 ≤ 4 GiB：**
- **推理资源 ≤ 64 MiB（未压缩）：**

## 5. 准备提交与复现同伴

[项目指南](../GUIDE_zh.md) 中说明了截止日期与网站提交流程。请在你的不可变代码仓库中包含以下内容：

- **报告**，**至多 10 页**（含图表与参考文献）
- **复现说明**

最终的网站提交必须链接到本代码与匹配的完整检查点包。网站会自动生成 Issue JSON。请保证所有推理资源均可下载以供核查。

要核查同伴的工作，请获取其精确的代码版本与检查点，按照其安装说明操作，并使用提供的评分器运行其冻结模型：

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

请将复现得到的 BPB 与报告分数进行比对。提交 **Peer Review Report** 时必须填写复现分数；可选地附带命令、环境、差异与证据/日志链接。差异由教师裁定。在为期 7 天的公开评审期内，经确认的差异将依据已公布的评分政策获得加分。

## 6. 数据来源声明

WikiText-2 由 Stephen Merity、Caiming Xiong、James Bradbury 与 Richard Socher 在 [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843) 中提出。文本来源于 Wikipedia 贡献者。[原始数据集](https://huggingface.co/datasets/Salesforce/wikitext) 标注为 [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) 与 [GNU 自由文档许可证](https://www.gnu.org/licenses/fdl-1.3.html)；在再分发数据时请保留上述声明。

所提供的 `wikitext-2-raw-v1` 划分保留了修订版本 `b08601e04326c79dfdd32d625aee71d232d685c3`。各行以换行符连接，并以 UTF-8 编码；分词器仅基于训练文本进行拟合。数据集哈希见 `data/manifest.json`。上述数据声明并不对周围课程代码施加新的许可。