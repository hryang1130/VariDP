# DASC7606C 小组项目 · Track 3：Simulation Experiments

在 **≥6 个 ManiSkill 任务**上自采/生成专家演示，训练并评测 **Diffusion Policy（DP）** 基线，
并围绕一个研究问题做受控实验（**数据效率** / **网络结构对比**）。

> **本机环境**：Windows + RTX 4060 Ti 8 GB；PyTorch 2.14.0+cu126；ManiSkill 3.0.1 / SAPIEN 3.0.3（已移植 pinocchio 4.1.0）。
> 训练走 **GPU**，评测只能走 **CPU 单环境仿真** —— 无 CUDA Toolkit → `physx_cuda` 不可用；无 Vulkan → 不出视频。

---

## 目录结构

```
pythonProject1/
├── README.md                          ← 本文件：项目总览 + 导航 + 快速上手
├── dp/                                ← ★ 可复用库（两台机器、所有脚本共用同一份）
│   ├── backbones.py                   ← 三种噪声预测主干：MLP / 官方 UNet / 官方 Transformer(DP-T)
│   ├── dp_lib.py                      ← DP 模型（DDPM 训练 / DDIM 采样）+ 数据集 + 归一化统计
│   └── utils.py                       ← EMA / loss 曲线（两套 train.py 共用，不再是复制粘贴）
├── tools/                             ← ★ 机器无关的工具脚本
│   ├── run_local.py                   ← 一键：训练 → 评测 → 出 SUMMARY（--train-dir 可指向任一套）
│   └── summarize_runs.py              ← 汇总某个 runs/ → SUMMARY_all.md（--runs-dir 指哪扫哪）
├── train_local/                       ← ★ 本机 Windows（RTX 4060 Ti）入口脚本
│   ├── README.md                      ← 参数、产物、排错、实测结果
│   ├── train.py                       ← 单任务训练（默认 --num-workers 0，Windows 安全）
│   ├── eval.py                        ← held-out seeds 评测（physx_cpu 单环境）
│   └── runs/                          ← 本机实验产物（checkpoint / log.csv / 评测 JSON）
├── train/                             ← ★ 学院 GPU（Linux, RTX 4080）入口脚本
│   ├── README.md
│   ├── train.py                       ← 默认 --device cuda / --num-workers 8，产物存 train/runs
│   └── eval.py                        ← 批量评测（--num-envs 并行环境 + --sim-backend gpu/cpu）
├── pyproject.toml                     ← 依赖声明（uv sync 或 pip install -e . 一把装好）
├── 实验三_Track3_完整教程.md           ← 主文档：全流程教程（附录 A = 三种主干结构对比实验）
├── GPU农场_uv安装指南.md               ← 学院 GPU 环境搭建（uv / 原生 venv 两种）
├── DASC7606C_cleaned.txt              ← 课程项目说明（清理后的纯文本，便于检索）
├── DASC7606C_Group_Project_0921.pdf   ← 课程项目说明原文
├── intro.pdf                          ← 课程 / 项目介绍
├── .ref/                              ← 论文与官方源码对照资料 + 结构等价性验证脚本（scratch，可删）
├── wheels/  venv/  .venv/             ← 离线依赖包与虚拟环境
└── .workbuddy/                        ← 工作记录（勿删）
```

## 快速上手

```bash
# 0) 装依赖（二选一，详见 GPU农场_uv安装指南.md）
uv sync                                  # 按 pyproject.toml 建 .venv 并装齐依赖
# 或不装包、直接用现成 venv：脚本会自动把仓库根挂到 sys.path

# 1) 一键：检查数据 → 训练 → 评测 → 出汇总（主干默认 mlp）
python tools/run_local.py --env-id PickCube-v1

# 2) 换主干做结构对比（只换噪声预测网络，其余全不变）
python train_local/train.py --env-id PickCube-v1 --backbone unet
python train_local/eval.py  --ckpt train_local/runs/PickCube-v1_frac1.0_unet_seed0/best.pt -n 50 --seed0 2000

# 3) 数据效率实验（1.0 / 0.5 / 0.25 / 0.1 四档）
python tools/run_local.py --env-id PickCube-v1 --data-efficiency --backbone unet

# 4) 换任务（需先按教程第 4 节生成该任务的数据集）
python train_local/train.py --env-id StackCube-v1 --backbone unet --max-episode-steps 200

# 5) 汇总 runs/ 下所有实验 → train_local/runs/SUMMARY_all.md（写报告收尾用）
python tools/summarize_runs.py --csv
```

**学院 GPU 机器（RTX 4080, Linux）**：用 `train/` 目录代替 `train_local/`（环境搭建见
`GPU农场_uv安装指南.md`）。训练命令一致；评测额外支持并行仿真：

```bash
cd train
python train.py --env-id PickCube-v1 --backbone unet   # 默认 --device cuda
python eval.py  --ckpt runs/PickCube-v1_frac1.0_unet_seed0/best.pt -n 50 --num-envs 32 --sim-backend gpu

# 汇总农场结果（回到仓库根目录）
python tools/summarize_runs.py --runs-dir train/runs --out train/runs/SUMMARY_all.md
```

> 注意：`train/runs` 与 `train_local/runs` 是两套独立目录，别混放；评测 JSON 文件名带
> `_gpuN`/`_cpuN` 后缀，不会覆盖本机 `cpu` 结果。对比实验仍须统一仿真后端（建议 GPU 机器
> 全部用 `--sim-backend gpu` 一组做完，或全部拷回本机用 `cpu` 评）。

前提：数据集已生成到 `~/.maniskill/demos/<Task>-v1/` 下（如
`PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5`）。没有就按教程第 4 节转换/生成。

## 三条铁律（评分表的分主要在这）

| 铁律 | 说明 |
|---|---|
| **held-out 评测** | 训练 seeds 用 `0,1,2...`，评测 seeds 用 `2000~2049`，**绝不重叠** |
| **同一后端** | 所有对比必须在同一仿真后端上做（本机统一 `physx_cpu`），否则不可比 |
| **报 n_ok/n_total** | 每任务 ≥50 episode，写清 `48/50` 而不是只写百分比；失败要分类 |

## 文档导航（3 份，职责不重叠）

| 文档 | 什么时候看 |
|---|---|
| 根 `README.md`（本文件） | 想快速知道项目是什么、文件在哪、怎么跑 |
| `实验三_Track3_完整教程.md` | 要完整走一遍流程（任务选型 → 演示生成 → 训练 → 评测 → 报告）；**附录 A = 网络结构对比实验**。注意：文内代码块是**历史版本**，代码以 `dp/` + `train_local/` 为准 |
| `train_local/README.md` | 本机代码怎么用（参数 / 产物 / 排错 / 实测数据 / 主干结构） |
| `GPU农场_uv安装指南.md` | 要在学院 GPU 机器（RTX 4080, Linux）上用 uv 搭环境 |
| `train/README.md` | 在学院 GPU 机器上训练 / 并行评测的具体命令 |
| `dp/__init__.py` 与各脚本头注释 | 只想看代码接口（库 ↔ 入口脚本的分工） |

## 当前进展（PickCube-v1，state 观测 / `pd_ee_delta_pos`）

全部在 held-out seeds `2000..2049` 上评测（50 episode）：

| 主干 | 参数量 | 演示条数 | 总迭代数 | 成功率 | 成功/总数 | 训练用时 |
|---|---|---|---|---|---|---|
| MLP | 0.353 M | 900 | 82,200（`--epochs 300`） | 96.0% | 48/50 | ~849 s |
| **1D-UNet** | 2.806 M | 900（+100 val） | 29,866 | **100%** | **50/50** | ~684 s |
| Transformer | 8.972 M | — | 待跑 | — | — | — |

> **值得注意**：UNet 只用了约 **1/3 的迭代数**（29.9k vs 82.2k）就拿到 100%，明显超过 MLP 的 96% ——
> 这与文献趋势（难任务上 UNet ≫ MLP）一致，是结构对比实验的一个亮点。三种主干代码均已冒烟验证可训练。
>
> ⚠️ **上表的 1D-UNet 行（2.806 M / 100%）用的是旧版小型 UNet**。2026-09-25 起 `unet` 主干已换成
> **官方 `ConditionalUnet1D` 完整移植**（A1：每级 2 个残差块、2 个 mid 块、ConvTranspose1d 上采样、
> 时间嵌入与条件拼接后 FiLM；无 obs encoder，`global_cond` = 归一化 obs 展平），参数量 **66.418 M**；
> `transformer` 主干也换成 **官方 `TransformerForDiffusion`（DP-T）完整移植**，参数量 **8.972 M**。
> 因此 UNet 行需要重训后才能引用（Transformer 行本就未跑），只有 mlp 那一行的代码与参数量完全没变。
