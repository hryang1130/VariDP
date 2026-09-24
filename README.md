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
├── 实验三_Track3_完整教程.md           ← 主文档：从任务选型到报告结构的全流程教程
│                                         （附录 A = 三种主干结构对比实验方案）
├── train_local/                       ← ★ 实际代码：DP 训练 / 评测流水线
│   ├── README.md                      ← 代码级说明（参数、产物、排错、实测结果）
│   ├── dp_lib.py                      ← DP 模型（DDPM 训练 / DDIM 采样）+ 数据集
│   ├── backbones.py                   ← 三种噪声预测主干：MLP / 1D-UNet / Transformer
│   ├── train.py                       ← 单任务训练
│   ├── eval.py                        ← held-out seeds 评测
│   ├── run_local.py                   ← 一键流程 + 数据效率实验
│   ├── summarize_runs.py              ← 汇总 runs/ 全部产物 → runs/SUMMARY_all.md
│   └── runs/                          ← 实验产物（checkpoint / log.csv / 评测 JSON）
├── DASC7606C_cleaned.txt              ← 课程项目说明（清理后的纯文本，便于检索）
├── DASC7606C_Group_Project_0921.pdf   ← 课程项目说明原文
├── intro.pdf                          ← 课程 / 项目介绍
├── wheels/  venv/  .venv/             ← 离线依赖包与虚拟环境
└── .workbuddy/                        ← 工作记录（勿删）
```

## 快速上手

```bash
cd train_local

# 1) 一键：检查数据 → 训练 → 评测 → 出汇总（主干默认 mlp）
python run_local.py --env-id PickCube-v1

# 2) 换主干做结构对比（只换噪声预测网络，其余全不变）
python train.py --env-id PickCube-v1 --backbone unet
python eval.py  --ckpt runs\PickCube-v1_frac1.0_unet_seed0\best.pt -n 50 --seed0 2000

# 3) 数据效率实验（1.0 / 0.5 / 0.25 / 0.1 四档）
python run_local.py --env-id PickCube-v1 --data-efficiency

# 4) 换任务（需先按教程第 4 节生成该任务的数据集）
python train.py --env-id StackCube-v1 --backbone unet --max-episode-steps 200

# 5) 汇总 runs/ 下所有实验 → runs/SUMMARY_all.md（写报告收尾用）
python summarize_runs.py --csv
```

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
| `实验三_Track3_完整教程.md` | 要完整走一遍流程（任务选型 → 演示生成 → 训练 → 评测 → 报告）；**附录 A = 网络结构对比实验** |
| `train_local/README.md` | 只关心 `train_local/` 代码怎么用（参数 / 产物 / 排错 / 实测数据） |

## 当前进展（PickCube-v1，state 观测 / `pd_ee_delta_pos`）

全部在 held-out seeds `2000..2049` 上评测（50 episode）：

| 主干 | 参数量 | 演示条数 | 总迭代数 | 成功率 | 成功/总数 | 训练用时 |
|---|---|---|---|---|---|---|
| MLP | 0.353 M | 900 | 82,200（`--epochs 300`） | 96.0% | 48/50 | ~849 s |
| **1D-UNet** | 2.806 M | 900（+100 val） | 29,866 | **100%** | **50/50** | ~684 s |
| Transformer | 3.348 M | — | 待跑 | — | — | — |

> **值得注意**：UNet 只用了约 **1/3 的迭代数**（29.9k vs 82.2k）就拿到 100%，明显超过 MLP 的 96% ——
> 这与文献趋势（难任务上 UNet ≫ MLP）一致，是结构对比实验的一个亮点。三种主干代码均已冒烟验证可训练。
