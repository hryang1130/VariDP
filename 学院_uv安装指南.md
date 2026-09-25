# 学院GPU （RTX 4080）用 uv 搭建本项目环境

> 目标：在学院 GPU 机器（Linux）上跑通本项目的训练与评测（`dp/` 库 + `train/` 入口脚本）。
> 结论先行：**只需要 6 个第三方库**——`torch` / `numpy` / `h5py` / `gymnasium` / `sapien` / `mani_skill`（外加可选的 `matplotlib` 画 loss 曲线）。
> **不需要** diffusers / wandb / tensorboard / Vulkan / CUDA Toolkit / mplib。

---

## 0. 先看懂你那台机器（截图解读）

```
NVIDIA-SMI 580.178.04   Driver Version: 580.178.04   CUDA Version: 13.0
NVIDIA GeForce RTX 4080 ...  16376MiB
```

| 信息 | 含义 | 对安装的影响 |
|---|---|---|
| Driver 580.178.04 / CUDA 13.0 | **驱动**支持的 CUDA 上限是 13.0 | 装任何 `cu12x` 的 PyTorch 轮子都行（cu126 / cu128 都 OK），不需要装 CUDA Toolkit |
| RTX 4080（Ada, sm_89, 16 GB） | 显存 16 GB，比本机 4060 Ti 大一倍 | batch 可以开大；`physx_cuda` 并行仿真**有机会能用**（见 §6） |
| Linux（gpu-4080-402） | 系统是 Linux | 比 Windows 省事：**pinocchio 不用手动移植**（sapien 轮子里自带） |

> 如果集群用 SLURM 管理卡片，先 `srun`/`salloc` 申请到交互节点，确认 `nvidia-smi` 能看到卡再继续。

---

## 1. 装 uv（一次性）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env      # 或重新开一个 shell
uv --version
```

装不上（curl 被墙/无外网）就用 pip 装到用户目录：

```bash
python3 -m pip install --user uv -i https://pypi.tuna.tsinghua.edu.cn/simple
export PATH="$HOME/.local/bin:$PATH"
```

---

## 2. 推荐做法：uv 项目（可复现，全组统一）

**仓库根目录已经带了一份 `pyproject.toml`**（版本对齐本机 Windows 那份），所以把仓库传上去后在仓库根跑一条命令就行：

```bash
cd ~/dasc7606c      # 放仓库的目录，见 §4
uv sync             # 自动装 Python + 建 .venv + 装齐依赖，生成 uv.lock
```

> 它等价于下面这份手写配置；只有想在别的目录另起一份时才需要自己写：

```toml
[project]
name = "dasc7606c-track3"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "torch==2.14.0+cu126",
    "numpy==2.5.3",
    "h5py==3.16.0",
    "gymnasium==1.3.0",
    "sapien==3.0.3",
    "mani_skill==3.0.1",
    "matplotlib",
]

# 让 torch 只从 PyTorch 官方的 cu126 源拿（否则 uv 会去 PyPI 拿默认版）
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
```

> - 若提示 `torch==2.14.0+cu126` 在 Linux 没有对应轮子，把 `==2.14.0+cu126` 去掉、留 `torch`，uv 会自动选最新的 cu126 版；**锁进 `uv.lock` 的才是最终版本**。
> - PyPI 慢的话加镜像：`export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"`（老版本 uv 用 `UV_INDEX_URL`）。torch 的 cu126 源不在镜像里，保持上面那个 URL。
> - Python 版本：3.12 最稳；sapien 3.0.3 官方轮子覆盖 cp310–cp314，所以想和本机完全一致用 `uv python install 3.13` + `requires-python = ">=3.13"` 也行。

---

## 3. 快速做法：uv venv + uv pip（不想维护 pyproject 就用这个）

```bash
cd ~/dasc7606c
uv venv .venv --python 3.12
source .venv/bin/activate

# 顺序很重要：先钉死 torch，再装 mani_skill，避免它拉一个别的 torch 版本
uv pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
uv pip install "mani_skill==3.0.1" "numpy==2.5.3" "h5py==3.16.0" "gymnasium==1.3.0" matplotlib
```

---

## 3.5 uv 太慢：先提速，不行再用原生 venv

> 先说清楚：**uv 默认就是在项目目录下建 `.venv`**，这点和手建 venv 一样。慢通常不是 uv 的问题，而是
> ① torch 那个 ~2.5 GB 的大轮子，② **家目录在 NFS 网络盘上**（集群标配，缓存/解压 I/O 特别慢）。

先判断慢在哪：

```bash
df -hT ~            # 家目录若是 nfs/fuse 类型 → 就是 I/O 拖慢的
uv sync -v          # 卡在 Downloading = 网络；卡在 Installing/Linking = 磁盘 I/O
```

**招 1：把缓存和 venv 挪到本地盘（家目录是 NFS 时收益最大）**

```bash
export UV_CACHE_DIR=/tmp/$USER/uv-cache          # 有 /scratch、/data 就用那些，更持久
export UV_PROJECT_ENVIRONMENT=/tmp/$USER/dasc7606c/.venv
uv sync
```

注意 `/tmp` 重启会清空；venv 挪走后激活路径变成 `/tmp/$USER/dasc7606c/.venv/bin/activate`。

**招 2：PyPI 换国内镜像**

```bash
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
```

**招 3：torch 那个大文件单独处理**

- uv 有缓存，断了重跑 `uv sync` 会续传，不用从头来；
- 或者在带宽好的机器上先把 wheel 下下来，scp 过去离线装：

```bash
pip download torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126 -d wheels/
# 传到集群后：
uv pip install --find-links wheels/ torch==2.14.0+cu126
```

**招 4：看看集群有没有现成模块** `module avail | grep -i torch`。有的话能省掉整个下载，但版本未必和本机对齐，报告里要写清楚实际用的版本。

### 备选：不用 uv，直接在项目目录下原生 venv + pip

完全可以，等价可行：

```bash
cd ~/dasc7606c
python3 --version                 # 最好 >=3.12；太老就用 uv python install 3.12，或找管理员装
python3 -m venv .venv             # ← 这就是你说的「项目目录下的虚拟环境目录」，uv 本来也是建这里
source .venv/bin/activate
python -m pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 顺序同样重要：先钉 torch，再装 mani_skill
pip install torch==2.14.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install mani_skill==3.0.1 numpy==2.5.3 h5py==3.16.0 gymnasium==1.3.0 matplotlib \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 装完立刻固定版本（代替 uv.lock 的作用，全组统一 + 报告可复现）
pip freeze > requirements.txt
```

三点说明：

1. **pip 不会更快**——解析依赖和下载它通常比 uv 还慢。快慢主要取决于网络和磁盘，不是工具。
2. 原生 venv 没有锁文件，就用上面那份 `requirements.txt` 承担这个角色，全组都用它装。
3. 同样**别把 `.venv` 建在 NFS 家目录**，理由同招 1（`python3 -m venv /tmp/$USER/venv` 再 activate 也行）。

---

## 4. 把代码和数据传上去

```bash
# 在 Windows 上（PowerShell），hostname 换成你登录集群用的地址
# 只传代码：dp/ tools/ train_local/ train/ pyproject.toml（venv / wheels / runs 都不用传）
$proj = "D:\Code\python\dpl\demo\pythonProject1"      # ← 改成你的项目根目录（改了名也没关系）
$host = "u3684238@gpu-4080-402"

ssh $host "mkdir -p ~/dasc7606c"
scp -r "$proj\dp"           "$host`:~/dasc7606c/"
scp -r "$proj\tools"        "$host`:~/dasc7606c/"
scp -r "$proj\scripts"      "$host`:~/dasc7606c/"
scp -r "$proj\train_local"  "$host`:~/dasc7606c/"
scp -r "$proj\train"        "$host`:~/dasc7606c/"
scp    "$proj\pyproject.toml" "$host`:~/dasc7606c/"

# 演示数据（PickCube 那份 39 MB，几秒钟）
scp -r "$env:USERPROFILE\.maniskill\demos\PickCube-v1" "$host`:~/.maniskill/demos/"
```

> 以后新增任务数据也一样传到 `~/.maniskill/demos/<Task>-v1/` 下；`dp.dp_lib.find_dataset` 会自动按
> `*.state.pd_ee_delta_pos.physx_cpu.h5` 找到它。

---

## 5. 验证（三步，30 秒）

```bash
cd ~/dasc7606c
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：2.14.0+cu126 True NVIDIA GeForce RTX 4080 ...

uv run python -c "import mani_skill, sapien, gymnasium, h5py; print(mani_skill.__version__, sapien.__version__, gymnasium.__version__, h5py.__version__)"
# 期望：3.0.1 3.0.3 1.3.0 3.16.0

uv run python -c "import dp; print('dp 库可导入')"      # 验证 dp 包（装成 editable 包后任何目录都能 import）

# 真正的冒烟：农场那套入口脚本（产物落在 train/runs/）
uv run python train/train.py --env-id PickCube-v1 --backbone unet --total-iters 3000 --max-episodes 50
```

第三条是真正的冒烟：能开始训练、loss 在降、结束后 `runs/` 里出现 `loss_curve.png`，环境就通了。
（脚本会自动把仓库根挂到 `sys.path`，所以 `dp` 不必 `pip install -e .` 也能 import；但 torch / mani_skill 这些依赖仍要先装好。）

---

## 6. 和本机 Windows 环境的差异（重要的 4 条）

| 项 | 本机 Windows | GPU 农场 Linux | 说明 |
|---|---|---|---|
| pinocchio | 手动移植（conda-forge 拷包） | **什么都不用做** | sapien 的 Linux 轮子自带；`replay_trajectory` / `pd_ee_delta_pos` 直接能用 |
| Vulkan / 渲染 | 没有 → 不出视频 | 也**不需要装** | 本项目训练+数值评测不渲染；`train.py` 画图已强制 `Agg` 后端 |
| `physx_cuda` 并行仿真 | 不可用（缺 CUDA Toolkit） | **可能可用** | Linux 上 sapien 自带 PhysX CUDA 库，通常只要驱动。收益：评测可开 `-n 64` 并行，50 ep 从 ~1 分钟降到几秒 |
| 评测后端一致性 | 全部 `physx_cpu` | 混用会毁掉公平性 | **同一组实验必须同一后端**。农场用 `train/eval.py --num-envs 32 --sim-backend gpu`（本机 `train_local/eval.py` 固定 `physx_cpu` 单环境），且**所有对比都要重跑成同一后端** |

> 结论：训练随便用 GPU 农场（这是它最大的价值）；**评测后端要么全 cpu，要么全 gpu，别混**。
> 稳妥做法：农场训练，评测仍用 `physx_cpu`（单环境串行也就几十秒），和已有结果直接可比。

---

## 7. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `uv: command not found` | 安装后 PATH 没刷新 | `source ~/.local/bin/env` 或重开 shell |
| 下载 torch 超时 / 反复 retry | 到 `download.pytorch.org` 网络差 | 重试；或先在本机下好 wheel 再 `uv pip install <whl文件>` |
| `import sapien` 报 libvulkan 缺失 | 少系统库（少见，不渲染一般碰不到） | `sudo apt install libvulkan1`；无 sudo 用 `mamba install -c conda-forge vulkan` 或找管理员 |
| `ModuleNotFoundError: mani_skill` | 没在 `.venv` 里跑 | 用 `uv run python ...`（推荐，自动带环境），或先 `source .venv/bin/activate` |
| `没找到 xxx 的 state 数据集` | 数据没传 / 路径不对 | 确认 `~/.maniskill/demos/<Task>-v1/**/*.state.pd_ee_delta_pos.physx_cpu.h5` 存在 |
| 家目录配额爆了 | torch+CUDA 轮子 ~7 GB | `df -h ~` 看配额；`uv cache clean` 清缓存；或把 `.venv` 建在大盘上（`UV_PROJECT_ENVIRONMENT=/path/.venv uv sync`） |
| 改了项目目录名后 `activate` / pinocchio 报错 | venv 里存了创建时的绝对路径 | 跑 `python tools/fix_venv_paths.py --apply` 一键修（脚本在仓库里，只用标准库） |
| 训练比本机还慢 | 被分到共享节点 / CPU 核被限 | `nvidia-smi` 确认卡空闲；数据加载用 `--num-workers`（如支持） |

---

## 8. 版本对齐表（本机实测可用的一套，直接照抄）

| 包 | 版本 | 备注 |
|---|---|---|
| Python | 3.13.14（本机）/ **3.12（农场推荐）** | sapien 3.0.3 轮子覆盖 cp310–cp314 |
| torch | 2.14.0+cu126 | cu126 在 4080 上跑满，驱动 580 兼容 |
| numpy | 2.5.3 | |
| h5py | 3.16.0 | |
| gymnasium | 1.3.0 | |
| sapien | 3.0.3 | |
| mani_skill | 3.0.1 | |
| matplotlib | 任意最新 | 仅画 loss 曲线，已用 Agg 后端 |

> 报告里「computing resources」一节可直接写：训练在 RTX 4080 (16GB, driver 580.178.04) 上完成，
> 本机 RTX 4060 Ti (8GB) 做冒烟；软件版本如上表，环境用 uv 锁定（`uv.lock`）可复现。
