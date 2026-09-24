"""
run_local.py — 本地一键流程：检查数据 -> 训练 -> 评测 -> 出汇总

它会依次调用 train.py 和 eval.py（子进程），最后把结果写进
runs/<exp>/SUMMARY.md，可直接贴进报告。

用法：
  python run_local.py                                   # 默认 PickCube-v1
  python run_local.py --env-id PickCube-v1 --epochs 300
  python run_local.py --env-id PickCube-v1 --data-efficiency   # 跑数据效率曲线
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import os.path as osp
import sys

import subprocess

HERE = osp.dirname(osp.abspath(__file__))
PY = sys.executable


def run(cmd, cwd=HERE):
    print("\n" + "=" * 78)
    print("[cmd]", " ".join(str(c) for c in cmd))
    print("=" * 78, flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if r.returncode != 0:
        print(f"[warn] 子进程返回码 {r.returncode}（继续）", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description="本地一键训练+评测流程")
    ap.add_argument("--env-id", default="PickCube-v1")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo-frac", type=float, default=1.0)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--eval-seed0", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--data-efficiency", action="store_true",
                    help="跑 1.0/0.5/0.25/0.1 四档数据量，出数据效率曲线")
    ap.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    a = ap.parse_args()

    stages = []

    if a.data_efficiency:
        fracs = [1.0, 0.5, 0.25, 0.1]
    else:
        fracs = [a.demo_frac]

    all_rows = []
    for frac in fracs:
        exp = f"{a.env_id}_frac{frac}_seed{a.seed}"
        out_dir = osp.join(HERE, "runs", exp)

        cmd = [PY, "train.py", "--env-id", a.env_id, "--epochs", a.epochs,
               "--batch", a.batch, "--seed", a.seed, "--demo-frac", frac,
               "--device", a.device]
        if a.dry_run:
            print("[dry-run]", " ".join(cmd)); continue
        run(cmd)

        ckpt = osp.join(out_dir, "best.pt")
        if a.skip_eval or not osp.exists(ckpt):
            continue
        run([PY, "eval.py", "--ckpt", ckpt, "-n", a.eval_episodes,
             "--seed0", a.eval_seed0, "--device", a.device, "--quiet"])

        # 收集结果
        ev = glob.glob(osp.join(out_dir, f"eval_seed{a.eval_seed0}_n{a.eval_episodes}*.json"))
        if ev:
            d = json.load(open(ev[0], encoding="utf-8"))
            s = json.load(open(osp.join(out_dir, "train_summary.json"), encoding="utf-8"))
            all_rows.append(dict(demo_frac=frac, n_train_episodes=s["cfg"]["n_train_episodes"],
                                 success_rate=d["success_rate"], n_success=d["n_success"],
                                 n_episodes=d["n_episodes"],
                                 avg_steps=round(sum(r["steps"] for r in d["results"])
                                                 / len(d["results"]), 1),
                                 best_val=s.get("best_score"),
                                 train_seconds=s.get("seconds")))

    if a.dry_run or not all_rows:
        print("\n没有可汇总的结果。")
        return

    # ---------------- 汇总 ----------------
    lines = [f"# {a.env_id} 本地训练结果", "",
             f"- 设备：{a.device}（RTX 4060 Ti 8GB / CPU 仿真评测）",
             f"- epochs：{a.epochs}，batch：{a.batch}，训练 seed：{a.seed}",
             f"- 评测：{a.eval_episodes} episode，seeds {a.eval_seed0}~{a.eval_seed0+a.eval_episodes-1}（held-out）",
             "",
             "| 演示条数 | demo_frac | 成功率 | 成功/总数 | 平均步数 | best val loss | 训练用时(s) |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(all_rows, key=lambda x: -x["demo_frac"]):
        lines.append(f"| {r['n_train_episodes']} | {r['demo_frac']} | "
                     f"{r['success_rate']*100:.1f}% | {r['n_success']}/{r['n_episodes']} | "
                     f"{r['avg_steps']} | {r['best_val']:.5f} | {r['train_seconds']} |")

    summary = "\n".join(lines) + "\n"
    out_name = ("SUMMARY_data_efficiency.md" if a.data_efficiency
                else f"SUMMARY_{a.env_id}.md")
    path = osp.join(HERE, "runs", out_name)
    os.makedirs(osp.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(summary)
    print("\n" + summary)
    print(f"[done] 汇总已保存 {path}")


if __name__ == "__main__":
    main()
