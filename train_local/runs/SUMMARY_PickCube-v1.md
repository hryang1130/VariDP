# PickCube-v1 本地训练结果

- 设备：auto（RTX 4060 Ti 8GB / CPU 仿真评测）
- epochs：300，batch：256，训练 seed：0
- 评测：50 episode，seeds 2000~2049（held-out）

| 演示条数 | demo_frac | 成功率 | 成功/总数 | 平均步数 | best val loss | 训练用时(s) |
|---|---|---|---|---|---|---|
| 900 | 1.0 | 96.0% | 48/50 | 77.3 | 0.01530 | 849.2 |
