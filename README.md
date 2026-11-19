# 浮标日最高水温更正的分块重算效率研究

《海洋数据挖掘技术》课程实验。实验以补块重算为切入点，比较原始水温观测更正后更新日最高水温的重算范围和实际耗时。

[查看 AI 协作记录](AI协作记录.md)，含逐轮对话和智能体最终回复。

## 实验代码

四个脚本只使用 Python 标准库，复现环境为 Python 3.12。将它们放在同一目录，从该目录运行。

| 脚本 | 实验内容 |
|---|---|
| [recompute_daily_max.py](recompute_daily_max.py) | 全日、单级分块和两级汇总的重算与核对 |
| [benchmark_recompute.py](benchmark_recompute.py) | 四种重算方案的基础计时 |
| [benchmark_distribution.py](benchmark_distribution.py) | 更正分布与数据规模对照 |
| [benchmark_pruning.py](benchmark_pruning.py) | 按需更新、最大值计数与正确性验证 |

```bash
python -B recompute_daily_max.py
python -B benchmark_recompute.py
python -B benchmark_distribution.py --rounds 15 --pairs-per-round 128 --warmup-pairs 32
python -B benchmark_pruning.py --rounds 15 --pairs 128 --random-batches 1000
```

脚本使用合成水温数据并输出 JSON。耗时随机器和运行环境变化。
