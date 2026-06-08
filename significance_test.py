"""
显著性检验脚本：对比两个 eval 文件夹里的 per-dataset pass@1 结果。

用法：
    python3 significance_test.py \
        --dir_a /mnt/workspace/wxc/resum/logs/eval/20260430_121425 \
        --dir_b /mnt/workspace/wxc/resum/logs/eval/20260505_101532 \
        --metric math_pass@1:1_samples

原理：
    - 从 JSON 里读取 mean 和 stderr
    - stderr = std / sqrt(n)，所以 n = (std / stderr)^2
    - 对于 pass@1（0/1 分布），std = sqrt(p*(1-p))，所以 n = p*(1-p) / stderr^2
    - 用 two-proportion z-test 做显著性检验
    - 同时输出 Cohen's h 作为 effect size
"""

import argparse
import json
import math
import os
from pathlib import Path


def find_result_jsons(eval_dir: str) -> dict[str, str]:
    """
    在 eval_dir 下递归找所有 results_*.json，
    按 dataset 名（从 results key 里提取）分组，取最新的一个。
    返回 {dataset_name: json_path}
    """
    result_files = sorted(Path(eval_dir).rglob("results_*.json"))
    dataset_to_path = {}
    for path in result_files:
        try:
            with open(path) as f:
                data = json.load(f)
            for key in data.get("results", {}):
                if key == "all":
                    continue
                # key 格式: "custom|aime24|0" -> dataset = "aime24"
                parts = key.split("|")
                dataset = parts[1] if len(parts) >= 2 else key
                # 取最新的（sorted 已按文件名时间戳排序，后面的更新）
                dataset_to_path[dataset] = (str(path), data)
        except Exception as e:
            print(f"  [warn] failed to read {path}: {e}")
    return dataset_to_path


def extract_metric(data: dict, dataset: str, metric: str) -> tuple[float, float] | None:
    """从 results 里提取指定 dataset 和 metric 的 (mean, stderr)"""
    results = data.get("results", {})
    for key, values in results.items():
        if key == "all":
            continue
        parts = key.split("|")
        ds = parts[1] if len(parts) >= 2 else key
        if ds == dataset:
            mean = values.get(metric)
            stderr = values.get(f"{metric}_stderr")
            if mean is not None and stderr is not None:
                return float(mean), float(stderr)
    return None


def infer_n(mean: float, stderr: float) -> int:
    """
    从 pass@1 的 mean 和 stderr 反推样本量 n。
    对于 Bernoulli 分布：stderr = sqrt(p*(1-p)/n)
    所以 n = p*(1-p) / stderr^2
    """
    if stderr <= 0 or mean <= 0 or mean >= 1:
        return 0
    n = mean * (1 - mean) / (stderr ** 2)
    return max(1, round(n))


def two_proportion_z_test(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """
    Two-proportion z-test（双侧）。
    返回 (z_stat, p_value)
    """
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    # 用正态分布近似计算双侧 p-value
    # P(|Z| > |z|) = 2 * (1 - Phi(|z|))
    # 用 math.erfc 实现
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return z, p_value


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_a", required=True, help="第一个 eval 文件夹（model A）")
    parser.add_argument("--dir_b", required=True, help="第二个 eval 文件夹（model B）")
    parser.add_argument("--metric", default="math_pass@1:1_samples", help="要对比的指标")
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"显著性检验: {args.metric}")
    print(f"  Model A: {args.dir_a}")
    print(f"  Model B: {args.dir_b}")
    print(f"  alpha = {args.alpha}")
    print(f"{'='*70}\n")

    datasets_a = find_result_jsons(args.dir_a)
    datasets_b = find_result_jsons(args.dir_b)

    common_datasets = sorted(set(datasets_a.keys()) & set(datasets_b.keys()))
    if not common_datasets:
        print("❌ 两个文件夹没有共同的 dataset，请检查路径。")
        return

    header = f"{'Dataset':<20} {'A (mean±se)':<18} {'B (mean±se)':<18} {'Δ(B-A)':<10} {'z':<8} {'p-value':<12} {'h':<8} {'sig?'}"
    print(header)
    print("-" * len(header))

    for dataset in common_datasets:
        _, data_a = datasets_a[dataset]
        _, data_b = datasets_b[dataset]

        result_a = extract_metric(data_a, dataset, args.metric)
        result_b = extract_metric(data_b, dataset, args.metric)

        if result_a is None or result_b is None:
            print(f"{dataset:<20} {'N/A':<18} {'N/A':<18} {'—':<10} {'—':<8} {'—':<12} {'—':<8} —")
            continue

        mean_a, se_a = result_a
        mean_b, se_b = result_b
        n_a = infer_n(mean_a, se_a)
        n_b = infer_n(mean_b, se_b)

        z, p_val = two_proportion_z_test(mean_b, n_b, mean_a, n_a)
        h = cohens_h(mean_b, mean_a) if mean_a > 0 and mean_b > 0 else float("nan")
        delta = mean_b - mean_a
        sig = "✅" if (not math.isnan(p_val) and p_val < args.alpha) else "—"

        print(
            f"{dataset:<20} "
            f"{mean_a:.3f}±{se_a:.3f}    "
            f"{mean_b:.3f}±{se_b:.3f}    "
            f"{delta:+.3f}     "
            f"{z:+.2f}    "
            f"{p_val:.4f}      "
            f"{h:+.3f}   "
            f"{sig}"
        )

    print(f"\n注：z-test 基于 two-proportion z-test（双侧），n 由 stderr 反推。")
    print(f"    Cohen's h: |h|<0.2 小效应, 0.2-0.5 中等, >0.5 大效应")


if __name__ == "__main__":
    main()