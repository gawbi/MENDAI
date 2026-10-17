"""사진 단위 4지 판정. both를 예측한 오답도 재현율 분모에 포함한다."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .data import ORDER


def predict_classes(probabilities, thresholds=(.5, .5)):
    bits = np.asarray(probabilities) >= np.asarray(thresholds)
    return bits[:, 0].astype(int) + 2 * bits[:, 1].astype(int)


def metrics(y_true, probabilities, thresholds=(.5, .5)):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.shape != (len(y), 2) or not len(y):
        raise ValueError("정답·점수 크기 오류 또는 빈 평가셋")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.isin(y, range(4)).all():
        raise ValueError("정답·점수 범위 오류")
    pred = predict_classes(p, thresholds)
    cm = np.bincount(4 * y + pred, minlength=16).reshape(4, 4)
    counts = cm.sum(1)
    recall = np.divide(np.diag(cm), counts, out=np.full(4, np.nan), where=counts != 0)
    return {"images": len(y), "thresholds": list(map(float, thresholds)), "order": ORDER,
            "confusion_matrix": cm.tolist(), "accuracy": float(np.trace(cm) / cm.sum()),
            "recall": {k: None if np.isnan(v) else float(v) for k, v in zip(ORDER, recall)},
            "balanced_accuracy": float(np.nanmean(recall))}


def choose_thresholds(y, probabilities):
    """사전 고정한 99×99 그리드. 네 클래스 exact-match 재현율을 최적화한다."""
    y, p = np.asarray(y), np.asarray(probabilities)
    # 입력 검증도 공통 지표 함수와 일치시킨다.
    metrics(y, p)
    present = [c for c in range(4) if np.any(y == c)]
    grid = np.arange(1, 100) / 100
    best_key, best = None, None
    heel_bits = p[:, 0, None] >= grid[None, :]
    for t_outsole in grid:
        predictions = heel_bits.astype(int) + 2 * (p[:, 1, None] >= t_outsole).astype(int)
        correct = predictions == y[:, None]
        ba = np.mean([correct[y == c].mean(0) for c in present], axis=0)
        accuracy = correct.mean(0)
        for j, t_heel in enumerate(grid):
            key = (round(float(ba[j]), 12), round(float(accuracy[j]), 12),
                   -round(abs(t_heel-.5) + abs(t_outsole-.5), 12), -float(t_heel), -float(t_outsole))
            if best_key is None or key > best_key:
                best_key, best = key, [float(t_heel), float(t_outsole)]
    return best


def save_predictions(path, rows, probabilities):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "p_heel_lining", "p_outsole"])
        writer.writerows((r["file"], float(p[0]), float(p[1])) for r, p in zip(rows, probabilities))


def report_markdown(results, title="평가 결과"):
    lines = [f"### {title}", "", "| split | 사진 수 | 균형 정확도 | 전체 정확도 | normal 재현율 | heel 재현율 | outsole 재현율 | both 재현율 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def percent(value):
        return "N/A" if value is None else f"{100*value:.2f}%"
    for split, result in results.items():
        values = [result["balanced_accuracy"], result["accuracy"]] + [result["recall"][c] for c in ORDER]
        lines.append(f"| {split} | {result['images']} | " + " | ".join(map(percent, values)) + " |")
    for split, result in results.items():
        lines += ["", f"**{split} 혼동 행렬** — 행: 실제 정답 / 열: 예측", "",
                  "| 실제 / 예측 | normal | heel | outsole | both |", "|---|---:|---:|---:|---:|"]
        for name, row in zip(ORDER, result["confusion_matrix"]):
            lines.append(f"| {name} | " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="최종 가중치의 v5 val·val25만 평가")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    from .train import finalize, log, now, record, write_json
    run_dir = args.run_dir.resolve()
    try:
        finalize(run_dir)
    except Exception as exc:
        log(f"최종 평가 실패: {exc}")
        write_json(run_dir / "evaluation_failure.json", {"time": now(), "error": f"{type(exc).__name__}: {exc}"})
        record(run_dir, "학습 완료 · 최종 평가 중단", f"`{type(exc).__name__}: {exc}`\n\n저장한 last 가중치로 평가를 재개합니다. test는 미평가입니다.")
        raise


if __name__ == "__main__":
    main()
