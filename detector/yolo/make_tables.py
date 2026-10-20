"""Build the baseline + follow-up experiment tables from saved metrics.json files.

    python detector/yolo/make_tables.py --runs-root <runs> --out results.md

Rows whose metrics are missing are listed as not completed; numbers are never filled in by hand.
"""

import argparse
import json
from pathlib import Path

import eval_photo_level as ev

SETS = [("v5_val", "v5 val"), ("val25", "val25")]
BASELINES = [("B11", "yolo11m_1024_s0", "YOLO11m · 50 · 1024"), ("B26", "yolo26m_1024_s0", "YOLO26m · 50 · 1024")]
SELECTED = "yolo26m_1024_s0"
EXPERIMENTS = [
    ("A1", "yolo26m_1024_s0_mosaic0", "Mosaic 끔 (mosaic 1.0 → 0)", "재학습 50 · 1024"),
    ("A2", "yolo26m_1024_s0_scale025", "scale 증강 약하게 (0.5 → 0.25)", "재학습 50 · 1024"),
    ("A3", "yolo26m_1024_s0_lr5e-4", "초기 학습률 낮춤 (lr0 0.001 → 0.0005)", "재학습 50 · 1024"),
    ("T1", "yolo26m_1024_s0_T1conf", "conf 튜닝 (클래스별)", "재학습 없음"),
    ("T2", "yolo26m_1024_s0_T2sahi", "SAHI 타일 추론 (512 · 겹침 0.2 + 전체 사진)", "재학습 없음"),
]


def load(runs: Path, name: str, s: str):
    p = runs / name / "photo_eval" / s / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return None if m.get("limit") else m


def delta(x, b) -> str:
    if x is None or b is None:
        return "N/A"
    d = 100 * (x - b)
    return f"{d:+.1f}%p"


def row(cells) -> str:
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = Path(args.runs_root).resolve() / "yolo"

    L = ["# YOLO 실험 결과", "",
         "사진 단위 판정. 균형 정확도 = (Recall normal + Recall heel + Recall outsole) / 3. "
         "both 정답 0장 → Recall both N/A. 행 = 실제 정답, 열 = 모델 예측. 모두 검증셋(v5 val, val25)이며 test는 채점하지 않았습니다.", ""]

    L += ["## 1. 기본 학습 비교 (conf 0.25)", ""]
    hdr = ["평가셋", "실험", "모델", "균형 정확도", "전체 정확도", "Recall normal", "Recall heel", "Recall outsole", "Recall both"]
    L += [row(hdr), row(["---"] * 3 + ["--:"] * 6)]
    for s, sl in SETS:
        for eid, name, desc in BASELINES:
            m = load(runs, name, s)
            if m is None:
                L.append(row([sl, eid, desc] + ["미완료"] * 6))
                continue
            r = m["recall"]
            L.append(row([sl, eid, desc, f"**{ev.pct(m['balanced_acc'])}**", ev.pct(m["overall_acc"]),
                          ev.pct(r["normal"]), ev.pct(r["heel"]), ev.pct(r["outsole"]), ev.pct(r["both"])]))
    L.append("")

    L += ["## 2. 추가 실험 (선정 모델 B26 = YOLO26m 기준)", "",
          "괄호 안은 B26 기본선 대비 차이(%p)입니다.", ""]
    hdr = ["평가셋", "실험", "바꾼 것", "학습", "conf", "균형 정확도", "전체 정확도", "Recall normal", "Recall heel",
           "Recall outsole", "Recall both"]
    L += [row(hdr), row(["---"] * 5 + ["--:"] * 6)]
    for s, sl in SETS:
        b = load(runs, SELECTED, s)
        br = b["recall"] if b else {}
        L.append(row([sl, "B26", "기본선", "50 · 1024", str(b["conf"]) if b else "-",
                      f"**{ev.pct(b['balanced_acc'])}**" if b else "미완료", ev.pct(b["overall_acc"]) if b else "-",
                      ev.pct(br.get("normal")), ev.pct(br.get("heel")), ev.pct(br.get("outsole")), "N/A"]))
        for eid, name, what, how in EXPERIMENTS:
            m = load(runs, name, s)
            if m is None:
                L.append(row([sl, eid, what, how, "-", "**미완료**"] + ["-"] * 5))
                continue
            r = m["recall"]
            cell = lambda k: f"{ev.pct(r[k])} ({delta(r[k], br.get(k))})"
            L.append(row([sl, eid, what, how, str(m["conf"]),
                          f"**{ev.pct(m['balanced_acc'])}** ({delta(m['balanced_acc'], b and b['balanced_acc'])})",
                          f"{ev.pct(m['overall_acc'])} ({delta(m['overall_acc'], b and b['overall_acc'])})",
                          cell("normal"), cell("heel"), cell("outsole"), ev.pct(r["both"])]))
    L.append("")

    L += ["## 3. 혼동 행렬", ""]
    for eid, name, desc in [(e, n, d) for e, n, d in BASELINES] + [(e, n, w) for e, n, w, _ in EXPERIMENTS]:
        for s, sl in SETS:
            m = load(runs, name, s)
            L += [f"### {eid} · {sl} — {desc}" + (f" (conf {m['conf']}, N={m['n']})" if m else ""), ""]
            L += [ev.cm_markdown(m) if m else "**미완료** — 결과 없음", ""]

    Path(args.out).write_text("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
