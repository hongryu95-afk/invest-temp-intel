#!/usr/bin/env python3
"""미국 지수 5일 방향 예측 → intel.json / history.csv.

GitHub Actions에서 매일 22:00 UTC(07:00 KST)에 돈다. 투자온도 웹앱이 이 저장소의
intel.json을 직접 읽는다. **앱이 읽는 파일을 쓰는 곳은 여기 하나뿐이다** —
PC(WSL us-stock) 쪽에서는 intel.json을 만들지 않는다.

배경: live 99일 실측에서 종목 픽·등급·시장 게이트는 근거가 없었고, 기준선을 넘긴
건 지수 방향 예측 하나였다. 그래서 그것만 떼어 왔다.
상세 = myClaude/toss_app/invest-temp/진짜성적표_2026-09-19.md
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from index_predictor import IndexPredictor

ROOT = Path(__file__).resolve().parent
INTEL = ROOT / "intel.json"
HISTORY = ROOT / "history.csv"
SCORECARD = ROOT / "scorecard.json"
FIELDS = ["date", "asof", "spy_direction", "spy_probability", "spy_predicted_return",
          "qqq_direction", "qqq_probability", "qqq_predicted_return", "model_accuracy"]


def data_asof(predictor):
    """실제로 쓴 가격 데이터의 마지막 날짜. 실행일이 아니라 이게 신선도다.

    ponytail: 가격을 두 번 받는다(예측 내부 1회 + 여기 1회, 각 2초 남짓).
    index_predictor가 asof를 같이 반환하게 고치면 한 번으로 줄지만,
    그러려면 us-stock 원본까지 건드려야 해서 지금은 두 번 받는다.
    """
    try:
        px = predictor._fetch_price_data(start_date="2026-01-01")
        if px is not None and not px.empty:
            return str(px.index[-1])[:10]
    except Exception as e:  # noqa: BLE001 — 신선도 확인 실패가 예측을 막진 않는다
        print(f"[warn] asof 확인 실패: {e}", file=sys.stderr)
    return None


def append_history(row):
    """하루 한 줄. 같은 날짜가 이미 있으면 안 쓴다(재실행해도 안전)."""
    seen = set()
    if HISTORY.exists():
        with HISTORY.open(encoding="utf-8", newline="") as f:
            seen = {r["date"] for r in csv.DictReader(f)}
    if row["date"] in seen:
        print(f"[skip] history에 {row['date']} 이미 있음")
        return
    new = not HISTORY.exists()
    with HISTORY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[ok] history += {row['date']}")


def main():
    p = IndexPredictor(data_dir=str(ROOT))
    result = p.predict_next_week()

    # 값을 못 구했으면 **아무것도 발행하지 않는다**. 옛 값을 새 값처럼 내보내는 게
    # 이 프로젝트에서 제일 비싸게 배운 실패였다(NaN → "전량 매도").
    if not result or result.get("error") or not result.get("predictions", {}).get("spy"):
        print(f"[fail] 예측 실패, 발행 안 함: {result and result.get('error')}", file=sys.stderr)
        sys.exit(1)

    preds = result["predictions"]
    asof = data_asof(p)

    append_history({
        "date": result["date"], "asof": asof,
        **{f"{k}_{f}": (preds.get(k) or {}).get(src)
           for k in ("spy", "qqq")
           for f, src in (("direction", "direction"),
                          ("probability", "probability_up"),
                          ("predicted_return", "predicted_return"))},
        "model_accuracy": (preds.get("spy") or {}).get("model_accuracy"),
    })

    scorecard = None
    if SCORECARD.exists():
        try:
            scorecard = json.loads(SCORECARD.read_text(encoding="utf-8"))
        except ValueError as e:
            print(f"[warn] scorecard.json 파싱 실패: {e}", file=sys.stderr)

    intel = {
        "source": "invest-temp-intel (GitHub Actions)",
        "date": result["date"],
        "asof": asof,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "index_prediction": {"date": result["date"], "predictions": preds},
        "scorecard": scorecard,
    }
    # NaN 리터럴은 JS JSON.parse가 거부한다 → 앱이 통째로 죽는다. 여기서 막는다.
    INTEL.write_text(
        json.dumps(intel, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8")
    print(f"[ok] intel.json (date={result['date']} asof={asof} "
          f"spy={preds['spy']['direction']})")


if __name__ == "__main__":
    main()
