#!/usr/bin/env python3
"""GitHub Actions(데이터센터 IP)에서 야후 시세를 긁을 수 있는지 재보는 일회성 탐침.

묻는 것 하나: **스크리닝 파이프라인을 클라우드로 옮길 수 있나?**
집 인터넷에서 잘 되는 수집이 데이터센터 IP에서는 막히는 일이 흔하다
(이 프로젝트에서 FRED가 Vercel IP에 막힌 전례가 있다).

그래서 us-stock의 `USStockDataFetcher`와 **같은 방식**으로 긁는다:
curl_cffi 크롬 임퍼소네이션 + `history()` + `info()` 두 종류.
`info()`는 재무 항목이 쓰는 호출이고 레이트리밋이 더 빡빡하다.

성공 여부만 찍고 끝낸다. 실패해도 exit 0 — 이건 판단 재료지 파이프라인이 아니다.
"""
import random
import sys
import time

import yfinance as yf

OUT = []
_p = print


def print(*a, **k):  # noqa: A001 — 화면과 파일에 같이 남긴다
    _p(*a, **k)
    OUT.append(" ".join(str(x) for x in a))


N = int(sys.argv[1]) if len(sys.argv) > 1 else 100

session = None
try:
    from curl_cffi import requests as curl_requests
    session = curl_requests.Session(impersonate="chrome")
    print("curl_cffi 크롬 임퍼소네이션 ON (us-stock와 동일)")
except ImportError:
    print("⚠ curl_cffi 없음 — 맨몸 요청이라 실제보다 불리한 조건")

# S&P500 목록 자체를 못 받으면 그것도 결과다
try:
    import pandas as pd
    tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    universe = [t.replace(".", "-") for t in tables[0]["Symbol"].tolist()]
    print(f"S&P500 목록 수집 OK: {len(universe)}종목")
except Exception as e:
    print(f"S&P500 목록 수집 실패({type(e).__name__}) — 하드코딩 표본으로 진행")
    universe = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "JPM", "V",
                "UNH", "XOM", "MA", "PG", "JNJ", "HD", "COST", "ABBV", "MRK", "WMT"]

random.seed(7)
sample = random.sample(universe, min(N, len(universe)))

hist_ok = hist_fail = info_ok = info_fail = 0
empties = []
t0 = time.time()

for i, tk in enumerate(sample, 1):
    t = yf.Ticker(tk, session=session)
    try:
        df = t.history(period="6mo")
        if df is None or df.empty:
            hist_fail += 1
            empties.append(tk + "(빈 프레임)")
        else:
            hist_ok += 1
    except Exception as e:
        hist_fail += 1
        empties.append(f"{tk}({type(e).__name__})")
    try:
        info = t.info
        if info and info.get("marketCap") is not None:
            info_ok += 1
        else:
            info_fail += 1
    except Exception:
        info_fail += 1
    if i % 25 == 0:
        print(f"  {i}/{len(sample)} · {time.time()-t0:.0f}초 · "
              f"history {hist_ok}OK/{hist_fail}실패 · info {info_ok}OK/{info_fail}실패")

el = time.time() - t0
n = len(sample)
print()
print("=" * 56)
print(f"표본 {n}종목 · {el:.0f}초 (종목당 {el/n:.2f}초)")
print(f"history(6mo): {hist_ok} 성공 / {hist_fail} 실패  → {hist_ok/n*100:.0f}%")
print(f"info()      : {info_ok} 성공 / {info_fail} 실패  → {info_ok/n*100:.0f}%")
if empties:
    print("실패 목록(최대 10):", ", ".join(empties[:10]))
print()
print(f"503종목 환산 예상 소요: {el/n*503/60:.0f}분")
print("=" * 56)
if hist_ok / n >= 0.95 and info_ok / n >= 0.90:
    print("판정: 통과 — 데이터센터 IP에서도 긁힌다. 이전 검토 가능")
elif hist_ok / n >= 0.95:
    print("판정: 반쪽 — 시세는 되는데 info(재무)가 막힌다. 재무 항목 20%가 죽는다")
else:
    print("판정: 막힘 — 클라우드 이전 불가. PC에서 계속 돌려야 한다")

import pathlib  # noqa: E402
pathlib.Path("tools/yahoo_probe_result.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
