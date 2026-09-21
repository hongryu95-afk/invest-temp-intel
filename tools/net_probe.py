#!/usr/bin/env python3
"""데이터센터 IP에서 FRED·Gemini에 닿는지만 본다. 키는 쓰지 않는다.

키가 없어도 **차단**과 **인증 거부**는 구분된다:
  - 서버가 자기 형식대로 400/403을 돌려주면 → 네트워크는 뚫린 것(키만 붙이면 됨)
  - 연결 거부·타임아웃·WAF의 HTML 차단 페이지면 → 막힌 것

FRED는 이 프로젝트에서 Vercel IP에 막힌 전례가 있어서 반드시 재봐야 한다.
"""
import pathlib
import urllib.error
import urllib.request

OUT = []


def log(s):
    print(s)
    OUT.append(s)


def probe(name, url, expect_hint):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read(400).decode("utf-8", "replace")
            log(f"{name}: HTTP {r.status} — 응답 옴")
            log(f"  {body[:200]}")
            log(f"  판정: 뚫림 ({expect_hint})")
    except urllib.error.HTTPError as e:
        body = e.read(400).decode("utf-8", "replace")
        blocked = e.code in (403, 451) and "<html" in body.lower()
        log(f"{name}: HTTP {e.code} — 서버가 자기 형식으로 거절")
        log(f"  {body[:200].strip()}")
        log(f"  판정: {'막힘(WAF 차단 페이지)' if blocked else '뚫림 — 키만 붙이면 됨'}")
    except Exception as e:
        log(f"{name}: {type(e).__name__} — {e}")
        log("  판정: 막힘 (연결 자체가 안 됨)")
    log("")


log("=" * 56)
probe("FRED",
      "https://api.stlouisfed.org/fred/series/observations"
      "?series_id=DGS10&api_key=0000000000000000000000000000000d&file_type=json",
      "가짜 키라 400이 정상")
probe("Gemini",
      "https://generativelanguage.googleapis.com/v1beta/models?key=INVALID_KEY_FOR_PROBE",
      "가짜 키라 400이 정상")
probe("Telegram API",
      "https://api.telegram.org/bot000000:INVALID/getMe",
      "가짜 토큰이라 401이 정상")
log("=" * 56)

pathlib.Path("tools/net_probe_result.txt").write_text("\n".join(OUT) + "\n", encoding="utf-8")
