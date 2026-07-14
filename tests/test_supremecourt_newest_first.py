import asyncio
import hashlib
import json
import sqlite3
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from scrapy import Request
from scrapy.http import HtmlResponse
from scrapy.settings import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPER_ROOT = PROJECT_ROOT / "scraper"
sys.path.insert(0, str(SCRAPER_ROOT))

from legal_scrapers.middlewares import (  # noqa: E402
    SupremecourtRetryAfterMiddleware,
    _retry_after_seconds,
)
from legal_scrapers.extensions import RotatingSpiderLogExtension  # noqa: E402
from legal_scrapers.pipelines import SupremecourtDurablePipeline  # noqa: E402
from legal_scrapers.run import crawl_quality_issues, parse_args  # noqa: E402
from legal_scrapers.spiders.supremecourt_spider import (  # noqa: E402
    CHAMBER_NAMES,
    DateWindow,
    NewestFirstPlanner,
    SupremecourtSpider,
    parse_authoritative_total,
)


class _Stats:
    def __init__(self):
        self.values = {}

    def inc_value(self, key, count=1):
        self.values[key] = self.values.get(key, 0) + count

    def set_value(self, key, value):
        self.values[key] = value

    def max_value(self, key, value):
        self.values[key] = max(self.values.get(key, value), value)

    def get_stats(self):
        return dict(self.values)


def _crawler(settings=None):
    return SimpleNamespace(
        stats=_Stats(),
        settings=settings or Settings({"RETRY_TIMES": 8}),
        engine=None,
    )


def _response(body, *, meta=None, status=200, headers=None, url=None):
    url = url or "https://www.supremecourt.ge/ka/getCases?palata=0&page=1"
    request = Request(url, meta=meta or {})
    return HtmlResponse(
        url=url,
        body=body.encode("utf-8"),
        encoding="utf-8",
        status=status,
        headers=headers or {},
        request=request,
    )


def _card(case_id, palata, decision_date):
    return (
        '<div class="cases">'
        f"<div><span>საქმის ნომერი:</span> საქმე-{case_id}</div>"
        f"<div><span>თარიღი:</span> {decision_date}</div>"
        "<span><span>დავის საგანი:</span> დავა</span>"
        "<div><span>შედეგი:</span> შედეგი</div>"
        "<div><span>საჩივრის სახე:</span> საკასაციო</div>"
        f'<a href="/ka/fullcase/{case_id}/{palata}">სრულად</a>'
        "</div>"
    )


def _ready_window(spider, window_id, chamber, start, end, cards):
    window = DateWindow(window_id, chamber, start, end, continues=False)
    window.authoritative_total = len(cards)
    window.pages_seen.add(1)
    window.listed_identities = {card["identity"] for card in cards}
    window.cards = cards
    window.status = "ready_for_details"
    spider._planner.windows[window_id] = window
    spider._ready_window_ids.add(window_id)
    return window


def _planned_card(window_id, case_id, palata, when):
    chamber = CHAMBER_NAMES[str(palata)]
    return {
        "identity": f"{case_id}:{chamber}",
        "fields": {
            "case_id": str(case_id),
            "chamber": chamber,
            "date": when.isoformat(),
        },
        "url": f"https://www.supremecourt.ge/ka/fullcase/{case_id}/{palata}",
        "palata": str(palata),
        "decision_date": when,
        "window_id": window_id,
        "page": 1,
        "card_index": 0,
        "dispatched": False,
    }


def test_authoritative_total_parser_and_adaptive_boundaries():
    response = _response("<div>სულ მოიძებნა 1,234 გადაწყვეტილება</div>")
    assert parse_authoritative_total(response) == 1234

    planner = NewestFirstPlanner(date(2026, 1, 1), initial_days=7)
    window = planner._new_window(0, date(2026, 7, 7), date(2026, 7, 13))
    assert planner.adapted_days(window, 0) == 14
    assert planner.adapted_days(window, 10) > window.days
    assert planner.adapted_days(window, 25) == window.days

    newer, older = planner.split(window)
    assert newer.end == date(2026, 7, 13)
    assert older.end + timedelta(days=1) == newer.start
    assert newer.start <= newer.end
    assert older.start == date(2026, 7, 7)


def test_same_frontier_chambers_are_probed_before_any_detail():
    spider = SupremecourtSpider(start_date="2026-01-01", end_date="2026-07-13")
    spider.crawler = _crawler()
    card = _planned_card("ready", 10, 1, date(2026, 7, 12))
    _ready_window(spider, "ready", 1, date(2026, 7, 7), date(2026, 7, 13), [card])
    spider._planner._new_window(0, date(2026, 7, 7), date(2026, 7, 13))

    requests = spider._advance_frontier()

    assert len(requests) == 1
    assert "/getCases" in requests[0].url
    assert "/fullcase/" not in requests[0].url


def test_pending_older_window_is_safe_cutoff_for_decision_order():
    spider = SupremecourtSpider(start_date="2026-01-01", end_date="2026-07-13")
    spider.crawler = _crawler()
    cards = [
        _planned_card("civil", 12, 1, date(2026, 7, 12)),
        _planned_card("civil", 7, 1, date(2026, 7, 7)),
    ]
    _ready_window(spider, "civil", 1, date(2026, 7, 7), date(2026, 7, 13), cards)
    spider._planner._new_window(0, date(2026, 7, 7), date(2026, 7, 10))

    requests = spider._advance_frontier()

    assert [request.meta["identity"].split(":", 1)[0] for request in requests] == ["12"]
    assert cards[0]["dispatched"] is True
    assert cards[1]["dispatched"] is False


def test_virtual_successor_keeps_unequal_chamber_windows_globally_newest_first():
    spider = SupremecourtSpider(start_date="2026-01-01", end_date="2026-07-13")
    spider.crawler = _crawler()
    civil_cards = [
        _planned_card("civil", 2201, 1, date(2026, 6, 22)),
        _planned_card("civil", 1601, 1, date(2026, 6, 16)),
    ]
    criminal_cards = [
        _planned_card("criminal", 2202, 2, date(2026, 6, 22)),
        _planned_card("criminal", 1502, 2, date(2026, 6, 15)),
        _planned_card("criminal", 902, 2, date(2026, 6, 9)),
    ]
    civil = _ready_window(
        spider,
        "civil",
        1,
        date(2026, 6, 16),
        date(2026, 6, 22),
        civil_cards,
    )
    criminal = _ready_window(
        spider,
        "criminal",
        2,
        date(2026, 6, 9),
        date(2026, 6, 22),
        criminal_cards,
    )
    civil.continues = True
    criminal.continues = True

    first_batch = spider._advance_frontier()
    first_dates = [request.meta["fields"]["date"] for request in first_batch]
    assert first_dates == ["2026-06-22", "2026-06-22", "2026-06-16"]
    assert criminal_cards[-1]["dispatched"] is False

    for request in first_batch:
        window = spider._planner.windows[request.meta["window_id"]]
        window.pending_identities.discard(request.meta["identity"])
        window.new_identities.add(request.meta["identity"])
    spider._detail_batch_active = False

    successor_probe = spider._advance_frontier()
    assert len(successor_probe) == 1
    assert successor_probe[0].meta["kind"] == "window_list"
    assert successor_probe[0].meta["palata"] == 1
    assert "tarigiMde=2026%2F06%2F15" in successor_probe[0].url
    assert criminal_cards[-1]["dispatched"] is False

    successor = spider._planner.windows[successor_probe[0].meta["window_id"]]
    successor_response = _response(
        "<div>სულ მოიძებნა 2 გადაწყვეტილება</div>"
        + _card(1501, 1, "2026-06-15")
        + _card(1201, 1, "2026-06-12"),
        meta={
            "window_id": successor.window_id,
            "palata": 1,
            "page": 1,
        },
        url=successor_probe[0].url,
    )
    second_batch = list(spider.parse_list(successor_response))
    second_dates = [request.meta["fields"]["date"] for request in second_batch]
    assert second_dates == [
        "2026-06-15",
        "2026-06-15",
        "2026-06-12",
        "2026-06-09",
    ]
    emitted = [date.fromisoformat(value) for value in first_dates + second_dates]
    assert emitted == sorted(emitted, reverse=True)


def test_combined_cross_chamber_details_are_strictly_newest_first():
    spider = SupremecourtSpider(start_date="2026-01-01", end_date="2026-07-13")
    spider.crawler = _crawler()
    admin = _planned_card("admin", 13, 0, date(2026, 7, 13))
    civil = _planned_card("civil", 12, 1, date(2026, 7, 12))
    _ready_window(spider, "admin", 0, date(2026, 7, 11), date(2026, 7, 13), [admin])
    _ready_window(spider, "civil", 1, date(2026, 7, 7), date(2026, 7, 13), [civil])
    spider._planner._new_window(2, date(2026, 7, 1), date(2026, 7, 6))

    requests = spider._advance_frontier()

    assert [request.meta["fields"]["date"] for request in requests] == [
        "2026-07-13",
        "2026-07-12",
    ]
    assert requests[0].priority > requests[1].priority


def test_known_identity_is_skipped_before_detail_request():
    spider = SupremecourtSpider(start_date="2026-07-01", end_date="2026-07-13")
    spider.crawler = _crawler()
    card = _planned_card("known", 99, 0, date(2026, 7, 13))
    spider.dedup_enabled = True
    spider._seen_keys = {card["identity"]}
    window = _ready_window(
        spider, "known", 0, date(2026, 7, 13), date(2026, 7, 13), [card]
    )

    requests = spider._advance_frontier()

    assert not [request for request in requests if "/fullcase/" in request.url]
    assert card["identity"] in window.known_identities


def test_window_splits_above_30_and_paginates_only_irreducible_day():
    spider = SupremecourtSpider(start_date="2026-07-01", end_date="2026-07-13")
    spider.crawler = _crawler()
    broad = DateWindow("broad", 0, date(2026, 7, 7), date(2026, 7, 13))
    spider._planner.windows[broad.window_id] = broad
    response = _response(
        "<div>სულ მოიძებნა 31 გადაწყვეტილება</div>",
        meta={"window_id": "broad", "palata": 0, "page": 1},
    )

    list(spider.parse_list(response))
    assert broad.status == "split"

    day_spider = SupremecourtSpider(start_date="2026-07-13", end_date="2026-07-13")
    day_spider.crawler = _crawler()
    single = DateWindow("single", 0, date(2026, 7, 13), date(2026, 7, 13))
    day_spider._planner.windows[single.window_id] = single
    html = "<div>სულ მოიძებნა 31 გადაწყვეტილება</div>" + "".join(
        _card(1000 + index, 0, "2026-07-13") for index in range(30)
    )
    page_one = _response(
        html,
        meta={"window_id": "single", "palata": 0, "page": 1},
    )
    requests = list(day_spider.parse_list(page_one))
    assert single.expected_pages == 2
    assert len(requests) == 1
    assert requests[0].meta["page"] == 2

    bounded_spider = SupremecourtSpider(start_date="2026-07-13", end_date="2026-07-13")
    bounded_spider.crawler = _crawler()
    bounded = DateWindow("bounded", 0, date(2026, 7, 13), date(2026, 7, 13))
    bounded_spider._planner.windows[bounded.window_id] = bounded
    bounded_html = "<div>სულ მოიძებნა 30 გადაწყვეტილება</div>" + "".join(
        _card(2000 + index, 0, "2026-07-13") for index in range(30)
    )
    bounded_response = _response(
        bounded_html,
        meta={"window_id": "bounded", "palata": 0, "page": 1},
    )
    bounded_requests = list(bounded_spider.parse_list(bounded_response))
    assert bounded.expected_pages == 1
    assert not [
        request
        for request in bounded_requests
        if request.meta.get("kind") == "window_list" and request.meta.get("page") == 2
    ]


def test_recent_window_requests_always_bypass_http_cache():
    spider = SupremecourtSpider(start_date="2026-07-01", end_date="2026-07-13")
    window = DateWindow("fresh", 0, date(2026, 7, 7), date(2026, 7, 13))
    request = spider.request_window(window)
    assert request.meta["dont_cache"] is True
    assert request.dont_filter is True


def test_retry_after_numeric_and_http_date(monkeypatch):
    assert _retry_after_seconds("17") == 17
    now = datetime(2026, 7, 13, 8, 0, tzinfo=UTC)
    target = now + timedelta(seconds=23)
    assert (
        _retry_after_seconds(target.strftime("%a, %d %b %Y %H:%M:%S GMT"), now=now)
        == 23
    )

    settings = Settings({"RETRY_TIMES": 8, "SUPREMECOURT_RETRY_AFTER_MAX_SECONDS": 60})
    spider = SupremecourtSpider()
    spider.crawler = _crawler(settings)
    request = Request(
        "https://www.supremecourt.ge/ka/getCases", meta={"kind": "window_list"}
    )
    response = HtmlResponse(
        request.url,
        status=429,
        headers={"Retry-After": "9"},
        request=request,
    )
    import twisted.internet.task

    delays = []
    monkeypatch.setattr(
        twisted.internet.task,
        "deferLater",
        lambda _reactor, delay, callback: delays.append(delay) or callback(),
    )
    retry = SupremecourtRetryAfterMiddleware().process_response(
        request, response, spider
    )
    assert isinstance(retry, Request)
    assert retry.meta["supremecourt_retry_after_times"] == 1
    assert retry.dont_filter is True
    assert delays == [9]


def test_durable_pipeline_persists_before_marking_seen():
    events = []
    identity = f"1:{CHAMBER_NAMES['0']}"
    spider = SimpleNamespace(
        dedup_key=lambda _item: identity,
        _seen_keys=set(),
        crawler=SimpleNamespace(stats=_Stats()),
        persist_item=lambda _item, _identity: events.append("persist"),
        mark_seen=lambda _item: events.append("seen"),
        item_persisted=lambda _identity: events.append("settled"),
        item_failed=lambda *_args: events.append("failed"),
        record_quality_failure=lambda *_args, **_kwargs: None,
    )
    item = {"case_id": "1", "chamber": CHAMBER_NAMES["0"], "body_markdown": "full"}
    SupremecourtDurablePipeline().process_item(item, spider)
    assert events == ["persist", "seen", "settled"]


def test_empty_modal_retries_then_stays_unseen_and_unresolved():
    spider = SupremecourtSpider(start_date="2026-07-13", end_date="2026-07-13")
    spider.crawler = _crawler()
    identity = f"77:{CHAMBER_NAMES['0']}"
    window = DateWindow(
        "empty", 0, date(2026, 7, 13), date(2026, 7, 13), continues=False
    )
    window.pending_identities.add(identity)
    spider._planner.windows[window.window_id] = window
    spider._ready_window_ids.add(window.window_id)
    spider._identity_to_window[identity] = window.window_id
    meta = {
        "fields": {
            "case_id": "77",
            "chamber": CHAMBER_NAMES["0"],
            "date": "2026-07-13",
        },
        "palata": "0",
        "identity": identity,
        "window_id": window.window_id,
        "kind": "detail",
    }
    first = _response(
        "<html><body>metadata only</body></html>",
        meta=meta,
        url="https://www.supremecourt.ge/ka/fullcase/77/0",
    )
    retry = list(spider.parse_detail(first))
    assert len(retry) == 1
    assert retry[0].meta["detail_parse_retry_times"] == 1

    exhausted_meta = {**meta, "detail_parse_retry_times": 2}
    exhausted = _response(
        "<html><body>still empty</body></html>",
        meta=exhausted_meta,
        url="https://www.supremecourt.ge/ka/fullcase/77/0",
    )
    assert list(spider.parse_detail(exhausted)) == []
    assert identity not in spider._seen_keys
    assert identity not in window.pending_identities
    assert window.failures[0]["kind"] == "empty_fullcase"


def test_failure_manifest_retains_exact_count_with_bounded_examples():
    spider = SupremecourtSpider(start_date="2026-07-13", end_date="2026-07-13")
    window = DateWindow(
        "bounded", 0, date(2026, 7, 13), date(2026, 7, 13), continues=False
    )
    window.status = "incomplete"
    spider._planner.windows[window.window_id] = window

    for index in range(500):
        spider._window_failure(
            window,
            "detail_http",
            f"failure {index}",
            f"https://example.invalid/{index}",
        )

    manifest = spider._manifest(final=False)
    assert window.failure_count == 500
    assert len(window.failures) == 100
    assert manifest["completed_windows"][0]["failures"] == 500
    assert manifest["unresolved_failure_count"] == 500
    assert manifest["unresolved_failures_truncated"] is True
    assert len(manifest["unresolved_failures"]) == 100


def test_graceful_timeout_is_an_expected_partial_finish():
    args = parse_args(["--only", "supremecourt", "--max-runtime-seconds", "14400"])
    assert args.max_runtime_seconds == 14400
    crawler = SimpleNamespace(
        spider=SimpleNamespace(name="supremecourt", partial_by_design=True),
        spidercls=SimpleNamespace(name="supremecourt"),
        stats=SimpleNamespace(
            get_stats=lambda: {"finish_reason": "closespider_timeout"}
        ),
    )
    assert crawl_quality_issues([crawler]) == []


def test_rotating_log_close_accepts_scrapy_signal_keyword_names():
    extension = RotatingSpiderLogExtension(_crawler())
    extension.spider_closed(spider=SimpleNamespace(), reason="shutdown")


def _artifact_item(case_id, chamber="0", when="2026-07-13"):
    return {
        "case_id": str(case_id),
        "chamber": CHAMBER_NAMES[chamber],
        "case_number": f"საქმე-{case_id}",
        "date": when,
        "source_url": f"https://www.supremecourt.ge/ka/fullcase/{case_id}/{chamber}",
        "docx_url": f"https://www.supremecourt.ge/ka/download/{case_id}/{chamber}",
        "body_markdown": "სრული გადაწყვეტილების ტექსტი",
    }


def _write_finalized_parent(root, run_id="20260713T100000Z_parent"):
    run = root / "supremecourt" / "runs" / run_id
    run.mkdir(parents=True, mode=0o700)
    rows = [
        _artifact_item("100", "0", "2026-07-12"),
        _artifact_item("200", "1", "2026-07-12"),
        _artifact_item("49251", "2", "2026-07-12"),
    ]
    rows[-1]["body_markdown"] += " პირველი ინსტანციის ნომერი 330100122006207137"
    rows.sort(
        key=lambda item: (item["date"], item["chamber"], item["case_id"]),
        reverse=True,
    )
    payload = b"".join(
        (json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        for item in rows
    )
    items_path = run / "items.jsonl"
    journal_path = run / "items.journal.jsonl"
    items_path.write_bytes(payload)
    journal_path.write_bytes(payload)
    per_chamber = {}
    windows = []
    for chamber_id, chamber in CHAMBER_NAMES.items():
        chamber_rows = [item for item in rows if item["chamber"] == chamber]
        per_chamber[chamber] = {
            "known_items": 0,
            "new_items": 1,
            "total_items": 1,
            "newest_date": chamber_rows[0]["date"],
            "oldest_date": chamber_rows[0]["date"],
            "resume_cursor": "2026-07-09",
        }
        windows.append(
            {
                "id": f"w{int(chamber_id) + 1:06d}",
                "chamber": chamber,
                "start": "2026-07-10",
                "end": "2026-07-13",
                "status": "completed",
                "authoritative_total": 1,
                "known": 0,
                "new": 1,
                "failures": 0,
            }
        )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": "2026-07-13T10:00:00+00:00",
        "finished_at": "2026-07-13T14:00:01+00:00",
        "finish_reason": "closespider_timeout",
        "partial_by_design": True,
        "date_order": "newest_first",
        "frontier_start_date": "2026-07-13",
        "lower_bound": "2026-07-01",
        "max_runtime_seconds": 14400,
        "elapsed_time_seconds": 14401.0,
        "items_file": str(items_path.absolute()),
        "journal_file": str(journal_path.absolute()),
        "items_sha256": hashlib.sha256(payload).hexdigest(),
        "known_items": 0,
        "new_items": 3,
        "total_items": 3,
        "known_encountered": 0,
        "per_chamber": per_chamber,
        "oldest_fully_completed_global_date_frontier": "2026-07-10",
        "per_chamber_resume_cursors": {
            chamber: "2026-07-09" for chamber in CHAMBER_NAMES.values()
        },
        "completed_windows": windows,
        "retries": {"total": 0, "retry_after": 0, "parse": 0, "detail_parse": 0},
        "unresolved_failure_count": 0,
        "unresolved_failures_truncated": False,
        "unresolved_failures": [],
    }
    manifest_path = run / "partial_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in (items_path, journal_path, manifest_path):
        path.chmod(0o600)
    return run


def test_validated_final_parent_seeds_each_chamber_and_advances_child_cursors(
    tmp_path, monkeypatch
):
    root = tmp_path / "artifacts"
    parent = _write_finalized_parent(root)
    monkeypatch.setattr(SupremecourtSpider, "ARTIFACTS_ROOT", root)
    settings = Settings({"DEDUP_ENABLED": True, "CLOSESPIDER_TIMEOUT": 14400})
    spider = SupremecourtSpider(start_date="2026-07-01", end_date="2026-07-13")
    spider.crawler = _crawler(settings)

    spider.configure_run_outputs(settings)

    assert set(spider._chamber_start_cursors.values()) == {date(2026, 7, 9)}
    assert {window.end for window in spider._planner.windows.values()} == {
        date(2026, 7, 9)
    }
    assert spider._resume_parent["run_id"] == parent.name
    assert spider._resume_parent["manifest_sha256"] == hashlib.sha256(
        (parent / "partial_manifest.json").read_bytes()
    ).hexdigest()

    for window in spider._planner.windows.values():
        window.status = "completed"
        window.authoritative_total = 0
        window.pages_seen.add(1)
        spider._completed_intervals[window.chamber].append((window.start, window.end))
    manifest = spider._manifest(final=False)
    assert set(manifest["per_chamber_start_cursors"].values()) == {"2026-07-09"}
    assert set(manifest["per_chamber_resume_cursors"].values()) == {"2026-07-02"}
    assert manifest["oldest_fully_completed_global_date_frontier"] == "2026-07-03"
    spider._dedup_conn.close()


def test_invalid_newest_final_parent_fails_closed_without_older_fallback(
    tmp_path, monkeypatch
):
    root = tmp_path / "artifacts"
    _write_finalized_parent(root, "20260712T100000Z_parent")
    newest = _write_finalized_parent(root, "20260713T100000Z_parent")
    path = newest / "partial_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    chamber = CHAMBER_NAMES["0"]
    manifest["per_chamber"][chamber]["resume_cursor"] = "2026-07-08"
    manifest["per_chamber_resume_cursors"][chamber] = "2026-07-08"
    path.write_text(json.dumps(manifest, ensure_ascii=False) + "\n", encoding="utf-8")
    path.chmod(0o600)

    monkeypatch.setattr(SupremecourtSpider, "ARTIFACTS_ROOT", root)
    settings = Settings({"DEDUP_ENABLED": True})
    spider = SupremecourtSpider(start_date="2026-07-01", end_date="2026-07-13")
    spider.crawler = _crawler(settings)

    with pytest.raises(RuntimeError, match="newest finalized.*failed validation"):
        spider.configure_run_outputs(settings)
    assert len(list((root / "supremecourt" / "runs").iterdir())) == 2


def test_nonfinal_manifest_is_ignored_and_preserves_fresh_frontier(
    tmp_path, monkeypatch
):
    root = tmp_path / "artifacts"
    live = root / "supremecourt" / "runs" / "20260713T100000Z_live"
    live.mkdir(parents=True)
    path = live / "partial_manifest.json"
    path.write_text('{"finished_at":null,"items_sha256":null}\n', encoding="utf-8")
    path.chmod(0o600)

    monkeypatch.setattr(SupremecourtSpider, "ARTIFACTS_ROOT", root)
    settings = Settings({"DEDUP_ENABLED": True})
    spider = SupremecourtSpider(start_date="2026-07-01", end_date="2026-07-13")
    spider.crawler = _crawler(settings)
    spider.configure_run_outputs(settings)

    assert spider._resume_parent is None
    assert set(spider._chamber_start_cursors.values()) == {date(2026, 7, 13)}
    assert {window.end for window in spider._planner.windows.values()} == {
        date(2026, 7, 13)
    }
    spider._dedup_conn.close()


def test_valid_parent_with_a_different_end_date_starts_from_the_new_frontier(
    tmp_path, monkeypatch
):
    root = tmp_path / "artifacts"
    _write_finalized_parent(root)
    monkeypatch.setattr(SupremecourtSpider, "ARTIFACTS_ROOT", root)
    settings = Settings({"DEDUP_ENABLED": True})
    spider = SupremecourtSpider(start_date="2026-07-01", end_date="2026-07-14")
    spider.crawler = _crawler(settings)

    spider.configure_run_outputs(settings)

    assert spider._resume_parent is None
    assert set(spider._chamber_start_cursors.values()) == {date(2026, 7, 14)}
    assert {window.end for window in spider._planner.windows.values()} == {
        date(2026, 7, 14)
    }
    spider._dedup_conn.close()


def test_seen_reconciliation_removes_only_ghosts_and_adds_exported(
    tmp_path, monkeypatch
):
    root = tmp_path / "artifacts"
    run = root / "supremecourt" / "runs" / "old"
    run.mkdir(parents=True)
    item = _artifact_item("100")
    (run / "items.jsonl").write_text(json.dumps(item, ensure_ascii=False) + "\n")
    db = root / "supremecourt" / "seen.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE seen (key TEXT PRIMARY KEY, run_id TEXT, ts TEXT)")
    conn.execute(
        "INSERT INTO seen VALUES (?, ?, ?)", (f"ghost:{CHAMBER_NAMES['0']}", "x", "x")
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(SupremecourtSpider, "ARTIFACTS_ROOT", root)
    settings = Settings({"DEDUP_ENABLED": True})
    spider = SupremecourtSpider(start_date="2026-01-01", end_date="2026-07-13")
    spider.crawler = _crawler(settings)
    spider.configure_run_outputs(settings)

    keys = {row[0] for row in spider._dedup_conn.execute("SELECT key FROM seen")}
    assert keys == {f"100:{CHAMBER_NAMES['0']}"}
    assert spider.crawler.stats.values["dedup/ghosts_removed"] == 1
    spider._dedup_conn.close()


def test_reconciliation_metrics_wait_for_scrapy_stats_initialization(
    tmp_path, monkeypatch
):
    root = tmp_path / "artifacts"
    run = root / "supremecourt" / "runs" / "old"
    run.mkdir(parents=True)
    item = _artifact_item("100")
    (run / "items.jsonl").write_text(json.dumps(item, ensure_ascii=False) + "\n")

    monkeypatch.setattr(SupremecourtSpider, "ARTIFACTS_ROOT", root)
    settings = Settings({"DEDUP_ENABLED": True})
    spider = SupremecourtSpider(start_date="2026-01-01", end_date="2026-07-13")
    # Scrapy 2.16 constructs the spider before Crawler._apply_settings installs stats.
    spider.crawler = SimpleNamespace(stats=None, settings=settings, engine=None)
    spider.configure_run_outputs(settings)
    assert spider._reconcile_counts == (1, 0, 1)

    spider.crawler.stats = _Stats()
    request = asyncio.run(anext(spider.start()))
    assert isinstance(request, Request)
    assert spider.crawler.stats.values["dedup/reconciled_exported"] == 1
    assert spider.crawler.stats.values["dedup/exported_keys_added"] == 1
    spider._dedup_conn.close()


def test_partial_manifest_and_cumulative_items_are_atomic_and_exact(
    tmp_path, monkeypatch
):
    root = tmp_path / "artifacts"
    old_run = root / "supremecourt" / "runs" / "old"
    old_run.mkdir(parents=True)
    old = _artifact_item("100", "0", "2026-07-12")
    (old_run / "items.jsonl").write_text(json.dumps(old, ensure_ascii=False) + "\n")

    monkeypatch.setattr(SupremecourtSpider, "ARTIFACTS_ROOT", root)
    settings = Settings({"DEDUP_ENABLED": True, "CLOSESPIDER_TIMEOUT": 14400})
    spider = SupremecourtSpider(start_date="2026-07-01", end_date="2026-07-13")
    spider.crawler = _crawler(settings)
    spider.crawler.stats.set_value("elapsed_time_seconds", 14400.4)
    spider.configure_run_outputs(settings)
    new = _artifact_item("200", "1", "2026-07-13")
    identity = f"200:{CHAMBER_NAMES['1']}"
    spider.persist_item(new, identity)
    for chamber in (0, 1, 2):
        spider._completed_intervals[chamber].append(
            (date(2026, 7, 10), date(2026, 7, 13))
        )
    spider.closed("closespider_timeout")

    payload = spider.items_path.read_bytes()
    manifest = json.loads(spider.partial_manifest_path.read_text())
    items = [json.loads(line) for line in payload.decode().splitlines()]
    assert [item["case_id"] for item in items] == ["200", "100"]
    assert manifest["items_sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["known_items"] == 1
    assert manifest["new_items"] == 1
    assert manifest["total_items"] == 2
    assert manifest["max_runtime_seconds"] == 14400
    assert manifest["partial_by_design"] is True
    assert manifest["oldest_fully_completed_global_date_frontier"] == "2026-07-10"
    assert set(manifest["per_chamber_resume_cursors"].values()) == {"2026-07-09"}
    assert not list(spider.run_dir.glob(".*.tmp"))


def test_final_manifest_has_monotonic_elapsed_before_corestats_close(
    monkeypatch,
):
    spider = SupremecourtSpider(start_date="2026-07-01", end_date="2026-07-13")
    spider.crawler = _crawler(Settings({"CLOSESPIDER_TIMEOUT": 14400}))
    spider._started_monotonic = 100.0
    monkeypatch.setattr(
        "legal_scrapers.spiders.supremecourt_spider.time.monotonic", lambda: 14500.5
    )

    manifest = spider._manifest(final=True)

    assert manifest["elapsed_time_seconds"] == 14400.5
