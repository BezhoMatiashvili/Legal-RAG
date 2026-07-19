import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRAPY_PROJECT_ROOT = PROJECT_ROOT / "scraper"
sys.path.insert(0, str(SCRAPY_PROJECT_ROOT))

from legal_scrapers.completion import CompletionError  # noqa: E402
from legal_scrapers.run import (  # noqa: E402
    FIXED_RELEASE_BUNDLE,
    REPOSITORY_ROOT,
    crawl_attestation_issues,
    crawl_quality_issues,
    main,
    parse_args,
    select_spiders,
)

ALL = ["matsne", "ecd", "constcourt", "napr", "supremecourt", "tas", "tbappeal"]


class SelectSpidersTests(unittest.TestCase):
    def test_default_is_all_in_canonical_order(self):
        # Available reported in arbitrary order -> returned in SPIDER_ORDER.
        self.assertEqual(select_spiders(sorted(ALL), None), ALL)

    def test_only_subset_keeps_canonical_order(self):
        self.assertEqual(
            select_spiders(ALL, ["tbappeal", "ecd"]), ["ecd", "tbappeal"]
        )

    def test_unknown_spider_raises(self):
        with self.assertRaises(SystemExit):
            select_spiders(ALL, ["nope"])

    def test_unordered_extra_spider_appended_deterministically(self):
        avail = ALL + ["zzz_future"]
        self.assertEqual(select_spiders(avail, None), ALL + ["zzz_future"])


class ParseArgsTests(unittest.TestCase):
    def test_dates_and_flags(self):
        args = parse_args(
            ["--start-date", "2026-06-01", "--end-date", "2026-06-30", "--no-dedup"]
        )
        self.assertEqual(args.start_date, "2026-06-01")
        self.assertEqual(args.end_date, "2026-06-30")
        self.assertTrue(args.no_dedup)
        self.assertFalse(args.no_progress)
        self.assertIsNone(args.only)

    def test_only_list(self):
        args = parse_args(["--only", "ecd", "tbappeal"])
        self.assertEqual(args.only, ["ecd", "tbappeal"])


class EvidenceReleaseGateTests(unittest.TestCase):
    @staticmethod
    def _argv(*, include_bundle: bool = True, code_revision: str = "a" * 40):
        argv = [
            "--evidence-crawl",
            "--code-revision",
            code_revision,
            "--only",
            "matsne",
            "ecd",
            "constcourt",
            "napr",
            "tas",
            "tbappeal",
        ]
        if include_bundle:
            argv[3:3] = ["--release-bundle", str(FIXED_RELEASE_BUNDLE)]
        return argv

    def test_missing_release_bundle_cannot_initialize_crawler_process(self):
        with (
            patch("legal_scrapers.run._load_release_validator") as load_validator,
            patch("legal_scrapers.run._bootstrap_project_dir") as bootstrap,
            patch("scrapy.crawler.CrawlerProcess") as crawler_process,
            self.assertRaisesRegex(SystemExit, "requires --release-bundle"),
        ):
            main(self._argv(include_bundle=False))

        load_validator.assert_not_called()
        bootstrap.assert_not_called()
        crawler_process.assert_not_called()

    def test_release_validation_error_cannot_initialize_crawler_process(self):
        validator = Mock(side_effect=ValueError("release bundle is absent"))
        with (
            patch(
                "legal_scrapers.run._load_release_validator",
                return_value=validator,
            ),
            patch("legal_scrapers.run._bootstrap_project_dir") as bootstrap,
            patch("scrapy.crawler.CrawlerProcess") as crawler_process,
            self.assertRaisesRegex(SystemExit, "validation failed"),
        ):
            main(self._argv())

        validator.assert_called_once_with(
            FIXED_RELEASE_BUNDLE,
            repo_root=REPOSITORY_ROOT,
        )
        bootstrap.assert_not_called()
        crawler_process.assert_not_called()

    def test_release_revision_drift_cannot_initialize_crawler_process(self):
        validator = Mock(
            return_value=SimpleNamespace(
                repository_revision="b" * 40,
                code_identity_sha256="c" * 64,
            )
        )
        with (
            patch(
                "legal_scrapers.run._load_release_validator",
                return_value=validator,
            ),
            patch("legal_scrapers.run._bootstrap_project_dir") as bootstrap,
            patch("scrapy.crawler.CrawlerProcess") as crawler_process,
            self.assertRaisesRegex(SystemExit, "does not match"),
        ):
            main(self._argv(code_revision="a" * 40))

        validator.assert_called_once_with(
            FIXED_RELEASE_BUNDLE,
            repo_root=REPOSITORY_ROOT,
        )
        bootstrap.assert_not_called()
        crawler_process.assert_not_called()

    def test_existing_ordinary_run_blocks_all_crawlers_before_bootstrap(self):
        validator = Mock(
            return_value=SimpleNamespace(
                repository_revision="a" * 40,
                code_identity_sha256="c" * 64,
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source-evidence"
            (root / "ecd" / "runs" / "prior-run").mkdir(parents=True)
            with (
                patch(
                    "legal_scrapers.run._load_release_validator",
                    return_value=validator,
                ),
                patch("legal_scrapers.run.EVIDENCE_ARTIFACTS_ROOT", root),
                patch("legal_scrapers.run._bootstrap_project_dir") as bootstrap,
                patch("scrapy.crawler.CrawlerProcess") as crawler_process,
                self.assertRaisesRegex(SystemExit, "each be run exactly once"),
            ):
                main(self._argv())

        bootstrap.assert_not_called()
        crawler_process.assert_not_called()


class CrawlQualityGateTests(unittest.TestCase):
    @staticmethod
    def _crawler(name, **stats):
        return SimpleNamespace(
            spider=SimpleNamespace(name=name),
            spidercls=SimpleNamespace(name=name),
            stats=SimpleNamespace(get_stats=lambda: stats),
        )

    def test_clean_finished_crawl_has_no_issues(self):
        self.assertEqual(crawl_quality_issues([
            self._crawler("ecd", finish_reason="finished", item_scraped_count=3)
        ]), [])

    def test_quality_failures_and_callback_errors_fail_gate(self):
        issues = crawl_quality_issues([
            self._crawler(
                "ecd",
                finish_reason="finished",
                **{"quality/failures": 2, "spider_exceptions/ValueError": 1},
            )
        ])
        self.assertTrue(any("2 completeness" in issue for issue in issues))
        self.assertTrue(any("1 callback" in issue for issue in issues))

    def test_aggregate_exception_counter_is_not_double_counted(self):
        issues = crawl_quality_issues([
            self._crawler(
                "ecd",
                finish_reason="finished",
                **{
                    "spider_exceptions/count": 1,
                    "spider_exceptions/ValueError": 1,
                },
            )
        ])
        self.assertTrue(any("1 callback" in issue for issue in issues))
        self.assertFalse(any("2 callback" in issue for issue in issues))


class CrawlAttestationGateTests(unittest.TestCase):
    @staticmethod
    def _crawler():
        spider = SimpleNamespace(
            name="ecd",
            run_id="run-exact",
            run_metadata_path=Path("/tmp/run-exact/run.json"),
            items_path=Path("/tmp/run-exact/items.jsonl"),
        )
        return SimpleNamespace(
            spider=spider,
            spidercls=SimpleNamespace(name="ecd"),
        )

    def test_runner_reloads_the_exact_run_and_items_binding(self):
        crawler = self._crawler()
        with patch("legal_scrapers.run.verify_terminal_record") as verify:
            self.assertEqual(crawl_attestation_issues([crawler]), [])
        verify.assert_called_once_with(
            crawler.spider.run_metadata_path,
            expected_source="ecd",
            expected_run_id="run-exact",
            expected_items_path=crawler.spider.items_path,
        )

    def test_runner_rejects_startup_failed_changed_or_malformed_evidence(self):
        crawler = self._crawler()
        with patch(
            "legal_scrapers.run.verify_terminal_record",
            side_effect=CompletionError("completion outcome does not qualify"),
        ):
            issues = crawl_attestation_issues([crawler])
        self.assertEqual(len(issues), 1)
        self.assertIn("exact completion evidence rejected", issues[0])
        self.assertIn("does not qualify", issues[0])


if __name__ == "__main__":
    unittest.main()
