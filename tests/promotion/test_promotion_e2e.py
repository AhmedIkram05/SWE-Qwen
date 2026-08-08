"""Offline end-to-end tests for the promotion pipeline entrypoints (task 9.7).

Drives ``promotion.run.main`` — the decide job and the deploy job — end to
end with in-memory runs and monkeypatched exec/network edges: no LLM, no
Ollama, no Modal, no W&B, no GCS.  Every scenario is deterministic (fixed
seed-42 paired vectors, no sleeps, no real subprocesses, no timeouts) and
CI-safe (``pytestmark = pytest.mark.unit`` at module level).

Coverage strategy: the exec/network lines of ``promotion/run.py`` are
``# pragma: no cover`` (plan §4.11); the surrounding control flow is
exercised here with recording fakes so the 95% branch gate is a real gate.
"""

from __future__ import annotations

import json
import runpy
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evaluation.config import EvalConfig
from evaluation.metrics import aggregate_metrics
from evaluation.schema import EvalResult, EvalRun, PatchApplicationResult
from inference.config import ServeConfig
from promotion import run as run_mod
from promotion.deploy import ProbeError
from promotion.registry import ChampionRecord, write_champion

pytestmark = pytest.mark.unit

# A real ``ServeConfig.variants`` member: the decide job's config-gap gate
# aborts on anything else before the eval is even considered.
CANDIDATE = "higher_rank_14b"
CHAMPION = "champion_a"
CHAMPION_KEY = f"qwen3-14b:{CHAMPION}"
INSTANCES = [f"i{index}" for index in range(1, 11)]


# ── in-memory EvalRun/EvalResult fixtures (mirrors tests/test_promotion.py) ──


def _result(instance_id: str, f2p: float, variant: str, p2p: float = 1.0) -> EvalResult:
    """Minimal valid ``EvalResult`` for a single instance."""
    return EvalResult(
        instance_id=instance_id,
        repo="owner/repo",
        model_name="qwen3-14b",
        variant=variant,
        prompt_template="chat",
        generated_patch="",
        patch_application=PatchApplicationResult(success=True, method_used="git_apply"),
        tests_before=[],
        tests_after=[],
        f2p=f2p,
        p2p=p2p,
        latency_seconds=1.0,
        timestamp=datetime.now(UTC),
    )


def _make_run(run_id: str, config: EvalConfig, results: list[EvalResult]) -> EvalRun:
    """In-memory completed ``EvalRun`` with derived models_evaluated/aggregate."""
    return EvalRun(
        run_id=run_id,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        config=config,
        models_evaluated=sorted({f"{r.model_name}:{r.variant}" for r in results}),
        results=results,
        aggregate=[aggregate_metrics(results)] if results else [],
        status="completed",
    )


def _pair_config() -> EvalConfig:
    """Default paired-eval window (dev tier: expanded-repos, seed 42)."""
    return EvalConfig(dataset_run_id="expanded-repos", tier_seed=42)


def _f2p_run(run_id: str, variant: str, solved: set[str]) -> EvalRun:
    """Run over i1..i10 where only the instances in *solved* pass F2P (p2p all 1.0)."""
    results = [
        _result(name, f2p=1.0 if name in solved else 0.0, variant=variant) for name in INSTANCES
    ]
    return _make_run(run_id, _pair_config(), results)


def _record(**overrides: Any) -> ChampionRecord:
    """Fake 2026-08-08 champion record, overridable per field."""
    fields: dict[str, Any] = {
        "variant": CHAMPION,
        "model_ref": CHAMPION_KEY,
        "f2p_rate": 0.3,
        "p2p_rate": 0.9,
        "dataset_run_id": "expanded-repos",
        "tier": "full",
        "seed": 42,
        "promoted_at": "2026-08-08T00:00:00+00:00",
        "previous": None,
    }
    fields.update(overrides)
    return ChampionRecord(**fields)


def _seed_champion(tmp_path: Path, record: ChampionRecord | None = None) -> Path:
    """Write champion.json into *tmp_path* via the real registry writer."""
    path = tmp_path / "champion.json"
    write_champion(path, record or _record())
    return path


def _patch_decide(monkeypatch: pytest.MonkeyPatch, champion_run: EvalRun, candidate_run: EvalRun):
    """Stub every exec/network edge of the decide path; record audit calls."""
    calls: dict[str, list[Any]] = {"decisions": []}
    monkeypatch.setattr(run_mod, "_has_challenger_alias", lambda _variant, _serve: True)
    monkeypatch.setattr(run_mod, "_launch_evals", lambda _pairs, _mode, _base: [])
    monkeypatch.setattr(
        run_mod, "_wait_for_pair", lambda *args, **kwargs: ((champion_run, candidate_run), "ok")
    )
    # list.append can't take the entity=/project= kwargs _finalize passes
    monkeypatch.setattr(
        run_mod.audit,
        "write_decision_record",
        lambda record, **kwargs: calls["decisions"].append(record),
    )
    monkeypatch.setattr(run_mod.audit, "log_decision_metrics", lambda *args, **kwargs: None)
    return calls


def _github_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point $GITHUB_OUTPUT at a tmp file (spec §4.6 step 6)."""
    out = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    return out


def _parse_output(path: Path) -> dict[str, str]:
    """Parse the $GITHUB_OUTPUT file into ``key=value`` pairs ({} when absent)."""
    if not path.exists():
        return {}
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())


# ── decide job (spec §4.6): accept / reject / abort paths ──────────────────


class TestDecideJobE2E:
    """``main([...])`` decide path: promote, reject, and metrics-gap outcomes."""

    def test_promote_writes_full_github_output(self, tmp_path, monkeypatch, capsys):
        # Champion solves {i1,i2} (F2P 0.2); candidate solves the strict
        # superset {i1..i6} (F2P 0.6): gain 0.4, b01=0, b10=4 — a clean,
        # deterministic win (ci_lower = 0.1 > 0 with seed 42).
        path = _seed_champion(tmp_path)
        champion_run = _f2p_run("champion-run", CHAMPION, solved={"i1", "i2"})
        candidate_run = _f2p_run("candidate-run", CANDIDATE, solved=set(INSTANCES[:6]))
        calls = _patch_decide(monkeypatch, champion_run, candidate_run)
        out = _github_output(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)  # F1: keep data/promotion_decisions out of the repo

        rc = run_mod.main(
            ["--candidate-variant", CANDIDATE, "--champion-path", str(path), "--mode", "dev"]
        )

        assert rc == 0
        assert "promote=True" in capsys.readouterr().out
        outputs = _parse_output(out)
        assert outputs["promote"] == "true"
        assert outputs["candidate"] == CANDIDATE
        # champion output carries the PROMOTED candidate, so the deploy job's
        # --variant (and the ChampionRecord it writes) hits the gate winner.
        assert outputs["champion"] == CANDIDATE
        assert outputs["decision_id"].startswith("promote-")
        # spec §4.10: the rates decide() saw must reach the deploy job
        assert float(outputs["champion_f2p"]) == 0.2
        assert float(outputs["candidate_f2p"]) == 0.6
        assert float(outputs["champion_p2p"]) == 1.0
        assert float(outputs["candidate_p2p"]) == 1.0
        record = calls["decisions"][0]
        assert record["outcome"] == "promote"
        assert record["metrics"]["candidate"]["f2p_rate"] == 0.6
        assert record["metrics"]["incumbent"]["f2p_rate"] == 0.2

    def test_reject_writes_no_promote_and_exits_zero(self, tmp_path, monkeypatch, capsys):
        # Candidate solves {i1} only: F2P 0.1 < the 0.15 floor -> fatal-flaw.
        # Rejection is a successful outcome: exit 0, no $GITHUB_OUTPUT lines.
        path = _seed_champion(tmp_path)
        champion_run = _f2p_run("champion-run", CHAMPION, solved={"i1", "i2"})
        candidate_run = _f2p_run("candidate-run", CANDIDATE, solved={"i1"})
        calls = _patch_decide(monkeypatch, champion_run, candidate_run)
        out = _github_output(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)  # F1: keep data/promotion_decisions out of the repo

        rc = run_mod.main(["--candidate-variant", CANDIDATE, "--champion-path", str(path)])

        assert rc == 0
        assert "promote=False fatal-flaw" in capsys.readouterr().out
        assert "promote=true" not in _parse_output(out)
        record = calls["decisions"][0]
        assert record["outcome"] == "reject"
        assert record["reasons"] == ["fatal-flaw"]
        assert record["metrics"]["candidate"]["f2p_rate"] == 0.1

    def test_missing_metrics_rejects_fatal_flaw(self, tmp_path, monkeypatch, capsys):
        # Stale models_evaluated makes the aggregate lookup miss: the gate
        # must never fabricate a gain from unpaired metrics (spec §4.6 step 4).
        path = _seed_champion(tmp_path)
        champion_run = _f2p_run("champion-run", CHAMPION, solved={"i1", "i2"})
        for result in champion_run.results:
            result.model_name = "other-model"  # aggregate key diverges from models_evaluated
        candidate_run = _f2p_run("candidate-run", CANDIDATE, solved=set(INSTANCES[:6]))
        calls = _patch_decide(monkeypatch, champion_run, candidate_run)
        _github_output(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)  # F1: keep data/promotion_decisions out of the repo

        rc = run_mod.main(["--candidate-variant", CANDIDATE, "--champion-path", str(path)])

        assert rc == 0
        assert "fatal-flaw" in capsys.readouterr().out
        assert calls["decisions"][0]["outcome"] == "reject"

    def test_promote_writes_step_summary_when_env_set(self, tmp_path, monkeypatch, capsys):
        # F1 (spec §4.10 L179): the full decision markdown reaches
        # $GITHUB_STEP_SUMMARY in the decide job, so the human approving the
        # deploy gate reads exactly what they approve (terraform-plan parity).
        path = _seed_champion(tmp_path)
        champion_run = _f2p_run("champion-run", CHAMPION, solved={"i1", "i2"})
        candidate_run = _f2p_run("candidate-run", CANDIDATE, solved=set(INSTANCES[:6]))
        calls = _patch_decide(monkeypatch, champion_run, candidate_run)
        summary = tmp_path / "step_summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
        monkeypatch.chdir(tmp_path)  # F1: keep data/promotion_decisions out of the repo

        rc = run_mod.main(["--candidate-variant", CANDIDATE, "--champion-path", str(path)])

        assert rc == 0
        decision_id = calls["decisions"][0]["decision_id"]
        text = summary.read_text(encoding="utf-8")
        assert f"# Promotion decision: {decision_id}" in text
        assert "**Outcome**: `promote`" in text
        assert "- **Candidate**: higher_rank_14b" in text
        assert "## Metrics" in text
        assert "**F2P gain**:" in text
        assert "## Reasons" in text

    def test_promote_persists_decision_files_for_upload(self, tmp_path, monkeypatch, capsys):
        # F1 (spec §4.10 L179): decision.md/decision.json land in
        # data/promotion_decisions/ so the decide job's actions/upload-artifact
        # step (name promotion-decision-<id>, retention 7) ships them.
        path = _seed_champion(tmp_path)
        champion_run = _f2p_run("champion-run", CHAMPION, solved={"i1", "i2"})
        candidate_run = _f2p_run("candidate-run", CANDIDATE, solved=set(INSTANCES[:6]))
        calls = _patch_decide(monkeypatch, champion_run, candidate_run)
        monkeypatch.chdir(tmp_path)  # F1: keep data/promotion_decisions out of the repo

        rc = run_mod.main(["--candidate-variant", CANDIDATE, "--champion-path", str(path)])

        assert rc == 0
        decisions = tmp_path / "data" / "promotion_decisions"
        md = (decisions / "decision.md").read_text(encoding="utf-8")
        record = json.loads((decisions / "decision.json").read_text(encoding="utf-8"))
        assert record == calls["decisions"][0]
        assert f"# Promotion decision: {record['decision_id']}" in md

    def test_step_summary_noop_when_env_unset(self, tmp_path, monkeypatch):
        # F1: unset $GITHUB_STEP_SUMMARY (local runs) — silent no-op, False
        # return, nothing appended (same contract as _write_github_output).
        target = tmp_path / "summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
        assert run_mod._write_step_summary("# Promotion decision: promote-x") is True
        assert target.read_text(encoding="utf-8") == "# Promotion decision: promote-x\n"
        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
        assert run_mod._write_step_summary("# ignored") is False
        assert target.read_text(encoding="utf-8") == "# Promotion decision: promote-x\n"


class TestDecideAbortPaths:
    """Every non-eval abort of the decide job returns a non-zero exit."""

    def _patch_audit(self, monkeypatch):
        monkeypatch.setattr(run_mod.audit, "write_decision_record", lambda *a, **k: None)
        monkeypatch.setattr(run_mod.audit, "log_decision_metrics", lambda *a, **k: None)

    def test_config_gap_abort_exits_one(self, tmp_path, monkeypatch, capsys):
        # decision 6: v1 promotes only among trained ServeConfig.variants.
        path = _seed_champion(tmp_path)
        self._patch_audit(monkeypatch)
        out = _github_output(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)  # F1: keep data/promotion_decisions out of the repo

        rc = run_mod.main(["--candidate-variant", "ghost_variant", "--champion-path", str(path)])

        assert rc == 1
        assert "promote=False config-gap" in capsys.readouterr().out
        assert _parse_output(out) == {}

    def test_no_eval_kill_switch_exits_zero(self, tmp_path, monkeypatch, capsys):
        # RUN_MODAL_EVAL=false kill switch (decision 11): no eval, no GITHUB_OUTPUT.
        monkeypatch.setitem(sys.modules, "wandb", None)  # note_gating_off degrades
        rc = run_mod.main(
            [
                "--candidate-variant",
                CANDIDATE,
                "--no-eval",
                "--champion-path",
                str(tmp_path / "x.json"),
            ]
        )
        assert rc == 0
        assert "promote=false gating-off" in capsys.readouterr().out

    def test_no_challenger_abort_exits_one(self, tmp_path, monkeypatch, capsys):
        path = _seed_champion(tmp_path)
        monkeypatch.setattr(run_mod, "_has_challenger_alias", lambda _variant, _serve: False)
        self._patch_audit(monkeypatch)
        monkeypatch.chdir(tmp_path)  # F1: keep data/promotion_decisions out of the repo

        rc = run_mod.main(["--candidate-variant", CANDIDATE, "--champion-path", str(path)])

        assert rc == 1
        assert "no-challenger" in capsys.readouterr().out

    def test_missing_champion_record_abort_exits_one(self, tmp_path, monkeypatch, capsys):
        missing = tmp_path / "no-champion.json"
        monkeypatch.setattr(run_mod, "_download_champion_from_gcs", lambda _dest: False)

        rc = run_mod.main(["--candidate-variant", CANDIDATE, "--champion-path", str(missing)])

        assert rc == 1
        assert "no champion record" in capsys.readouterr().err

    def test_eval_failed_abort_exits_one(self, tmp_path, monkeypatch, capsys):
        path = _seed_champion(tmp_path)
        monkeypatch.setattr(run_mod, "_has_challenger_alias", lambda _variant, _serve: True)
        monkeypatch.setattr(run_mod, "_launch_evals", lambda _pairs, _mode, _base: [])
        monkeypatch.setattr(run_mod, "_wait_for_pair", lambda *a, **k: (None, "eval-failed"))
        self._patch_audit(monkeypatch)
        monkeypatch.chdir(tmp_path)  # F1: keep data/promotion_decisions out of the repo

        rc = run_mod.main(["--candidate-variant", CANDIDATE, "--champion-path", str(path)])

        assert rc == 1
        assert "promote=False eval-failed" in capsys.readouterr().out


# ── champion loading (local file + best-effort GCS fallback) ────────────────


class TestLoadChampion:
    """``_load_champion`` / ``_download_champion_from_gcs`` control flow."""

    def test_invalid_local_record_returns_none(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "champion.json"
        path.write_text("{not json", encoding="utf-8")
        assert run_mod._load_champion(path) is None
        assert "invalid champion record" in capsys.readouterr().err

    def test_gcs_fallback_downloads_on_local_miss(self, tmp_path, monkeypatch):
        path = tmp_path / "champion.json"
        record = _record()
        monkeypatch.setattr(
            run_mod,
            "_download_champion_from_gcs",
            lambda dest: write_champion(dest, record) or True,
        )
        assert run_mod._load_champion(path) == record

    def test_invalid_download_returns_none(self, tmp_path, monkeypatch, capsys):
        path = tmp_path / "champion.json"
        monkeypatch.setattr(
            run_mod,
            "_download_champion_from_gcs",
            lambda dest: Path(dest).write_text("{bad", encoding="utf-8") or True,
        )
        assert run_mod._load_champion(path) is None
        assert "invalid champion record downloaded" in capsys.readouterr().err

    def test_download_success_without_network(self, tmp_path, monkeypatch):
        # Fake google.cloud module: the import + mkdir + return-True lines run,
        # nothing touches the network (create_anonymous_client is pragma'd).
        seen: dict[str, Any] = {}

        class _FakeBlob:
            def download_to_filename(self, destination):
                Path(destination).write_text("{}", encoding="utf-8")

        class _FakeBucket:
            def blob(self, name):
                seen["blob"] = name
                return _FakeBlob()

        class _FakeClient:
            @staticmethod
            def create_anonymous_client():
                return _FakeClient()

            def bucket(self, name):
                seen["bucket"] = name
                return _FakeBucket()

        fake = SimpleNamespace(Client=_FakeClient)
        monkeypatch.setitem(sys.modules, "google.cloud", SimpleNamespace(storage=fake))
        monkeypatch.setitem(sys.modules, "google.cloud.storage", fake)

        dest = tmp_path / "sub" / "champion.json"
        assert run_mod._download_champion_from_gcs(dest) is True
        assert dest.is_file()
        assert seen["bucket"] == run_mod.GCS_BUCKET
        assert seen["blob"] == run_mod.GCS_CHAMPION_BLOB

    def test_download_failure_returns_false(self, tmp_path, monkeypatch):
        class _BrokenClient:
            @staticmethod
            def create_anonymous_client():
                raise RuntimeError("no network")

        fake = SimpleNamespace(Client=_BrokenClient)
        monkeypatch.setitem(sys.modules, "google.cloud", SimpleNamespace(storage=fake))
        monkeypatch.setitem(sys.modules, "google.cloud.storage", fake)
        assert run_mod._download_champion_from_gcs(tmp_path / "c.json") is False


# ── challenger alias check and run-id uniqueness (pure, deterministic) ──────


class TestChallengerAlias:
    """``_has_challenger_alias`` with a fake wandb module."""

    def test_true_when_artifact_carries_challenger_alias(self, monkeypatch):
        class _FakeArtifact:
            aliases = ["challenger", "latest"]

        class _FakeApi:
            def __init__(self, **kwargs) -> None:
                pass

            def artifact(self, name):
                assert name == "model-qwen3-14b-candidate:latest"
                return _FakeArtifact()

        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=_FakeApi))
        assert run_mod._has_challenger_alias("candidate", ServeConfig()) is True

    def test_false_when_wandb_unavailable(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "wandb", None)
        assert run_mod._has_challenger_alias("candidate", ServeConfig()) is False


class TestUniqueRunIds:
    """Two distinct eval run ids even on make_run_id (second-resolution) collision."""

    def test_collision_gets_suffix(self, monkeypatch):
        ids = iter(["eval-20260808-120000", "eval-20260808-120000"])
        monkeypatch.setattr(run_mod, "make_run_id", lambda: next(ids))
        assert run_mod._unique_run_ids() == ["eval-20260808-120000", "eval-20260808-120000-2"]

    def test_distinct_ids_pass_through(self, monkeypatch):
        ids = iter(["eval-20260808-120000", "eval-20260808-120001"])
        monkeypatch.setattr(run_mod, "make_run_id", lambda: next(ids))
        assert run_mod._unique_run_ids() == ["eval-20260808-120000", "eval-20260808-120001"]


# ── deploy job (spec §4.10): pin → deploy → probe → record → rollback ──────


class TestDeployJobE2E:
    """``main(["--deploy", ...])`` with recording fakes — nothing executes."""

    def _patch_deploy(self, monkeypatch, *, probe: float | None = 0.5, deploy_rc: int = 0):
        """Recording fakes for every exec/network edge of the deploy path."""
        calls: dict[str, Any] = {"order": []}

        monkeypatch.setitem(sys.modules, "wandb", None)  # _append_deploy_status degrades
        monkeypatch.setattr(run_mod.deploy_mod, "assert_variant_known", lambda *a, **k: None)

        def _deploy(champion_key, _serve, **kwargs):
            calls["champion_key"] = champion_key
            calls["order"].append("deploy")
            return subprocess.CompletedProcess(
                args=[], returncode=deploy_rc, stdout="deployed", stderr=""
            )

        def _probe(base_url, _token):
            calls["order"].append("health_check")
            if probe is None:
                raise ProbeError(f"chat probe failed: {base_url}")
            return probe

        def _write(_path, record):
            calls["order"].append("write_champion")
            calls["written"] = record

        def _sync(_key, _config):
            calls["order"].append("sync_alias_or_abort")

        def _rollback(previous, _serve, **kwargs):
            calls["order"].append("rollback")
            calls["rollback"] = previous
            return {"outcome": "rollback", "variant": previous.variant}

        monkeypatch.setattr(run_mod.deploy_mod, "deploy", _deploy)
        monkeypatch.setattr(run_mod.deploy_mod, "health_check", _probe)
        monkeypatch.setattr(run_mod.deploy_mod, "sync_alias_or_abort", _sync)
        monkeypatch.setattr(run_mod, "write_champion", _write)
        monkeypatch.setattr(run_mod.deploy_mod, "rollback", _rollback)
        return calls

    def _deploy_args(self, path: Path, *extra: str) -> list[str]:
        return ["--deploy", "--variant", CHAMPION, "--champion-path", str(path), *extra]

    def test_deploy_success_records_new_champion(self, tmp_path, monkeypatch, capsys):
        incumbent = _record(variant="baseline_14b", model_ref="qwen3-14b:baseline_14b")
        champion = _record(previous=incumbent)
        path = _seed_champion(tmp_path, champion)
        calls = self._patch_deploy(monkeypatch)

        rc = run_mod.main(self._deploy_args(path, "--decision-id", "test-123"))

        assert rc == 0
        captured = capsys.readouterr()  # readouterr drains; capture once
        assert "deployed champion=champion_a probe=ok alias=ok champion.json written" in (
            captured.out
        )
        # champion_key is pinned from the requested variant (spec §4.5)
        assert calls["champion_key"] == "qwen3-14b:champion_a"
        # probe-before-record: deploy → probe → write → alias, in code order
        assert calls["order"] == [
            "deploy",
            "health_check",
            "write_champion",
            "sync_alias_or_abort",
        ]
        written = calls["written"]
        assert written.variant == CHAMPION
        assert written.model_ref == CHAMPION_KEY
        assert written.previous == champion  # the replaced record, JSON round-trip
        assert written.previous.previous == incumbent
        assert written.tier == "dev"
        assert written.dataset_run_id == incumbent.dataset_run_id
        # no --f2p-rate/--p2p-rate: incumbent rates carry forward (warning)
        assert written.f2p_rate == incumbent.f2p_rate
        assert written.p2p_rate == incumbent.p2p_rate
        assert "carries forward" in captured.err

    def test_deploy_uses_provided_rates(self, tmp_path, monkeypatch, capsys):
        incumbent = _record(variant="baseline_14b", model_ref="qwen3-14b:baseline_14b")
        path = _seed_champion(tmp_path, _record(previous=incumbent))
        calls = self._patch_deploy(monkeypatch)

        rc = run_mod.main(
            self._deploy_args(
                path, "--decision-id", "test-123", "--f2p-rate", "0.4", "--p2p-rate", "1.0"
            )
        )

        assert rc == 0
        assert calls["written"].f2p_rate == 0.4
        assert calls["written"].p2p_rate == 1.0
        assert "carries forward" not in capsys.readouterr().err

    def test_deploy_probe_failure_rolls_back_and_exits_one(self, tmp_path, monkeypatch, capsys):
        incumbent = _record(variant="baseline_14b", model_ref="qwen3-14b:baseline_14b")
        champion = _record(previous=incumbent)
        path = _seed_champion(tmp_path, champion)
        calls = self._patch_deploy(monkeypatch, probe=None)

        rc = run_mod.main(self._deploy_args(path, "--decision-id", "test-123"))

        assert rc == 1
        assert "deploy probe failed" in capsys.readouterr().err
        # rollback re-promotes the displaced incumbent `old`, not old.previous
        assert calls["order"] == ["deploy", "health_check", "rollback", "write_champion"]
        assert calls["rollback"] == champion
        assert calls["written"] == champion  # probe path: rewrite is a no-op

    def test_deploy_failure_without_previous_exits_one(self, tmp_path, monkeypatch, capsys):
        # First-generation champion (previous=None): the displaced incumbent is
        # itself, so rollback still re-promotes it (spec decision 7).
        champion = _record()
        path = _seed_champion(tmp_path, champion)
        calls = self._patch_deploy(monkeypatch, probe=None)

        rc = run_mod.main(self._deploy_args(path, "--decision-id", "test-123"))

        assert rc == 1
        assert calls["order"] == ["deploy", "health_check", "rollback", "write_champion"]
        assert calls["rollback"] == champion

    def test_deploy_command_failure_rolls_back(self, tmp_path, monkeypatch, capsys):
        incumbent = _record(variant="baseline_14b", model_ref="qwen3-14b:baseline_14b")
        champion = _record(previous=incumbent)
        path = _seed_champion(tmp_path, champion)
        calls = self._patch_deploy(monkeypatch, deploy_rc=1)

        rc = run_mod.main(self._deploy_args(path, "--decision-id", "test-123"))

        assert rc == 1
        assert calls["order"] == ["deploy", "rollback", "write_champion"]
        assert calls["rollback"] == champion  # displaced incumbent, not old.previous
        assert calls["written"] == champion  # champion.json restored to old
        assert "rollback to champion_a" in capsys.readouterr().out

    def test_deploy_rollback_failure_still_exits_one(self, tmp_path, monkeypatch, capsys):
        incumbent = _record(variant="baseline_14b", model_ref="qwen3-14b:baseline_14b")
        path = _seed_champion(tmp_path, _record(previous=incumbent))
        calls = self._patch_deploy(monkeypatch, probe=None)
        calls["_rollback_raise"] = True

        def _broken_rollback(previous, _serve, **kwargs):
            calls["order"].append("rollback")
            raise RuntimeError("modal down")

        monkeypatch.setattr(run_mod.deploy_mod, "rollback", _broken_rollback)

        rc = run_mod.main(self._deploy_args(path, "--decision-id", "test-123"))

        assert rc == 1
        assert "rollback failed: modal down" in capsys.readouterr().err
        # no champion.json rewrite when the rollback itself fails
        assert calls["order"] == ["deploy", "health_check", "rollback"]

    def test_deploy_alias_sync_failure_rolls_back(self, tmp_path, monkeypatch, capsys):
        incumbent = _record(variant="baseline_14b", model_ref="qwen3-14b:baseline_14b")
        champion = _record(previous=incumbent)
        path = _seed_champion(tmp_path, champion)
        calls = self._patch_deploy(monkeypatch)

        def _broken_sync(_key, _config):
            calls["order"].append("sync_alias_or_abort")
            raise RuntimeError("alias sync failed")

        monkeypatch.setattr(run_mod.deploy_mod, "sync_alias_or_abort", _broken_sync)

        rc = run_mod.main(self._deploy_args(path, "--decision-id", "test-123"))

        assert rc == 1
        assert "deploy alias sync failed" in capsys.readouterr().err
        assert calls["order"] == [
            "deploy",
            "health_check",
            "write_champion",
            "sync_alias_or_abort",
            "rollback",
            "write_champion",
        ]
        # champion.json restored to the displaced incumbent, with its own chain
        assert calls["written"] == champion
        assert calls["rollback"] == champion

    def test_deploy_config_gap_aborts(self, tmp_path, monkeypatch, capsys):
        path = _seed_champion(tmp_path)

        rc = run_mod.main(["--deploy", "--variant", "ghost_variant", "--champion-path", str(path)])

        assert rc == 1
        assert "deploy aborted" in capsys.readouterr().err

    def test_deploy_missing_champion_record_aborts(self, tmp_path, monkeypatch, capsys):
        rc = run_mod.main(
            [
                "--deploy",
                "--variant",
                CANDIDATE,
                "--champion-path",
                str(tmp_path / "none.json"),
                "--decision-id",
                "test-123",
            ]
        )
        assert rc == 1
        assert "cannot read champion record" in capsys.readouterr().err


# ── deploy_status artifact append (best-effort, never raises) ──────────────


class TestAppendDeployStatus:
    """``_append_deploy_status`` with a fake wandb module."""

    def test_success_updates_artifact_metadata(self, monkeypatch):
        artifacts: list[Any] = []

        class _FakeArtifact:
            def __init__(self, name):
                artifacts.append(self)
                self.metadata = {}

            def save(self):
                pass

        class _FakeApi:
            def __init__(self, **kwargs) -> None:
                pass

            def artifact(self, name):
                return _FakeArtifact(name)

        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=_FakeApi))
        run_mod._append_deploy_status("dec-1", "success")  # must not raise
        assert len(artifacts) == 1
        assert artifacts[0].metadata["deploy_status"] == "success"
        assert artifacts[0].metadata["deployed_at"]  # ISO timestamp present

    def test_noop_without_decision_id(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "wandb", None)
        run_mod._append_deploy_status(None, "success")  # must not raise

    def test_degrades_without_wandb(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "wandb", None)
        run_mod._append_deploy_status("dec-1", "rollback")  # must not raise


# ── CLI surface ────────────────────────────────────────────────────────────


class TestCliSurface:
    """Argument validation and the module entrypoint."""

    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["--deploy"],
            ["--deploy", "--variant", "v", "--candidate-variant", "c"],
            ["--deploy", "--variant", "v", "--no-eval"],
        ],
    )
    def test_invalid_argument_combinations_exit(self, argv):
        with pytest.raises(SystemExit):
            run_mod.main(argv)

    def test_module_entrypoint_raises_system_exit(self):
        # `python -m promotion.run` with no args: argparse errors, exit 2.
        with pytest.raises(SystemExit):
            runpy.run_module("promotion.run", run_name="__main__")
