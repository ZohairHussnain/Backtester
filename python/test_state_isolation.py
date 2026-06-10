"""Phase 1 tests: sim/paper state isolation, per-file duplicate guard,
sim-state reset, and one-time legacy migration.

Zero-dependency assert harness (mirrors the C++ test style).
Run:  python test_state_isolation.py

These tests use temporary files only. They do NOT read or write the real
output/ state files.
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from portfolio import Portfolio
import run_daily

_passed = 0
_failed = 0


def check(cond: bool, msg: str) -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS: {msg}")
    else:
        _failed += 1
        print(f"  FAIL: {msg}")


# ---------------------------------------------------------------------------
# Item 1: separate state files per mode
# ---------------------------------------------------------------------------

def test_state_file_for_mode_mapping():
    print("\n[A] state_file_for_mode mapping")
    sim = config.state_file_for_mode("sim")
    dry = config.state_file_for_mode("ibkr_dry_run")
    paper = config.state_file_for_mode("ibkr_paper")
    unknown = config.state_file_for_mode("something_new")

    check(sim != paper, "sim and paper resolve to different files")
    check(dry == paper, "ibkr_dry_run shares the paper state file")
    check(sim == config.PORTFOLIO_STATE_FILE_SIM, "sim -> PORTFOLIO_STATE_FILE_SIM")
    check(paper == config.PORTFOLIO_STATE_FILE_PAPER, "ibkr_paper -> PORTFOLIO_STATE_FILE_PAPER")
    check("sim" in sim.name, "sim file name marks it as sim")
    check("paper" in paper.name, "paper file name marks it as paper")
    check(unknown == config.PORTFOLIO_STATE_FILE_SIM, "unknown mode falls back to sim (never paper)")


def test_sim_does_not_contaminate_paper():
    print("\n[B] sim writes do not touch paper state")
    with tempfile.TemporaryDirectory() as d:
        sim = Path(d) / "portfolio_state.sim.json"
        paper = Path(d) / "portfolio_state.paper.json"

        p_sim = Portfolio(sim, 10000.0)
        p_sim.record_entry("AAPL", 10, 100.0, 95.0, 110.0, "2026-06-06", 0.35)
        p_sim.save()

        check(sim.exists(), "sim save created the sim file")
        check(not paper.exists(), "paper file was NOT created by a sim run")

        # A paper portfolio on its own file sees none of the sim positions.
        p_paper = Portfolio(paper, 10000.0)
        check("AAPL" not in p_paper.open_positions, "paper portfolio has no sim position")
        check(p_paper.cash == 10000.0, "paper cash untouched by sim fill")

        # And the sim state really did persist to the sim file.
        p_sim_reload = Portfolio(sim, 10000.0)
        check("AAPL" in p_sim_reload.open_positions, "sim position persisted to sim file")


# ---------------------------------------------------------------------------
# Item 4: duplicate-run guard scoped per state file
# ---------------------------------------------------------------------------

def test_duplicate_guard_scoped_per_file():
    print("\n[C] duplicate guard is scoped per state file")
    with tempfile.TemporaryDirectory() as d:
        sim = Path(d) / "portfolio_state.sim.json"
        paper = Path(d) / "portfolio_state.paper.json"
        today = datetime.now().strftime("%Y-%m-%d")

        # A sim run saves -> sim file's last_updated advances to now.
        p_sim = Portfolio(sim, 10000.0)
        p_sim.save()

        # Paper portfolio is fresh (last_updated == "").
        p_paper = Portfolio(paper, 10000.0)

        check(run_daily.check_duplicate(p_sim, today) is True,
              "sim flagged as duplicate after its own save")
        check(run_daily.check_duplicate(p_paper, today) is False,
              "paper NOT gated by today's sim run")


# ---------------------------------------------------------------------------
# Item 5: --reset-sim-state
# ---------------------------------------------------------------------------

def test_reset_removes_sim_only():
    print("\n[D] reset_sim_state removes sim files only")
    with tempfile.TemporaryDirectory() as d:
        sim = Path(d) / "portfolio_state.sim.json"
        paper = Path(d) / "portfolio_state.paper.json"
        sim_bak = Path(str(sim) + ".bak")

        sim.write_text("{}")
        sim_bak.write_text("{}")
        paper.write_text("{}")

        removed = run_daily.reset_sim_state(sim=sim)

        check(not sim.exists(), "sim state file removed")
        check(not sim_bak.exists(), "sim .bak sidecar removed")
        check(paper.exists(), "paper state file NOT removed by reset")
        check(len(removed) == 2, "reset reported exactly the 2 removed files")


def test_reset_on_clean_state_is_noop():
    print("\n[E] reset on already-clean state is a no-op")
    with tempfile.TemporaryDirectory() as d:
        sim = Path(d) / "portfolio_state.sim.json"
        removed = run_daily.reset_sim_state(sim=sim)
        check(removed == [], "reset returns empty list when nothing to remove")


# ---------------------------------------------------------------------------
# Item 1 (rollout): one-time legacy migration
# ---------------------------------------------------------------------------

def test_migrate_legacy_to_sim_only():
    print("\n[F] legacy state migrates to sim, never to paper")
    with tempfile.TemporaryDirectory() as d:
        legacy = Path(d) / "portfolio_state.json"
        sim = Path(d) / "portfolio_state.sim.json"
        paper = Path(d) / "portfolio_state.paper.json"

        legacy.write_text(json.dumps({
            "cash": 1234.0,
            "open_positions": {"X": {"shares": 1, "entry_price": 10.0}},
            "trade_history": [],
            "last_updated": "2026-06-06T10:00:00",
        }))

        migrated = run_daily.migrate_legacy_sim_state(legacy=legacy, sim=sim)
        check(migrated is True, "legacy migrated to sim")
        check(sim.exists(), "sim file created from legacy")
        data = json.loads(sim.read_text())
        check("X" in data.get("open_positions", {}), "legacy positions copied into sim")
        check(not paper.exists(), "paper file NOT created by migration")


def test_migrate_is_idempotent():
    print("\n[G] migration does not re-copy once sim exists")
    with tempfile.TemporaryDirectory() as d:
        legacy = Path(d) / "portfolio_state.json"
        sim = Path(d) / "portfolio_state.sim.json"

        legacy.write_text(json.dumps({"cash": 1.0, "open_positions": {}}))
        first = run_daily.migrate_legacy_sim_state(legacy=legacy, sim=sim)

        # Mutate sim so we can detect any unwanted overwrite.
        sim.write_text(json.dumps({"cash": 999.0, "open_positions": {}}))
        second = run_daily.migrate_legacy_sim_state(legacy=legacy, sim=sim)

        check(first is True, "first migration performed")
        check(second is False, "second migration is a no-op")
        check(json.loads(sim.read_text())["cash"] == 999.0, "existing sim file not overwritten")


def test_migrate_noop_without_legacy():
    print("\n[H] no migration when there is no legacy file")
    with tempfile.TemporaryDirectory() as d:
        legacy = Path(d) / "portfolio_state.json"   # does not exist
        sim = Path(d) / "portfolio_state.sim.json"
        migrated = run_daily.migrate_legacy_sim_state(legacy=legacy, sim=sim)
        check(migrated is False, "no legacy -> no migration")
        check(not sim.exists(), "sim file not created when there is nothing to migrate")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  PHASE 1 STATE-ISOLATION TESTS")
    print("=" * 60)

    test_state_file_for_mode_mapping()
    test_sim_does_not_contaminate_paper()
    test_duplicate_guard_scoped_per_file()
    test_reset_removes_sim_only()
    test_reset_on_clean_state_is_noop()
    test_migrate_legacy_to_sim_only()
    test_migrate_is_idempotent()
    test_migrate_noop_without_legacy()

    print("\n" + "=" * 60)
    print(f"  RESULTS: {_passed} passed, {_failed} failed")
    print("=" * 60)
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
