#!/usr/bin/env python3
"""Nightly consolidation — ZERO's sleep. Replays the day, keeps what mattered.

Steps (each independent: one failing never blocks the rest; all idempotent —
re-running the same night is safe):

  distill   unconsolidated episodes -> long-term memory, with importance
            weighted by reward (turns) and surprise (scene events); episodes
            are then marked consolidated (the idempotency marker).
  decay     memory.consolidate(): old, unimportant, never-recalled memories
            fade (existing mechanism; reward already raised what mattered).
  corpus    reward-weighted training export: data/train/chat.jsonl rebuilt
            in full (overwrite = idempotent). Sessions that went well are
            oversampled; sessions that went badly are dropped.
  vocab     if the LIVE detector vocabulary (server/vision/vocab.runtime.txt,
            grown via POST /perceive/vocab) is newer than the Pi's ONNX
            export, re-export so the Pi reflex eye learns the new words.
  objects   learned-objects store maintenance (VACUUM; caps are enforced at
            teach time). NOTE: raw crops aren't stored, so embeddings cannot
            be recomputed offline — flagged, not faked.
  lora      run the configured fine-tune command (learning.training.cmd) on
            the exported corpus, if set. Unset (default) -> SKIPPED: LoRA
            training needs GPU headroom this card doesn't have live.

State: data/consolidation/journal.json (per-step ok/ts/detail). A flock on
data/consolidation/.lock guarantees a single instance. Exit code: number of
failed steps (0 = clean night) — visible to systemd.

Run nightly via scripts/systemd/zero-consolidate.timer, or by hand:
    .venv/bin/python scripts/consolidate.py [--dry-run] [--steps distill decay ...]
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from zero.config import load_config                     # noqa: E402
from zero.utils.logging import get_logger               # noqa: E402

log = get_logger("consolidate")

STATE_DIR = REPO_ROOT / "data" / "consolidation"


# ── steps ────────────────────────────────────────────────────────────────────
def step_distill(cfg, dry: bool) -> str:
    from zero.factory import build_memory
    from zero.learning.episodes import EpisodeStore

    episodes = EpisodeStore(str(cfg.resolve_path(
        "learning.episodes.db_path", "zero_episodes.sqlite")))
    memory = build_memory(cfg)
    if memory is None:
        return "memory disabled — nothing to distill into"
    pending = episodes.unconsolidated()
    if not pending:
        return "no unconsolidated episodes"

    # Turns: one episodic memory per (person, day) with reward-weighted
    # importance — the session's emotional residue, not its transcript.
    by_person_day: dict[tuple, list[dict]] = {}
    scene: list[dict] = []
    for ep in pending:
        if ep["kind"] == "turn":
            day = time.strftime("%Y-%m-%d", time.localtime(ep["ts"]))
            by_person_day.setdefault((ep["person_id"], day), []).append(ep)
        elif ep["kind"] == "scene":
            scene.append(ep)
    written = 0
    for (pid, day), turns in by_person_day.items():
        rewards = [t["reward"] for t in turns if t["reward"] is not None]
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        feel = ("it went well" if mean_r > 0.15 else
                "it went badly" if mean_r < -0.15 else "")
        summary = (f"Talked on {day} ({len(turns)} exchange"
                   f"{'s' if len(turns) != 1 else ''}"
                   + (f"; {feel}" if feel else "") + ").")
        importance = max(1.0, min(9.0, 4.0 + 3.0 * mean_r))
        if not dry:
            memory.add_episode(summary, person_id=pid, importance=importance)
        written += 1
    # Scene events: only the genuinely surprising made it here (gated);
    # importance scales with bits of surprise.
    for ep in sorted(scene, key=lambda e: -(e["surprise"] or 0))[:10]:
        p = ep["payload"]
        summary = (f"Noticed: the {p.get('label', 'something')} "
                   f"{p.get('event', 'changed')} "
                   f"(unusual — {ep['surprise']:.1f} bits).")
        importance = max(3.0, min(9.0, 3.0 + (ep["surprise"] or 0) / 2.0))
        if not dry:
            memory.add_episode(summary, importance=importance)
        written += 1
    if not dry:
        episodes.mark_consolidated([e["id"] for e in pending])
    return (f"{written} memories from {len(pending)} episodes "
            f"({len(by_person_day)} person-days, {len(scene)} scene events)")


def step_decay(cfg, dry: bool) -> str:
    from zero.factory import build_memory

    memory = build_memory(cfg)
    if memory is None:
        return "memory disabled"
    if dry:
        return "dry-run: skipped"
    stats = memory.consolidate()
    return f"forgot {stats.get('forgotten', 0)}, insights {stats.get('insights', 0)}"


def step_corpus(cfg, dry: bool) -> str:
    """Reward-weighted rebuild of the training set. Weighting: a corpus
    record inherits the mean reward of the same speaker's turn episodes in
    the hours before it was flushed. reward <= -0.4 -> dropped; >= +0.3 ->
    doubled; else kept once."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from export_corpus import SYSTEM_TEMPLATE, to_chat

    from zero.learning.episodes import EpisodeStore

    src = REPO_ROOT / "data" / "corpus" / "interactions.jsonl"
    out = REPO_ROOT / "data" / "train" / "chat.jsonl"
    if not src.exists():
        return "no corpus yet"
    episodes = EpisodeStore(str(cfg.resolve_path(
        "learning.episodes.db_path", "zero_episodes.sqlite")))
    turns = episodes.recent("turn", within_s=45 * 86400.0, limit=100000)

    def session_reward(speaker, ts: float) -> float | None:
        window = [t["reward"] for t in turns
                  if t["person_id"] == (speaker if isinstance(speaker, int)
                                        and speaker > 0 else None)
                  and ts - 6 * 3600 <= t["ts"] <= ts + 3600
                  and t["reward"] is not None]
        return sum(window) / len(window) if window else None

    kept = dropped = doubled = 0
    lines: list[str] = []
    system = SYSTEM_TEMPLATE if isinstance(SYSTEM_TEMPLATE, str) else str(SYSTEM_TEMPLATE)
    for raw in src.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(raw)
        except ValueError:
            continue
        example = to_chat(rec, system)
        if example is None:
            continue
        r = session_reward(rec.get("speaker"), rec.get("ts", 0.0))
        copies = 1
        if r is not None and r <= -0.4:
            dropped += 1
            continue
        if r is not None and r >= 0.3:
            copies, doubled = 2, doubled + 1
        kept += 1
        lines.extend([json.dumps(example, ensure_ascii=False)] * copies)
    if not dry:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""),
                       encoding="utf-8")
        tmp.replace(out)
    return f"{kept} kept ({doubled} doubled, {dropped} dropped) -> {out.name}"


def step_vocab(cfg, dry: bool) -> str:
    runtime = REPO_ROOT / "server" / "vision" / "vocab.runtime.txt"
    seed = REPO_ROOT / "scripts" / "vocab_indoor.txt"
    vocab = runtime if runtime.exists() else seed
    model_path = cfg.resolve_path("vision.detect.model_path",
                                  "yolov8s-worldv2-480.onnx")
    imgsz = int(cfg.get("vision.detect.imgsz", 480))
    if model_path.exists() and model_path.stat().st_mtime >= vocab.stat().st_mtime:
        return f"ONNX newer than {vocab.name} — no re-export needed"
    if dry:
        return f"dry-run: would re-export {model_path.name} from {vocab.name}"
    # The exporter writes <checkpoint>.onnx next to the weights; export in a
    # temp dir, then move onto the deployed filename atomically.
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        ckpt = cfg.get("world.vocab_export.checkpoint", "yolov8s-worldv2.pt")
        r = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "export_yolo_onnx.py"),
             ckpt, str(imgsz), "--vocab", str(vocab)],
            cwd=td, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"export failed: {r.stderr[-400:]}")
        exported = Path(td) / Path(ckpt).with_suffix(".onnx").name
        names = exported.with_suffix("").with_suffix(".names.json")
        shutil.move(str(exported), str(model_path))
        shutil.move(str(names), str(model_path.with_suffix("")
                                    .with_suffix(".names.json")))
    return f"re-exported {model_path.name} from {vocab.name} ({imgsz}px)"


def step_objects(cfg, dry: bool) -> str:
    import sqlite3

    path = cfg.resolve_path("learning.objects.db_path", "zero_objects.sqlite")
    if path is None or not path.exists():
        return "no learned-objects store"
    if dry:
        return "dry-run: skipped"
    db = sqlite3.connect(str(path))
    names = db.execute("SELECT COUNT(DISTINCT name) FROM objects").fetchone()[0]
    db.execute("VACUUM")
    db.close()
    # Honest limitation: raw crops are not stored, so embeddings cannot be
    # recomputed against a new embedder offline — re-teaching does that live.
    return f"vacuumed ({names} learned names); re-embedding N/A (no raw crops)"


def step_lora(cfg, dry: bool) -> str:
    cmd = cfg.get("learning.training.cmd")
    if not cmd:
        return ("SKIPPED: no learning.training.cmd configured (LoRA needs GPU "
                "headroom — see docs/TRAINING.md)")
    if dry:
        return f"dry-run: would run {cmd!r}"
    r = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT),
                       capture_output=True, text=True, timeout=6 * 3600)
    if r.returncode != 0:
        raise RuntimeError(f"training cmd failed ({r.returncode}): "
                           f"{r.stderr[-400:]}")
    return "training command completed"


STEPS = [("distill", step_distill), ("decay", step_decay),
         ("corpus", step_corpus), ("vocab", step_vocab),
         ("objects", step_objects), ("lora", step_lora)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--steps", nargs="*", default=None,
                    help="subset of steps to run (default: all)")
    args = ap.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_f = (STATE_DIR / ".lock").open("w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[consolidate] another instance is running — exiting")
        return 0

    cfg = load_config()
    journal_path = STATE_DIR / "journal.json"
    journal: dict = {}
    if journal_path.exists():
        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
        except ValueError:
            journal = {}

    failures = 0
    for name, fn in STEPS:
        if args.steps and name not in args.steps:
            continue
        t0 = time.time()
        try:
            detail = fn(cfg, args.dry_run)
            ok = True
        except Exception as e:
            detail, ok = f"{type(e).__name__}: {e}", False
            failures += 1
            log.warning("step %s FAILED: %s", name, detail)
        journal[name] = {"ok": ok, "ts": time.time(),
                         "took_s": round(time.time() - t0, 1),
                         "detail": str(detail)[:400]}
        print(f"[consolidate] {name:8s} {'ok  ' if ok else 'FAIL'} "
              f"({journal[name]['took_s']}s) {detail}")
    if not args.dry_run:
        tmp = journal_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(journal, indent=2), encoding="utf-8")
        tmp.replace(journal_path)
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
