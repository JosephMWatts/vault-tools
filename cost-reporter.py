#!/usr/bin/env python3
"""
cost-reporter.py
Post-hoc enrichment agent. Walks the Agent Runs corpus, computes USD per
run-log from token counts and _pricing.yml rates, and writes the result
into each run-log's frontmatter under a namespaced enrichment: block.

Implements the Harness / Enrichment Layer Pattern (Figure 49.1). The
harness layer is immutable: cost-reporter NEVER edits any field above
the enrichment: boundary. Re-runs are idempotent — the enrichment block
is replaced wholesale on each pass.

Mutations are minimal-modification text patches, not YAML round-trips.
The toolkit deliberately avoids PyYAML; the harness layer is the source
of truth and any reformatting risk is unacceptable.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import harness

VAULT_ROOT      = Path("/Users/josephwatts/Documents/joseph_vault")
PRICING_PATH    = VAULT_ROOT / "_pricing.yml"
AGENT_RUNS_DIR  = VAULT_ROOT / "Agent Runs"

AGENT_ID        = "cost-reporter"
AGENT_VERSION   = "1.0.0"
PRICING_SCHEMA_EXPECTED = 1

ALLOWED_FIELDS = {
    "cost_usd",
    "cost_status",
    "priced_at",
    "pricing_schema_version",
}


# --- Pricing -----------------------------------------------------------
def load_pricing(path: Path) -> tuple[int, dict[str, dict[str, float]]]:
    """Hand-rolled parser for the _pricing.yml subset cost-reporter depends on.

    The file has a fixed two-level shape: top-level scalars and a single
    `models:` mapping with one nested level. A general YAML parser would
    pull in a dependency the toolkit deliberately avoids, so a small
    indent-aware reader does the job.
    """
    schema_version: int | None = None
    models: dict[str, dict[str, float]] = {}
    current_model: str | None = None
    in_models = False

    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped_full = raw.rstrip()
        if not stripped_full or stripped_full.lstrip().startswith("#"):
            continue
        indent = len(stripped_full) - len(stripped_full.lstrip(" "))
        content = stripped_full.strip()

        if indent == 0:
            in_models = False
            current_model = None
            if content == "models:":
                in_models = True
                continue
            if content.startswith("schema_version:"):
                schema_version = int(content.split(":", 1)[1].strip())
            continue

        if not in_models:
            continue

        if indent == 2 and content.endswith(":"):
            current_model = content[:-1].strip()
            models[current_model] = {}
            continue

        if indent == 4 and current_model is not None and ":" in content:
            key, _, val = content.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key in ("input_per_million_usd", "output_per_million_usd"):
                try:
                    models[current_model][key] = float(val)
                except ValueError:
                    pass

    if schema_version is None:
        raise RuntimeError(f"pricing file {path} missing schema_version")
    return schema_version, models


# --- Frontmatter slicing -----------------------------------------------
def split_frontmatter(text: str) -> tuple[list[str], list[str]]:
    """Return (fm_lines, post_lines). fm_lines excludes the opening ---
    and closing --- delimiters. post_lines starts at the closing --- and
    includes everything after it. The opening --- is dropped because it
    is always written back unchanged."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise RuntimeError("file has no opening frontmatter delimiter")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return lines[1:i], lines[i:]
    raise RuntimeError("file has no closing frontmatter delimiter")


def _raw_value(fm_lines: list[str], key: str) -> str | None:
    """Return the raw post-colon string for a top-level fm key, or None
    if the key is absent. Top-level means column 0, no leading whitespace."""
    prefix = f"{key}:"
    for raw in fm_lines:
        line = raw.rstrip("\n")
        if line.startswith(prefix) and (len(line) == len(prefix) or line[len(prefix)] in (" ", "\t")):
            return line[len(prefix):].strip()
    return None


def get_string(fm_lines: list[str], key: str) -> str | None:
    raw = _raw_value(fm_lines, key)
    if raw is None or raw == "" or raw == "null":
        return None
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        return raw[1:-1]
    return raw


def get_int(fm_lines: list[str], key: str) -> int | None:
    raw = _raw_value(fm_lines, key)
    if raw is None or raw == "" or raw == "null":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def locate_enrichment_block(fm_lines: list[str]) -> tuple[int, int] | None:
    """Return (start, end) indices of an existing enrichment: block, or
    None. start is the 'enrichment:' line, end is one past the last
    indented child line."""
    start = None
    for i, raw in enumerate(fm_lines):
        line = raw.rstrip("\n")
        if line == "enrichment:" or line.startswith("enrichment:"):
            if not raw.startswith((" ", "\t")) and line.split(":", 1)[0] == "enrichment":
                start = i
                break
    if start is None:
        return None
    end = start + 1
    while end < len(fm_lines) and fm_lines[end].startswith((" ", "\t")):
        end += 1
    return start, end


# --- Cost computation --------------------------------------------------
def compute_cost(
    model_id: str | None,
    tokens_in: int | None,
    tokens_out: int | None,
    pricing: dict[str, dict[str, float]],
) -> tuple[float | None, str | None]:
    """Returns (cost_usd, cost_status). See docstring at module top for
    the three cases."""
    if model_id is None:
        return None, "rule-engine"
    if model_id not in pricing:
        return None, f"unpriced: {model_id}"
    rate = pricing[model_id]
    ti = tokens_in or 0
    to = tokens_out or 0
    cost = (
        ti * rate.get("input_per_million_usd", 0.0)
        + to * rate.get("output_per_million_usd", 0.0)
    ) / 1_000_000
    return cost, None


# --- Enrichment block construction -------------------------------------
def build_enrichment_block(
    cost_usd: float | None,
    cost_status: str | None,
    priced_at_iso: str,
    pricing_schema: int,
) -> list[str]:
    """Build the enrichment: block as a list of lines (each ending in \n)."""
    out = ["enrichment:\n"]
    if cost_usd is not None:
        out.append(f"  cost_usd: {cost_usd:.6f}\n")
    if cost_status is not None:
        out.append(f"  cost_status: {harness._yaml_quote(cost_status)}\n")
    out.append(f"  priced_at: {harness._yaml_quote(priced_at_iso)}\n")
    out.append(f"  pricing_schema_version: {pricing_schema}\n")
    return out


# --- Smoke test + atomic write -----------------------------------------
def _strip_enrichment(fm_lines: list[str]) -> list[str]:
    bounds = locate_enrichment_block(fm_lines)
    if bounds is None:
        return list(fm_lines)
    s, e = bounds
    return fm_lines[:s] + fm_lines[e:]


def _validate_enrichment_fields(block_lines: list[str], filename: str) -> None:
    """Hard fail if any field outside ALLOWED_FIELDS appears in the new
    enrichment block."""
    # block_lines[0] is "enrichment:\n"; children are indented.
    for raw in block_lines[1:]:
        if not raw.startswith((" ", "\t")):
            continue
        body = raw.lstrip(" \t").rstrip("\n")
        if not body or ":" not in body:
            continue
        key = body.split(":", 1)[0].strip()
        if key not in ALLOWED_FIELDS:
            raise RuntimeError(
                f"smoke test failed for {filename}: enrichment field "
                f"{key!r} is not in ALLOWED_FIELDS"
            )


def write_run_log_enrichment(
    path: Path,
    old_fm_lines: list[str],
    new_fm_lines: list[str],
    post_lines: list[str],
    new_block_range: tuple[int, int],
) -> None:
    """Smoke-test then atomically write. Hard fails on any drift in the
    harness layer (anything outside the enrichment block)."""
    # 1. Harness-layer byte identity: pre vs. post, modulo enrichment block.
    old_harness = "".join(_strip_enrichment(old_fm_lines))
    new_harness_lines = list(new_fm_lines)
    s, e = new_block_range
    del new_harness_lines[s:e]
    new_harness = "".join(new_harness_lines)
    if old_harness != new_harness:
        raise RuntimeError(
            f"smoke test failed for {path.name}: harness layer drift "
            f"detected — refusing to write"
        )

    # 2. Allowed-fields check on the new enrichment block.
    new_block = new_fm_lines[s:e]
    _validate_enrichment_fields(new_block, path.name)

    # 3. Atomic write: temp file in same directory, then os.replace.
    new_text = "---\n" + "".join(new_fm_lines) + "".join(post_lines)
    fd, tmp_str = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".cost-reporter-",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise


# --- Per-file processing -----------------------------------------------
def process_run_log(
    path: Path,
    pricing: dict[str, dict[str, float]],
    pricing_schema: int,
    priced_at_iso: str,
    dry_run: bool,
) -> tuple[float | None, str | None, str | None]:
    """Compute cost for one run-log, write the enrichment block (unless
    --dry-run). Returns (cost_usd, cost_status, model_id)."""
    original = path.read_text(encoding="utf-8")
    fm_lines, post_lines = split_frontmatter(original)

    model_id  = get_string(fm_lines, "model_id")
    tokens_in = get_int(fm_lines, "token_cost_input")
    tokens_ou = get_int(fm_lines, "token_cost_output")

    cost_usd, cost_status = compute_cost(model_id, tokens_in, tokens_ou, pricing)

    new_block = build_enrichment_block(
        cost_usd, cost_status, priced_at_iso, pricing_schema)

    existing = locate_enrichment_block(fm_lines)
    if existing is None:
        new_fm = fm_lines + new_block
        block_range = (len(fm_lines), len(fm_lines) + len(new_block))
    else:
        s, e = existing
        new_fm = fm_lines[:s] + new_block + fm_lines[e:]
        block_range = (s, s + len(new_block))

    if not dry_run:
        write_run_log_enrichment(path, fm_lines, new_fm, post_lines, block_range)

    return cost_usd, cost_status, model_id


# --- Entry point -------------------------------------------------------
def _print_dry_run_table(rows: list[tuple[str, str | None, float | None, str | None]]) -> None:
    print()
    header = f"{'Run log':<58}  {'Model':<22}  {'USD':>10}  Status"
    print(header)
    print("-" * len(header))
    for name, model_id, cost, status in rows:
        cost_disp = f"{cost:.6f}" if cost is not None else "-"
        print(
            f"{name[:58]:<58}  {(model_id or '-')[:22]:<22}  "
            f"{cost_disp:>10}  {status or ''}"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute USD per run-log and write enrichment blocks."
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Parse and compute, print a summary table, write nothing.",
    )
    args = ap.parse_args()

    started_at = datetime.now().astimezone()

    try:
        pricing_schema, pricing = load_pricing(PRICING_PATH)
        if pricing_schema != PRICING_SCHEMA_EXPECTED:
            raise RuntimeError(
                f"_pricing.yml schema_version is {pricing_schema}, "
                f"this script expects {PRICING_SCHEMA_EXPECTED}"
            )

        run_logs = sorted(
            p for p in AGENT_RUNS_DIR.glob("*.md") if p.name != "_index.md"
        )

        # One timestamp per cost-reporter run, shared across all priced_at fields.
        priced_at_iso = (
            datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        )

        total = priced = rule_engine = unpriced = 0
        unpriced_models: set[str] = set()
        rows: list[tuple[str, str | None, float | None, str | None]] = []

        for p in run_logs:
            total += 1
            cost_usd, cost_status, model_id = process_run_log(
                p, pricing, pricing_schema, priced_at_iso, args.dry_run)
            rows.append((p.name, model_id, cost_usd, cost_status))
            if cost_status is None:
                priced += 1
            elif cost_status == "rule-engine":
                rule_engine += 1
            else:
                unpriced += 1
                if model_id:
                    unpriced_models.add(model_id)
                print(
                    f"WARNING unpriced model_id {model_id!r} in {p.name}",
                    file=sys.stderr,
                )

        if args.dry_run:
            _print_dry_run_table(rows)

        summary = (
            f"Walked {total} run-log(s): {priced} priced, "
            f"{rule_engine} rule-engine, {unpriced} unpriced"
        )
        if unpriced_models:
            summary += f" (unpriced models: {', '.join(sorted(unpriced_models))})"
        summary += "."
        print(summary)

        if args.dry_run:
            return

        harness.write_run_log(
            vault_path=str(VAULT_ROOT),
            agent_id=AGENT_ID,
            agent_version=AGENT_VERSION,
            started_at=started_at,
            completed_at=datetime.now().astimezone(),
            status="success",
            trigger="manual",
            input_summary=f"Walked {total} run-log(s) in Agent Runs/",
            output_summary=summary,
            output_paths=[],
            model_id=None,
        )

    except Exception as e:
        if not args.dry_run:
            harness.write_run_log(
                vault_path=str(VAULT_ROOT),
                agent_id=AGENT_ID,
                agent_version=AGENT_VERSION,
                started_at=started_at,
                completed_at=datetime.now().astimezone(),
                status="failed",
                trigger="manual",
                input_summary="cost-reporter pass over Agent Runs/",
                output_summary=str(e),
                output_paths=[],
                error=str(e),
                model_id=None,
            )
        raise


if __name__ == "__main__":
    main()
