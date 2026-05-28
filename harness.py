"""Harness telemetry — v1 run-log writer for Joseph's agent fleet.

Fleet-portable. One public function, write_run_log, emits a single run-log per
agent run to <vault>/Agent Runs/ per the Harness Pattern v1 schema. Deliberately
decoupled from any one agent: imports nothing from a specific agent's mail/ or
vault/ layers and takes everything it needs as arguments.
"""

from __future__ import annotations

import os
from datetime import datetime

SCHEMA_VERSION = 2
HOST = "personal"


def _yaml_quote(value: str) -> str:
    """Double-quoted YAML scalar with the minimal escapes the schema needs."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _yaml_list(items: list[str]) -> str:
    """Render a block list, or inline [] when empty, matching the schema."""
    if not items:
        return " []"
    return "\n" + "\n".join(f"  - {_yaml_quote(item)}" for item in items)


def _vault_relative(path: str, vault_path: str) -> str:
    """Path relative to the vault root. An entry outside the vault keeps its
    ../ form — a diagnostic signal that an artifact landed off-vault."""
    return os.path.relpath(path, vault_path)


def write_run_log(
    vault_path: str,
    agent_id: str,
    agent_version: str,
    started_at: datetime,
    completed_at: datetime,
    status: str,
    trigger: str,
    input_summary: str,
    output_summary: str,
    output_paths: list[str],
    error: str | None = None,
    parent_run_id: str | None = None,
    tags: list[str] | None = None,
    notes: str = "",
    model_id: str | None = None,
    token_cost_input: int | None = None,
    token_cost_output: int | None = None,
) -> str:
    """Write a v1 harness run-log to <vault_path>/Agent Runs/. Returns the path written."""
    tags = tags or []

    slug = agent_id.replace("-", "")
    run_id = f"{started_at:%Y-%m-%d-%H%M}-{slug}"
    # duration keeps full precision; only the displayed timestamps are truncated.
    duration_ms = int((completed_at - started_at).total_seconds() * 1000)
    started_display = started_at.replace(microsecond=0)
    completed_display = completed_at.replace(microsecond=0)

    rel_paths = [_vault_relative(p, vault_path) for p in output_paths]

    runs_dir = os.path.join(vault_path, "Agent Runs")
    os.makedirs(runs_dir, exist_ok=True)
    out_path = os.path.join(runs_dir, f"{started_at:%Y-%m-%d %H%M} {agent_id}.md")

    error_field = "null" if error is None else _yaml_quote(error)
    parent_field = "null" if parent_run_id is None else _yaml_quote(parent_run_id)
    model_field = "null" if model_id is None else _yaml_quote(model_id)
    token_in_field = "null" if token_cost_input is None else str(token_cost_input)
    token_out_field = "null" if token_cost_output is None else str(token_cost_output)

    frontmatter = (
        "---\n"
        f"schema_version: {SCHEMA_VERSION}\n"
        f"agent_id: {agent_id}\n"
        f"agent_version: {agent_version}\n"
        f"model_id: {model_field}\n"
        f"run_id: {run_id}\n"
        f"parent_run_id: {parent_field}\n"
        f"host: {HOST}\n"
        f"started_at: {started_display.isoformat()}\n"
        f"completed_at: {completed_display.isoformat()}\n"
        f"duration_ms: {duration_ms}\n"
        f"status: {status}\n"
        f"trigger: {trigger}\n"
        f"input_summary: {_yaml_quote(input_summary)}\n"
        f"output_summary: {_yaml_quote(output_summary)}\n"
        f"output_paths:{_yaml_list(rel_paths)}\n"
        f"token_cost_input: {token_in_field}\n"
        f"token_cost_output: {token_out_field}\n"
        f"tags:{_yaml_list(tags)}\n"
        f"error: {error_field}\n"
        f"notes: {_yaml_quote(notes)}\n"
        "---\n"
    )

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(frontmatter)

    return out_path
