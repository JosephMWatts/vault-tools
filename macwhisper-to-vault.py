#!/usr/bin/env python3
"""
macwhisper-to-vault.py
Stage two of the MacWhisper meeting pipeline.

Reads each MacWhisper Segments.json export from the inbox, sends it to
the Claude API for cleanup and enrichment using the locked v2 prompt,
writes a raw note and an enriched note into the vault, and moves the
consumed export to processed/.

The Anthropic API key is never stored here. It loads at runtime from
an untracked config file outside this repository.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import anthropic

import harness

# --- Paths --------------------------------------------------------------
HOME           = Path.home()
PIPELINE       = HOME / "Documents" / "meeting-pipeline"
CONFIG_PATH    = PIPELINE / "config.json"
INBOX          = PIPELINE / "inbox"
PROCESSED      = PIPELINE / "processed"
FAILED         = PIPELINE / "failed"
ATTEMPTS_PATH  = PIPELINE / "attempts.json"
LOG_PATH       = PIPELINE / "pipeline.log"
VAULT_ROOT     = HOME / "Documents" / "joseph_vault"
VAULT_RAW      = VAULT_ROOT / "Meetings" / "Raw"
VAULT_ENRICHED = VAULT_ROOT / "Meetings" / "Enriched"

DEFAULT_MODEL  = "claude-sonnet-4-6"
DEFAULT_MAXTOK = 16000
MAX_ATTEMPTS   = 3
STABILITY_WINDOW = 30   # seconds a file must be untouched before processing
TITLE_SCAN_LINES = 15   # how far into the response to look for the TITLE marker

CONVERTER_AGENT_ID = "macwhisper-converter"
ENRICHER_AGENT_ID  = "macwhisper-enricher"
AGENT_VERSION      = "1.0.0"
TRIGGER            = os.environ.get("TRIGGER", "manual")

# --- Locked enrichment prompt, v2 --------------------------------------
ENRICHMENT_PROMPT = """ROLE
You are a meeting-notes editor. You receive a raw, machine-generated
transcript of a single meeting and produce a clean, structured meeting
note. The transcript came from an automatic speech-to-text system and
has known defects you must fix.

INPUT
A JSON array of segment objects. Each object has:
  speaker    a label, either a real name or a generic ID like "Speaker 0"
  text       the spoken words for that segment
  timestamp  the segment's start and end time, formatted
             MM:SS.mmm-MM:SS.mmm

CLEANUP RULES
Fix these three defects before doing anything else.

1. Acoustic bleed echo. When a participant was not on headphones, the
   open microphone re-recorded the remote audio. This creates near-
   duplicate segments: the same words appear twice, once attributed to
   the real remote speaker and once to the local speaker labeled
   "Joseph Watts". The echo is offset 30 to 100 milliseconds and may be
   garbled or split. Keep the original, delete the echo. The original
   is attributed to the actual speaker. The echo is the copy on the
   "Joseph Watts" channel that contains another person's words.

2. Inconsistent speaker labels. One person may appear as a real name in
   one place and a generic ID elsewhere. Reconcile each person to one
   consistent name. If no real name is determinable, keep one stable
   generic label and note it in Attendees.

3. Diarization over-split. One real person may be split across two or
   more speaker IDs. Merge them into one speaker.

Also correct obvious transcription errors in proper nouns where context
makes the intended word unambiguous. Do not invent content. Do not
paraphrase or summarize inside the transcript. Preserve the speakers'
actual words, fixing only the defects above. If a passage is too
garbled to interpret, mark it [unclear] rather than guessing.

ENRICHMENT RULES
After cleaning, produce a Meeting Intelligence section:
  Attendees       each person, with role or affiliation if stated or
                  inferable from context
  Topics          bulleted, the substantive subjects covered
  Decisions       bulleted, explicit decisions or conclusions reached
  Action Items    a markdown table, columns Action, Owner, Due
  Summary         one short prose paragraph, what the meeting achieved
If a section has no content, write "None identified". Never omit a
section.

OUTPUT
Return markdown only, no preamble, in exactly this structure. The first
line is a machine-readable title the calling script will parse out.

TITLE: <short descriptive meeting subject, no date>

## Meeting Intelligence
*Auto-enriched by Claude on {{DATE}} — review and correct as needed*

**Attendees:** ...

**Topics Discussed:**
- ...

**Decisions Made:**
- ...

**Action Items:**

| Action | Owner | Due |
|---|---|---|
| ... | ... | ... |

**Summary:** ...

---

## Cleaned Transcript

**[timestamp] Speaker Name:** text

GUARDRAILS
Do not fabricate attendees, decisions, or action items unsupported by
the transcript. Flag ambiguity in the relevant section rather than
resolving it silently. The {{DATE}} placeholder is filled by the script.
"""


# --- Helpers ------------------------------------------------------------
def log(msg):
    """Print a timestamped line and append it to the pipeline log."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def load_config():
    """Read the untracked config file. Exit on any problem."""
    if not CONFIG_PATH.exists():
        log(f"ERROR config file not found at {CONFIG_PATH}")
        sys.exit(1)
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
    except json.JSONDecodeError as e:
        log(f"ERROR config file is not valid JSON: {e}")
        sys.exit(1)
    if not cfg.get("anthropic_api_key"):
        log("ERROR config file has no anthropic_api_key")
        sys.exit(1)
    return cfg


def is_stable(path):
    """True if the file has not been modified within the stability window.

    A file still being written by MacWhisper has a fresh mtime. Skipping
    it this run avoids reading a half-written export. The next run picks
    it up once the file has settled.
    """
    age = time.time() - path.stat().st_mtime
    return age >= STABILITY_WINDOW


def slugify(text):
    """Turn a title into a filesystem-safe slug."""
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s or "untitled-meeting"


def unique_basename_in_raw(meeting_date, name):
    """Return a filename free in Raw/. Source stem feeds the slug; Raw and
    Enriched are now named independently so each checks only its own dir."""
    slug = slugify(name)
    candidate = f"{meeting_date}-{slug}.md"
    n = 2
    while (VAULT_RAW / candidate).exists():
        candidate = f"{meeting_date}-{slug}-{n}.md"
        n += 1
    return candidate


def unique_basename_in_enriched(meeting_date, name):
    """Return a filename free in Enriched/. Claude-derived title feeds the
    slug; mirrors unique_basename_in_raw but only checks Enriched/."""
    slug = slugify(name)
    candidate = f"{meeting_date}-{slug}.md"
    n = 2
    while (VAULT_ENRICHED / candidate).exists():
        candidate = f"{meeting_date}-{slug}-{n}.md"
        n += 1
    return candidate


def ts_to_seconds(ts_part):
    """Convert MM:SS.mmm (or HH:MM:SS.mmm) to total seconds."""
    seconds = 0.0
    for piece in ts_part.split(":"):
        seconds = seconds * 60 + float(piece)
    return seconds


def compute_duration(segments):
    """Derive a human-readable duration from the segment timestamps."""
    try:
        start = segments[0]["timestamp"].split("-")[0]
        end = segments[-1]["timestamp"].split("-")[1]
        total = int(round(ts_to_seconds(end) - ts_to_seconds(start)))
        return f"{total // 60} min {total % 60} sec"
    except (KeyError, IndexError, ValueError):
        return "unknown"


def render_raw_note(segments, source_name, meeting_date, duration):
    """Build a faithful, unprocessed markdown copy of the export."""
    out = [
        f"# MacWhisper Transcript — {meeting_date}",
        "",
        f"**Date:** {meeting_date}",
        f"**Duration:** {duration}",
        f"**Source file:** {source_name}",
        f"**Segments:** {len(segments)}",
        "",
        "---",
        "",
        "## Transcript",
        "",
    ]
    for seg in segments:
        start = seg.get("timestamp", "").split("-")[0]
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        out.append(f"**[{start}] {speaker}:** {text}")
        out.append("")
    out += ["---", "", "*Raw MacWhisper export, unprocessed. "
            "Generated by macwhisper-to-vault.py.*"]
    return "\n".join(out)


def call_claude(client, model, max_tokens, raw_json_text):
    """Send the export to Claude and return (markdown, usage).

    The enricher needs usage.input_tokens / output_tokens to populate
    the v2 run-log's token_cost_* fields, so usage rides back with the
    text rather than being logged-and-discarded.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    system_prompt = ENRICHMENT_PROMPT.replace("{{DATE}}", today)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": raw_json_text}],
    )
    usage = resp.usage
    log(f"  TOKENS in={usage.input_tokens} out={usage.output_tokens}")
    if resp.stop_reason == "max_tokens":
        log(f"  WARNING response hit the {max_tokens}-token ceiling, "
            f"the note is probably truncated and needs review")
    return resp.content[0].text, usage


def parse_response(text):
    """Split the machine-readable TITLE line from the note body.

    The enrichment prompt asks for TITLE: on the first line, but the model
    sometimes emits a short preamble above it. Scan the opening lines for
    the marker rather than trusting line one, and discard everything up to
    and including the TITLE line so no preamble leaks into the note body.
    """
    text = text.strip()
    lines = text.split("\n")
    for i, line in enumerate(lines[:TITLE_SCAN_LINES]):
        stripped = line.strip()
        if stripped.upper().startswith("TITLE:"):
            title = stripped.split(":", 1)[1].strip()
            body = "\n".join(lines[i + 1:]).strip()
            return title or "Untitled Meeting", body
    return "Untitled Meeting", text


def build_enriched_note(title, body, source_name, meeting_date, duration):
    """Assemble the title, metadata block, Claude body, and footer."""
    today = datetime.now().strftime("%Y-%m-%d")
    header = "\n".join([
        f"# Meeting — {meeting_date}: {title}",
        "",
        f"**Date:** {meeting_date}",
        f"**Duration:** {duration}",
        f"**Source file:** {source_name}",
        "",
        "---",
        "",
    ])
    footer = "\n".join([
        "",
        "",
        "---",
        "",
        f"*Auto-generated by macwhisper-to-vault.py | "
        f"Enriched by Claude on {today}*",
    ])
    return header + body + footer


# --- Per-file processing, split into two harness-instrumented agents ----
@dataclass
class ConverterResult:
    """State the converter hands to the enricher. On failure, only status
    and source_name are meaningful; the rest are None."""
    status: str                       # "success" or "failed"
    run_id: str | None
    segments: list | None
    raw_json_text: str | None         # original file contents, fed to Claude
    meeting_date: str | None
    duration: str | None
    raw_basename: str | None
    source_name: str


def _compute_run_id(started_at, agent_id):
    """Mirror harness.write_run_log's internal recipe so the converter can
    hand its own run_id to the enricher as parent_run_id without round-
    tripping through the filesystem."""
    return f"{started_at:%Y-%m-%d-%H%M}-{agent_id.replace('-', '')}"


def run_converter(path, cfg):
    """Stage 1: parse the MacWhisper export, write the raw note, move the
    source to processed/, emit a v2 run-log. Returns a ConverterResult."""
    log(f"CONVERTER {path.name}")
    started_at = datetime.now().astimezone()
    run_id = _compute_run_id(started_at, CONVERTER_AGENT_ID)
    try:
        raw_json_text = path.read_text()
        try:
            segments = json.loads(raw_json_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"malformed JSON: {e}") from e
        if not isinstance(segments, list) or not segments:
            raise ValueError("empty or non-array JSON")

        meeting_date = datetime.fromtimestamp(
            path.stat().st_mtime).strftime("%Y-%m-%d")
        duration = compute_duration(segments)

        raw_md = render_raw_note(segments, path.name, meeting_date, duration)
        VAULT_RAW.mkdir(parents=True, exist_ok=True)
        raw_basename = unique_basename_in_raw(meeting_date, path.stem)
        raw_path = VAULT_RAW / raw_basename
        raw_path.write_text(raw_md)
        log(f"  WROTE Raw/{raw_basename}")

        PROCESSED.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(PROCESSED / path.name))
        log(f"  MOVED {path.name} to processed/")

        harness.write_run_log(
            vault_path=str(VAULT_ROOT),
            agent_id=CONVERTER_AGENT_ID,
            agent_version=AGENT_VERSION,
            started_at=started_at,
            completed_at=datetime.now().astimezone(),
            status="success",
            trigger=TRIGGER,
            input_summary=f"{len(segments)} segments from {path.name}",
            output_summary=f"Raw note: {raw_basename}",
            output_paths=[str(raw_path)],
            parent_run_id=None,
            model_id=None,
        )
        return ConverterResult(
            status="success",
            run_id=run_id,
            segments=segments,
            raw_json_text=raw_json_text,
            meeting_date=meeting_date,
            duration=duration,
            raw_basename=raw_basename,
            source_name=path.name,
        )
    except Exception as e:
        harness.write_run_log(
            vault_path=str(VAULT_ROOT),
            agent_id=CONVERTER_AGENT_ID,
            agent_version=AGENT_VERSION,
            started_at=started_at,
            completed_at=datetime.now().astimezone(),
            status="failed",
            trigger=TRIGGER,
            input_summary=f"file: {path.name}",
            output_summary=str(e),
            output_paths=[],
            error=str(e),
            parent_run_id=None,
            model_id=None,
        )
        log(f"  CONVERTER FAILED, left in inbox: {e}")
        return ConverterResult(
            status="failed",
            run_id=None,
            segments=None,
            raw_json_text=None,
            meeting_date=None,
            duration=None,
            raw_basename=None,
            source_name=path.name,
        )


def run_enricher(cr, cfg, client):
    """Stage 2: send the export to Claude, write the enriched note, emit a
    chained v2 run-log linked to the converter via parent_run_id. Returns
    True on success, False on any failure."""
    log(f"ENRICHER {cr.source_name}")
    started_at = datetime.now().astimezone()
    model_id = cfg.get("model", DEFAULT_MODEL)
    try:
        text, usage = call_claude(
            client, model_id,
            cfg.get("max_tokens", DEFAULT_MAXTOK), cr.raw_json_text)
        title, body = parse_response(text)
        enriched_md = build_enriched_note(
            title, body, cr.source_name, cr.meeting_date, cr.duration)
        VAULT_ENRICHED.mkdir(parents=True, exist_ok=True)
        enriched_basename = unique_basename_in_enriched(cr.meeting_date, title)
        enriched_path = VAULT_ENRICHED / enriched_basename
        enriched_path.write_text(enriched_md)
        log(f"  WROTE Enriched/{enriched_basename}")

        harness.write_run_log(
            vault_path=str(VAULT_ROOT),
            agent_id=ENRICHER_AGENT_ID,
            agent_version=AGENT_VERSION,
            started_at=started_at,
            completed_at=datetime.now().astimezone(),
            status="success",
            trigger="chained",
            input_summary=f"Enrich {cr.source_name}",
            output_summary=f"Enriched note: {enriched_basename}",
            output_paths=[str(enriched_path)],
            parent_run_id=cr.run_id,
            model_id=model_id,
            token_cost_input=usage.input_tokens,
            token_cost_output=usage.output_tokens,
        )
        return True
    except Exception as e:
        harness.write_run_log(
            vault_path=str(VAULT_ROOT),
            agent_id=ENRICHER_AGENT_ID,
            agent_version=AGENT_VERSION,
            started_at=started_at,
            completed_at=datetime.now().astimezone(),
            status="failed",
            trigger="chained",
            input_summary=f"Enrich {cr.source_name}",
            output_summary=str(e),
            output_paths=[],
            error=str(e),
            parent_run_id=cr.run_id,
            model_id=model_id,
        )
        log(f"  ENRICHER FAILED: {e}")
        return False


# --- Retry cap, quarantine, and notification ---------------------------
def load_attempts():
    """Read the per-file failure counter. Return {} if absent or bad."""
    if not ATTEMPTS_PATH.exists():
        return {}
    try:
        return json.loads(ATTEMPTS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        log("WARNING attempts file unreadable, starting the counter fresh")
        return {}


def save_attempts(attempts):
    """Persist the per-file failure counter."""
    try:
        ATTEMPTS_PATH.write_text(json.dumps(attempts, indent=2))
    except OSError as e:
        log(f"WARNING could not save the attempts file: {e}")


def notify(title, message):
    """Fire a macOS desktop notification. Never fatal."""
    try:
        subprocess.run(
            ["osascript", "-e",
             f"display notification {json.dumps(message)} "
             f"with title {json.dumps(title)}"],
            check=False, capture_output=True)
    except Exception as e:
        log(f"  WARNING desktop notification failed: {e}")


def quarantine(path):
    """Move a file that has failed MAX_ATTEMPTS times out of the inbox."""
    FAILED.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(path), str(FAILED / path.name))
        log(f"  QUARANTINED {path.name} to failed/ after "
            f"{MAX_ATTEMPTS} failed attempts")
    except OSError as e:
        log(f"  ERROR could not quarantine {path.name}: {e}")
    notify("MacWhisper pipeline",
           f"{path.name} failed {MAX_ATTEMPTS} times and was moved to "
           f"failed/. It needs manual review.")


# --- Entry point --------------------------------------------------------
def main():
    cfg = load_config()
    client = anthropic.Anthropic(api_key=cfg["anthropic_api_key"])

    if not INBOX.exists():
        log(f"ERROR inbox not found at {INBOX}")
        sys.exit(1)

    files = sorted(INBOX.glob("*.json"))
    if not files:
        log("Inbox empty, nothing to do.")
        return

    ready = [p for p in files if is_stable(p)]
    for p in files:
        if p not in ready:
            log(f"SKIP {p.name} still settling, will retry next run")
    if not ready:
        log("Inbox has only files still being written, nothing to do.")
        return
    files = ready

    attempts = load_attempts()
    log(f"Found {len(files)} file(s) in inbox.")
    succeeded = 0
    for path in files:
        ok = False
        try:
            cr = run_converter(path, cfg)
            if cr.status == "success":
                ok = run_enricher(cr, cfg, client)
        except Exception as e:
            log(f"  ERROR unhandled while processing {path.name}, "
                f"left in inbox: {e}")
        if ok:
            succeeded += 1
            attempts.pop(path.name, None)
        else:
            count = attempts.get(path.name, 0) + 1
            attempts[path.name] = count
            log(f"  ATTEMPT {count} of {MAX_ATTEMPTS} failed "
                f"for {path.name}")
            if count >= MAX_ATTEMPTS:
                quarantine(path)
                attempts.pop(path.name, None)
    save_attempts(attempts)
    log(f"Done. {succeeded}/{len(files)} processed.")


if __name__ == "__main__":
    main()
