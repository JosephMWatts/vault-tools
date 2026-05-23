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

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import anthropic

# --- Paths --------------------------------------------------------------
HOME           = Path.home()
PIPELINE       = HOME / "Documents" / "meeting-pipeline"
CONFIG_PATH    = PIPELINE / "config.json"
INBOX          = PIPELINE / "inbox"
PROCESSED      = PIPELINE / "processed"
FAILED         = PIPELINE / "failed"
ATTEMPTS_PATH  = PIPELINE / "attempts.json"
LOG_PATH       = PIPELINE / "pipeline.log"
VAULT_RAW      = HOME / "Documents" / "joseph_vault" / "Meetings" / "Raw"
VAULT_ENRICHED = HOME / "Documents" / "joseph_vault" / "Meetings" / "Enriched"

DEFAULT_MODEL  = "claude-sonnet-4-6"
DEFAULT_MAXTOK = 16000
MAX_ATTEMPTS   = 3

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


def slugify(text):
    """Turn a title into a filesystem-safe slug."""
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s_]+", "-", s).strip("-")
    return s or "untitled-meeting"


def unique_basename(meeting_date, title):
    """Return a note filename not already used in Raw/ or Enriched/.

    Guards against silently overwriting an existing vault note when two
    meetings share a date and title. Appends -2, -3, ... until free.
    """
    slug = slugify(title)
    candidate = f"{meeting_date}-{slug}.md"
    n = 2
    while (VAULT_RAW / candidate).exists() or \
            (VAULT_ENRICHED / candidate).exists():
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
    """Send the export to Claude and return the markdown response.

    Logs token usage for spend visibility, and warns if the response
    hit the token ceiling and was probably truncated.
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
    return resp.content[0].text


def parse_response(text):
    """Split the machine-readable TITLE line from the note body."""
    text = text.strip()
    lines = text.split("\n")
    if lines and lines[0].upper().startswith("TITLE:"):
        title = lines[0].split(":", 1)[1].strip()
        body = "\n".join(lines[1:]).strip()
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


# --- Per-file processing ------------------------------------------------
def process_file(path, cfg, client):
    """Process one export. Return True only on full success."""
    log(f"Processing {path.name}")
    raw_text = path.read_text()
    try:
        segments = json.loads(raw_text)
    except json.JSONDecodeError as e:
        log(f"  SKIP malformed JSON, left in inbox: {e}")
        return False
    if not isinstance(segments, list) or not segments:
        log("  SKIP empty or non-array JSON, left in inbox")
        return False

    meeting_date = datetime.fromtimestamp(
        path.stat().st_mtime).strftime("%Y-%m-%d")
    duration = compute_duration(segments)

    try:
        response = call_claude(
            client, cfg.get("model", DEFAULT_MODEL),
            cfg.get("max_tokens", DEFAULT_MAXTOK), raw_text)
    except Exception as e:
        log(f"  SKIP Claude API error, left in inbox: {e}")
        return False

    title, body = parse_response(response)
    raw_md = render_raw_note(segments, path.name, meeting_date, duration)
    enriched_md = build_enriched_note(
        title, body, path.name, meeting_date, duration)

    VAULT_RAW.mkdir(parents=True, exist_ok=True)
    VAULT_ENRICHED.mkdir(parents=True, exist_ok=True)
    basename = unique_basename(meeting_date, title)
    (VAULT_RAW / basename).write_text(raw_md)
    (VAULT_ENRICHED / basename).write_text(enriched_md)
    log(f"  WROTE {basename} to Raw/ and Enriched/")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(PROCESSED / path.name))
    log(f"  MOVED {path.name} to processed/")
    return True


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

    attempts = load_attempts()
    log(f"Found {len(files)} file(s) in inbox.")
    succeeded = 0
    for path in files:
        ok = False
        try:
            ok = process_file(path, cfg, client)
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
