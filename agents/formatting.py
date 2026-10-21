"""
Shared output formatting for the analytics agents.

The deterministic agents return their answer as a structured JSON object; this
module parses it and renders a clean GitHub-flavored markdown table so every
agent produces the SAME tidy format (one row per record, no cells spilling
across rows).

Expected LLM JSON shape:
    {
      "columns": ["Col A", "Col B", ...],     # ordered headers (optional)
      "rows":    [ {"Col A": v, "Col B": v}, ... ],
      "message": "optional one-line note / 'Not Available'"
    }
"""

import json
import re
from collections import Counter

try:
    import markdown as _md
    _HAS_MARKDOWN = True
except ImportError:
    _HAS_MARKDOWN = False

# Matches bare https?:// URLs not already inside an href="..." attribute
_BARE_URL_RE = re.compile(r'(?<!["\'>])https?://\S+')
# Matches [label](https://...) markdown links (fallback path only)
_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^\s)]+)\)')


def parse_structured(content: str) -> dict:
    """Parse the LLM response into {columns, rows, message}, tolerating fences."""
    content = (content or "").strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    try:
        data = json.loads(content)
    except Exception:
        return {"columns": [], "rows": [], "message": content}

    if isinstance(data, list):
        return {"columns": [], "rows": data, "message": ""}
    if isinstance(data, dict):
        return {
            "columns": data.get("columns") or [],
            "rows": data.get("rows") or [],
            "message": data.get("message") or "",
        }
    return {"columns": [], "rows": [], "message": str(data)}


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(x) for x in value)
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()


def _coerce_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _build_table_summary(columns: list, rows: list) -> str:
    if not rows:
        return ""
    if not columns:
        columns = list(rows[0].keys())

    label_column = next(
        (c for c in columns if c.lower() in {"department", "store", "name", "item", "category", "product", "region", "employee"}),
        None,
    )
    if not label_column and columns:
        label_column = columns[0]

    numeric_column = None
    numeric_values = []
    for col in columns:
        if col.lower() in {"transactiondate", "date", "colour", "color", "status", "performancecomparison"}:
            continue
        values = [_coerce_number(r.get(col)) for r in rows if r.get(col) is not None]
        values = [v for v in values if v is not None]
        if len(values) >= 2:
            numeric_column = col
            numeric_values = values
            break

    parts = [f"{len(rows)} row{'s' if len(rows) != 1 else ''} returned."]

    if numeric_column and numeric_values:
        max_value = max(numeric_values)
        min_value = min(numeric_values)
        avg_value = round(sum(numeric_values) / len(numeric_values), 2)
        if label_column:
            best_row = max(
                rows,
                key=lambda r: (_coerce_number(r.get(numeric_column)) if _coerce_number(r.get(numeric_column)) is not None else float("-inf")),
            )
            best_label = _cell(best_row.get(label_column)) or "a row"
            parts.append(f"{best_label} shows the highest {numeric_column} at {max_value}, while the weakest value is {min_value} and the average is {avg_value}.")
        else:
            parts.append(f"The {numeric_column} values range from {min_value} to {max_value}, with an average of {avg_value}.")
    else:
        parts.append("The numeric trend is not strong enough to highlight a clear best or weakest value.")

    color_column = next((c for c in columns if c.lower() in {"colour", "color", "status"}), None)
    if color_column:
        values = [str(r.get(color_column) or "").strip().upper() for r in rows if r.get(color_column) is not None]
        counts = Counter(v for v in values if v)
        if counts:
            top_color, top_count = counts.most_common(1)[0]
            parts.append(f"Most rows are {top_color.lower()} ({top_count} of {len(rows)}), indicating a generally underperforming pattern.")
        else:
            parts.append("The status mix is not available for this dataset.")
    else:
        parts.append("The status mix is not available for this dataset.")

    if color_column:
        values = [str(r.get(color_column) or "").strip().upper() for r in rows if r.get(color_column) is not None]
        counts = Counter(v for v in values if v)
        top_color = counts.most_common(1)[0][0].lower() if counts else ""
        if top_color in {"red", "yellow"}:
            parts.append("Overall, the data suggests that action should be taken to improve the weaker areas and lift performance.")
        else:
            parts.append("Overall, the data appears stable and should be monitored regularly rather than changed immediately.")
    else:
        parts.append("Overall, the table should be reviewed carefully for the next decision.")

    return " ".join(parts)


def render_markdown_table(columns: list, rows: list) -> str:
    """Render rows (list of dicts) as a clean markdown table."""
    if not rows:
        return ""
    if not columns:
        columns = list(rows[0].keys())
    header = "| " + " | ".join(str(c) for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_cell(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def render_html_table(columns: list, rows: list) -> str:
    """Render rows (list of dicts) as an HTML table."""
    if not rows:
        return ""
    if not columns:
        columns = list(rows[0].keys())
    thead = "<thead><tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr></thead>"
    tbody_rows = []
    for r in rows:
        cells = "".join(f"<td>{_cell(r.get(c))}</td>" for c in columns)
        tbody_rows.append(f"<tr>{cells}</tr>")
    tbody = "<tbody>" + "".join(tbody_rows) + "</tbody>"
    return f"<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">{thead}{tbody}</table>"


def render_reply(content: str) -> dict:
    """Parse an LLM structured response and return {reply, columns, rows, summary}."""
    data = parse_structured(content)
    columns, rows, message = data["columns"], data["rows"], data["message"]
    summary = _build_table_summary(columns, rows)
    table = render_markdown_table(columns, rows)
    summary_text = f"Summary:\n{summary}" if summary else ""
    parts = [p for p in (message, summary_text, table) if p]
    reply = "\n\n".join(parts) if parts else "Not Available"
    return {
        "reply": reply,
        "columns": columns or (list(rows[0].keys()) if rows else []),
        "rows": rows,
        "summary": summary,
    }


def render_html_reply(content: str) -> dict:
    """Parse structured LLM response and return HTML reply plus structured data."""
    data = parse_structured(content)
    columns, rows, message = data["columns"], data["rows"], data["message"]
    summary = _build_table_summary(columns, rows)
    table = render_html_table(columns, rows)
    parts = []
    if message:
        msg = str(message).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        parts.append(f"<p>{msg}</p>")
    if summary:
        safe_summary = str(summary).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        summary_html = safe_summary.replace("\n", "<br>")
        parts.append(f"<p><strong>Summary:</strong><br>{summary_html}</p>")
    if table:
        parts.append(table)
    reply_html = "".join(parts) if parts else "<p>Not Available</p>"
    return {
        "reply_html": reply_html,
        "columns": columns or (list(rows[0].keys()) if rows else []),
        "rows": rows,
        "summary": summary,
    }


def _ensure_table_spacing(text: str) -> str:
    """Guarantee a blank line before and after every pipe-table block.

    The markdown `tables` extension only recognises a pipe table when it is a
    standalone block (preceded and followed by a blank line).  LLMs often write
    tables immediately after a sentence with only a single newline, so the
    parser sees them as paragraph text and nl2br turns every row into a <br>.

    Blank lines are injected only at BLOCK TRANSITIONS (non-table → table, or
    table → non-table), never between consecutive table rows — inserting a blank
    between the header and separator rows would break the table entirely.
    """
    lines = text.split("\n")
    out: list = []
    prev_was_table = False
    for line in lines:
        stripped = line.strip()
        is_table = stripped.startswith("|") and stripped.endswith("|")

        # Blank line BEFORE the first row of a new table block
        if is_table and not prev_was_table and out and out[-1].strip() != "":
            out.append("")

        # Blank line BEFORE the first non-table content after a table block
        if not is_table and prev_was_table and stripped and out and out[-1].strip() != "":
            out.append("")

        out.append(line)
        prev_was_table = is_table
    return "\n".join(out)


def text_to_html(text: str) -> str:
    """Convert a plain-text / markdown agent reply into rendered HTML.

    When the `markdown` package is installed (preferred):
      - ### headings     → <h3>
      - **bold**         → <strong>
      - *italic*         → <em>
      - - bullet lists   → <ul><li>
      - | pipe tables |  → <table>
      - [label](url)     → <a href="url" target="_blank">
      - bare https://    → auto-linked <a>

    Falls back to basic escaping + paragraph split when the package is absent.
    """
    if not text:
        return "<p>Not Available</p>"
    s = str(text)

    if _HAS_MARKDOWN:
        s = _ensure_table_spacing(s)
        html = _md.markdown(
            s,
            extensions=["tables", "nl2br", "sane_lists"],
        )
        # Add target="_blank" to every <a> that doesn't already have it
        html = re.sub(r'<a\b(?![^>]*\btarget=)', '<a target="_blank"', html)
        # Auto-link bare https:// URLs the markdown parser left as plain text
        html = _BARE_URL_RE.sub(
            lambda m: f'<a href="{m.group()}" target="_blank">{m.group()}</a>',
            html,
        )
        return html

    # ── Fallback (no markdown package) ───────────────────────────────────────
    # Extract [label](url) links before escaping so brackets aren't mangled.
    _anchors: dict[str, str] = {}

    def _stash(m: re.Match) -> str:
        key = f"\x00L{len(_anchors)}\x00"
        _anchors[key] = f'<a href="{m.group(2)}" target="_blank">{m.group(1)}</a>'
        return key

    s = _MD_LINK_RE.sub(_stash, s)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for key, anchor in _anchors.items():
        s = s.replace(key, anchor)
    s = _BARE_URL_RE.sub(
        lambda m: f'<a href="{m.group()}" target="_blank">{m.group()}</a>', s
    )
    paragraphs = [p.replace("\n", "<br>") for p in s.split("\n\n")]
    return "".join(f"<p>{p}</p>" for p in paragraphs)
