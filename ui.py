"""Helpers that render the custom HTML/CSS front end inside Streamlit."""
from __future__ import annotations

from html import escape
from pathlib import Path
from string import Template

import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"
PRETTY = {"pothole": "Pothole", "speed_breaker": "Speed breaker"}


def _compact(text: str) -> str:
    """Strip indentation / blank lines so Markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def inject_css() -> None:
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{_compact(css)}</style>", unsafe_allow_html=True)


def _template(name: str, **values) -> str:
    tpl = Template((ASSETS / "templates" / name).read_text(encoding="utf-8"))
    return _compact(tpl.safe_substitute(**values))


def render(name: str, **values) -> None:
    """Render assets/templates/<name> with $placeholders filled in."""
    st.markdown(_template(name, **values), unsafe_allow_html=True)


def stat_grid(items: list[dict]) -> None:
    """items: [{'label', 'value', 'sub', 'tone'}] -> responsive grid of stat cards."""
    cards = "".join(
        _template(
            "stat_card.html",
            label=escape(str(i["label"])),
            value=escape(str(i["value"])),
            sub=escape(str(i.get("sub", ""))),
            tone=i.get("tone", ""),
        )
        for i in items
    )
    st.markdown(f'<div class="stat-grid">{cards}</div>', unsafe_allow_html=True)


def feature_grid(items: list[dict]) -> None:
    cards = "".join(
        _template(
            "feature_card.html",
            icon=i["icon"], title=escape(i["title"]), body=escape(i["body"]),
        )
        for i in items
    )
    st.markdown(f'<div class="feature-grid">{cards}</div>', unsafe_allow_html=True)


def events_table(events: pd.DataFrame, limit: int = 200) -> None:
    rows = []
    for r in events.head(limit).itertuples():
        pct = int(round(r.confidence * 100))
        rows.append(
            "<tr>"
            f'<td class="mono">{r.peak_time_s:.1f} s</td>'
            f'<td><span class="badge {r.type}">{PRETTY[r.type]}</span></td>'
            f'<td class="mono">{r.peak_ms2:.1f} m/s²</td>'
            f'<td><span class="sev {r.severity.lower()}">{r.severity}</span></td>'
            f'<td><div class="meter"><span style="width:{pct}%"></span></div>'
            f'<span class="mono">{pct}%</span></td>'
            "</tr>"
        )
    note = ""
    if len(events) > limit:
        note = f'<div class="table-note">Showing first {limit} of {len(events)} events - download the CSV for all.</div>'
    html = (
        '<div class="table-wrap"><table class="evt-table"><thead><tr>'
        "<th>Time</th><th>Type</th><th>Peak shock</th><th>Severity</th><th>Confidence</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>{note}"
    )
    st.markdown(html, unsafe_allow_html=True)
