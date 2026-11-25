"""Reusable Streamlit glass panels and read-only visual elements."""
from html import escape
from urllib.parse import quote, urlsplit

import streamlit as st

from .features import ROOT

ICONS = {
    "shield": '<path d="M12 22s8-3.5 8-11V5l-8-3-8 3v6c0 7.5 8 11 8 11Z"/><path d="m9 12 2 2 4-4"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
    "search": '<circle cx="10" cy="10" r="6"/><path d="m14.5 14.5 5 5"/>',
    "warning": '<path d="M12 3 2 20h20L12 3Zm0 6v5m0 3h.01"/>',
    "pending": '<path d="M6 3h12M6 21h12M7 3v3c0 3 5 6 5 6s5-3 5-6V3M7 21v-3c0-3 5-6 5-6s5 3 5 6v3"/>',
    "data": '<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5v13c0 4 16 4 16 0V5M4 11c0 4 16 4 16 0"/>',
    "chart": '<path d="M4 20v-5m5 5V9m6 11v-7m5 7V4"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/>',
    "file": '<path d="M6 3h8l4 4v14H6V3Zm8 0v5h4M9 12h6m-6 4h6"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4m6-4v4m-6 12v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4"/>',
    "cloud": '<path d="M6 18h12a4 4 0 0 0 0-8h-1A6 6 0 0 0 5 11a3.5 3.5 0 0 0 1 7Z"/>',
    "pie": '<path d="M11 3v10h10A9 9 0 1 1 11 3Zm3 0v7h7a9 9 0 0 0-7-7Z"/>',
}


def icon(name):
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
           'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' + ICONS[name] + '</svg>')
    return f'<span class="fs-icon" style="--fs-symbol:url(data:image/svg+xml,{quote(svg, safe="")})" aria-hidden="true"></span>'


@st.cache_data
def stylesheet():
    return (ROOT / "assets" / "styles" / "glassmorphism.css").read_text(encoding="utf-8")


def install():
    st.html(f"<style>{stylesheet()}</style>")


def brand():
    with st.sidebar:
        st.html(f'<div class="fs-brand">{icon("shield")}<div><strong>FraudShield <b>AI</b></strong>'
                '<small>Fraud Intelligence Platform</small></div></div>')


def contact_footer(name="", email="", phone="", linkedin=""):
    name, email, phone, linkedin = (value.strip() for value in (name, email, phone, linkedin))
    links = []
    if email:
        links.append(f'<a href="mailto:{quote(email, safe="@.+-")}">{escape(email)}</a>')
    if phone:
        dial = ("+" if phone.startswith("+") else "") + "".join(c for c in phone if c in "0123456789")
        if dial.lstrip("+"):
            links.append(f'<a href="tel:{dial}">{escape(phone)}</a>')
    if linkedin:
        try:
            url = urlsplit(linkedin)
            if (url.scheme == "https" and url.netloc.lower() in {"linkedin.com", "www.linkedin.com"}
                    and url.path.startswith("/in/") and url.path.strip("/") != "in"):
                links.append(f'<a href="{escape(linkedin, quote=True)}" target="_blank" rel="noopener noreferrer">LinkedIn</a>')
        except ValueError:
            pass
    contact = f'<nav aria-label="Contact links">{"".join(links)}</nav>' if links else ""
    st.html(f'<footer class="fs-footer" aria-label="Project contact">'
            f'<div><strong>FraudShield AI</strong>{f"<span>{escape(name)}</span>" if name else ""}</div>'
            f'{contact}</footer>')


def topbar(model, provider, status, workspace="Investigation Command Center"):
    ready = status in {"Model verified", "Available offline"}
    st.html(f'<div class="fs-top"><span>{escape(workspace)}</span><div class="fs-health">'
            f'<span class="fs-chip">{icon("cpu")}{escape(model)} <span class="fs-state ready">Ready</span></span>'
            f'<span class="fs-chip">{icon("cloud")}{escape(provider)} '
            f'<span class="fs-state{" ready" if ready else ""}">{escape(status)}</span></span></div></div>')


def panel_heading(title, symbol, subtitle="", tag=""):
    return (f'<div class="fs-panel-title"><span class="fs-panel-icon">{icon(symbol)}</span>'
            f'<div><strong>{escape(title)}</strong><small>{escape(subtitle)}</small></div>'
            f'<span class="fs-panel-tag">{escape(tag)}</span></div>')


def band(title, symbol, subtitle, items, tag="", historical=False):
    cards = "".join(f'<div class="fs-kpi {tone}"><span class="fs-kpi-icon">{icon(mark)}</span>'
                    f'<div><strong>{escape(str(value))}</strong><small>{escape(caption)}</small></div></div>'
                    for caption, value, mark, tone in items)
    st.html(f'<section class="fs-glass fs-band{" fs-historical" if historical else ""}" aria-label="{escape(title)}">'
            f'{panel_heading(title, symbol, subtitle, tag)}'
            f'<div class="fs-metrics" style="--fs-count:{len(items)}">{cards}</div></section>')
