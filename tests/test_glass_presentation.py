"""Presentation checks for real data, honest status and escaped HTML."""
from fraudshield import dashboard, glass


def test_footer_links_and_safe_profile_url(monkeypatch):
    rendered = []
    monkeypatch.setattr(glass.st, "html", rendered.append)
    glass.contact_footer("Alex & Sam", "example@gmail.com", "+1 (202) 555-0123",
                         "https://www.linkedin.com/in/example/")
    assert "Alex &amp; Sam" in rendered[0]
    assert 'href="mailto:example@gmail.com"' in rendered[0]
    assert 'href="tel:+12025550123"' in rendered[0]
    assert 'href="https://www.linkedin.com/in/example/"' in rendered[0]
    glass.contact_footer(linkedin="javascript:alert(1)")
    assert "<a " not in rendered[1]


def test_status_only_marks_verified_provider_ready(monkeypatch):
    rendered = []
    monkeypatch.setattr(glass.st, "html", rendered.append)
    glass.topbar("XGBoost", "Groq Cloud", "Model configured. Verified on first request")
    assert rendered[0].count("fs-state ready") == 1
    glass.topbar("XGBoost", "Groq Cloud", "Model verified")
    assert rendered[1].count("fs-state ready") == 2
    glass.topbar("XGBoost", "Groq Cloud", "Last provider request used Python fallback")
    assert rendered[2].count("fs-state ready") == 1


def test_workspace_glass_escapes_labels_and_displays_supplied_values(monkeypatch):
    rendered = []
    monkeypatch.setattr(glass.st, "html", rendered.append)
    glass.band("Workspace", "layers", "", [("<b>Pending</b>", "7", "pending", "amber")])
    assert "<strong>7</strong>" in rendered[0] and "&lt;b&gt;Pending&lt;/b&gt;" in rendered[0]
    dashboard.recent_activity([
        {"transaction_id": "<script>bad</script>", "analyzed_at": "2026-11-25T06:05:00Z",
         "suspected": 1, "risk": .831843},
    ])
    assert "&lt;script&gt;bad&lt;/script&gt;" in rendered[1]
    assert "83.18%" in rendered[1] and "Suspected fraud" in rendered[1]
