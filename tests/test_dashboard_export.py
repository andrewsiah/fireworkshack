import importlib.util
from pathlib import Path


def test_portable_export_binds_sanitized_data_without_external_assets(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "export_dashboard", Path("scripts/export_dashboard.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = module.export_dashboard(Path.cwd(), tmp_path / "dashboard.html")
    body = output.read_text()
    assert "window.FIREWORKSHACK_SNAPSHOT =" in body
    assert 'src="/app.js"' not in body
    assert 'href="/styles.css"' not in body
    assert "Saved snapshot" in body
    assert "FIREWORKS_API_KEY=" not in body
