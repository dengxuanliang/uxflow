import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2]
WEB = ROOT / "src" / "service" / "web"


def test_problem_confidence_label_is_explicit():
    app_js = (WEB / "app.js").read_text()

    assert "置信度" in app_js
    assert "Module0 子问题编译置信度" in app_js


def test_detail_highlight_toolbar_is_present():
    index_html = (WEB / "index.html").read_text()
    app_js = (WEB / "app.js").read_text()
    style_css = (WEB / "style.css").read_text()

    assert 'id="detail-tools"' in index_html
    assert 'id="next-highlight"' in index_html
    assert "jumpToNextHighlight" in app_js
    assert ".step.hit-current" in style_css
