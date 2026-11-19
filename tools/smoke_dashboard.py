#!/usr/bin/env python3
"""Streamlit app smoke test using the repository's own app without browser use."""
import sys
from pathlib import Path


def main():
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError as exc:
        raise RuntimeError('Streamlit must be installed for dashboard testing') from exc
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'app'))
    dashboard = root / 'app' / 'app.py'
    if not dashboard.is_file():
        raise RuntimeError('app/app.py is missing')
    app = AppTest.from_file(str(dashboard), default_timeout=35).run()
    if len(app.exception):
        msg = '\n'.join(str(e.message) for e in app.exception)
        raise RuntimeError('Streamlit dashboard raised exceptions: ' + msg)
    if len(app.error):
        msg = '\n'.join(str(e.message) for e in app.error)
        raise RuntimeError('Streamlit dashboard displayed errors: ' + msg)
    print('STREAMLIT DASHBOARD SMOKE: PASS')


if __name__ == '__main__':
    sys.exit(main())
