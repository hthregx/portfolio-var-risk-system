"""Release-only tests: final README and frozen C1 evidence stay in sync."""
import json
from pathlib import Path
from tools.prepare_final_release import check

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_data_and_c1_metrics():
    _, result = check(ROOT)
    assert result['holdout']['breaches'] == 25
    assert result['holdout']['n'] == 468
    assert 0.00163 < result['holdout']['pinball_loss'] < 0.00164


def test_release_documentation_is_coherent():
    r = (ROOT / 'README.md').read_text(encoding='utf-8')
    d = (ROOT / 'docs/submission/FINAL_SUBMISSION_2026_11_19.md').read_text(encoding='utf-8')
    m = json.loads((ROOT / 'docs/submission/FINAL_SUBMISSION_MANIFEST.json').read_text(encoding='utf-8'))
    assert '<!-- FINAL_SUBMISSION_2026_11_19 -->' in r
    assert 'Phase 1' in d
    assert 'C1' in d
    assert m['selected_model'] == 'C1_12F_FROZEN_PHASE2_2026_11_06'
    assert m['holdout_breaches'] == 25


def test_line_endings_only_are_accepted_not_tampered_data(tmp_path):
    import hashlib
    from tools.prepare_final_release import eol_equivalent, portable_sha
    actual = b'col1,col2\n1,2\n3,4\n'
    expected = hashlib.sha256(actual).hexdigest()
    item = tmp_path/'feature_test.csv'
    item.write_bytes(actual.replace(b'\n', b'\r\n'))
    assert eol_equivalent(item, expected)
    assert portable_sha(item) == expected
    item.write_bytes(actual.replace(b'3,4', b'3,5').replace(b'\n',b'\r\n'))
    assert not eol_equivalent(item, expected)
    binary = tmp_path/'model_test.joblib'
    binary.write_bytes(actual.replace(b'\n',b'\r\n'))
    assert not eol_equivalent(binary,expected)
