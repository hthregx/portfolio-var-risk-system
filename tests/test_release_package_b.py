from scripts.check_release_package_b import (
    OPTIONAL_DAY22_B,
    REQUIRED_FILES,
    ROOT,
    check_day22_b_assets,
    check_required_files,
)


def test_required_release_assets_exist():
    assert check_required_files()


def test_required_paths_are_files():
    for relative in REQUIRED_FILES:
        assert (ROOT / relative).is_file()


def test_day22_b_check_is_non_blocking():
    result = check_day22_b_assets()

    assert set(result) == set(OPTIONAL_DAY22_B)
    assert all(
        isinstance(value, bool)
        for value in result.values()
    )