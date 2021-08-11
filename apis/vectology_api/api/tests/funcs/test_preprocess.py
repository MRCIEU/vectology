from app.funcs.preprocess import (
    common_preproc,
    last_matching_group,
    last_matching_group_with_prefix,
)


def test_common_preproc():
    orig_text = r"Foo\tb  aR"
    res = common_preproc(orig_text)
    expected_res = "foo b ar"
    assert res == expected_res


def test_last_matching_group():
    pattern = r"^diagnoses - (main|secondary) icd(9|10): (\w+\.?\w*) (.*)"
    orig_text = "Diagnoses - Main icd10: D12.6 Colon, unspecified"
    res = last_matching_group(common_preproc(orig_text), pattern)
    expected_res = "colon, unspecified"
    assert res == expected_res


def test_last_matching_group_with_prefix():
    pattern = r"^diagnoses - (main|secondary) icd(9|10): (\w+\.?\w*) (.*)"
    orig_text = "Diagnoses - Main icd10: D12.6 Colon, unspecified"
    prefix = "diagnoses"
    res = last_matching_group_with_prefix(
        common_preproc(orig_text), pattern, prefix
    )
    expected_res = "diagnoses colon, unspecified"
    assert res == expected_res
