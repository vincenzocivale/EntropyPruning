from src.data.wsi.corpora import _histai_case_id, _histai_rank, _is_histai_he


def test_histai_he_selection_and_case_id():
    path = "train/case_0042/slide_20x_H&E_0.tiff"
    assert _histai_case_id(path) == "case_0042"
    assert _is_histai_he(path)
    assert _histai_rank(path)[0] == 0


def test_histai_rejects_non_he():
    assert not _is_histai_he("case_1/slide_IHC_0.tiff")
