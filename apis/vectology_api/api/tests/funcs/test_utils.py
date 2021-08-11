from app.funcs.utils import complete_pairs, nest_pairs

pairs = [
    {"idx1": 0, "idx2": 1, "cosine_similarity": 0.1},
    {"idx1": 0, "idx2": 2, "cosine_similarity": 0.2},
    {"idx1": 1, "idx2": 2, "cosine_similarity": 0.3},
]


def test_common_pairs():
    expected_res = [
        {"idx1": 0, "idx2": 1, "cosine_similarity": 0.1},
        {"idx1": 0, "idx2": 2, "cosine_similarity": 0.2},
        {"idx1": 1, "idx2": 2, "cosine_similarity": 0.3},
        {"idx1": 1, "idx2": 0, "cosine_similarity": 0.1},
        {"idx1": 2, "idx2": 0, "cosine_similarity": 0.2},
        {"idx1": 2, "idx2": 1, "cosine_similarity": 0.3},
        {"idx1": 0, "idx2": 0, "cosine_similarity": 1},
        {"idx1": 1, "idx2": 1, "cosine_similarity": 1},
        {"idx1": 2, "idx2": 2, "cosine_similarity": 1},
    ]
    res = complete_pairs(pairs)
    assert expected_res == res


def test_nested_pairs():
    expected_res = [
        {
            "name": "0",
            "data": [
                {"x": "0", "y": 1.0},
                {"x": "1", "y": 0.1},
                {"x": "2", "y": 0.2},
            ],
        },
        {
            "name": "1",
            "data": [
                {"x": "0", "y": 0.1},
                {"x": "1", "y": 1.0},
                {"x": "2", "y": 0.3},
            ],
        },
        {
            "name": "2",
            "data": [
                {"x": "0", "y": 0.2},
                {"x": "1", "y": 0.3},
                {"x": "2", "y": 1.0},
            ],
        },
    ]
    res = nest_pairs(pairs)
    assert expected_res == res
