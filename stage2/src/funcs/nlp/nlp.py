from string import punctuation
from typing import List

import numpy as np
import sent2vec
from loguru import logger
from nltk import word_tokenize
from nltk.corpus import stopwords

ECHO_STEP = 200


def preprocess_sentence(text: str) -> str:

    stop_words = set(stopwords.words("english"))
    text = text.replace("/", " / ")
    text = text.replace(".-", " .- ")
    text = text.replace(".", " . ")
    text = text.replace("'", " ' ")
    text = text.lower()

    tokens = [
        token
        for token in word_tokenize(text)
        if token not in punctuation and token not in stop_words
    ]

    return " ".join(tokens)


def ascii_fy(text: str) -> str:
    text = text.encode("ascii", "ignore").decode()
    for _ in punctuation:
        text = text.replace(_, " ")
    return text


def harmonize_vectors(
    main_vector: np.ndarray,
    addons: List[np.ndarray],
) -> np.ndarray:
    addon_shape = (100,)
    main_vector_shape = (700,)
    pad_width = int((main_vector_shape[0] - addon_shape[0]) / 2)
    addons_padded = [
        np.pad(_, pad_width, mode="constant", constant_values=(0)).reshape(
            main_vector_shape
        )
        for _ in addons
    ]
    res_vector = main_vector
    for _ in addons_padded:
        res_vector = res_vector + _
    res_vector = res_vector.reshape(main_vector_shape)
    return res_vector


def biosentvec_encode_terms(
    text_list: List[str], biosentvec_model: sent2vec.Sent2vecModel
) -> np.ndarray:
    def _embed(idx: int, total: int, text: str) -> np.ndarray:
        if idx % ECHO_STEP == 0:
            logger.info(f"#{idx} / {total}")
        preprocessed = preprocess_sentence(text)
        embed = biosentvec_model.embed_sentence(preprocessed)
        res = embed[0]
        return res

    res = np.array(
        [
            _embed(idx=idx, total=len(text_list), text=_)
            for idx, _ in enumerate(text_list)
        ]
    )
    return res
