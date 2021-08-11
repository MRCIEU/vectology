import re
from enum import Enum
from typing import Callable, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, validator

from app.funcs.preprocess import common_preproc
from app.funcs.ukbb_filtering_rules import PatternItem, ukbb_pattern_items

router = APIRouter()


class SourceName(str, Enum):
    ukbb = "ukbb"


class PreprocInput(BaseModel):
    text_list: List[str]
    source: SourceName = SourceName.ukbb

    @validator("text_list", whole=True)
    def text_list_length(cls, v):
        limit = 50
        if len(v) > limit:
            raise HTTPException(
                status_code=400, detail=f"Too many items. Limit: {limit}."
            )
        return v


class PreprocResponse(BaseModel):
    result: str
    rule: str
    pattern: Optional[str]


@router.get("/preprocess", response_model=PreprocResponse)
def get_preproc(text: str, source: SourceName = SourceName.ukbb):
    """Preprocess a text.

    `source`: The origin of this text -- to determine the
    appropriate preprocessing steps.
    """
    if source.value == "ukbb":
        res = preprocess_text_ukbb(text)
    return res


@router.post("/preprocess", response_model=List[PreprocResponse])
def post_preproc(input: PreprocInput):
    """Preprocess a text.
    """
    if input.source.value == "ukbb":
        res = [preprocess_text_ukbb(text) for text in input.text_list]
    return res


def match_text_to_pattern(
    text: str, pattern_items: List[PatternItem]
) -> Tuple[Optional[str], Optional[Callable], Optional[str]]:
    pattern = None
    func = None
    rule = "asis"
    for pattern_item in pattern_items:
        match_res = re.match(pattern_item.pattern, text)
        if match_res:
            pattern = pattern_item.pattern
            func = pattern_item.func
            if pattern_item.rule is not None:
                rule = pattern_item.rule
            break
    return (pattern, func, rule)


def preprocess_text_ukbb(text: str):
    # common preprocessing
    text = common_preproc(text)
    # determine the pattern group of the text, and the assoc pattern and func
    pattern, func, rule = match_text_to_pattern(
        text=text, pattern_items=ukbb_pattern_items
    )
    # apply processing
    if func is not None:
        text = func(text, pattern)
    res = {"result": text, "rule": rule, "pattern": pattern}
    return res
