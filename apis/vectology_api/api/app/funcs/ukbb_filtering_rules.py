# flake8: noqa
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .preprocess import last_matching_group, last_matching_group_with_prefix

ukbb_pattern_list: List[Dict[str, Any]] = [
    {
        "pattern": r"^blood clot,? dvt,? bronchitis,? emphysema,? asthma,? rhinitis,? eczema,? allergy diagnosed by doctor: (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^cancer code,? self-reported: (.*)",
        "func": lambda x, pattern: last_matching_group_with_prefix(
            x, pattern, "cancer"
        ),
        "rule": "last_matching_group_with_prefix",
    },
    {
        "pattern": r"^contributory \(secondary\) causes of death: icd10: \w+\.?\w* (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^destinations on discharge from hospital \(recoded\): (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^diagnoses - (main|secondary) icd(9|10): \w+\.?\w* (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^external causes: \w+\.?\w* (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {"pattern": r"^eye problems/disorders: (.*)", "func": None, "rule": None},
    {
        "pattern": r"^illness,? injury,? bereavement,? stress in last 2 years: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^illnesses of (father|mother|siblings): (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^main speciality of consultant \(recoded\): (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^medication for cholesterol,? blood pressure or diabetes: (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^medication for cholesterol,? blood pressure,? diabetes,? or take exogenous hormones: (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^medication for pain relief,? constipation,? heartburn: (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^medication for smoking cessation, constipation, heartburn, allergies \(pilot\): (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^methods of discharge from hospital \(recoded\): (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^mouth/teeth dental problems: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^non-cancer illness code,? self-reported: (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^operation code: (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^operative procedures - (main|secondary) opcs: \w+\.?\w* (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^treatment speciality of consultant \(recoded\): (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^treatment/medication code: (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^type of cancer: icd10: \w+\.?\w* (.*)",
        "func": lambda x, pattern: last_matching_group_with_prefix(
            x, pattern, "cancer"
        ),
        "rule": "last_matching_group_with_prefix",
    },
    {
        "pattern": r"^types? of fat/oil used in cooking: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^types? of physical activity in last 4 weeks: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^underlying \(primary\) cause of death: icd10: \w+\.?\w* (.*)",
        "func": last_matching_group,
        "rule": "last_matching_group",
    },
    {
        "pattern": r"^vascular/heart problems diagnosed by doctor: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^attendance/disability/mobility allowance: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^blood sample #, note contents: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^current employment status: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^how are people in household related to participant: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^mineral and other dietary supplements: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^pct responsible for patient data: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^pct where patients gp was registered: (.*)",
        "func": None,
        "rule": None,
    },
    {"pattern": r"^qualifications: (.*)", "func": None, "rule": None},
    {
        "pattern": r"^reason for glasses/contact lenses: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^sources of admission to hospital \(recoded\): (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^thickness of butter/margarine spread on .+?: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^types? of large bap eaten: (.*)",
        "func": None,
        "rule": None,
    },
    {"pattern": r"^types? of meals eaten: (.*)", "func": None, "rule": None},
    {
        "pattern": r"^types? of sliced bread eaten: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^types? of special diet followed: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^types? of spread used on .+?: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^vitamin and mineral supplements: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^vitamin and/or mineral supplement use: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^why (reduced|stopped) smoking: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^types? of transport used .+?: (.*)",
        "func": None,
        "rule": None,
    },
    {"pattern": r"^doctor diagnosed (.*)", "func": None, "rule": None},
    {
        "pattern": r"^ingredients in canned soup: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^intended management of patient \(recoded\): (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^methods of admission to hospital \(recoded\): (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^pain type\(s\) experienced in last month: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^transport type for commuting to job workplace: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^types? of (baguette|bread roll|yogurt) eaten: (.*)",
        "func": None,
        "rule": None,
    },
    {
        "pattern": r"^types? of spreads/sauces consumed: (.*)",
        "func": None,
        "rule": None,
    },
]


@dataclass
class PatternItem:
    pattern: str
    func: Optional[Callable]
    rule: str


ukbb_pattern_items = [
    PatternItem(pattern=item["pattern"], func=item["func"], rule=item["rule"])
    for item in ukbb_pattern_list
]
