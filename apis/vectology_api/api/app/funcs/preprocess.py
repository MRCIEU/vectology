import re


def common_preproc(x: str) -> str:
    """Common pre-processing steps.

    - convert to lowercase
    - replace tabs with spaces
    - remove multiple spaces
    - strip leading trailing spaces
    """
    x = x.lower()
    x = x.replace(r"\t", " ")
    x = re.sub(r"\s+", " ", x)
    x = x.strip()
    return x


def last_matching_group(x: str, pattern: str) -> str:
    """Return last matching group.
    """
    match = re.match(pattern, x)
    if not match:
        return x
    else:
        groups = match.groups()
        return match.group(len(groups))


def last_matching_group_with_prefix(x: str, pattern: str, prefix: str) -> str:
    """Return last matching group.
    """
    match = re.match(pattern, x)
    if not match:
        return x
    else:
        groups = match.groups()
        res = "{prefix} {text}".format(
            prefix=prefix, text=match.group(len(groups))
        )
        return res
