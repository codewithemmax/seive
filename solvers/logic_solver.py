"""Layer 1 deterministic logic/deduction solver — zero tokens.

Handles syllogisms and multiple-choice questions with clear structural patterns.
"""
import re


def _extract_choices(text: str) -> dict[str, str]:
    choices = {}
    for m in re.finditer(r"\(([A-E])\)\s*([^(]+)", text):
        choices[m.group(1)] = m.group(2).strip().rstrip(".")
    return choices


# Matches: "All X are Y." where X and Y can include hyphens/spaces
_ALL_ARE = re.compile(
    r"[Aa]ll\s+([\w][\w\s-]*?)\s+are\s+([\w][\w\s-]+?)\s*\.",
    re.IGNORECASE,
)
# Matches: "Z is/are X" (the minor premise)
_IS_ARE = re.compile(r"([\w][\w\s-]+?)\s+(?:is|are)\s+([\w][\w\s-]+?)\s*\.", re.IGNORECASE)
# Matches: "No X are Y."
_NO_ARE = re.compile(
    r"[Nn]o\s+([\w][\w\s-]*?)\s+are\s+([\w][\w\s-]+?)\s*\.",
    re.IGNORECASE,
)


def _find_yes_choice(choices: dict) -> str | None:
    for k, v in choices.items():
        if re.search(r"\byes\b", v, re.IGNORECASE):
            return k
    return None


def _find_no_choice(choices: dict) -> str | None:
    for k, v in choices.items():
        if re.search(r"\bno\b|\bnot\b", v, re.IGNORECASE):
            return k
    return None


def _detect_syllogism(text: str) -> str | None:
    """Modus ponens: All A are B. C is/are A. → C is B (Yes)."""
    major = _ALL_ARE.search(text)
    if not major:
        return None
    subject_class = major.group(1).strip().lower()   # e.g. "mammals"
    predicate = major.group(2).strip()               # e.g. "warm-blooded"

    # Find minor premise: "Z are/is <subject_class>"
    minor_pat = re.compile(
        rf"([\w][\w\s-]+?)\s+(?:is|are)\s+{re.escape(subject_class)}\s*[.,]",
        re.IGNORECASE,
    )
    minor = minor_pat.search(text)
    if not minor:
        return None

    choices = _extract_choices(text)
    yes_key = _find_yes_choice(choices)
    if yes_key:
        return choices[yes_key]   # return full text e.g. "Yes"
    return f"Yes, {minor.group(1).strip()} are {predicate}."


def _detect_negation_syllogism(text: str) -> str | None:
    """No A are B. C is A. → C is not B."""
    major = _NO_ARE.search(text)
    if not major:
        return None
    subject_class = major.group(1).strip().lower()
    predicate = major.group(2).strip()

    minor_pat = re.compile(
        rf"([\w][\w\s-]+?)\s+(?:is|are)\s+{re.escape(subject_class)}\s*[.,]",
        re.IGNORECASE,
    )
    minor = minor_pat.search(text)
    if not minor:
        return None

    choices = _extract_choices(text)
    no_key = _find_no_choice(choices)
    if no_key:
        return choices[no_key]
    return f"No, {minor.group(1).strip()} are not {predicate}."


def _detect_conditional(text: str) -> str | None:
    """If P then Q. P. → Q (modus ponens)."""
    m = re.search(
        r"[Ii]f\s+(.+?),?\s+then\s+(.+?)\.\s*(.+?)\s+is\s+true",
        text, re.IGNORECASE,
    )
    if not m:
        return None
    consequent = m.group(2).strip()
    choices = _extract_choices(text)
    for v in choices.values():
        if consequent.lower() in v.lower():
            return v
    return f"{consequent} is true."


def solve_logic(task: dict) -> dict | None:
    text = task.get("input", "")

    for fn in (_detect_syllogism, _detect_negation_syllogism, _detect_conditional):
        result = fn(text)
        if result:
            return {
                "answer": result,
                "confidence": 0.95,
                "tokens_used": 0,
                "source": "logic_solver",
            }

    return None
