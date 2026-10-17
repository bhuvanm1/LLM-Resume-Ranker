"""Bias-aware evaluation.

1. BLIND SCREENING  - redact PII and protected-attribute signals before scoring/ranking.
2. AUDIT            - compare scores on raw vs. redacted text and run a counterfactual
                      gender-swap test; large deltas would reveal the pipeline reacting
                      to protected attributes (or proxies for them).
"""
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<!\d)\+?\d[\d ().-]{8,}\d(?!\d)")
URL = re.compile(r"(?:https?://|www\.|linkedin\.com|github\.com)\S+", re.I)
PRONOUN = re.compile(r"\b(he|him|his|she|her|hers|mr|mrs|ms|miss)\b\.?", re.I)
GENDER_WORDS = re.compile(r"\b(male|female|man|woman|husband|wife|father|mother|mom|dad|gender)\b", re.I)
AGE_WORDS = re.compile(r"\b(age[d]?:?\s*\d{2}|\d{2}\s*years?\s*old|date of birth|dob:?\s*[\d/.-]+|born\s+(?:in\s+)?\d{4})", re.I)
MARITAL = re.compile(r"\b(married|single|unmarried|divorced|marital status|children|kids|pregnan\w*)\b", re.I)
NATIONAL = re.compile(r"\b(nationality|citizenship|religion|race|ethnicity|caste|visa status)\b[:\s]*\w*", re.I)
GRAD_YEAR = re.compile(r"\b(class of\s*)?(19[5-9]\d|20[0-2]\d)\b(?=[^\n]{0,20}(graduat|bachelor|master|b\.?s|m\.?s|degree))", re.I)

_SWAP = {"he": "she", "she": "he", "him": "her", "her": "him", "his": "her", "hers": "his",
         "mr": "ms", "ms": "mr", "mrs": "mr", "male": "female", "female": "male",
         "man": "woman", "woman": "man", "husband": "wife", "wife": "husband",
         "father": "mother", "mother": "father"}


def _first_line_name(text: str) -> str | None:
    first = next((l.strip() for l in text.splitlines() if l.strip()), "")
    if 2 <= len(first.split()) <= 4 and not re.search(r"[\d@:/]", first) and len(first) < 40:
        return first
    return None


def redact(text: str, candidate_name: str | None = None) -> tuple[str, list[str]]:
    flagged: list[str] = []
    out = text

    def sub(pat, label, repl=" "):
        nonlocal out
        for m in pat.finditer(out):
            flagged.append(f"{label}: {m.group(0).strip()[:30]}")
        out = pat.sub(repl, out)

    def sub_phone():
        nonlocal out
        def r(m):
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) < 10 or re.fullmatch(r"\s*(19|20)\d{2}\s*[-–—]\s*((19|20)\d{2})\s*", m.group(0)):
                return m.group(0)
            flagged.append(f"phone: {m.group(0).strip()[:30]}")
            return " "
        out = PHONE.sub(r, out)

    name = _first_line_name(text)
    for n in filter(None, {name, candidate_name}):
        if n in out:
            flagged.append(f"name: {n}")
            out = out.replace(n, " ")
            for part in n.split():
                if len(part) > 2:
                    out = re.sub(rf"\b{re.escape(part)}\b", " ", out)
    sub(EMAIL, "email"); sub_phone(); sub(URL, "url")
    sub(AGE_WORDS, "age/dob"); sub(MARITAL, "marital/family"); sub(NATIONAL, "nationality/religion")
    sub(GENDER_WORDS, "gender"); sub(PRONOUN, "gendered pronoun"); sub(GRAD_YEAR, "graduation year (age proxy)")
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out, list(dict.fromkeys(flagged))


def gender_swap(text: str) -> str:
    def rep(m):
        w = m.group(0)
        s = _SWAP.get(w.lower().rstrip("."), w)
        return s.capitalize() if w[0].isupper() else s
    return re.sub(r"\b(" + "|".join(_SWAP) + r")\b", rep, text, flags=re.I)


def verdict(delta_counterfactual: float, flagged: list[str]) -> str:
    if abs(delta_counterfactual) < 1.0:
        base = "No sensitivity to gendered wording detected."
    else:
        base = f"Score moved {delta_counterfactual:+.1f} pts under a gender-swap test - review for bias."
    if flagged:
        base += f" {len(flagged)} protected/PII signal(s) were redacted before ranking."
    return base
