import re
import numpy as np
import pandas as pd

NAME_ALLOWED = re.compile(r"[^A-Z ]+")

def norm_name(s: str) -> str:
    """
    Uppercase, strip, replace NBSP/tabs, remove punctuation/digits,
    collapse multiple spaces -> single space.
    """
    if pd.isna(s):
        return ""
    s = str(s).upper()
    s = s.replace("\xa0", " ").replace("\u200b", "").replace("\t", " ")
    s = NAME_ALLOWED.sub(" ", s)            # keep only A-Z and spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_ribu(val) -> float:
    """
    Parse 'ribu jiwa' in both styles:
      - 1,464.64   (comma thousands, dot decimals)
      - 1.464,64   (dot thousands, comma decimals)
      - 1464.64 / 1464,64 (no thousands sep)
    Returns float in ribu (thousands), or np.nan.
    """
    if pd.isna(val): return np.nan
    s = str(val).strip()

    # remove spaces and NBSPs
    s = s.replace("\xa0", "").replace(" ", "")

    # pattern: comma thousands, dot decimals
    if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")  # kill thousands commas
        try: return float(s)    # dot is decimal
        except: return np.nan

    # pattern: dot thousands, comma decimals (ID style)
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", s):
        s = s.replace(".", "").replace(",", ".")
        try: return float(s)
        except: return np.nan

    # plain number with either , or .
    s2 = s.replace(",", ".")
    try: return float(s2)
    except: return np.nan
