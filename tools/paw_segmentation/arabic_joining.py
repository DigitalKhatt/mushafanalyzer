"""Arabic joining behaviour: base-letter extraction, joining forms, PAW split.

The joining sets and `joining_form` are copied verbatim from
`tools/build_instance_index.py` so the two agree; that file is deliberately
left untouched for now (it is the current pilot index builder).

A PAW ("piece of Arabic word") is one connected run of ink in a word's body.
Arabic breaks a word into PAWs after every letter that does not join to its
left, so the PAW decomposition is derivable from the text alone -- which is
what makes it a hard constraint for letter segmentation.
"""

import unicodedata
from typing import List, Tuple

DUAL_JOIN = set("بتثجحخسشصضطظعغفقكلمنهيئى")
RIGHT_JOIN = set("آاٱأإدذرزوؤة")
BASES = DUAL_JOIN | RIGHT_JOIN | set("ءـ٠١٢٣٤٥٦٧٨٩۝۞")

# U+0640 TATWEEL elongates the connector between two letters; it carries a mark
# (a hamza, in ٱلْـَٔاخِرِ) but has no body of its own. Left in the base sequence it
# is a letter the segmenter is asked to locate and cannot, so every PAW holding
# one fails. It still has to be present while joining forms are computed --
# making its neighbours join is its whole purpose -- so it is dropped only
# afterwards, by ``describe_word``.
TRANSPARENT = frozenset("ـ")


def joining_form(bases, index):
  current = bases[index]
  joins_previous = index > 0 and bases[index - 1] in DUAL_JOIN | {"ـ"} and current != "ء"
  joins_next = index + 1 < len(bases) and current in DUAL_JOIN | {"ـ"} and bases[index + 1] != "ء"
  if joins_previous and joins_next:
    return "medial"
  if joins_previous:
    return "final"
  if joins_next:
    return "initial"
  return "isolated"


def base_letters(text: str) -> List[str]:
  """Drop combining marks and anything that is not a recognized base letter."""
  out = []
  for ch in text:
    if unicodedata.category(ch) == "Mn":
      continue
    if ch in BASES:
      out.append(ch)
  return out


def split_paws(bases: List[str]) -> List[Tuple[int, int]]:
  """Split base letters into PAWs, returned as [start, end) index ranges.

  A PAW ends after any letter whose joining form is final or isolated, i.e.
  any letter that does not join leftwards to its successor.
  """
  paws: List[Tuple[int, int]] = []
  start = 0
  for i in range(len(bases)):
    if joining_form(bases, i) in ("final", "isolated"):
      paws.append((start, i + 1))
      start = i + 1
  if start < len(bases):
    paws.append((start, len(bases)))
  return paws


def describe_word(text: str):
  """Return (bases, forms, paw_ranges) for a word's expected text.

  Transparent bases are removed after the joining forms and the PAW split have
  been decided on the full sequence, so a tatweel still joins what it sits
  between while contributing no letter to locate. The ranges are remapped onto
  the letters that survive; a PAW that was nothing but tatweel would leave no
  range at all, which is the honest answer -- there is no letter in it.
  """
  full = base_letters(text)
  forms = [joining_form(full, i) for i in range(len(full))]
  paws = split_paws(full)
  keep = [ch not in TRANSPARENT for ch in full]
  if all(keep):
    return full, forms, paws
  kept_before = [0]
  for flag in keep:
    kept_before.append(kept_before[-1] + int(flag))
  return ([base for base, flag in zip(full, keep) if flag],
          [form for form, flag in zip(forms, keep) if flag],
          [(kept_before[lo], kept_before[hi]) for lo, hi in paws
           if kept_before[hi] > kept_before[lo]])
