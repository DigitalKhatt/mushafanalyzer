"""Shape classes for base letters.

Two letters that differ only by dots share one written body, so for shape
analysis they are the same instance class -- the dots are separate mark
components in the analyzer's output anyway. The class sets are the ones
already used for search in ``oldmadinaanalyzer/src/renderer.ts``; the joining
sets themselves live in ``tools/paw_segmentation/arabic_joining.py`` and are
not duplicated here.

The interesting case, also taken from the renderer: in initial and medial
position beh/teh/theh/noon/yeh all reduce to the same *senna* tooth, so a
query for "initial beh" that means "this shape" wants all of them, while a
query for final beh does not (final noon and final yeh are different bowls).
``shape_class`` encodes exactly that.
"""

from typing import Dict, List

# One body shape per entry; the members differ only in dots or hamza, which
# the analyzer segments as separate mark components.
SHAPE_CLASSES: Dict[str, str] = {
    "alef": "اآٱأإ",
    "beh": "بتث",
    "hah": "جحخ",
    "dal": "دذ",
    "reh": "رز",
    "seen": "سش",
    "sad": "صض",
    "tah": "طظ",
    "ain": "عغ",
    "feh": "ف",
    "qaf": "ق",
    "kaf": "ك",
    "lam": "ل",
    "meem": "م",
    "noon": "ن",
    "heh": "ةه",
    "waw": "وؤ",
    "yeh": "يىئ",
    "hamza": "ء",
    "tatweel": "ـ",
}

# The shared tooth. Only in initial and medial position: their final and
# isolated forms are genuinely different shapes.
SENNA = set("بتثنيئى")
SENNA_CLASS = "senna"

_CLASS_OF_CHAR = {ch: name for name, chars in SHAPE_CLASSES.items() for ch in chars}


def shape_class(char: str, form: str) -> str:
  """Return the body-shape class of one base letter in one joining form."""
  if not char:
    return ""
  if form in ("initial", "medial") and char in SENNA:
    return SENNA_CLASS
  return _CLASS_OF_CHAR.get(char, char)


def folds_to() -> Dict[str, str]:
  """Classes that give way to another one in initial and medial position.

  ``{"beh": "senna", "noon": "senna", "yeh": "senna"}``. A caller that filters
  on class and form needs this to tell "no such letters" from "they are stored
  under a different class here", which is the difference between a bug and the
  design.
  """
  return {_CLASS_OF_CHAR.get(char, char): SENNA_CLASS for char in SENNA}


def class_members(name: str) -> List[str]:
  """Characters belonging to a shape class, for building a query."""
  if name == SENNA_CLASS:
    return sorted(SENNA)
  return list(SHAPE_CLASSES.get(name, ""))


def class_names() -> List[str]:
  """Every class a stored row can carry, for the GUI's class picker."""
  return [SENNA_CLASS] + list(SHAPE_CLASSES)


# How many ink components one instance of a mark class is drawn with. The
# analyzer groups mark components per PAW *per class*, so a group of two
# components means two fathas but a single dammatan -- the difference is the
# class, not the geometry. Anything absent is a single-component mark.
#
# Components can also merge in the scan (two dots touching read as one blob),
# so a group smaller than the part count is still one instance. A group
# *larger* than the part count would be several instances of a multi-part
# class in one PAW, which cannot be split without a mark-assignment model;
# those rows are recorded whole and flagged ambiguous instead of guessed at.
MARK_PARTS: Dict[str, int] = {
    "dot_above_two": 2,
    "dot_below_two": 2,
    "dot_above_three": 3,
    "dot_below_three": 3,
    "fathatan": 2,
    "kasratan": 2,
    "dammatan": 2,
    "waqf_three_dots": 3,
}


def mark_parts(class_name: str) -> int:
  return MARK_PARTS.get(class_name, 1)
