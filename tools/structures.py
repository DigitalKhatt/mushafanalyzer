from dataclasses import dataclass

@dataclass
class WordRec:
    page: int
    line: int
    qpc: str
    text: str