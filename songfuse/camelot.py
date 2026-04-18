"""Camelot Wheel harmonic compatibility.

Codes like '8B' (C major) / '8A' (A minor). Compatible moves: same code, ±1 (same
letter), or same number with letter flipped (relative switch)."""

from __future__ import annotations

from dataclasses import dataclass

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# (pitch_class, mode) -> (number, letter). mode: 0 major, 1 minor.
_MAJOR = {0: 8, 1: 3, 2: 10, 3: 5, 4: 12, 5: 7, 6: 2, 7: 9, 8: 4, 9: 11, 10: 6, 11: 1}
_MINOR = {0: 5, 1: 12, 2: 7, 3: 2, 4: 9, 5: 4, 6: 11, 7: 6, 8: 1, 9: 8, 10: 3, 11: 10}


@dataclass(frozen=True)
class Camelot:
    number: int  # 1..12
    letter: str  # 'A' minor / 'B' major

    def __str__(self) -> str:
        return f"{self.number}{self.letter}"

    @property
    def mode(self) -> str:
        return "minor" if self.letter == "A" else "major"


def from_key(pitch_class: int, mode: str) -> Camelot:
    pc = pitch_class % 12
    if mode.startswith("min"):
        return Camelot(_MINOR[pc], "A")
    return Camelot(_MAJOR[pc], "B")


def key_name(pitch_class: int, mode: str) -> str:
    root = PITCH_NAMES[pitch_class % 12]
    return f"{root} {'minor' if mode.startswith('min') else 'major'}"


def is_compatible(a: Camelot, b: Camelot) -> bool:
    """Standard harmonic-mixing rules: same, ±1 on the wheel (same letter), or relative."""
    if a == b:
        return True
    if a.letter == b.letter:
        diff = (a.number - b.number) % 12
        if diff in (1, 11):
            return True
    if a.number == b.number and a.letter != b.letter:
        return True
    return False


def semitone_shift_to_match(src_pc: int, src_mode: str, dst_pc: int, dst_mode: str,
                            max_shift: int = 2) -> int:
    """Return minimum semitone shift for `src` track so it becomes Camelot-compatible
    with `dst`. Returns 0 if already compatible or if no shift within ±max_shift works."""
    src = from_key(src_pc, src_mode)
    dst = from_key(dst_pc, dst_mode)
    if is_compatible(src, dst):
        return 0
    # Try ±1, ±2… semitone shifts; prefer the smallest magnitude.
    for magnitude in range(1, max_shift + 1):
        for sign in (-1, 1):
            shift = sign * magnitude
            shifted = from_key(src_pc + shift, src_mode)
            if is_compatible(shifted, dst):
                return shift
    return 0
