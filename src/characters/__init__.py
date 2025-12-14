"""
Character module for defining character configurations.
"""

from .base import CharacterBase
from .paul_graham import PaulGraham
from .palki import Palki
from .tony_stark import TonyStark
from .yann import YannLeCun
from .donald_trump import DonaldTrump
from .barack_obama import BarackObama
from .elon_musk import ElonMusk
from .joe_biden import JoeBiden
from .bill_gates import BillGates
from .oprah_winfrey import OprahWinfrey
from .mark_zuckerberg import MarkZuckerberg

__all__ = [
    "CharacterBase", 
    "PaulGraham", "Palki", "TonyStark", "YannLeCun",
    "DonaldTrump", "BarackObama",
    "ElonMusk", "JoeBiden", "BillGates", "OprahWinfrey", "MarkZuckerberg"
]


