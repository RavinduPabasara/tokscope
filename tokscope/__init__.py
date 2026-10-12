"""tokscope — measure how efficiently tokenizers handle your language.

>>> import tokscope
>>> report = tokscope.analyze("මම සිංහල භාෂාව ඉගෙන ගනිමි.", "ws:")
>>> report.fertility
1.0
"""

from .adapters import load
from .metrics import Report
from .runner import analyze, compare

__version__ = "0.2.0"

__all__ = ["analyze", "compare", "load", "Report", "__version__"]
