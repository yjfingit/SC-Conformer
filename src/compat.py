"""Shim old moabb dataset names so braindecode's stale compat layer imports.

Import this BEFORE `import braindecode`.
"""
import moabb.datasets as _md

_ALIASES = {
    "BNCI2014001": "BNCI2014_001",
    "BNCI2014004": "BNCI2014_004",
    "BNCI2015001": "BNCI2015_001",
    "BNCI2015004": "BNCI2015_004",
    "HGD": "Schirrmeister2017",
    "Weibo2014": "Weibo2014",
}
for _old, _new in _ALIASES.items():
    if not hasattr(_md, _old) and hasattr(_md, _new):
        setattr(_md, _old, getattr(_md, _new))
