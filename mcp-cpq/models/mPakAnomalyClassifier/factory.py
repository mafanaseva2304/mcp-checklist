from importlib import import_module
import re


def _normalize_unit(unit: str) -> str:
    """только цифры+буквы  (QR-116B -> qr116b)"""
    return re.sub(r"[^0-9a-zA-Z]", "", unit)


def make_strategy(unit: str):
    base = _normalize_unit(unit)
    mod = import_module(
        f"models.mPakAnomalyClassifier.pak_strategies.{base.lower()}"
    )
    class_name = f"{base.upper()}Strategy"
    return getattr(mod, class_name)()
