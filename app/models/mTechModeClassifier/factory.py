from importlib import import_module

def make_strategy(unit: str):
    mod = import_module(f"app.models.mTechModeClassifier.mode_strategies.{unit.lower()}")
    class_name = f"{unit.upper()}Strategy"
    return getattr(mod, class_name)()
