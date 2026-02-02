import numpy as np

def gbm_predict(model, X2d: np.ndarray) -> np.ndarray:
    """
    - lightgbm.basic.Booster (Из lgb.train)
    - lightgbm.sklearn.LGBMRegressor (из .fit)
    """
    if hasattr(model, "num_trees"):          # Booster
        return model.predict(X2d)
    return model.booster_.predict(X2d)       # LGBMRegressor
