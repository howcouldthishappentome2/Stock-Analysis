"""
Fama-French-Carhart 6-Factor Model implementation
Factors: MKT, SMB, HML, RMW, CMA, MOM
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple

class FamaFrenchCarhart6Factor:
    def __init__(self, returns: pd.Series, factors: pd.DataFrame):
        """
        returns: pd.Series of asset returns (index: datetime)
        factors: pd.DataFrame with columns ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM'] (index: datetime)
        """
        self.returns = returns
        self.factors = factors
        self.model = None
        self.coef_ = None
        self.alpha_ = None
        self.r2_ = None

    def fit(self) -> None:
        X = self.factors.copy()
        X = sm.add_constant(X)
        y = self.returns.loc[X.index]
        model = sm.OLS(y, X, missing='drop').fit()
        self.model = model
        self.coef_ = model.params
        self.alpha_ = model.params['const']
        self.r2_ = model.rsquared

    def summary(self) -> Optional[str]:
        if self.model is not None:
            return self.model.summary().as_text()
        return None

    def predict(self, factors: Optional[pd.DataFrame] = None) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fit yet.")
        if factors is None:
            factors = self.factors
        X = sm.add_constant(factors)
        return self.model.predict(X)

try:
    import statsmodels.api as sm
except ImportError:
    sm = None
    print("statsmodels is required for Fama-French-Carhart 6-factor regression.")
