from dataclasses import dataclass
from typing import Any, Optional

from algo.binance.fit import ProductFitResult, FitResults


@dataclass
class ProductModel:
    stacked_lms: list[Any]
    rescaling: float
    column_names: list[str]
    caps: Optional[tuple[float, float]]

    @classmethod
    def from_fit_results(cls, prod_res: ProductFitResult,
                         glob_res: FitResults,
                         column_names: list[str]) -> 'ProductModel':
        return cls(stacked_lms=[glob_res.fitted_model, prod_res.res.fitted_model],
                   rescaling=prod_res.rescaling,
                   caps=prod_res.caps,
                   column_names=column_names)

    def predict(self, X):
        pred = sum(lm.predict(X) for lm in self.stacked_lms)
        if self.caps is not None:
            pred[pred < self.caps[0]] = self.caps[0]
            pred[pred > self.caps[1]] = self.caps[1]
        pred *= self.rescaling
        return pred
