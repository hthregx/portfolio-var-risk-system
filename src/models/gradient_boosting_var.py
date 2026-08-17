from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


DEFAULT_ALPHA = 0.05
DEFAULT_N_ESTIMATORS = 100
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_MAX_DEPTH = 2
DEFAULT_MIN_SAMPLES_LEAF = 5
DEFAULT_SUBSAMPLE = 1.0
DEFAULT_RANDOM_STATE = 42


class GradientBoostingVaR:
    """Gradient Boosting quantile model for one-day portfolio VaR."""

    def __init__(
        self,
        *,
        alpha: float = DEFAULT_ALPHA,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        max_depth: int = DEFAULT_MAX_DEPTH,
        min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
        subsample: float = DEFAULT_SUBSAMPLE,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.alpha = self._validate_alpha(alpha)
        self.n_estimators = self._validate_positive_integer(
            n_estimators,
            "n_estimators",
        )
        self.learning_rate = self._validate_positive_float(
            learning_rate,
            "learning_rate",
        )
        self.max_depth = self._validate_positive_integer(
            max_depth,
            "max_depth",
        )
        self.min_samples_leaf = self._validate_positive_integer(
            min_samples_leaf,
            "min_samples_leaf",
        )
        self.subsample = self._validate_subsample(
            subsample
        )
        self.random_state = self._validate_random_state(
            random_state
        )

        self._model: GradientBoostingRegressor | None = None
        self._feature_names: tuple[str, ...] | None = None

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        if isinstance(alpha, bool) or not isinstance(
            alpha,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError(
                "Alpha must be numeric."
            )

        value = float(alpha)

        if not np.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(
                "Alpha must be finite and strictly between 0 and 1."
            )

        return value

    @staticmethod
    def _validate_positive_integer(
        value: int,
        name: str,
    ) -> int:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

        normalized = int(value)

        if normalized <= 0:
            raise ValueError(
                f"{name} must be a positive integer."
            )

        return normalized

    @staticmethod
    def _validate_positive_float(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError(
                f"{name} must be a positive finite number."
            )

        normalized = float(value)

        if not np.isfinite(normalized) or normalized <= 0.0:
            raise ValueError(
                f"{name} must be a positive finite number."
            )

        return normalized

    @staticmethod
    def _validate_subsample(
        subsample: float,
    ) -> float:
        value = GradientBoostingVaR._validate_positive_float(
            subsample,
            "subsample",
        )

        if value > 1.0:
            raise ValueError(
                "subsample must be less than or equal to 1.0."
            )

        return value

    @staticmethod
    def _validate_random_state(
        random_state: int,
    ) -> int:
        if isinstance(random_state, bool) or not isinstance(
            random_state,
            (int, np.integer),
        ):
            raise ValueError(
                "random_state must be an integer."
            )

        return int(random_state)

    @staticmethod
    def _validate_features(
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise ValueError(
                "Features must be provided as a pandas DataFrame."
            )

        if features.empty:
            raise ValueError(
                "Features cannot be empty."
            )

        if features.columns.has_duplicates:
            raise ValueError(
                "Feature columns must be unique."
            )

        if features.index.has_duplicates:
            raise ValueError(
                "Feature index must be unique."
            )

        if not all(
            isinstance(column, str)
            for column in features.columns
        ):
            raise ValueError(
                "Feature column names must be strings."
            )

        try:
            numeric_features = (
                features
                .apply(
                    pd.to_numeric,
                    errors="raise",
                )
                .astype("float64")
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "All feature values must be numeric."
            ) from error

        if numeric_features.isna().any().any():
            raise ValueError(
                "Features cannot contain missing values."
            )

        if not np.isfinite(
            numeric_features.to_numpy()
        ).all():
            raise ValueError(
                "Features must contain only finite values."
            )

        return numeric_features.copy()

    @staticmethod
    def _validate_target(
        target: Sequence[float] | pd.Series | np.ndarray,
        expected_rows: int,
        expected_index: pd.Index,
    ) -> np.ndarray:
        if isinstance(target, pd.DataFrame):
            raise ValueError(
                "Target must be one-dimensional."
            )

        if isinstance(target, pd.Series):
            if not target.index.equals(expected_index):
                raise ValueError(
                    "Target index must exactly match feature index."
                )

        try:
            values = np.asarray(
                target,
                dtype="float64",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Target values must be numeric."
            ) from error

        if values.ndim != 1:
            raise ValueError(
                "Target must be one-dimensional."
            )

        if values.size != expected_rows:
            raise ValueError(
                "Feature and target row counts must match."
            )

        if values.size == 0:
            raise ValueError(
                "Target cannot be empty."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "Target must contain only finite values."
            )

        return values.copy()

    def fit(
        self,
        features: pd.DataFrame,
        target: Sequence[float] | pd.Series | np.ndarray,
    ) -> "GradientBoostingVaR":
        validated_features = self._validate_features(
            features
        )

        validated_target = self._validate_target(
            target,
            expected_rows=len(validated_features),
            expected_index=validated_features.index,
        )

        self._feature_names = tuple(
            validated_features.columns
        )

        self._model = GradientBoostingRegressor(
            loss="quantile",
            alpha=self.alpha,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
        )

        self._model.fit(
            validated_features,
            validated_target,
        )

        return self

    def _validate_prediction_features(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        if self._model is None or self._feature_names is None:
            raise ValueError(
                "Model must be fitted before prediction."
            )

        validated_features = self._validate_features(
            features
        )

        feature_names = tuple(
            validated_features.columns
        )

        if feature_names != self._feature_names:
            raise ValueError(
                "Prediction feature columns must match "
                "the fitted feature columns and order."
            )

        return validated_features

    def predict_quantile(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        validated_features = (
            self._validate_prediction_features(
                features
            )
        )

        predictions = np.asarray(
            self._model.predict(
                validated_features
            ),
            dtype="float64",
        )

        if predictions.ndim != 1:
            raise RuntimeError(
                "Gradient Boosting returned an invalid "
                "prediction shape."
            )

        if not np.isfinite(predictions).all():
            raise RuntimeError(
                "Gradient Boosting returned non-finite predictions."
            )

        return predictions

    def predict(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        quantile_return = self.predict_quantile(
            features
        )

        var_values = np.maximum(
            0.0,
            -quantile_return,
        )

        return pd.DataFrame(
            {
                "quantile_return": quantile_return,
                "var": var_values,
            },
            index=features.index.copy(),
            dtype="float64",
        )

    def forecast(
        self,
        features: pd.DataFrame,
    ) -> dict[str, float]:
        if len(features) != 1:
            raise ValueError(
                "Forecast requires exactly one feature row."
            )

        prediction = self.predict(
            features
        ).iloc[0]

        return {
            "quantile_return": float(
                prediction["quantile_return"]
            ),
            "var": float(
                prediction["var"]
            ),
        }

    @property
    def feature_names(self) -> tuple[str, ...]:
        if self._feature_names is None:
            raise ValueError(
                "Model must be fitted before feature names are available."
            )

        return self._feature_names
