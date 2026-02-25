import ast
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted


def get_best_min_frac(pareto_df: pd.DataFrame, min_frac: float) -> Optional[pd.Series]:
    p = pareto_df[pareto_df.eff >= min_frac].sort_values("block_len")
    return None if p.empty else p.iloc[0]


def get_best_max_block_len(pareto_df: pd.DataFrame, max_block_len: int) -> Optional[pd.Series]:
    p = pareto_df[pareto_df.block_len <= max_block_len].sort_values("eff", ascending=False)
    return None if p.empty else p.iloc[0]


_pareto_frontier = pd.read_csv("pareto_block_t2.zip")
_pareto_frontier["blocks"] = _pareto_frontier["blocks"].apply(ast.literal_eval)


def _pick_blocks(min_data_frac: Optional[float], max_l1_fits: Optional[int]):
    if (min_data_frac is None) == (max_l1_fits is None):
        raise ValueError("Specify exactly one of min_data_frac or max_l1_fits.")

    row = (
        get_best_min_frac(_pareto_frontier, min_data_frac)
        if min_data_frac is not None
        else get_best_max_block_len(_pareto_frontier, max_l1_fits)
    )
    if row is None:
        raise ValueError("No valid block design found.")
    return row.blocks


class _BaseCoveringStack(BaseEstimator):
    _fit_splitter_cls = KFold

    def __init__(self, *, blocks, l1_estimator, l2_estimator, splitter_cls=None, splitter_kwargs=None):
        self.blocks = [set(b) for b in blocks]
        self.l1_estimator = l1_estimator
        self.l2_estimator = l2_estimator
        self.splitter_cls = splitter_cls
        self.splitter_kwargs = splitter_kwargs

        self.v = max(max(b) for b in self.blocks)
        self.route = {}
        self.test_l1 = {}
        for m, block in enumerate(self.blocks):
            for i in block:
                self.test_l1[i] = m
            for i, j in combinations(block, 2):
                self.route[frozenset((i, j))] = m

        self.M = len(self.blocks)
        self.l1_models = [[None] * len(self.l1_estimator) for _ in range(self.M)]
        self.l2_models = [[None] * len(self.l2_estimator) for _ in range(self.v + 1)]

    def _model_output(self, est, X):
        raise NotImplementedError

    def _rows(self, folds):
        return np.concatenate([self.fold_rows_[f] for f in folds])

    def fit(self, X, y):
        n = len(y)
        fit_splitter_cls = self.splitter_cls or self._fit_splitter_cls
        splitter = fit_splitter_cls(n_splits=self.v, **(self.splitter_kwargs or {}))

        self.fold_rows_ = {}
        for f, (_, te) in enumerate(splitter.split(X, y), start=1):
            self.fold_rows_[f] = te

        self.l1_oof_ = np.full((n, self.M, len(self.l1_estimator)), np.nan, dtype=float)
        all_folds = set(range(1, self.v + 1))
        for m, block in enumerate(self.blocks):
            tr_idx = self._rows(all_folds - block)
            te_idx = self._rows(block)
            for l, l1_est in enumerate(self.l1_estimator):
                l1 = clone(l1_est).fit(X[tr_idx], y[tr_idx])
                self.l1_models[m][l] = l1
                self.l1_oof_[te_idx, m, l] = self._model_output(l1, X[te_idx])

        self.oof_ = np.full((n, len(self.l2_estimator)), np.nan, dtype=float)
        for i in range(1, self.v + 1):
            train_folds = [j for j in range(1, self.v + 1) if j != i]
            train_rows = self._rows(train_folds)
            z_train = np.vstack([
                self.l1_oof_[self.fold_rows_[j], self.route[frozenset((i, j))]]
                for j in train_folds
            ])

            for r, l2_est in enumerate(self.l2_estimator):
                l2 = clone(l2_est).fit(np.hstack([X[train_rows], z_train]), y[train_rows])
                self.l2_models[i][r] = l2

            test_rows = self.fold_rows_[i]
            z_test = self.l1_oof_[test_rows, self.test_l1[i]]
            self.oof_[test_rows] = np.column_stack([
                self._model_output(l2, np.hstack([X[test_rows], z_test]))
                for l2 in self.l2_models[i]
            ])

        return self

    def _predict_raw(self, X):
        check_is_fitted(self, ["l1_oof_", "oof_", "fold_rows_"])
        return self.predict_uniform_and_routed_mean(X)[0]  # uniform mean

    def predict_uniform_and_routed_mean(self, X):
        """
        uniform_mean:
            (1/K) * sum_i (1/M) * sum_m P[i,m](x)

        routed_mean:
            (1/K) * sum_i sum_m w_i[m] * P[i,m](x),
            w_i[m] ∝ sum_{j != i, route(i,j)=m} |fold_j|, normalized over m.
        """
        check_is_fitted(self, ["fold_rows_"])

        X = np.asarray(X)
        n, p = X.shape
        K, M = self.v, self.M
        R = len(self.l2_estimator)

        Z = np.stack(
            [
                np.column_stack([self._model_output(l1, X) for l1 in l1_group])
                for l1_group in self.l1_models
            ],
            axis=1,
        )

        uniform_sum = np.zeros((n, R))
        routed_sum = np.zeros((n, R))

        Xz = np.empty((n, p + Z.shape[2]))
        Xz[:, :p] = X

        for i in range(1, K + 1):
            w = np.zeros(M)
            for j in range(1, K + 1):
                if j == i:
                    continue
                m = self.route[frozenset((i, j))]
                w[m] += len(self.fold_rows_[j])
            w /= w.sum()

            l2 = self.l2_models[i]
            for m in range(M):
                Xz[:, p:] = Z[:, m, :]
                p_im = np.column_stack([self._model_output(l2_model, Xz) for l2_model in l2])
                uniform_sum += p_im
                routed_sum += w[m] * p_im

        l1_pred = Z.mean(axis=1)

        return uniform_sum / (K * M), routed_sum / K, l1_pred


class CoveringStackRegressor(_BaseCoveringStack, RegressorMixin):
    def __init__(
        self,
        *,
        l1_estimator,
        l2_estimator,
        min_data_frac: Optional[float] = None,
        max_l1_fits: Optional[int] = None,
        splitter_cls=KFold,
        splitter_kwargs=None,
    ):
        super().__init__(
            blocks=_pick_blocks(min_data_frac, max_l1_fits),
            l1_estimator=l1_estimator,
            l2_estimator=l2_estimator,
            splitter_cls=splitter_cls,
            splitter_kwargs=splitter_kwargs,
        )

    def _model_output(self, est, X):
        return est.predict(X)

    def predict(self, X):
        return self._predict_raw(X)


class CoveringStackClassifier(_BaseCoveringStack, ClassifierMixin):
    _fit_splitter_cls = StratifiedKFold

    def __init__(
        self,
        *,
        l1_estimator,
        l2_estimator,
        min_data_frac: Optional[float] = None,
        max_l1_fits: Optional[int] = None,
        splitter_cls=StratifiedKFold,
        splitter_kwargs=None,
    ):
        super().__init__(
            blocks=_pick_blocks(min_data_frac, max_l1_fits),
            l1_estimator=l1_estimator,
            l2_estimator=l2_estimator,
            splitter_cls=splitter_cls,
            splitter_kwargs=splitter_kwargs,
        )

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.pos_label_ = self.classes_[1]
        return super().fit(X, y)

    def _model_output(self, est, X):
        proba = est.predict_proba(X)
        idx = np.where(est.classes_ == self.pos_label_)[0]
        if idx.size:
            return proba[:, int(idx[0])]
        return np.ones(X.shape[0], dtype=float) if est.classes_[0] == self.pos_label_ else np.zeros(X.shape[0], dtype=float)

    def predict_proba(self, X):
        p = self._predict_raw(X)
        return np.stack([1.0 - p, p], axis=1)

    def predict(self, X):
        p = self._predict_raw(X)
        return np.where(p >= 0.5, self.pos_label_, self.classes_[0])
