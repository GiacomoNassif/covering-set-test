import numpy as np
from itertools import combinations
from sklearn.base import clone
from sklearn.model_selection import KFold

from typing import Optional
import pandas as pd
import ast

def get_best_min_frac(pareto_df: pd.DataFrame, min_frac: float) -> Optional[pd.Series]:
    # eff >= min_frac, pick smallest block_len
    p = pareto_df[pareto_df.eff >= min_frac].sort_values("block_len")
    return None if p.empty else p.iloc[0]


def get_best_max_block_len(pareto_df: pd.DataFrame, max_block_len: int) -> Optional[pd.Series]:
    # block_len <= max_block_len, pick biggest eff
    p = pareto_df[pareto_df.block_len <= max_block_len].sort_values("eff", ascending=False)
    return None if p.empty else p.iloc[0]


# Load in the Pareto frontier
_pareto_frontier = pd.read_csv("pareto_block_t2.zip")
_pareto_frontier["blocks"] = _pareto_frontier["blocks"].apply(ast.literal_eval)



class _CoveringStackRegressor:
    """
    Minimal block-driven 2-level stacker (regression).

    blocks: list[iterable[int]] with 1-based fold ids (1..v)
      - one L1 model per block
      - each block B is the set of folds EXCLUDED by that L1 model

    Key constraint (t=2 logic):
      - For each pair of folds (i, j), we need at least one L1 model that leaves out BOTH i and j.
      - i.e. we need some block B_m such that {i, j} ⊆ B_m.

    L2 features:
      - L2 uses the original X plus meta-feature columns (the routed L1 OOF prediction).
    """

    def __init__(self, *, blocks, l1_estimator, l2_estimator, splitter_cls=KFold, splitter_kwargs=None):
        self.blocks = [set(b) for b in blocks]

        self.l1_estimator = l1_estimator
        self.l2_estimator = l2_estimator

        self.v = max(max(b) for b in self.blocks)
        self.fold_set = set(range(1, self.v + 1))

        self.splitter = splitter_cls(n_splits=self.v, **(splitter_kwargs or {}))

        # route[{i,j}] -> m  (0-based L1 id)
        self.route = {}

        # test_l1[i] -> m  (0-based L1 id)
        self.test_l1 = {}

        for m, B in enumerate(self.blocks):
            for i in B:
                self.test_l1[i] = m
            for i, j in combinations(B, 2):
                self.route[frozenset((i, j))] = m

        self.M = len(self.blocks)
        self.l1_models = [None] * self.M
        self.l2_models = [None] * (self.v + 1)

        self.fold_rows_ = None
        self.l1_oof_ = None   # (n, M, 1)
        self.oof_ = None      # (n,)

    def fit(self, X, y):
        n = len(y)

        # fold_rows_[i] = row indices in fold i
        self.fold_rows_ = [None] * (self.v + 1)
        for f, (_, te) in enumerate(self.splitter.split(X, y), start=1):
            self.fold_rows_[f] = te

        # ---- L1 ----
        # l1_oof_[row, m, 0] exists only when that row is in a fold excluded by L1_m.
        self.l1_oof_ = np.full((n, self.M, 1), np.nan, dtype=float)

        for m, B in enumerate(self.blocks):
            tr_idx = np.concatenate([self.fold_rows_[f] for f in (self.fold_set - B)])
            te_idx = np.concatenate([self.fold_rows_[f] for f in B])

            l1 = clone(self.l1_estimator).fit(X[tr_idx], y[tr_idx])
            self.l1_models[m] = l1
            self.l1_oof_[te_idx, m, 0] = l1.predict(X[te_idx])

        # ---- L2 ----
        self.oof_ = np.full(n, np.nan, dtype=float)

        for i in self.fold_set:
            train_folds = tuple(self.fold_set - {i})
            train_rows = np.concatenate([self.fold_rows_[j] for j in train_folds])

            # Meta-feature for rows in fold j uses L1_m where m = route[{i,j}]
            z_train = np.vstack([
                self.l1_oof_[self.fold_rows_[j], self.route[frozenset((i, j))]]  # (n_j, 1)
                for j in train_folds
            ])

            X2_train = np.hstack([X[train_rows], z_train])

            l2 = clone(self.l2_estimator).fit(X2_train, y[train_rows])
            self.l2_models[i] = l2

            test_rows = self.fold_rows_[i]
            z_test = self.l1_oof_[test_rows, self.test_l1[i]]  # (n_i, 1)
            X2_test = np.hstack([X[test_rows], z_test])

            self.oof_[test_rows] = l2.predict(X2_test)

        return self
    
    def predict(self, X):
        # TODO: there is no mathematical rigor here. Definitely improvable.
        preds = []
        for i in range(1, self.v + 1):
            m = self.test_l1[i]
            z = self.l1_models[m].predict(X).reshape(-1, 1)
            X2 = np.hstack([X, z])
            preds.append(self.l2_models[i].predict(X2))
        return np.mean(np.vstack(preds), axis=0)


class CoveringStackRegressor(_CoveringStackRegressor):
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
        if (min_data_frac is None) == (max_l1_fits is None):
            raise ValueError("Specify exactly one of min_data_frac or max_l1_fits.")

        row = (
            get_best_min_frac(_pareto_frontier, min_data_frac)
            if min_data_frac is not None
            else get_best_max_block_len(_pareto_frontier, max_l1_fits)
        )

        if row is None:
            raise ValueError("No valid block design found.")

        super().__init__(
            blocks=row.blocks,
            l1_estimator=l1_estimator,
            l2_estimator=l2_estimator,
            splitter_cls=splitter_cls,
            splitter_kwargs=splitter_kwargs,
        )



# Copy paste for the classifier

from sklearn.model_selection import StratifiedKFold


class _CoveringStackClassifier:
    """
    Minimal block-driven 2-level stacker (binary classification).

    Same block logic as CoveringStackRegressor.
    Only change: use positive-class probability as the meta-feature / prediction.
    """

    def __init__(self, *, blocks, l1_estimator, l2_estimator, splitter_cls=StratifiedKFold, splitter_kwargs=None):
        self.blocks = [set(b) for b in blocks]

        self.l1_estimator = l1_estimator
        self.l2_estimator = l2_estimator

        self.v = max(max(b) for b in self.blocks)
        self.fold_set = set(range(1, self.v + 1))

        self.splitter = splitter_cls(n_splits=self.v, **(splitter_kwargs or {}))

        # route[{i,j}] -> m  (0-based L1 id)
        self.route = {}

        # test_l1[i] -> m  (0-based L1 id)
        self.test_l1 = {}

        for m, B in enumerate(self.blocks):
            for i in B:
                self.test_l1[i] = m
            for i, j in combinations(B, 2):
                self.route[frozenset((i, j))] = m

        self.M = len(self.blocks)
        self.l1_models = [None] * self.M
        self.l2_models = [None] * (self.v + 1)

        self.fold_rows_ = None
        self.l1_oof_ = None   # (n, M, 1)  stores pos-class proba
        self.oof_ = None      # (n,)       stores pos-class proba

        self.classes_ = None
        self.pos_label_ = None

    def _pos_proba(self, est, X):
        proba = est.predict_proba(X)
        idx = np.where(est.classes_ == self.pos_label_)[0]
        if idx.size:
            return proba[:, int(idx[0])]
        # estimator trained on a single class that isn't pos_label_
        # => P(pos)=0 (or 1 if the single class is pos)
        return np.ones(X.shape[0], dtype=float) if est.classes_[0] == self.pos_label_ else np.zeros(X.shape[0], dtype=float)


    def fit(self, X, y):
        n = len(y)

        self.classes_ = np.unique(y)
        self.pos_label_ = self.classes_[1]  # assume binary; "positive" = second sorted class

        # fold_rows_[i] = row indices in fold i
        self.fold_rows_ = [None] * (self.v + 1)
        for f, (_, te) in enumerate(self.splitter.split(X, y), start=1):
            self.fold_rows_[f] = te

        # ---- L1 ----
        # l1_oof_[row, m, 0] exists only when that row is in a fold excluded by L1_m.
        self.l1_oof_ = np.full((n, self.M, 1), np.nan, dtype=float)

        for m, B in enumerate(self.blocks):
            tr_idx = np.concatenate([self.fold_rows_[f] for f in (self.fold_set - B)])
            te_idx = np.concatenate([self.fold_rows_[f] for f in B])

            l1 = clone(self.l1_estimator).fit(X[tr_idx], y[tr_idx])
            self.l1_models[m] = l1
            self.l1_oof_[te_idx, m, 0] = self._pos_proba(l1, X[te_idx])

        # ---- L2 ----
        self.oof_ = np.full(n, np.nan, dtype=float)

        for i in self.fold_set:
            train_folds = tuple(self.fold_set - {i})
            train_rows = np.concatenate([self.fold_rows_[j] for j in train_folds])

            # Meta-feature for rows in fold j uses L1_m where m = route[{i,j}]
            z_train = np.vstack([
                self.l1_oof_[self.fold_rows_[j], self.route[frozenset((i, j))]]  # (n_j, 1)
                for j in train_folds
            ])

            X2_train = np.hstack([X[train_rows], z_train])

            l2 = clone(self.l2_estimator).fit(X2_train, y[train_rows])
            self.l2_models[i] = l2

            test_rows = self.fold_rows_[i]
            z_test = self.l1_oof_[test_rows, self.test_l1[i]]  # (n_i, 1)
            X2_test = np.hstack([X[test_rows], z_test])

            self.oof_[test_rows] = self._pos_proba(l2, X2_test)

        return self

    def predict_proba(self, X):
        # TODO: there is no mathematical rigor here. Definitely improvable.
        preds = []
        for i in range(1, self.v + 1):
            m = self.test_l1[i]
            z = self._pos_proba(self.l1_models[m], X).reshape(-1, 1)
            X2 = np.hstack([X, z])
            preds.append(self._pos_proba(self.l2_models[i], X2))

        p = np.mean(np.vstack(preds), axis=0)  # avg pos-class proba
        return np.column_stack([1.0 - p, p])   # columns align to self.classes_ = [neg, pos]


class CoveringStackClassifier(_CoveringStackClassifier):
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
        if (min_data_frac is None) == (max_l1_fits is None):
            raise ValueError("Specify exactly one of min_data_frac or max_l1_fits.")

        row = (
            get_best_min_frac(_pareto_frontier, min_data_frac)
            if min_data_frac is not None
            else get_best_max_block_len(_pareto_frontier, max_l1_fits)
        )

        if row is None:
            raise ValueError("No valid block design found.")

        super().__init__(
            blocks=row.blocks,
            l1_estimator=l1_estimator,
            l2_estimator=l2_estimator,
            splitter_cls=splitter_cls,
            splitter_kwargs=splitter_kwargs,
        )