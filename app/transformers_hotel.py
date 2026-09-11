"""
Transformers custom del pipeline de ML_hotel_bookings.

Las clases son las mismas que están definidas en main.ipynb, en la celda del
pipeline. Están aquí, en un módulo importable, porque un pickle no guarda el
código de las clases: guarda una referencia del tipo "modulo.NombreDeClase".
Si solo existen dentro del notebook, fuera de él no hay nada que importar.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Elimina columnas con leakage / nulos altos."""
    def __init__(self, cols_a_eliminar):
        self.cols_a_eliminar = cols_a_eliminar

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        presentes = [c for c in self.cols_a_eliminar if c in X.columns]
        return X.drop(columns=presentes)


class NumericFeatureEngineer(BaseEstimator, TransformerMixin):
    """total_nights, is_family, total_guests, adr_per_person."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['total_nights'] = X['stays_in_week_nights'] + X['stays_in_weekend_nights']
        X['is_family'] = ((X['children'] > 0) | (X['babies'] > 0)).astype(int)
        X['total_guests'] = X['adults'] + X['children'] + X['babies']
        X['adr_per_person'] = X['adr'] / X['total_guests'].replace(0, 1)
        return X


class DateFeatureEngineer(BaseEstimator, TransformerMixin):
    """season, mes_sin, mes_cos. Elimina las columnas de fecha crudas."""
    MESES = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
             'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    SEASON_MAP = {
        'December': 'Winter', 'January': 'Winter', 'February': 'Winter',
        'March': 'Spring', 'April': 'Spring', 'May': 'Spring',
        'June': 'Summer', 'July': 'Summer', 'August': 'Summer',
        'September': 'Autumn', 'October': 'Autumn', 'November': 'Autumn'
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        mes_num = X['arrival_date_month'].str.strip().map(self.MESES)
        X['season'] = X['arrival_date_month'].map(self.SEASON_MAP)
        X['mes_sin'] = np.sin(2 * np.pi * mes_num / 12)
        X['mes_cos'] = np.cos(2 * np.pi * mes_num / 12)

        descartables = ['arrival_date_year', 'arrival_date_month',
                        'arrival_date_day_of_month', 'arrival_date_week_number']
        return X.drop(columns=[c for c in descartables if c in X.columns])


class QuantileBinner(BaseEstimator, TransformerMixin):
    """Discretiza lead_time, adr, total_nights en cuartiles + ordinal encoding."""
    LABELS = {
        'lead_time': ['very_short', 'short', 'medium', 'long'],
        'adr': ['low', 'medium_low', 'medium_high', 'high'],
        'total_nights': ['short', 'medium', 'long', 'extended'],
    }

    def __init__(self, cols=('lead_time', 'adr', 'total_nights')):
        self.cols = list(cols)

    def fit(self, X, y=None):
        self.bin_edges_ = {
            col: pd.qcut(X[col], q=4, retbins=True, duplicates='drop')[1]
            for col in self.cols
        }
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            labels = self.LABELS[col]
            cats = pd.cut(X[col], bins=self.bin_edges_[col], labels=labels, include_lowest=True)
            mapping = {v: i for i, v in enumerate(labels)}
            X[f'{col}_bin'] = cats.map(mapping).astype("float32")
        return X


class ManualImputer(BaseEstimator, TransformerMixin):
    """children/agent -> 0; country -> moda de train. Recalcula derivadas."""
    def fit(self, X, y=None):
        self.imp_country_ = SimpleImputer(strategy='most_frequent')
        self.imp_country_.fit(X[['country']])
        return self

    def transform(self, X):
        X = X.copy()
        X['children'] = X['children'].fillna(0)
        X['agent'] = X['agent'].fillna(0)
        X['country'] = self.imp_country_.transform(X[['country']]).ravel()
        X['total_guests'] = X['adults'] + X['children'] + X['babies']
        X['adr_per_person'] = X['adr'] / X['total_guests'].replace(0, 1)
        return X


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Capea adr al percentil 99 de train."""
    def fit(self, X, y=None):
        self.p99_ = X['adr'].quantile(0.99)
        return self

    def transform(self, X):
        X = X.copy()
        X['adr'] = X['adr'].clip(upper=self.p99_)
        return X


class CountryGrouper(BaseEstimator, TransformerMixin):
    """country -> Top-20 de train + 'Other'."""
    def fit(self, X, y=None):
        self.top20_ = X['country'].value_counts().nlargest(20).index
        return self

    def transform(self, X):
        X = X.copy()
        X['country'] = X['country'].where(X['country'].isin(self.top20_), 'Other')
        return X


class AgentFlagger(BaseEstimator, TransformerMixin):
    """agent -> tiene_agente (binaria), elimina agent."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['tiene_agente'] = (X['agent'] != 0).astype(int)
        return X.drop(columns=['agent'])


class OneHotWrapper(BaseEstimator, TransformerMixin):
    """One-Hot para todas las categoricas (country ya agrupada y season incluidas)."""
    def __init__(self, cols):
        self.cols = list(cols)

    def fit(self, X, y=None):
        self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.ohe_.fit(X[self.cols])
        return self

    def transform(self, X):
        X = X.copy()
        ohe_arr = self.ohe_.transform(X[self.cols])
        ohe_df = pd.DataFrame(
            ohe_arr,
            columns=self.ohe_.get_feature_names_out(self.cols),
            index=X.index
        ).astype(np.float32)

        X = pd.concat([X.drop(columns=self.cols), ohe_df], axis=1)
        return X


class Scaler(BaseEstimator, TransformerMixin):
    """RobustScaler para adr/lead_time, StandardScaler para el resto de numericas."""
    def __init__(self, cols_robust=('adr', 'lead_time')):
        self.cols_robust = list(cols_robust)

    def fit(self, X, y=None):
        self.robust_scaler_ = RobustScaler().fit(X[self.cols_robust])
        self.cols_num_ = X.select_dtypes(include='number').columns.difference(self.cols_robust)
        self.scaler_ = StandardScaler().fit(X[self.cols_num_])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.cols_robust] = self.robust_scaler_.transform(X[self.cols_robust])
        X[self.cols_num_] = self.scaler_.transform(X[self.cols_num_])
        return X