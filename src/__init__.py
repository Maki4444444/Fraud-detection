"""
src/__init__.py
---------------
Exposes all Task 1 modules for clean imports in notebooks.

Usage in notebooks:
    from src.data_loader import load_fraud_data, load_creditcard
    from src.data_cleaner import clean_fraud_data, inspect
    from src.eda import plot_class_imbalance, plot_numeric_distributions
    from src.geolocation import enrich_with_country
    from src.feature_engineering import engineer_all_features
    from src.transformation import scale_numeric, encode_categorical
    from src.imbalance_handler import handle_imbalance
"""

from src import data_loader
from src import data_cleaner
from src import eda
from src import geolocation
from src import feature_engineering
from src import transformation
from src import imbalance_handler