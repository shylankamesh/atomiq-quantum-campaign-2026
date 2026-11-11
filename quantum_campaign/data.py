"""Dataset validation and fully disclosed synthetic demonstration data."""
from __future__ import annotations
import numpy as np
import pandas as pd

TARGET_ALIASES = ('subscribed', 'subscription', 'target', 'y', 'label', 'converted', 'deposit')


def normalize_target(series):
    """Organizer labels are yes/no; never infer labels from test features."""
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        mapped = series.astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0, '1': 1, '0': 0})
    else:
        mapped = pd.to_numeric(series, errors='coerce')
    if mapped.isna().any() or set(mapped.unique()) != {0, 1}:
        raise ValueError('Training target must contain both yes/no or 0/1 classes')
    return mapped.astype(int)


def join_metadata(test, metadata, id_col='customer_id'):
    if id_col not in metadata or metadata[id_col].isna().any() or metadata[id_col].duplicated().any():
        raise ValueError('Stage 2 metadata must have unique nonmissing IDs')
    overlap = (set(test.columns) & set(metadata.columns)) - {id_col}
    if overlap:
        raise ValueError(f'Ambiguous overlapping Stage 2 columns: {sorted(overlap)}')
    joined = test.merge(metadata, on=id_col, how='left', validate='one_to_one', indicator=True, sort=False)
    if not joined['_merge'].eq('both').all():
        raise ValueError('Stage 2 metadata missing test customer IDs')
    return joined.drop(columns='_merge')


def load_organizer_workbook(path):
    sheets = pd.read_excel(path, sheet_name=None, dtype={'customer_id': str})
    required = {'README', 'Stage1_Train', 'Stage1_Test', 'Stage2_Data', 'Stage2_Constraints'}
    if not required.issubset(sheets):
        raise ValueError(f'Missing organizer sheets: {required-set(sheets)}')
    train, test = sheets['Stage1_Train'], sheets['Stage1_Test']
    if len(train) != 300 or len(test) != 100:
        raise ValueError('Official run requires exactly 300 training and 100 test customers')
    constraints = sheets['Stage2_Constraints'].set_index('symbol')['value']
    if int(constraints['N']) != 10 or int(constraints['K']) != 6:
        raise ValueError('Expected official N=10, K=6')
    train['y'] = normalize_target(train['y'])
    return train, join_metadata(test, sheets['Stage2_Data']), float(constraints['C']), float(constraints['H'])

def resolve_target(frame: pd.DataFrame, explicit: str | None) -> str:
    if explicit:
        if explicit not in frame.columns:
            raise ValueError(f"Target column {explicit!r} not found in training data: {list(frame.columns)}")
        return explicit
    found = [c for c in frame.columns if c.lower() in TARGET_ALIASES]
    if len(found) != 1:
        raise ValueError(f"Cannot unambiguously detect target. Specify --target. Columns: {list(frame.columns)}")
    return found[0]

def validate_inputs(train: pd.DataFrame, test: pd.DataFrame, *, id_col: str, target: str,
                    cost_col: str, capacity_col: str, value_col: str | None,
                    expected_test_rows: int = 100) -> list[str]:
    for name, df in [('train', train), ('test', test)]:
        if id_col not in df:
            raise ValueError(f"Missing {id_col!r} in {name} dataset")
        if df[id_col].isna().any() or df[id_col].astype(str).duplicated().any():
            raise ValueError(f"Missing or duplicate customer IDs in {name} dataset")
    if set(train[id_col].astype(str)) & set(test[id_col].astype(str)):
        raise ValueError('Training and test customer IDs overlap')
    for col in (cost_col, capacity_col, *([value_col] if value_col else [])):
        if col not in test:
            raise ValueError(f"Test dataset missing {col!r}; provide actual organizer metadata")
        numbers = pd.to_numeric(test[col], errors='coerce')
        if numbers.isna().any() or not np.isfinite(numbers).all() or (numbers < 0).any():
            raise ValueError(f"{col!r} must contain finite nonnegative numbers")
    if target not in train:
        raise ValueError(f"Training data missing target {target!r}")
    y = pd.to_numeric(train[target], errors='coerce')
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError('Training target must be binary 0/1 with both classes present')
    if target in test and test[target].notna().any():
        raise ValueError('Test set contains target labels; remove them to avoid test leakage')
    if len(test) != expected_test_rows:
        raise ValueError(f"Expected {expected_test_rows} test rows, received {len(test)}")
    excluded = {target, id_col, cost_col, capacity_col, value_col, 'customer_segment', 'duration'}
    feature_cols = [c for c in train.columns if c not in excluded and c in test.columns]
    if not feature_cols:
        raise ValueError('No shared predictive feature columns')
    return feature_cols

def generate_demo(seed: int = 42, n_train: int = 320, n_test: int = 100):
    """GENERATED SAMPLES ONLY; NOT organizers' data. Labels sampled from a known synthetic model."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    age = rng.integers(18, 72, size=n)
    tenure_months = rng.integers(1, 90, size=n)
    digital_visits = rng.poisson(5, size=n)
    customer_engagement = rng.beta(2.2, 3.0, size=n)
    income = rng.lognormal(10.45, 0.55, size=n)
    prior_orders = rng.poisson(1.8, size=n)
    region = rng.choice(['north', 'south', 'east', 'west'], n)
    contact_cost = rng.integers(8, 21, size=n)
    capacity_units = rng.integers(1, 6, size=n)
    expected_value = rng.integers(85, 201, size=n)
    logit = (-1.8 + .23 * digital_visits + 2.0 * customer_engagement
             + .33 * prior_orders + .005 * (age - 40)
             + .006 * tenure_months + .15 * (region == 'west')
             - .24 * (region == 'east'))
    probability = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, probability)
    all_data = pd.DataFrame({
        'customer_id': [f'DEMO_{i:04d}' for i in range(n)],
        'age': age, 'tenure_months': tenure_months,
        'digital_visits': digital_visits, 'engagement_score': customer_engagement,
        'income': np.round(income, 2), 'prior_orders': prior_orders,
        'region': region, 'contact_cost': contact_cost,
        'capacity_units': capacity_units, 'expected_value': expected_value,
    })
    training = all_data.iloc[:n_train].copy()
    training['subscribed'] = y[:n_train]
    test = all_data.iloc[n_train:].copy()
    return training, test
