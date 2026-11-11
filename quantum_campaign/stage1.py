"""Leakage-free holdout model selection: classical baselines + quantum-kernel SVC blend."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, log_loss
from sklearn.svm import SVC
from .feature_map import state_embeddings, fidelity_kernel


def top_k_metrics(y_true, p, k=10):
    indices = np.argsort(-np.asarray(p), kind='stable')[:min(k, len(p))]
    positives = max(int(sum(y_true)), 1)
    return dict(pr_auc=float(average_precision_score(y_true, p)),
                roc_auc=float(roc_auc_score(y_true, p)),
                log_loss=float(log_loss(y_true, np.clip(p, 1e-8, 1-1e-8))),
                brier=float(brier_score_loss(y_true, p)),
                precision_at_10=float(np.mean(np.asarray(y_true)[indices])),
                recall_at_10=float(np.asarray(y_true)[indices].sum()/positives))


def preprocessor(frame: pd.DataFrame):
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    sections = []
    if numeric:
        sections.append(('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                         ('scale', StandardScaler())]), numeric))
    if categorical:
        sections.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                         ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical))
    return ColumnTransformer(sections, sparse_threshold=0)


def rank_predictions(ids, probabilities):
    result = pd.DataFrame({'customer_id': np.asarray(ids).astype(str),
                           'predicted_probability': probabilities})
    if result.customer_id.duplicated().any() or not np.isfinite(result.predicted_probability).all():
        raise ValueError('Invalid IDs or probabilities')
    if not result.predicted_probability.between(0, 1).all():
        raise ValueError('Probabilities outside [0,1]')
    result = result.sort_values(['predicted_probability', 'customer_id'], ascending=[False, True], kind='stable').reset_index(drop=True)
    result['rank'] = np.arange(1, len(result)+1)
    result['is_top_10'] = result['rank'] <= 10
    return result


def quantum_rows(y, seed, limit=64):
    if len(y) <= limit:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    np_ = min(len(pos), max(5, round(limit*len(pos)/len(y))))
    nn = min(len(neg), limit-np_)
    return np.sort(np.r_[rng.choice(pos, np_, replace=False), rng.choice(neg, nn, replace=False)])


def fit_models(frame, y, seed, engine, qml_backend):
    prep = preprocessor(frame)
    X = np.asarray(prep.fit_transform(frame), dtype=float)
    models = {
        'logistic_regression': LogisticRegression(C=0.3, max_iter=2000, random_state=seed),
        'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=8, l2_regularization=2., random_state=seed),
    }
    for model in models.values():
        model.fit(X, y)
    rows = quantum_rows(y, seed+22)
    # Selection/scaling fit only on this training partition.
    selector = SelectKBest(f_classif, k=min(4, X.shape[1]))
    qX = selector.fit_transform(X[rows], y[rows])
    scaler = MinMaxScaler(clip=True)
    qX = scaler.fit_transform(qX)
    if qml_backend == 'qiskit_ml':
        from .qml_qiskit_ml import make_qsvc
        quantum = make_qsvc(qX.shape[1], seed=seed)
        quantum.fit(qX, y[rows])
        states = None
    else:
        states = state_embeddings(qX, engine)
        quantum = SVC(C=2., kernel='precomputed', probability=True, random_state=seed)
        quantum.fit(fidelity_kernel(states, states), y[rows])
    return dict(preprocessor=prep, classical=models, selector=selector, scaler=scaler,
                quantum=quantum, states=states, engine=engine, qml_backend=qml_backend,
                quantum_training_rows=rows.tolist())


def predict_models(bundle, frame):
    X = np.asarray(bundle['preprocessor'].transform(frame), dtype=float)
    result = {name: model.predict_proba(X)[:, 1] for name, model in bundle['classical'].items()}
    qX = bundle['scaler'].transform(bundle['selector'].transform(X))
    if bundle['qml_backend'] != 'qiskit_ml':
        qX = fidelity_kernel(state_embeddings(qX, bundle['engine']), bundle['states'])
    result['quantum_kernel_svc'] = bundle['quantum'].predict_proba(qX)[:, 1]
    return result


def calibration_bins(y, p):
    edges = np.linspace(0, 1, 6)
    bins = np.minimum(np.digitize(p, edges)-1, 4)
    return [dict(lower=float(edges[i]), upper=float(edges[i+1]), count=int(sum(bins==i)),
                 mean_probability=float(np.mean(p[bins==i])), observed_fraction=float(np.mean(y[bins==i])))
            for i in range(5) if np.any(bins==i)]


def train_and_predict(train, test, feature_cols, *, id_col='customer_id', target='subscribed',
                      seed=42, engine='auto', qml_backend='auto', model_path=None):
    from sklearn.model_selection import StratifiedKFold
    import time
    started = time.perf_counter()
    y = train[target].astype(int).to_numpy()
    if min(np.bincount(y)) < 8:
        raise ValueError('Need at least 8 observations per class for three-fold validation')
    if qml_backend == 'auto':
        import importlib.util
        qml_backend = 'qiskit_ml' if engine != 'numpy' and importlib.util.find_spec('qiskit_machine_learning') else 'native'
    if qml_backend not in ('native', 'qiskit_ml'):
        raise ValueError('Unsupported quantum backend')
    names = ['logistic_regression', 'hist_gradient_boosting', 'quantum_kernel_svc']
    oof = {name: np.zeros(len(y)) for name in names}
    folds = []
    for fold, (tr, val) in enumerate(StratifiedKFold(3, shuffle=True, random_state=seed).split(train, y)):
        print(f'Stage 1 validation fold {fold+1}/3', flush=True)
        bundle = fit_models(train.iloc[tr][feature_cols], y[tr], seed+fold, engine, qml_backend)
        predicted = predict_models(bundle, train.iloc[val][feature_cols])
        for name in names:
            oof[name][val] = predicted[name]
        folds.append(dict(fold=fold, training_rows=tr.tolist(), validation_rows=val.tolist(),
                          seed=seed+fold, quantum_fit_samples=len(bundle['quantum_training_rows'])))
    metrics = {name: top_k_metrics(y, p) for name, p in oof.items()}
    chosen = min(names[:2], key=lambda name: (metrics[name]['brier'], -metrics[name]['pr_auc']))
    scores = {w: top_k_metrics(y, w*oof[names[2]]+(1-w)*oof[chosen]) for w in (0., .25, .5, .75, 1.)}
    weight = min(scores, key=lambda w: (scores[w]['brier'], -scores[w]['pr_auc']))
    # Freeze all selection before accessing test feature values.
    print('Stage 1 frozen selection; refitting on all training customers', flush=True)
    final = fit_models(train[feature_cols], y, seed, engine, qml_backend)
    final.update(chosen_classical=chosen, quantum_weight=weight, feature_columns=feature_cols)
    predicted = predict_models(final, test[feature_cols])
    p = weight*predicted[names[2]]+(1-weight)*predicted[chosen]
    if model_path:
        import joblib
        joblib.dump(final, model_path)
    diag = dict(classical_chosen=chosen, quantum_weight=float(weight), quantum_backend=qml_backend,
        quantum_library='FidelityQuantumKernel + QSVC' if qml_backend=='qiskit_ml' else 'native fidelity + sklearn SVC',
        validation_samples=len(y), training_fit_samples=len(y), quantum_kernel_fit_samples=len(final['quantum_training_rows']),
        metrics=metrics, hybrid_validation_metrics=scores[weight], quantum_qubits=min(4,len(final['selector'].get_support(indices=True))),
        selection_criterion='Lowest out-of-fold Brier score, then highest average precision; fixed weight grid',
        validation_caveat='Three-fold OOF estimates are used for model selection, not an independent final performance test. Precision@10 is pooled OOF top ten.',
        calibration_policy='QSVC uses training-only internal sigmoid probability fitting. Feature selection is within each outer training fold; inner SVC calibration reuses those fold transforms.',
        calibration={name:calibration_bins(y, pp) for name,pp in oof.items()},
        folds=folds, out_of_fold_probabilities={name:pp.tolist() for name,pp in oof.items()},
        training_labels=y.tolist(), training_customer_ids=train[id_col].astype(str).tolist(),
        quantum_feature_map='RY(pi*x), RZ(pi*x), nearest-neighbor RZZ(pi*x_i*x_j)',
        quantum_sampler='StatevectorSampler, 1024 shots per fidelity, seeded' if qml_backend=='qiskit_ml' else 'exact native statevector',
        final_preprocessing_feature_names=final['preprocessor'].get_feature_names_out().tolist(),
        quantum_selected_features=final['selector'].get_support(indices=True).tolist(),
        quantum_training_customer_ids=train.iloc[final['quantum_training_rows']][id_col].astype(str).tolist(),
        nonleakage_policy='All selection uses training OOF predictions only. Final preprocessing/classical models refit on all training rows; quantum classifier uses disclosed stratified 64-row training subset.',
        runtime_seconds=time.perf_counter()-started)
    return rank_predictions(test[id_col], p), diag
