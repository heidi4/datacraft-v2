from celery import Celery
import pandas as pd
import numpy as np
import os
import json
from redis import Redis
from datetime import datetime, timezone
from scipy import stats
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ai_service import get_ai_interpretation, get_treatment_plan_hypotheses
from data_type_detector import detect_data_type

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, mean_squared_error

import pandas as pd
import numpy as np

class NumpyJSONEncoder(json.JSONEncoder):
    """
    A comprehensive JSON encoder that handles all common NumPy and pandas data types.
    This prevents TypeError exceptions for types like numpy.int64, numpy.float64, 
    pandas.Timestamp, etc.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat() # Convert Timestamp to standard ISO string
        return super(NumpyJSONEncoder, self).default(obj)


celery_app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')
redis_cache = Redis(host='localhost', port=6379, db=1, decode_responses=True)

@celery_app.task(time_limit=900) # 15 minute time limit for huge files
def generate_comprehensive_stats(file_path: str):
    # This function remains unchanged.
    try:
        file_name = os.path.basename(file_path)
        cache_key = f"statistics:{file_name}"
        
        common_na_values = ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan',
                            '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a',
                            'nan', 'null', 'None']
        
        df = pd.read_csv(file_path, on_bad_lines='skip', na_values=common_na_values)

        if df.empty:
            redis_cache.delete(cache_key)
            return

        # --- Perform all calculations in one pass ---
        rows, columns = df.shape
        missing_cells = df.isnull().sum().sum()
        total_cells = rows * columns if rows > 0 else 1
        missing_pct = (missing_cells / total_cells) * 100
        duplicate_rows = df.duplicated().sum()
        duplicate_pct = (duplicate_rows / rows) * 100 if rows > 0 else 0
        quality_score = max(0, 100 - missing_pct - duplicate_pct)
        status = "RAW"
        if quality_score > 90: status = "CLEANED"
        elif quality_score > 60: status = "CLEANING"
        
        column_stats = []
        numeric_column_count = 0
        text_column_count = 0

        for header in df.columns:
            series = df[header]
            clean_series = series.dropna()
            null_count = int(series.isnull().sum())
            data_type = detect_data_type(series)

            if data_type in ['integer', 'float', 'identifier']:
                numeric_column_count += 1
            else:
                text_column_count += 1
            
            stat = {
                "column": header, "dataType": data_type, "nullCount": null_count,
                "nullPercentage": (null_count / rows) * 100 if rows > 0 else 0,
                "uniqueValues": series.nunique(), "totalValues": len(clean_series),
                "mean": "N/A", "median": "N/A", "mode": "N/A"
            }
            if data_type in ['integer', 'float'] and not clean_series.empty:
                stat["mean"] = round(clean_series.mean(), 2)
                stat["median"] = round(clean_series.median(), 2)
                modes = clean_series.mode()
                if not modes.empty:
                    stat["mode"] = ", ".join(modes.astype(str).tolist())
            column_stats.append(stat)

        comprehensive_result = {
            "filename": file_name,
            "lastModified": datetime.fromtimestamp(os.path.getmtime(file_path)).strftime('%Y-%m-%d'),
            "size": f"{os.path.getsize(file_path) / (1024*1024):.1f}MB",
            "rows": rows, "columns": columns, "totalCells": total_cells,
            "status": status, "qualityScore": round(quality_score),
            "missing_pct": round(missing_pct), "duplicates_pct": round(duplicate_pct),
            "overallNullCount": int(missing_cells),
            "columnStats": column_stats,
            "numericColumnCount": numeric_column_count,
            "textColumnCount": text_column_count
        }

        redis_cache.set(cache_key, json.dumps(comprehensive_result, cls=NumpyJSONEncoder), ex=86400)
        return comprehensive_result
    except Exception as e:
        print(f"CRITICAL ERROR in generate_comprehensive_stats for {file_path}: {e}")
        raise e

@celery_app.task(time_limit=1800)
def generate_diagnostic_report(file_path: str):
    try:
        file_name = os.path.basename(file_path)
        cache_key = f"diagnostics:{file_name}"

        df = pd.read_csv(
            file_path,
            on_bad_lines='skip',
            na_values=['', 'NA', 'N/A', 'NULL', 'None', 'nan', 'NaN'],
            encoding='utf-8',
            low_memory=False
        )

        if df.empty:
            redis_cache.delete(cache_key)
            return {"status": "ERROR", "message": "Dataset is empty."}

        rows, columns = df.shape
        duplicate_row_count = int(df.duplicated().sum())
        dataset_summary = {
            "row_count": rows,
            "column_count": columns,
            "duplicate_row_count": duplicate_row_count,
        }

        column_diagnostics = []
        for header in df.columns:
            series = df[header]

            # Infer data type robustly (float, integer, or categorical)
            if pd.api.types.is_numeric_dtype(series):
                data_type = "float" if pd.api.types.is_float_dtype(series) else "integer"
            else:
                data_type = "categorical"

            missing_count = int(series.isnull().sum())
            missing_percentage = round(series.isnull().mean() * 100, 2)
            constant_flag = bool(series.nunique(dropna=True) == 1)

            col_diag = {
                "column_name": header,
                "data_type": data_type,
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
                "constant_flag": constant_flag
            }

            # Numeric columns: only add allowed numeric metrics if at least 3 unique values
            if data_type in ["integer", "float"]:
                clean_series = series.dropna()
                if clean_series.nunique() > 2:
                    col_diag["skewness"] = round(float(clean_series.skew()), 2)
                    col_diag["kurtosis"] = round(float(clean_series.kurtosis()), 2)

            # Categorical columns: only add allowed categorical metrics if non-empty
            if data_type == "categorical":
                clean_series = series.dropna()
                if len(clean_series) > 0:
                    unique_count = int(clean_series.nunique())
                    unique_ratio = round(unique_count / len(clean_series), 4) if len(clean_series) > 0 else 0
                    col_diag["unique_count"] = unique_count
                    col_diag["unique_ratio"] = unique_ratio

            column_diagnostics.append(col_diag)

        diagnostic_report = {
            "filename": file_name,
            "dataset_summary": dataset_summary,
            "column_diagnostics": column_diagnostics,
        }

        redis_cache.set(
            cache_key,
            json.dumps(diagnostic_report, cls=NumpyJSONEncoder),
            ex=86400
        )
        return diagnostic_report

    except Exception as e:
        print(f"CRITICAL ERROR in generate_diagnostic_report: {e}")
        raise e

@celery_app.task(time_limit=1800)
def generate_treatment_plans_task(dataset_name: str, target_variable: str, goal: str):
    """
    Generates three competing data cleaning plans by passing the diagnostic report to an LLM.
    """
    try:
        cache_key = f"diagnostics:{dataset_name}"
        report_str = redis_cache.get(cache_key)
        
        if not report_str:
            return {"status": "FAILURE", "error": f"Diagnostic report for {dataset_name} not found in cache."}
            
        diagnostic_report = json.loads(report_str)
        
        # Add context for the AI, which can be used in more advanced prompts later
        diagnostic_report['modeling_context'] = {
            'target_variable': target_variable,
            'goal': goal
        }

        plans = get_treatment_plan_hypotheses(diagnostic_report)

        if "error" in plans:
             return {"status": "FAILURE", "error": plans.get("details", "AI service failed to generate plans.")}

        return {"status": "SUCCESS", "result": plans}
        
    except Exception as e:
        print(f"CRITICAL ERROR in generate_treatment_plans_task for {dataset_name}: {e}")
        return {"status": "FAILURE", "error": str(e)}

def get_temporal_profile(df: pd.DataFrame, col: str) -> dict:
    time_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
    if not time_cols:
        return {"is_time_series": False}
    time_col = time_cols[0]
    try:
        ts_data = pd.to_datetime(df[time_col], errors='coerce').dropna()
        if ts_data.empty: return {"is_time_series": False}
    except Exception:
        return {"is_time_series": False}
    ts_series = df.set_index(time_col)[col].sort_index().dropna()
    acf_1 = None
    if ts_series.nunique() > 1 and len(ts_series) > 1:
        try:
            acf_1 = round(acf(ts_series, nlags=1, fft=False)[1], 2)
        except Exception:
            acf_1 = None
    return {"is_time_series": True, "temporal_stability_acf1": acf_1}

def get_mnar_indicators(df: pd.DataFrame, col: str) -> dict:
    correlations = {}
    missing_indicator = df[col].isnull().astype(int)
    for other_col in df.columns:
        if other_col == col: continue
        try:
            if pd.api.types.is_numeric_dtype(df[other_col]) and df[other_col].nunique() > 1:
                corr = missing_indicator.corr(df[other_col])
                if pd.notnull(corr) and abs(corr) > 0.3:
                    correlations[other_col] = round(corr, 2)
        except Exception:
            continue
    return correlations

def get_statistical_profile(df: pd.DataFrame, column_name: str) -> dict:
    detected_type = detect_data_type(df[column_name])

    missing_count = int(df[column_name].isnull().sum())
    total_count = len(df[column_name])
    missing_pct = (missing_count / total_count) * 100 if total_count > 0 else 0

    profile = {
        "column": column_name,
        "missing_count": missing_count, 
        "missing_pct": round(missing_pct, 4),
        "data_type": detected_type,
        "unique_values": df[column_name].nunique()
    }
    if detected_type in ['integer', 'float', 'identifier']:
        clean_data = df[column_name].dropna()
        if not clean_data.empty:
            profile["mean"] = round(clean_data.mean(), 2)
            profile["median"] = round(clean_data.median(), 2)
    profile["mnar_indicators"] = get_mnar_indicators(df, column_name)
    profile.update(get_temporal_profile(df, column_name))
    return profile

def perform_standardization(df: pd.DataFrame, column_name: str, method: str, file_path: str) -> dict:
    new_col_name = f"{column_name}_{method}_scaled"
    if new_col_name in df.columns:
        return {"status": "SKIPPED", "message": f"Column '{new_col_name}' already exists."}
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise ValueError(f"Column '{column_name}' is not numeric.")
    if df[column_name].isnull().any():
        raise ValueError(f"Column '{column_name}' contains missing values. Impute first.")
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    df[new_col_name] = scaler.fit_transform(df[[column_name]].values.astype(np.float32))
    df.to_csv(file_path, index=False)
    q1 = float(df[column_name].quantile(0.25))
    q3 = float(df[column_name].quantile(0.75))
    audit_report = {
        "new_column_added": new_col_name, "message": f"Successfully created new column: '{new_col_name}'.",
        "method": method, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "row_count_affected": len(df),
        "input_stats_robust": {"median": float(df[column_name].median()), "interquartile_range": q3 - q1}
    }
    return audit_report

def perform_delete_column(df: pd.DataFrame, column_name: str, file_path: str):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found.")
    df.drop(columns=[column_name], inplace=True)
    df.to_csv(file_path, index=False)
    return {"message": f"Successfully deleted column '{column_name}' and updated the dataset."}

@celery_app.task
def perform_dataset_cleaning_task(file_path: str, action_type: str):
    try:
        if not os.path.exists(file_path):
            return {"status": "FAILURE", "error": "File not found."}

        df = pd.read_csv(file_path, on_bad_lines='skip')
        original_rows = len(df)

        if action_type == 'drop_na_rows':
            df.dropna(inplace=True)
            rows_affected = original_rows - len(df)
            message = f"Successfully dropped {rows_affected} rows with missing values."
        
        elif action_type == 'drop_duplicate_rows':
            df.drop_duplicates(inplace=True)
            rows_affected = original_rows - len(df)
            message = f"Successfully dropped {rows_affected} duplicate rows."
            
        else:
            return {"status": "FAILURE", "error": f"Unknown cleaning action: {action_type}"}

        # Overwrite the original file with the cleaned data
        df.to_csv(file_path, index=False)

        return {"status": "SUCCESS", "message": message, "rows_affected": rows_affected}

    except Exception as e:
        print(f"CRITICAL ERROR in perform_dataset_cleaning_task for {file_path}: {e}")
        return {"status": "FAILURE", "error": str(e)}
    
def perform_imputation(df: pd.DataFrame, column_name: str, method: str, value=None) -> dict:
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found.")

    original_missing_count = int(df[column_name].isnull().sum())
    if original_missing_count == 0:
        return {"message": "No missing values to impute.", "rows_affected": 0}

    if method == 'mean':
        fill_value = df[column_name].mean()
        df[column_name].fillna(fill_value, inplace=True)
    elif method == 'median':
        fill_value = df[column_name].median()
        df[column_name].fillna(fill_value, inplace=True)
    elif method == 'mode':
        fill_value = df[column_name].mode()[0]
        df[column_name].fillna(fill_value, inplace=True)
    elif method == 'constant':
        dtype = df[column_name].dtype
        try:
            fill_value = pd.Series([value]).astype(dtype).iloc[0]
        except (ValueError, TypeError):
            fill_value = value
        df[column_name].fillna(fill_value, inplace=True)
    else:
        raise ValueError(f"Invalid imputation method: {method}")

    return {"message": f"Successfully imputed {original_missing_count} missing values in '{column_name}'.", "rows_affected": original_missing_count}

def _apply_plan_steps(df: pd.DataFrame, steps: list) -> pd.DataFrame:
    """Applies a list of cleaning steps to a dataframe."""
    df_copy = df.copy()
    for step in steps:
        func = step['function_name']
        cols = step['target_columns']
        
        if func == 'delete_column':
            df_copy.drop(columns=cols, inplace=True, errors='ignore')
        elif func == 'impute_median' and cols:
            for col in cols:
                if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
                    median_val = df_copy[col].median()
                    df_copy[col].fillna(median_val, inplace=True)
        elif func == 'impute_mode' and cols:
            for col in cols:
                if col in df_copy.columns:
                    mode_val = df_copy[col].mode()
                    if not mode_val.empty:
                        df_copy[col].fillna(mode_val[0], inplace=True)
    return df_copy

def execute_ai_transformation(df: pd.DataFrame, code_str: str) -> pd.DataFrame:
    """
    Executes AI-generated Python code on a dataframe in a restricted scope.
    Includes robustness checks for Feature Engineering.
    """
    if not code_str:
        return df

    # Create a copy so we don't mess up the original data
    local_df = df.copy()
    initial_col_count = len(local_df.columns)
    
    # Define the 'Safe Box'
    local_scope = {
        "df": local_df,
        "pd": pd,
        "np": np
    }

    try:
        # Run the code
        exec(code_str, {}, local_scope)
        
        # Get the modified dataframe back
        result_df = local_scope.get("df")
        
        if not isinstance(result_df, pd.DataFrame):
            print("Error: AI code killed the dataframe.")
            return df 
        
        # --- ROBUSTNESS CHECK: EXPLOSION PROTECTION ---
        # If the Architect creates >50 new columns (e.g., bad One-Hot), block it.
        final_col_count = len(result_df.columns)
        if final_col_count > initial_col_count + 50:
             print(f"Safety Limit Triggered: AI created {final_col_count - initial_col_count} columns. Reverting.")
             return df 
            
        return result_df
    except Exception as e:
        print(f"AI Code Execution Failed: {e}")
        return df

def _validate_plan_robust(df: pd.DataFrame, plan: dict, target: str, goal: str) -> dict:
    """
    Returns a dict: {"score": float, "error": str/None}
    """
    try:
        # --- EXECUTION ---
        if 'python_code' in plan and plan['python_code']:
            df_processed = execute_ai_transformation(df, plan['python_code'])
        else:
            df_processed = _apply_plan_steps(df, plan.get('steps', []))

        # --- PREPARATION ---
        df_processed = df_processed.dropna(subset=[target])
        if df_processed.empty:
            return {"score": -np.inf, "error": "Dataset became empty after cleaning."}
        
        # --- ROBUSTNESS: SANITIZE INFINITY ---
        # Feature Engineering (e.g. division) often creates inf. Treat as NaN for imputation.
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)

        X = df_processed.drop(columns=[target])
        y = df_processed[target]

        # Check for single class (Crash prevention for ROC AUC)
        if goal == 'classification' and y.nunique() < 2:
             return {"score": -np.inf, "error": "Target has only 1 class (needs 2+ for classification)."}

        # --- PIPELINE SETUP ---
        numeric_cols = X.select_dtypes(include=['number']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

        if goal == 'classification':
            model = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
            # Handle text targets
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            metric = roc_auc_score
        else:
            model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42, n_jobs=-1)
            metric = lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))

        # --- TRAINING ---
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pipeline.fit(X_train, y_train)
        
        # --- SCORING ---
        if goal == 'classification':
            # Handle binary vs multiclass
            if len(np.unique(y)) > 2:
                probs = pipeline.predict_proba(X_test)
                score = metric(y_test, probs, multi_class='ovr')
            else:
                probs = pipeline.predict_proba(X_test)[:, 1]
                score = metric(y_test, probs)
        else:
            preds = pipeline.predict(X_test)
            score = metric(y_test, preds)

        return {"score": score, "error": None}

    except Exception as e:
        return {"score": -np.inf, "error": str(e)}

def detect_data_leakage(df: pd.DataFrame, target: str) -> list:
    print(f"DEBUG: Running Leakage Check on {target}") 
    warnings = []

    try:
        if df[target].dtype == 'object':
            df[target] = pd.to_numeric(df[target])
    except:
        pass
    
    # Check 1: Perfect Correlation (Leakage)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if target in numeric_cols:
        correlations = df[numeric_cols].corrwith(df[target]).abs()
        # Filter out the target itself
        correlations = correlations.drop(target, errors='ignore')
        
        # Debug print to prove math is working
        max_corr = correlations.max() if not correlations.empty else 0
        print(f"DEBUG: Max Correlation found: {max_corr}")
        
        # Flag anything > 0.95 (Extremely suspicious)
        suspicious_cols = correlations[correlations > 0.95].index.tolist()
        
        if suspicious_cols:
            # --- FIX IS HERE: Define 'msg' before printing it ---
            msg = f"High Leakage Risk: Columns {suspicious_cols} are >95% correlated with the target."
            print(f"DEBUG: Found Warning: {msg}")
            warnings.append(msg)

    # Check 2: ID Column Detection
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col == target: continue
        
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.95 and len(df) > 100:
            msg = f"Potential ID Column: '{col}' is nearly unique. It may cause overfitting."
            warnings.append(msg)

    return warnings

# --- START: NEW MAIN SIMULATION TASK ---
@celery_app.task(time_limit=3600)
def run_impact_simulation_task(dataset_name: str, plans: dict, target_variable: str, goal: str):
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'public', dataset_name)
        df_raw = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)

        leakage_warnings = detect_data_leakage(df_raw, target_variable)

        # 1. Baseline
        baseline_res = _validate_plan_robust(df_raw, {}, target_variable, goal)
        baseline_score = baseline_res["score"]
        
        simulation_results = []
        plan_keys = ['conservative_plan', 'balanced_plan', 'aggressive_plan', 'architect_plan']
        metric_name = "AUC" if goal == 'classification' else "Neg RMSE"

        for key in plan_keys:
            if key not in plans: continue
            plan = plans[key]
            
            # 2. Plan Score
            plan_res = _validate_plan_robust(df_raw, plan, target_variable, goal)
            plan_score = plan_res["score"]
            error_msg = plan_res["error"]
            
            # 3. Handle Errors/Deltas
            if error_msg or not np.isfinite(plan_score) or not np.isfinite(baseline_score):
                 impact_data = {
                    "metric_name": metric_name,
                    "baseline_score": "N/A",
                    "plan_score": "Error",
                    "delta_percent": 0,
                    # Show the REAL error in the UI
                    "impact_string": f"<b>Simulation Failed:</b> {error_msg or baseline_res.get('error') or 'Unknown error'}"
                }
            else:
                delta = ((plan_score - baseline_score) / abs(baseline_score)) * 100 if baseline_score != 0 else 0
                sign = "+" if delta >= 0 else ""
                impact_data = {
                    "metric_name": metric_name,
                    "baseline_score": round(baseline_score, 4),
                    "plan_score": round(plan_score, 4),
                    "delta_percent": round(delta, 2),
                    "impact_string": f"{metric_name} changed by **{sign}{delta:.2f}%**"
                }

            plan['measured_impact'] = impact_data
            simulation_results.append(plan)

        sorted_results = sorted(
            simulation_results, 
            key=lambda p: p['measured_impact'].get('delta_percent', -999), 
            reverse=True
        )

        return {"status": "SUCCESS", "result": sorted_results,"warnings": leakage_warnings}

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return {"status": "FAILURE", "error": str(e)}

@celery_app.task
def apply_ai_plan_task(dataset_name: str, python_code: str, note: str = "Applied AI Plan"):
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'public', dataset_name)
        if not os.path.exists(file_path):
            return {"status": "FAILURE", "error": "File not found."}

        # 1. Load Data
        df = pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
        original_rows = len(df)

        # 2. Execute the AI Code (REUSING the safe executor we made)
        # Note: We must define 'execute_ai_transformation' if it's not globally available in this scope,
        # but since we defined it in the previous step in this file, it works.
        df_clean = execute_ai_transformation(df, python_code)
        
        # 3. Save Over the Original File (Or you could version it)
        # For this stage, overwriting is expected behavior for "Cleaning"
        df_clean.to_csv(file_path, index=False)

        # 4. Invalidate Cache (CRITICAL)
        # If we don't do this, the UI will still show the old "Dirty" stats
        redis_cache.delete(f"statistics:{dataset_name}")
        redis_cache.delete(f"diagnostics:{dataset_name}")

        return {
            "status": "SUCCESS", 
            "message": f"Successfully applied plan. Dataset updated.",
            "rows_remaining": len(df_clean),
            "original_rows": original_rows
        }

    except Exception as e:
        print(f"CRITICAL ERROR in apply_ai_plan_task: {e}")
        return {"status": "FAILURE", "error": str(e)}


@celery_app.task
def route_task(dataset_name: str, column_name: str, task_type: str, task_params: dict = None):
    try:
        file_path = os.path.join(os.path.dirname(__file__), '..', 'public', dataset_name)
        df = pd.read_csv(file_path)
        if task_type == 'diagnosis':
            profile = get_statistical_profile(df, column_name)
            result = get_ai_interpretation(profile)
            return {"status": "SUCCESS", "result": result}
        elif task_type == 'delete_column':
            result = perform_delete_column(df, column_name, file_path)
            return {"status": "SUCCESS", "result": result}
        elif task_type.startswith('impute_'):
            method = task_type.split('_')[1]
            custom_value = task_params.get('value') if task_params else None
            result = perform_imputation(df, column_name, method, value=custom_value)
            df.to_csv(file_path, index=False)
            return {"status": "SUCCESS", "result": result}
        elif task_type in ['standard_scale', 'minmax_scale']:
            method = 'standard' if task_type == 'standard_scale' else 'minmax'
            result = perform_standardization(df, column_name, method, file_path)
            return {"status": "SUCCESS", "result": result}
        else:
            return {"status": "ERROR", "message": "Unknown task type."}
    except Exception as e:
        return {"status": "FAILURE", "error": str(e)}