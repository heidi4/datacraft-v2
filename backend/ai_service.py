import json
import os
import requests
import re 
import time

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY","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")
if not OPENROUTER_API_KEY:
    raise ValueError("CRITICAL ERROR: OPENROUTER_API_KEY environment variable is not set.")

def _call_openrouter_api(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
    """
    A private helper function to handle the actual API call to OpenRouter.
    Includes RETRY LOGIC to handle malformed JSON responses from the AI.
    """
    MAX_RETRIES = 3
    
    for attempt in range(MAX_RETRIES):
        try:
            # We use a slight backoff if it's a retry
            if attempt > 0:
                time.sleep(1)
                
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "DataCraft Studio"
                },
                data=json.dumps({
                    "model": "nvidia/nemotron-3-nano-30b-a3b:free",
                    "temperature": temperature,
                    # asking for json_object can help if the model supports it
                    "response_format": { "type": "json_object" },
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }),
                timeout=120 
            )
            response.raise_for_status()
            response_data = response.json()
            
            if not response_data.get('choices'):
                raise ValueError("AI Service returned no content choices.")

            ai_content_string = response_data['choices'][0]['message']['content']
            
            # --- Cleaning Step 1: Remove Markdown Code Blocks ---
            # This handles ```json ... ``` or just ``` ... ``` wrapping
            cleaned_string = re.sub(r'^```[a-z]*\s*', '', ai_content_string, flags=re.MULTILINE)
            cleaned_string = re.sub(r'\s*```$', '', cleaned_string, flags=re.MULTILINE)
            cleaned_string = cleaned_string.strip()

            # --- Cleaning Step 2: Find JSON boundaries ---
            # We look for the first '{' and the last '}'
            json_match = re.search(r'\{.*\}', cleaned_string, re.DOTALL)
            
            if not json_match:
                # If we can't find braces, the output is definitely not JSON.
                print(f"DEBUG (Attempt {attempt+1}): No JSON braces found in output: {ai_content_string[:100]}...")
                continue # Retry
            
            json_string = json_match.group(0)
            
            # --- Validation Step ---
            return json.loads(json_string)

        except json.JSONDecodeError as e:
            print(f"JSON Parse Error in _call_openrouter_api (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
            # If it's the last attempt, we let the loop finish to return the error
            if attempt == MAX_RETRIES - 1:
                print(f"DEBUG: Failed Content was:\n{ai_content_string}")
        except Exception as e:
            print(f"Network/API Error in _call_openrouter_api (Attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES - 1:
                return {
                    "error": "AI Service Connection Failed", 
                    "details": str(e)
                }

    # If we exit the loop, we failed to get valid JSON
    return {
        "error": "AI Generation Failed",
        "details": "The AI model failed to generate valid JSON after multiple attempts. Please try again."
    }

def get_ai_interpretation(profile: dict) -> dict:
    """
    Sends a detailed statistical profile to an LLM for expert interpretation (existing functionality).
    """
    system_prompt = """
    You are a principal data scientist with 20+ years experience. Your task is to analyze a statistical profile of a column and provide a professional recommendation that reflects how human experts think — not rigid rule-following.

    ## HOW REAL DATA SCIENTISTS THINK (NOT RULE ENGINES)
    When handling missing data, experienced professionals:
    - **Distinguish between count and percentage.** A `missing_count` of 3 is a minor issue in 40,000 rows (low `missing_pct`), but you must still question *why* even those few are missing. A `missing_count` of 3 in 10 rows is critical.
    - **Never treat thresholds as absolute** (e.g., "60% missing" is a signal, not a rule)
    - **Infer domain from data patterns** (e.g., "temperature" + high ACF → sensor data)
    - **Acknowledge uncertainty** ("Without domain knowledge, I'd verify X first")
    - **Explain why alternatives were rejected** ("ffill would distort volatility here")

    ## YOUR ANALYSIS WORKFLOW (CHAIN OF THOUGHT)
    Follow this reasoning pattern **in your <thinking> block**:

    1. **DOMAIN INFERENCE** → What domain does this likely belong to? (IoT, finance, healthcare, etc.)  
    → **Evidence**: "Column name='temperature' + high ACF(1)=0.88 → IoT sensor data"  
    → **Contradictions**: "But MNAR pattern suggests possible financial context"

    2. **MISSINGNESS PATTERN ASSESSMENT** → "MNAR indicators exist (humidity: -0.65) → systematic bias likely"  
    → "BUT high temporal stability (ACF=0.88) suggests gradual change"  
    → **Critical question**: "Is the correlation meaningful or coincidental?"

    3.  **RISK-BASED EVALUATION** → "For sensor data, bias could cause safety issues"  
    → "For financial data, bias could trigger regulatory penalties"  
    → "What's the cost of being wrong? (e.g., $10k vs $1M impact)"

    4.  **TECHNIQUE TRADEOFF ANALYSIS** → "ffill would be fast but assumes stability during gaps"  
    → "MICE would be accurate but requires sufficient data"  
    → **Key insight**: "For this domain, [X] matters more than [Y]"

    5.  **DECISION WITH UNCERTAINTY** → "Recommend [X], but only if [critical assumption] holds"  
    → "Without [domain knowledge], I'd verify [specific check] first"  
    → "This assumes [unstated condition] — flag if violated"

    ## CRITICAL SAFEGUARDS (NOT RULES)
    - **High missingness**: "60%+ missing is a red flag, but dropping may lose critical signals"  
    - **MNAR patterns**: "Correlation >0.3 suggests systematic bias, but could be coincidental"  
    - **Time-series**: "ACF>0.85 supports ffill, but only if gaps align with stable periods"  

    ## YOUR RESPONSE FORMAT
    <thinking>
    [Your step-by-step reasoning using the workflow above]
    </thinking>

    {
    "recommendation": "Specific technique with parameters (e.g., 'ffill with max gap=3h')",
    "reasoning_summary": "Concise justification with domain context",
    "assumptions": ["Domain: IoT/sensor (evidence: column name + ACF)", "Gaps occur during calibration"],
    "warning": "Critical risk: If gaps occur during equipment failure, ffill would distort readings"
    }
    """
    
    user_prompt = f"Here is the statistical profile to analyze:\n{json.dumps(profile, indent=2)}"
    
    return _call_openrouter_api(system_prompt, user_prompt)

# === HELPER: TOKEN-SAFE REPORT CONDENSATION ===
def _condense_diagnostic_report(report: dict, top_n: int = 25) -> dict:
    """
    Reduces token count by ~90% while preserving critical diagnostic signals.
    Keeps full details only for problematic columns.
    """
    # 1. Base Structure
    condensed = {
        "modeling_context": report.get("modeling_context", {}),
        "missingness_overview": report.get("missingness", {}),
        "skew_overview": report.get("distribution_skew", {}),
        "target_correlations": report.get("target_correlations", {}),
        "note": "Full column details condensed to top problematic columns. Healthy columns omitted."
    }
    
    # 2. Identify Critical Columns
    critical_cols = set()
    
    # Add top missing
    missing_cols = sorted(report.get('missingness', {}).items(), key=lambda x: x[1], reverse=True)[:top_n]
    critical_cols.update([c[0] for c in missing_cols])
    
    # Add high skew
    skew_cols = [col for col, val in report.get('distribution_skew', {}).items() if abs(val) > 1.5]
    critical_cols.update(skew_cols[:top_n])
    
    # Add strong correlations
    corr_cols = [col for col, val in report.get('target_correlations', {}).items() if abs(val) > 0.2]
    critical_cols.update(corr_cols[:top_n])
    
    # 3. Filter Detail Dictionary
    if 'column_details' in report:
        condensed['column_details'] = {
            col: details for col, details in report['column_details'].items()
            if col in critical_cols
        }
    
    return condensed

# === HELPER: SAFETY-HARDENED FALLBACK PLAN ===
def _generate_fallback_plan(report: dict, target: str, temporal_col: str = None) -> dict:
    """
    Hardcoded conservative plan with leakage prevention and edge case handling.
    Used when the LLM API fails or times out.
    """
    missingness = report.get('missingness', {})
    high_missing_cols = [col for col, pct in missingness.items() if pct > 0.95]
    
    # Safe temporal handling code injection
    date_parse_code = ""
    if temporal_col:
        date_parse_code = (
            f"if '{temporal_col}' in df.columns:\n"
            f"    df['{temporal_col}'] = pd.to_datetime(df['{temporal_col}'].astype(str), errors='coerce')\n"
        )
    
    python_code = (
        "import pandas as pd\nimport numpy as np\n"
        f"{date_parse_code}"
        f"drop_cols = {high_missing_cols}\n"
        "existing_drop_cols = [c for c in drop_cols if c in df.columns]\n"
        "if existing_drop_cols:\n"
        "    df = df.drop(columns=existing_drop_cols)\n"
        f"if '{target}' in df.columns:\n"
        f"    df = df.dropna(subset=['{target}'])\n"
    )

    return {
        "conservative_plan": {
            "name": "Failsafe Conservative Plan",
            "rationale": "API failure recovery - minimal safe operations with leakage prevention.",
            "steps": [{
                "function_name": "delete_column",
                "target_columns": high_missing_cols,
                "reasoning": "Automatic recovery: Columns with >95% missing dropped per production policy."
            }],
            "python_code": python_code
        },
        # We perform a shallow copy for others to ensure the UI doesn't break
        "balanced_plan": {"name": "Balanced Plan (Unavailable)", "rationale": "Service unavailable", "steps": [], "python_code": ""},
        "aggressive_plan": {"name": "Aggressive Plan (Unavailable)", "rationale": "Service unavailable", "steps": [], "python_code": ""},
        "architect_plan": {"name": "Architect Plan (Unavailable)", "rationale": "Service unavailable", "steps": [], "python_code": ""}
    }

def get_treatment_plan_hypotheses(diagnostic_report: dict) -> dict:
    """
    Generates FOUR statistically rigorous data preparation strategies.
    Strictly constrained to the Action Library to prevent hallucination.
    """
    # === 1. CONTEXT EXTRACTION & SAFETY DEFAULTS ===
    context = diagnostic_report.get('modeling_context', {})
    target_var = context.get('target_variable', 'target')
    problem_type = context.get('problem_type', 'regression')
    temporal_col = context.get('temporal_column', None)
    
    # === 2. TOKEN-SAFE REPORT CONDENSATION ===
    condensed_report = _condense_diagnostic_report(diagnostic_report, top_n=25)

    # === 3. SYSTEM PROMPT (STRICT ACTION LIBRARY & EXAMPLES) ===
    system_prompt = f"""
    **ROLE**: Principal Data Scientist. You generate data cleaning strategies based on evidence, NOT blind checklists.

    ### 1. THE ALLOWED ACTION LIBRARY (STRICT)
    **Actions**:
    - **Cleaning**: `delete_column`, `drop_rows_where_null` (target/ID only), `drop_duplicate_rows`.
    - **Imputation**: `impute_mean`, `impute_median`, `impute_mode`, `impute_constant`, `forward_fill` (time-series only).
    - **Encoding**: `one_hot_encode`, `label_encode`.
    - **Transformation**: `log_transform`, `standard_scale`, `min_max_scale`, `clip_outliers`.
    - **Creation**: `create_interaction`, `create_date_features`, `create_missing_flag`.

    **COLUMN ACTION COMPATIBILITY RULE (VIOLATION = HALLUCINATION)**:
    - `log_transform`, `clip_outliers`, `standard_scale`, `min_max_scale` → **NUMERIC COLUMNS ONLY**.
    - `impute_mean`, `impute_median` → **NUMERIC COLUMNS ONLY**.
    - `one_hot_encode`, `label_encode` → **CATEGORICAL/OBJECT COLUMNS ONLY**.
    - `impute_mode` → Any column.

    ### 2. STRATEGY ARCHETYPES (PHILOSOPHIES, NOT RULES)
    
    **Plan 1: CONSERVATIVE ("The Auditor")**
    * *Philosophy*: "Do no harm." Prefer deleting bad data over guessing (imputing).
    * *Example Thought*: "Column 'age' has 5% missing. Imputation might bias results. Better to drop these few rows if dataset is large, or impute strictly with median."
    
    **Plan 2: BALANCED ("The Engineer")**
    * *Philosophy*: "Standard Best Practices." Use Median for skew, Mean for normal. Handle outliers.
    * *Example Thought*: "'Income' is highly skewed (skew=5.2). Mean imputation is dangerous here; I will use Median."
    
    **Plan 3: AGGRESSIVE ("The Maximizer")**
    * *Philosophy*: "Keep every row." Never drop rows. Impute everything. Flag missing values as features.
    * *Example Thought*: "I can't afford to lose any rows. I will impute 'age' and also create a 'age_is_missing' column to capture the signal of missingness."
    * *Aggressive plans MAY explore new features, BUT they must justify them using:
    - Target correlation strength.
    - Explicit domain plausibility.
    - Variance amplification.
    * *If none apply, SKIP feature creation.*

    **Plan 4: ARCHITECT ("The Feature Forge")**
    * *Philosophy*: "Domain Specific." Focus on feature creation (interactions, date parts) over just cleaning.
    * *Example Thought*: "Since we have 'price' and 'quantity', the most predictive feature is likely 'revenue = price * qty'. I must create this."
    * *Architect plans MAY explore new features, BUT they must justify them using:
    - Target correlation strength.
    - Explicit domain plausibility.
    - Variance amplification.
    * *If none apply, SKIP feature creation.*

    ### 3. FINAL SELF-CHECK (MANDATORY)
    Before returning JSON, verify:
    - All `target_columns` are in the dataset, no hallucinated columns.
    - No numeric actions (log/scale) applied to categorical columns.
    - Every step cites a specific statistic (e.g. "Using IQR to clip and limit the effect of extreme values beyond –14.5 and 46.5" "Imputing missing ages with a median of 32 to avoid distortion from a few very large values.").

    ### 4. OUTPUT FORMAT (STRICT JSON)
    You MUST return this exact JSON structure. `steps` matches `python_code`.
    
    {{
      "conservative_plan": {{
        "name": "Conservative Strategy",
        "rationale": "High missingness in 'marketing_channel' (40%) warrants deletion to avoid noise.",
        "steps": [
            {{ 
                "function_name": "delete_column", 
                "target_columns": ["marketing_channel"], 
                "reasoning": "Missing > 40% (Actual: 42%)" 
            }},
            {{
                "function_name": "impute_median",
                "target_columns": ["age"],
                "reasoning": "Low missingness (5%), skew is high (2.1)"
            }}
        ],
        "python_code": "import pandas as pd\\nimport numpy as np\\n# Code that implements the steps above..."
      }},
      "balanced_plan": {{ ... }},
      "aggressive_plan": {{ ... }},
      "architect_plan": {{ ... }}
    }}

    """

    # === 4. USER PROMPT ===
    top_missing = sorted(condensed_report['missingness_overview'].items(), key=lambda x: x[1], reverse=True)[:5]
    
    user_prompt = f"""
    ### DATASET CONTEXT
    - **Target Variable**: "{target_var}" ({problem_type})
    - **Temporal Column**: {temporal_col if temporal_col else 'None'}
    - **Problem**: Missing values and potential outliers need handling.
    
    ### DIAGNOSTIC SUMMARY
    - **Top Missing Columns**: {top_missing}
    - **Skewed Columns**: {[k for k, v in condensed_report['skew_overview'].items() if abs(v) > 2.0]}
    
    ### FULL REPORT
    {json.dumps(condensed_report, indent=2)}
    
    Generate the FOUR plans. Ensure `steps` array is populated and `python_code` is valid pandas.
    """

    try:
        # Precision mode (low temp) for production safety
        return _call_openrouter_api(system_prompt, user_prompt, temperature=0.1)
    except Exception as e:
        # Automatic Fallback
        print(f"AI Service Failed: {e}. Reverting to Failsafe Plan.")
        return _generate_fallback_plan(diagnostic_report, target_var, temporal_col)