import json
import os
import requests
import re 

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY","")
if not OPENROUTER_API_KEY:
    raise ValueError("CRITICAL ERROR: OPENROUTER_API_KEY environment variable is not set.")

def _call_openrouter_api(system_prompt: str, user_prompt: str) -> dict:
    """
    A private helper function to handle the actual API call to OpenRouter.
    """
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "DataCraft Studio"
            },
            data=json.dumps({
                "model": "nvidia/nemotron-nano-9b-v2:free",
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }),
            timeout=120 # Increased timeout for more complex generation
        )
        response.raise_for_status()
        response_data = response.json()
        ai_content_string = response_data['choices'][0]['message']['content']
        
        json_match = re.search(r'\{.*\}', ai_content_string, re.DOTALL)
        if not json_match:
            # Fallback: sometimes the model just sends the code block
            print(f"DEBUG: AI Output was: {ai_content_string}")
            raise ValueError("AI response did not contain a valid JSON object.")
            
        json_string = json_match.group(0)
        return json.loads(json_string)

    except Exception as e:
        print(f"Error in _call_openrouter_api: {e}")
        # Return a safe error so the UI doesn't crash
        return {
            "error": "AI Service Error", 
            "details": str(e)
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

    1. **DOMAIN INFERENCE**  
    → What domain does this likely belong to? (IoT, finance, healthcare, etc.)  
    → **Evidence**: "Column name='temperature' + high ACF(1)=0.88 → IoT sensor data"  
    → **Contradictions**: "But MNAR pattern suggests possible financial context"

    2. **MISSINGNESS PATTERN ASSESSMENT**  
    → "MNAR indicators exist (humidity: -0.65) → systematic bias likely"  
    → "BUT high temporal stability (ACF=0.88) suggests gradual change"  
    → **Critical question**: "Is the correlation meaningful or coincidental?"

    3.  **RISK-BASED EVALUATION**  
    → "For sensor data, bias could cause safety issues"  
    → "For financial data, bias could trigger regulatory penalties"  
    → "What's the cost of being wrong? (e.g., $10k vs $1M impact)"

    4.  **TECHNIQUE TRADEOFF ANALYSIS**  
    → "ffill would be fast but assumes stability during gaps"  
    → "MICE would be accurate but requires sufficient data"  
    → **Key insight**: "For this domain, [X] matters more than [Y]"

    5.  **DECISION WITH UNCERTAINTY**  
    → "Recommend [X], but only if [critical assumption] holds"  
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
    
    try:
        return _call_openrouter_api(system_prompt, user_prompt)
    except Exception as e:
        return {
            "recommendation": "AI Service Error",
            "reasoning_summary": "Could not connect to the AI model or the response was invalid.",
            "assumptions": [],
            "warning": str(e)
        }

def get_treatment_plan_hypotheses(diagnostic_report: dict) -> dict:
    """
    Takes a full dataset diagnostic report and generates three competing
    data cleaning and feature engineering strategies.
    """
    system_prompt = """
    You are a Committee of Chief Data Scientists. Generate THREE distinct "Missing Value Treatment Plans".
    
    ### CRITICAL INSTRUCTION: DUAL OUTPUT MODE
    For every plan, you must provide TWO things:
    1. **"steps"**: A structured list for the UI (Human Readable).
    2. **"python_code"**: A valid, executable Python string that performs the cleaning.

    ### PYTHON CODE RULES
    - The code must assume a pandas DataFrame named `df` exists.
    - It must NOT import `os`, `sys` or read/write files (Security Risk). 
    - It must import `pandas as pd` and `numpy as np` inside the string if needed.
    - It must handle `SettingWithCopyWarning` by operating on `df`.
    - It must be self-contained.

    ### OUTPUT FORMAT (STRICT JSON)
    ```json
    {
      "conservative_plan": {
        "name": "Conservative Plan",
        "rationale": "Minimal intervention to preserve raw data integrity.",
        "steps": [{"function_name": "delete_column", "target_columns": ["col_A"], "reasoning": "..."}],
        "python_code": "import pandas as pd\\ndf.drop(columns=['col_A'], inplace=True, errors='ignore')"
      },
      "balanced_plan": {
        "name": "Balanced Plan",
        "rationale": "Standard imputation strategy.",
        "steps": [{"function_name": "impute_median", "target_columns": ["col_B"], "reasoning": "..."}],
        "python_code": "median_val = df['col_B'].median()\\ndf['col_B'].fillna(median_val, inplace=True)"
      },
      "aggressive_plan": { 
        "name": "Aggressive Plan",
        "rationale": "High-intervention strategy for maximum signal.",
        "steps": [...],
        "python_code": "..."
      },
      "architect_plan": {
        "name": "The Architect",
        "rationale": "Feature Engineering focus to extract hidden signals.",
        "steps": [
            {"function_name": "extract_date_features", "target_columns": ["order_date"], "reasoning": "Seasonality signals"},
            {"function_name": "create_interaction", "target_columns": ["price", "qty"], "reasoning": "Revenue proxy"}
        ],
        "python_code": "import pandas as pd\\nimport numpy as np\\n# Date Feature\\ndf['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')\\ndf['order_month'] = df['order_date'].dt.month\\n# Interaction\\ndf['revenue_proxy'] = df['price'] * df['qty']"
      }
    }
    ```
    """
    
    user_prompt = f"Here is the Diagnostic Report:\n{json.dumps(diagnostic_report, indent=2)}"
    
    try:
        return _call_openrouter_api(system_prompt, user_prompt)
    except Exception as e:
        return {
            "error": "Failed to generate treatment plans.",
            "details": str(e)
        }