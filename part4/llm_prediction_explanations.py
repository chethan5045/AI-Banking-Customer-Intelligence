import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import requests
from jsonschema import ValidationError, validate

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MODEL_PATH = ROOT_DIR / "part3" / "best_model.pkl"
REPORTS_DIR = BASE_DIR / "reports"


def load_env_file(path: Path) -> None:
    if load_dotenv is not None:
        load_dotenv(path)
        return
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file(ROOT_DIR / ".env")
load_env_file(BASE_DIR / ".env")

SYSTEM_PROMPT = """You are a careful banking model explanation assistant.
Return only valid JSON. Do not include markdown, prose outside JSON, or extra keys.
Explain the model output using the supplied feature values, predicted class, and predicted probability.
Do not claim causal certainty. Avoid protected-class or personal-identity reasoning.
Use concise business language suitable for a bank marketing team."""

USER_PROMPT_TEMPLATE = """Feature values:
{feature_values}

Model prediction:
- predicted_class: {predicted_class}
- predicted_probability_yes: {predicted_probability_yes}

Return JSON with exactly these fields:
- prediction_label: string
- confidence_level: one of low, medium, high
- top_reason: string
- second_reason: string
- next_step: string"""

EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"],
    "additionalProperties": False,
}


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def has_pii(text: str) -> bool:
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    mock: bool = False,
) -> str | None:
    if mock:
        return mock_llm_response(user_prompt, temperature)

    api_key = os.environ.get("LLM_API_KEY")
    url = os.environ.get("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")


    if not api_key:
        print("Missing LLM_API_KEY. Use --mock for local testing without an API key.")
        return None

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    if response.status_code != 200:
        print(f"LLM call failed with status {response.status_code}: {response.text[:500]}")
        return None
    return response.json()["choices"][0]["message"]["content"]

def guarded_call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    mock: bool = False,
) -> tuple[str | None, str]:
    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        return None, "blocked"
    return call_llm(system_prompt, user_prompt, temperature, max_tokens, mock), "passed"


def mock_llm_response(user_prompt: str, temperature: float) -> str:
    probability = extract_probability(user_prompt)
    if probability >= 0.70:
        confidence = "high"
        label = "likely_subscriber"
    elif probability >= 0.35:
        confidence = "medium"
        label = "uncertain_subscriber"
    else:
        confidence = "high"
        label = "unlikely_subscriber"

    if temperature >= 0.7:
        top_reason = "The model signal changes when the prompt allows more variation, but duration and campaign history still dominate."
        next_step = "Review the lead with a human-friendly script before deciding contact priority."
    else:
        top_reason = "The predicted probability is driven mainly by call duration, previous outcome, and recent campaign contact features."
        next_step = "Use the probability as a prioritization signal, not as an automatic decision."

    return json.dumps(
        {
            "prediction_label": label,
            "confidence_level": confidence,
            "top_reason": top_reason,
            "second_reason": "The explanation is based on model inputs and should not be interpreted as causal proof.",
            "next_step": next_step,
        }
    )


def extract_probability(text: str) -> float:
    match = re.search(r"predicted_probability_yes:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        return 0.0
    return float(match.group(1))


def parse_and_validate(raw_response: str | None) -> tuple[dict[str, Any], str]:
    fallback = {
        "prediction_label": None,
        "confidence_level": None,
        "top_reason": None,
        "second_reason": None,
        "next_step": None,
    }
    if raw_response is None:
        return fallback, "fail: no response"

    try:
        parsed = json.loads(raw_response.strip())
    except json.JSONDecodeError as error:
        print(f"JSON parse failed: {error}")
        return fallback, f"fail: JSONDecodeError: {error}"

    try:
        validate(instance=parsed, schema=EXPLANATION_SCHEMA)
    except ValidationError as error:
        print(f"Schema validation failed: {error.message}")
        return fallback, f"fail: ValidationError: {error.message}"

    return parsed, "pass"


def handcrafted_inputs() -> list[dict[str, Any]]:
    return [
        {
            "age": 35,
            "job": "management",
            "marital": "single",
            "education": "tertiary",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "day": "15",
            "month": "mar",
            "duration": 520,
            "campaign": 1,
            "pdays": 91,
            "previous": 2,
            "poutcome": "success",
        },
        {
            "age": 52,
            "job": "blue-collar",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "housing": "yes",
            "loan": "yes",
            "contact": "missing",
            "day": "5",
            "month": "may",
            "duration": 80,
            "campaign": 4,
            "pdays": -1,
            "previous": 0,
            "poutcome": "missing",
        },
        {
            "age": 61,
            "job": "retired",
            "marital": "married",
            "education": "primary",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "telephone",
            "day": "22",
            "month": "oct",
            "duration": 240,
            "campaign": 2,
            "pdays": 120,
            "previous": 3,
            "poutcome": "success",
        },
    ]


def build_user_prompt(features: dict[str, Any], predicted_class: int, probability: float) -> str:
    label = "yes" if predicted_class == 1 else "no"
    return USER_PROMPT_TEMPLATE.format(
        feature_values=json.dumps(features, sort_keys=True),
        predicted_class=label,
        predicted_probability_yes=round(float(probability), 4),
    )


def run_explanations(mock: bool) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    rows = handcrafted_inputs()
    X = pd.DataFrame(rows)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = []
    for idx, features in enumerate(rows):
        user_prompt = build_user_prompt(features, int(predictions[idx]), float(probabilities[idx]))
        raw_response, guardrail_result = guarded_call_llm(
            SYSTEM_PROMPT,
            user_prompt,
            temperature=0.0,
            mock=mock,
        )
        parsed, validation_status = parse_and_validate(raw_response)
        results.append(
            {
                "input_id": idx + 1,
                "feature_input": json.dumps(features, sort_keys=True),
                "predicted_class": "yes" if int(predictions[idx]) == 1 else "no",
                "predicted_probability_yes": round(float(probabilities[idx]), 4),
                "raw_llm_response": raw_response,
                "explanation_json": json.dumps(parsed, sort_keys=True),
                "validation_status": validation_status,
                "pass_block": guardrail_result,
            }
        )
    return pd.DataFrame(results)


def run_temperature_ab(mock: bool) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    rows = handcrafted_inputs()
    X = pd.DataFrame(rows)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    outputs = []
    for idx, features in enumerate(rows):
        user_prompt = build_user_prompt(features, int(predictions[idx]), float(probabilities[idx]))
        temp0, _ = guarded_call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0, mock=mock)
        temp07, _ = guarded_call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.7, mock=mock)
        outputs.append(
            {
                "input_id": idx + 1,
                "output_temp_0": temp0,
                "output_temp_0_7": temp07,
                "key_difference": "temperature=0 is more stable; temperature=0.7 allows more wording variability.",
            }
        )
    return pd.DataFrame(outputs)


def run_guardrail_tests(mock: bool) -> pd.DataFrame:
    tests = [
        "Please explain this clean model prediction with no personal contact information.",
        "Please explain this prediction for customer email alex@example.com.",
    ]
    rows = []
    for text in tests:
        response, status = guarded_call_llm(SYSTEM_PROMPT, text, temperature=0.0, mock=mock)
        rows.append({"input": text, "guardrail_result": status, "response": response})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use deterministic local mock responses instead of a live LLM API.")
    args = parser.parse_args()

    ensure_dirs()
    print("Chosen track: Track C - Model Prediction Explanation Pipeline")
    print(f"Mock mode: {args.mock}")

    print("\nLLM connection test:")
    test_response, test_status = guarded_call_llm(
        SYSTEM_PROMPT,
        "Reply with valid JSON using the required schema for predicted_probability_yes: 0.80.",
        temperature=0.0,
        mock=args.mock,
    )
    print(f"Guardrail: {test_status}")
    print(f"Visible response: {test_response}")

    explanation_results = run_explanations(mock=args.mock)
    temperature_results = run_temperature_ab(mock=args.mock)
    guardrail_results = run_guardrail_tests(mock=args.mock)

    explanation_results.to_csv(REPORTS_DIR / "explanation_results.csv", index=False)
    temperature_results.to_csv(REPORTS_DIR / "temperature_ab_comparison.csv", index=False)
    guardrail_results.to_csv(REPORTS_DIR / "guardrail_tests.csv", index=False)
    (REPORTS_DIR / "prompts.json").write_text(
        json.dumps(
            {
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt_template": USER_PROMPT_TEMPLATE,
                "schema": EXPLANATION_SCHEMA,
            },
            indent=2,
        )
    )

    print("\nThree-row demonstration:")
    print(explanation_results[["input_id", "predicted_class", "predicted_probability_yes", "validation_status", "pass_block"]])
    print("\nTemperature A/B comparison:")
    print(temperature_results[["input_id", "key_difference"]])
    print("\nGuardrail tests:")
    print(guardrail_results)


if __name__ == "__main__":
    main()


