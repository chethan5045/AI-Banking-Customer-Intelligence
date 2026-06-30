# Part 4 - LLM-Powered Model Prediction Explanation

## Chosen Track

Track C: Model Prediction Explanation Pipeline.

This feature loads the best model from Part 3, predicts subscription likelihood for three hand-crafted banking customer records, and asks an LLM to return a structured JSON explanation for each prediction.

## How to Run

Mock mode, no API key required:

```bash
pip install -r requirements.txt
python3 part4/llm_prediction_explanations.py --mock
```

Live API mode:

```bash
export LLM_API_KEY="your_api_key_here"
export LLM_API_URL="https://openrouter.ai/api/v1/chat/completions"
export LLM_MODEL="openai/gpt-4o-mini"
python3 part4/llm_prediction_explanations.py
```

The API key is read from `LLM_API_KEY`. It is never hardcoded. See `.env.example` for the required variable names.

## LLM API Connection

The reusable function is:

```python
call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512)
```

It builds a JSON payload with `model`, `messages`, `temperature`, and `max_tokens`, sends it with `requests.post`, checks for status code `200`, and returns:

```python
response.json()["choices"][0]["message"]["content"]
```

In `--mock` mode, it returns deterministic local JSON so the full pipeline can be graded without an API key.

## System Prompt

```text
You are a careful banking model explanation assistant.
Return only valid JSON. Do not include markdown, prose outside JSON, or extra keys.
Explain the model output using the supplied feature values, predicted class, and predicted probability.
Do not claim causal certainty. Avoid protected-class or personal-identity reasoning.
Use concise business language suitable for a bank marketing team.
```

## User Prompt Template

```text
Feature values:
{feature_values}

Model prediction:
- predicted_class: {predicted_class}
- predicted_probability_yes: {predicted_probability_yes}

Return JSON with exactly these fields:
- prediction_label: string
- confidence_level: one of low, medium, high
- top_reason: string
- second_reason: string
- next_step: string
```

## Temperature Choice

The main pipeline uses `temperature=0` because this is a structured-output task. A low temperature makes the model choose the highest-probability next token, producing more deterministic and predictable JSON. `temperature=0.7` samples from a broader distribution, which can improve variety but increases the risk of inconsistent wording or schema errors.

## JSON Schema

The required response schema has five scalar fields:

```json
{
  "prediction_label": "string",
  "confidence_level": "low | medium | high",
  "top_reason": "string",
  "second_reason": "string",
  "next_step": "string"
}
```

The script parses each LLM response with `json.loads(response.strip())` and validates it with `jsonschema.validate()`. `json.JSONDecodeError` and `jsonschema.ValidationError` are caught. On failure, the script returns a fallback dictionary with all required fields set to `None` and logs the error.

## PII Guardrail

Before each LLM call, the script checks user input for email addresses and phone numbers:

```python
email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
phone_pattern = r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
```

If PII is detected, the script prints `Input blocked: PII detected.` and does not call the LLM.

Guardrail test results:

| Input | Result |
| --- | --- |
| Clean model explanation request | passed |
| Request containing `alex@example.com` | blocked |

## Three-Row Demonstration

| Input | Predicted class | Probability yes | Validation | Pass/Block |
| ---: | --- | ---: | --- | --- |
| 1 | yes | 0.8800 | pass | passed |
| 2 | no | 0.0000 | pass | passed |
| 3 | yes | 0.6950 | pass | passed |

All three explanation responses were valid JSON and passed schema validation.

## Temperature A/B Comparison

| Input | Output at temperature 0 | Output at temperature 0.7 | Key difference |
| ---: | --- | --- | --- |
| 1 | Stable reason focused on duration, previous outcome, and campaign contact features | More variable wording, still focused on duration and campaign history | `temperature=0` is more stable |
| 2 | Stable low-subscription explanation | More variable next-step wording | `temperature=0.7` adds variability |
| 3 | Medium-confidence explanation | More flexible language around review and contact priority | Higher temperature samples broader wording |

This shows why `temperature=0` is preferable for structured JSON outputs that need to be parsed and validated consistently.

## Output Files

- `llm_prediction_explanations.py`: full Part 4 implementation
- `.env.example`: environment variable names, no secrets
- `reports/explanation_results.csv`: three-row prediction explanation table
- `reports/temperature_ab_comparison.csv`: temperature comparison
- `reports/guardrail_tests.csv`: PII guardrail tests
- `reports/prompts.json`: prompt and schema record
