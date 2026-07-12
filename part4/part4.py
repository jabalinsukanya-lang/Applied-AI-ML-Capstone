# ==========================================================
# PART 4
# MODEL PREDICTION EXPLANATION PIPELINE
# SECTION 1
# ==========================================================

import os
import json
import joblib
import requests
import pandas as pd

from dotenv import load_dotenv

# ----------------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------------

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")

if API_KEY is None:
    raise ValueError("LLM_API_KEY not found in .env")

print("API Key Loaded Successfully")

# ----------------------------------------------------------
# Load ML Model
# ----------------------------------------------------------

model = joblib.load("../part3/best_model.pkl")

print("Machine Learning Model Loaded Successfully")

# ----------------------------------------------------------
# OpenRouter Configuration
# ----------------------------------------------------------

URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {

    "Authorization": f"Bearer {API_KEY}",

    "Content-Type": "application/json",

    "HTTP-Referer": "http://localhost",

    "X-Title": "Applied AI ML Capstone"

}

# ----------------------------------------------------------
# LLM Function
# ----------------------------------------------------------

def call_llm(prompt,
             temperature=0):

    payload = {

        "model": "tencent/hy3:free",

        "messages":[

            {

                "role":"user",

                "content":prompt

            }

        ],

        "temperature":temperature,

        "max_tokens":300

    }

    response = requests.post(

        URL,

        headers=HEADERS,

        json=payload,

        timeout=60

    )

    if response.status_code != 200:

        raise Exception(

            f"OpenRouter Error {response.status_code}\n{response.text}"

        )

    data = response.json()

    try:

        answer = data["choices"][0]["message"]["content"]

        if answer is None:

            answer = "No content returned by model."

    except Exception:

        answer = json.dumps(data, indent=4)

    return answer

# ----------------------------------------------------------
# Test
# ----------------------------------------------------------

print("\nTesting LLM...\n")

reply = call_llm(

    "Explain in one sentence what machine learning is."

)

print(reply)

# PII GUARDRAILS & INPUT VALIDATION
# ==========================================================

import re

print("\n" + "=" * 70)
print("SECTION 2 - PII GUARDRAILS")
print("=" * 70)

# ----------------------------------------------------------
# Detect PII
# ----------------------------------------------------------

def detect_pii(text):

    findings = []

    # Email
    email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    if re.search(email_pattern, text):
        findings.append("Email Address")

    # Phone Number
    phone_pattern = r'(\+?\d{1,3}[- ]?)?\d{10}'
    if re.search(phone_pattern, text):
        findings.append("Phone Number")

    # Credit Card
    credit_pattern = r'\b(?:\d[ -]*?){13,16}\b'
    if re.search(credit_pattern, text):
        findings.append("Credit Card")

    # Aadhaar Number (India)
    aadhaar_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
    if re.search(aadhaar_pattern, text):
        findings.append("Aadhaar Number")

    return findings

# ----------------------------------------------------------
# Guardrail Function
# ----------------------------------------------------------

def validate_prompt(prompt):

    detected = detect_pii(prompt)

    if len(detected) > 0:

        print("\nPII Detected!")

        for item in detected:
            print("-", item)

        return False

    return True

# ----------------------------------------------------------
# Safe LLM Wrapper
# ----------------------------------------------------------

def safe_call_llm(prompt, temperature=0):

    if not validate_prompt(prompt):

        return {
            "status": "blocked",
            "reason": "Prompt contains personally identifiable information."
        }

    answer = call_llm(
        prompt,
        temperature
    )

    return {
        "status": "success",
        "response": answer
    }

# ----------------------------------------------------------
# Test Cases
# ----------------------------------------------------------

print("\nTesting Safe Prompt")

safe_prompt = "Explain why customer tenure affects churn."

result = safe_call_llm(safe_prompt)

print(result)

print("\nTesting Unsafe Prompt")

unsafe_prompt = """
Customer email is abc@gmail.com
Phone number is 9876543210
"""

result = safe_call_llm(unsafe_prompt)

print(result)

print("\n" + "=" * 70)
print("SECTION 3 - MODEL PREDICTION EXPLANATION")
print("=" * 70)

# ----------------------------------------------------------
# Load Sample Data
# ----------------------------------------------------------

sample_data = pd.read_csv("../part3/sample_predictions.csv")

# Remove prediction columns if present
drop_cols = [
    "Predicted_Class",
    "Probability_No",
    "Probability_Yes"
]

sample_data = sample_data.drop(
    columns=[col for col in drop_cols if col in sample_data.columns],
    errors="ignore"
)

print("\nSample Customer Data")

print(sample_data)

# ----------------------------------------------------------
# Predict
# ----------------------------------------------------------

predictions = model.predict(sample_data)

probabilities = model.predict_proba(sample_data)

print("\nPredictions")

print(predictions)

print("\nPrediction Probabilities")

print(probabilities)

# ----------------------------------------------------------
# Explain Each Prediction
# ----------------------------------------------------------

for i in range(len(sample_data)):

    probability = probabilities[i][1]

    prediction = "Likely to Churn" if predictions[i] == 1 else "Not Likely to Churn"

    prompt = f"""
You are an AI assistant helping customer-support staff.

A machine learning model predicted:

Prediction: {prediction}

Probability of Churn: {probability:.2f}

Customer Features:

{sample_data.iloc[i].to_dict()}

Explain:

1. Why the model may have reached this prediction.

2. Mention the important customer characteristics.

3. Suggest one business action to reduce churn.

Respond in simple English.
"""

    print("\n" + "-" * 60)

    print(f"Customer {i+1}")

    print("-" * 60)

    result = safe_call_llm(prompt)

    if result["status"] == "success":

        print(result["response"])

    else:

        print(result["reason"])

# JSON SCHEMA VALIDATION
# ==========================================================

from jsonschema import validate, ValidationError

print("\n" + "=" * 70)
print("SECTION 4 - JSON SCHEMA VALIDATION")
print("=" * 70)

# ----------------------------------------------------------
# JSON Schema
# ----------------------------------------------------------

response_schema = {

    "type": "object",

    "properties": {

        "prediction": {
            "type": "string"
        },

        "confidence": {
            "type": "number"
        },

        "explanation": {
            "type": "string"
        },

        "recommendation": {
            "type": "string"
        }

    },

    "required": [

        "prediction",

        "confidence",

        "explanation",

        "recommendation"

    ]

}

# ----------------------------------------------------------
# Structured LLM Call
# ----------------------------------------------------------

def explain_prediction_json(customer, prediction, probability):

    prompt = f"""
Return ONLY valid JSON.

The JSON must have exactly these fields:

prediction
confidence
explanation
recommendation

Prediction = {prediction}

Confidence = {probability:.2f}

Customer Information

{customer}

Do not include markdown.

Do not include code fences.

Only return JSON.
"""

    reply = call_llm(prompt)

    try:

        data = json.loads(reply)

        validate(
            instance=data,
            schema=response_schema
        )

        return data

    except (json.JSONDecodeError, ValidationError):

        print("Invalid JSON received from LLM.")

        return {

            "prediction": prediction,

            "confidence": probability,

            "explanation":
                "Fallback explanation generated because the LLM response was invalid.",

            "recommendation":
                "Review customer manually."

        }

# ----------------------------------------------------------
# Test
# ----------------------------------------------------------

print("\nTesting Structured Output\n")

customer = sample_data.iloc[0].to_dict()

prediction = "Not Likely to Churn"

confidence = 0.98

result = explain_prediction_json(

    customer,

    prediction,

    confidence

)

print(

    json.dumps(

        result,

        indent=4

    )

)

# TEMPERATURE COMPARISON
# ==========================================================

print("\n" + "=" * 70)
print("SECTION 5 - TEMPERATURE COMPARISON")
print("=" * 70)

comparison_prompt = """
Explain why customer tenure is one of the most important factors in predicting customer churn.

Keep the explanation under 100 words.
"""

# ----------------------------------------------------------
# Temperature = 0
# ----------------------------------------------------------

print("\nGenerating response with Temperature = 0")

response_temp0 = call_llm(
    comparison_prompt,
    temperature=0
)

print(response_temp0)

# ----------------------------------------------------------
# Temperature = 0.7
# ----------------------------------------------------------

print("\nGenerating response with Temperature = 0.7")

response_temp07 = call_llm(
    comparison_prompt,
    temperature=0.7
)

print(response_temp07)

# ----------------------------------------------------------
# Comparison Table
# ----------------------------------------------------------

comparison = pd.DataFrame({

    "Temperature":[
        0,
        0.7
    ],

    "Response":[
        response_temp0,
        response_temp07
    ]

})

print("\nTemperature Comparison")

print(comparison)

comparison.to_csv(

    "temperature_comparison.csv",

    index=False

)

# ----------------------------------------------------------
# Basic Evaluation
# ----------------------------------------------------------

print("\nEvaluation")

if response_temp0 == response_temp07:

    print("Responses are identical.")

else:

    print("Responses differ due to temperature variation.")

print("\nInterpretation")

print("""
Temperature 0 produces more deterministic and repeatable responses.

Temperature 0.7 allows the model to generate more creative and varied explanations.

For production systems that explain machine learning predictions,
Temperature 0 is generally preferred because it provides consistent outputs.
""")
