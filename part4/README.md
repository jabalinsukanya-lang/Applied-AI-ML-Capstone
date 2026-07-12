#part 4 - LLM Powered Intelligent prediction Explanation System

##Overview: This project demonstrates how a traditiona machine learning model can be integrated with a large language model to buil an intelligent prediction explanation system.
The machine learning model predicts whether a customer is likely to churn. The LLM then explains the prediction in natural language, making the results understandable for business users who may not have a technical background. To improve reliability, the system includes production guardrails such as Personally Identifiable Information (PII) detection, JSON schema validation, structured outputs, and fallback responses.

# Objectives:
    -Load a previously trained machine learning model.
    -Generate churn predictions.
    -Validate all generated responses.
    -Prevent accidental disclosure of sensitive information.
    -Demonstrate production ready AI system design

# Machine Learning model: 
    The trained model from part 3 was reused in this.
    The model predicts whether a customer is likely to churn based on customer attributes such as Tenure, MonthlyCharges,Contract typer, Internet Service amd payment method.
    The Prediction probablity produced by the model is included in the prompt sent to the LLm.
    
