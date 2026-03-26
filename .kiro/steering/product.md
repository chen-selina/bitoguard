# BitoGuard Product Overview

BitoGuard is an AI-powered fraud detection and compliance risk monitoring system for cryptocurrency exchanges. The system analyzes user behavior patterns across multiple transaction types (TWD transfers, crypto transfers, USDT trading) to identify suspicious accounts and money laundering activities.

## Core Capabilities

- Multi-source feature engineering from user KYC data, transaction history, and behavioral patterns
- Graph-based relationship analysis to detect fraud networks and accomplice structures
- Ensemble machine learning models (XGBoost, LightGBM) optimized for F1 score on highly imbalanced datasets
- SHAP-based explainability for individual risk assessments
- Interactive Streamlit dashboard for risk monitoring and user diagnostics

## Key Risk Signals

The system focuses on detecting:
- Rapid KYC completion (< 1 day, indicating rushed account setup)
- Fast money flow patterns (deposit → withdrawal within hours)
- Late-night transaction anomalies (23:00-05:59)
- Round amount patterns (multiples of 10,000 TWD, suggesting structured layering)
- Low KYC level with high transaction volumes (mule account indicators)
- Shared IP addresses and wallet addresses across multiple accounts
- Graph-based accomplice detection through 2-hop network analysis

## Target Users

Compliance teams and risk analysts at cryptocurrency exchanges who need to identify fraudulent accounts before they cause financial damage.
