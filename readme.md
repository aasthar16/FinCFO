# AI CFO: Autonomous Financial Intelligence Platform

An AI-powered CFO for startups that continuously tracks finances, predicts runway, simulates business decisions, and provides actionable recommendations using a state-driven hub-and-spoke LangGraph architecture.

## 🚀 Features

- **Hub-and-Spoke Architecture**: Multi-agent orchestration with cyclic reasoning
- **Real Financial Logic**: Gross/net burn, deferred revenue, burn multiple, Rule of 40
- **Production Checkpointing**: PostgreSQL persistence with LangGraph checkpoints
- **Time Series Forecasting**: Prophet-based forecasting with confidence intervals
- **Monte Carlo Simulation**: P10/P50/P90 runway projections
- **Full Observability**: LangSmith tracing with context propagation
- **Interactive UI**: Streamlit dashboard with real-time metrics
- **Assumptions Ledger**: Full audit trail of all financial assumptions

## 🏗️ Architecture
┌─────────────────────────────────────────────────────────────────┐
│ Streamlit UI │
└─────────────────────────┬───────────────────────────────────────┘
│
┌─────────────────────────▼───────────────────────────────────────┐
│ FastAPI Gateway │
└─────────────────────────┬───────────────────────────────────────┘
│
┌─────────────────────────▼───────────────────────────────────────┐
│ LangGraph (Hub-and-Spoke) │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Supervisor Agent │ │
│ └───────────────────────┬─────────────────────────────────┘ │
│ │ │
│ ┌─────────────────────┼─────────────────────┐ │
│ ▼ ▼ ▼ │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ │ Scenario │ │ Burn │ │ Forecast │ │
│ │ Simulator│ ◄──► │ Agent │ ◄──► │ Agent │ │
│ └──────────┘ └──────────┘ └──────────┘ │
│ │ │
│ ┌─────────────────────▼─────┐ │
│ │ Recommendation Agent │ │
│ └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
│
┌─────────────────────────▼───────────────────────────────────────┐
│ PostgreSQL │
│ (Checkpoints + Data) │
└─────────────────────────────────────────────────────────────────┘


## 📋 Prerequisites

- Python 3.10+
- PostgreSQL 14+
- OpenAI API Key (for LLM features)
- LangSmith API Key (optional, for tracing)