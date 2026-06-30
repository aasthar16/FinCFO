"""
Dashboard UI component using Streamlit and Plotly.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional


def render_dashboard(state: Dict[str, Any], currency: str = "USD"):
    """
    Render the main dashboard with financial metrics and charts.
    """
    metrics = state.get("computed_metrics")
    if not metrics:
        st.info("💡 No financial data available yet. Ask a question to get started.")
        return
    
    # === METRIC CARDS ===
    st.markdown("### 📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Cash Balance",
            f"{currency} {metrics.get('cash_balance', 0):,.0f}",
            delta=None
        )
    
    with col2:
        net_burn = metrics.get('net_burn_3m_avg', 0)
        st.metric(
            "🔥 Net Burn",
            f"{currency} {net_burn:,.0f}",
            delta=f"{metrics.get('net_burn', 0) - net_burn:,.0f}" if metrics.get('net_burn') else None,
            delta_color="inverse"
        )
    
    with col3:
        runway = metrics.get('cash_runway_months', 0)
        st.metric(
            "📈 Runway",
            f"{runway:.1f} months",
            delta=None
        )
    
    with col4:
        st.metric(
            "📊 Burn Multiple",
            f"{metrics.get('burn_multiple', 0):.1f}x",
            delta=None
        )
    
    st.divider()
    
    # === CHARTS SECTION ===
    st.markdown("### 📈 Financial Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Burn Trend Chart
        if state.get("monthly_breakdown"):
            df = pd.DataFrame(state["monthly_breakdown"])
            df['month'] = df['month'].astype(str)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['month'],
                y=df['amount'],
                name='Monthly Burn',
                marker_color='#FF6B6B',
                text=df['amount'].apply(lambda x: f"{currency} {abs(x):,.0f}"),
                textposition='outside'
            ))
            fig.update_layout(
                title='Monthly Burn Trend',
                xaxis_title='Month',
                yaxis_title=f'Amount ({currency})',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            fig.update_traces(textfont_size=10)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Expense Category Breakdown
        if state.get("transactions_df") is not None:
            df = state["transactions_df"].copy()
            category_expenses = df.groupby('category')['amount'].sum().abs().sort_values(ascending=False)
            
            if len(category_expenses) > 0:
                colors = px.colors.qualitative.Set3[:len(category_expenses)]
                fig = go.Figure(data=[go.Pie(
                    labels=category_expenses.index,
                    values=category_expenses.values,
                    hole=0.4,
                    marker=dict(colors=colors),
                    textinfo='label+percent',
                    textposition='inside'
                )])
                fig.update_layout(
                    title='Expense Breakdown by Category',
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # === DETAILED METRICS ===
    with st.expander("📊 Detailed Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Gross Burn", f"{currency} {metrics.get('gross_burn', 0):,.0f}")
            st.metric("Recurring Expenses", f"{currency} {metrics.get('recurring_expenses', 0):,.0f}")
            st.metric("One-Time Expenses", f"{currency} {metrics.get('one_time_expenses', 0):,.0f}")
        
        with col2:
            st.metric("Monthly Revenue", f"{currency} {metrics.get('monthly_revenue', 0):,.0f}")
            st.metric("Fully Loaded Ratio", f"{metrics.get('fully_loaded_ratio', 0):.2f}x")
            st.metric("Net Burn (3mo avg)", f"{currency} {metrics.get('net_burn_3m_avg', 0):,.0f}")
        
        with col3:
            st.metric("Gross Burn (3mo avg)", f"{currency} {metrics.get('gross_burn_3m_avg', 0):,.0f}")
            st.metric("Burn Multiple", f"{metrics.get('burn_multiple', 0):.1f}x")
            st.metric("Cash Runway", f"{metrics.get('cash_runway_months', 0):.1f} months")


def render_forecast_dashboard(forecast_results: Dict[str, Any], currency: str = "USD"):
    """
    Render forecast dashboard with projections using Plotly.
    """
    if not forecast_results:
        return
    
    st.markdown("### 🔮 Forecast Dashboard")
    
    # === REVENUE FORECAST ===
    if "revenue" in forecast_results and forecast_results["revenue"].get("results"):
        revenue_results = forecast_results["revenue"]["results"]
        
        # Prepare data
        df = pd.DataFrame([{
            'date': r['date'],
            'forecast': r['yhat'],
            'lower': r['yhat_lower'],
            'upper': r['yhat_upper']
        } for r in revenue_results])
        
        fig = go.Figure()
        
        # Historical data if available
        if forecast_results["revenue"].get("historical"):
            hist_df = pd.DataFrame(forecast_results["revenue"]["historical"])
            fig.add_trace(go.Scatter(
                x=hist_df['ds'],
                y=hist_df['revenue'],
                name='Historical',
                mode='lines+markers',
                line=dict(color='#3498db', width=2),
                marker=dict(size=8)
            ))
        
        # Forecast line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['forecast'],
            name='Forecast',
            mode='lines',
            line=dict(color='#2ecc71', width=2)
        ))
        
        # Confidence interval (fill between upper and lower)
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['upper'],
            name='Upper Bound',
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['lower'],
            name='Confidence Interval (95%)',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(46, 204, 113, 0.2)',
            fill='tonexty',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f'Revenue Forecast ({currency})',
            xaxis_title='Date',
            yaxis_title=f'Revenue ({currency})',
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # === RUNWAY FORECAST ===
    if "runway_forecast" in forecast_results:
        runway = forecast_results["runway_forecast"]
        
        st.markdown("#### ✈️ Runway Forecast (Monte Carlo Simulation)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "🔴 P10 (Pessimistic)",
                f"{runway.get('p10_days', 0) // 30} months",
                f"By {runway.get('p10_date', '')[:10]}"
            )
        
        with col2:
            st.metric(
                "🟡 P50 (Expected)",
                f"{runway.get('p50_days', 0) // 30} months",
                f"By {runway.get('p50_date', '')[:10]}"
            )
        
        with col3:
            st.metric(
                "🟢 P90 (Optimistic)",
                f"{runway.get('p90_days', 0) // 30} months",
                f"By {runway.get('p90_date', '')[:10]}"
            )
        
        # Runway distribution chart
        st.markdown("#### 📊 Runway Probability Distribution")
        
        # Simulate distribution for visualization
        import numpy as np
        np.random.seed(42)
        months = [runway.get('p10_days', 180)/30, runway.get('p50_days', 365)/30, runway.get('p90_days', 540)/30]
        
        fig = go.Figure()
        fig.add_trace(go.Violin(
            y=[months[0], months[1], months[2]],
            box_visible=True,
            meanline_visible=True,
            name='Runway Distribution',
            marker_color='#3498db',
            line_color='#2980b9'
        ))
        fig.update_layout(
            title='Runway Distribution (Months)',
            xaxis_title='',
            yaxis_title='Months',
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)