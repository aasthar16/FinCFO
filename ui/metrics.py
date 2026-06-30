"""
Metrics display UI components using Streamlit.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional


def render_metric_cards(metrics: Dict[str, Any], currency: str = "USD"):
    """
    Render metric cards in a grid layout with custom styling.
    """
    if not metrics:
        st.info("No metrics available")
        return
    
    # Define colors for different metrics
    colors = {
        'cash_balance': '#2ecc71',
        'net_burn': '#e74c3c',
        'runway': '#f39c12',
        'burn_multiple': '#3498db'
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 1.2rem; 
                    border-radius: 0.8rem; 
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 0.85rem; color: #666; margin-bottom: 0.3rem;">💰 Cash Balance</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {colors['cash_balance']};">
                {currency} {metrics.get('cash_balance', 0):,.0f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        net_burn = metrics.get('net_burn_3m_avg', 0)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 1.2rem; 
                    border-radius: 0.8rem; 
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 0.85rem; color: #666; margin-bottom: 0.3rem;">🔥 Net Burn</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {colors['net_burn']};">
                {currency} {net_burn:,.0f}
            </div>
            <div style="font-size: 0.8rem; color: #999;">3-month average</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        runway = metrics.get('cash_runway_months', 0)
        color = '#27ae60' if runway > 12 else '#f39c12' if runway > 6 else '#e74c3c'
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 1.2rem; 
                    border-radius: 0.8rem; 
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 0.85rem; color: #666; margin-bottom: 0.3rem;">📈 Runway</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {color};">
                {runway:.1f} months
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        burn_multiple = metrics.get('burn_multiple', 0)
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 1.2rem; 
                    border-radius: 0.8rem; 
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="font-size: 0.85rem; color: #666; margin-bottom: 0.3rem;">📊 Burn Multiple</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: {colors['burn_multiple']};">
                {burn_multiple:.1f}x
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_metric_table(metrics: Dict[str, Any], currency: str = "USD"):
    """
    Render detailed metrics as a table with color coding.
    """
    if not metrics:
        return
    
    data = {
        "Metric": [
            "💰 Gross Burn",
            "🔥 Net Burn",
            "📊 3-Month Avg Net Burn",
            "💳 Recurring Expenses",
            "🔴 One-Time Expenses",
            "💵 Monthly Revenue",
            "👥 Fully Loaded Ratio",
        ],
        "Value": [
            f"{currency} {metrics.get('gross_burn', 0):,.0f}",
            f"{currency} {metrics.get('net_burn', 0):,.0f}",
            f"{currency} {metrics.get('net_burn_3m_avg', 0):,.0f}",
            f"{currency} {metrics.get('recurring_expenses', 0):,.0f}",
            f"{currency} {metrics.get('one_time_expenses', 0):,.0f}",
            f"{currency} {metrics.get('monthly_revenue', 0):,.0f}",
            f"{metrics.get('fully_loaded_ratio', 0):.2f}x",
        ],
        "Status": [
            "✅" if metrics.get('gross_burn', 0) < 200000 else "⚠️",
            "✅" if metrics.get('net_burn', 0) < 150000 else "⚠️",
            "✅" if metrics.get('net_burn_3m_avg', 0) < 150000 else "⚠️",
            "✅" if metrics.get('recurring_expenses', 0) < 100000 else "⚠️",
            "✅" if metrics.get('one_time_expenses', 0) < 50000 else "⚠️",
            "✅" if metrics.get('monthly_revenue', 0) > 50000 else "⚠️",
            "✅" if 1.2 < metrics.get('fully_loaded_ratio', 0) < 1.4 else "⚠️",
        ]
    }
    
    df = pd.DataFrame(data)
    st.dataframe(
        df, 
        hide_index=True, 
        use_container_width=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric"),
            "Value": st.column_config.TextColumn("Value"),
            "Status": st.column_config.TextColumn("Status", width="small"),
        }
    )


def render_metric_trends(monthly_data: List[Dict[str, Any]], currency: str = "USD"):
    """
    Render metric trends over time using Plotly.
    """
    if not monthly_data:
        return
    
    df = pd.DataFrame(monthly_data)
    df['month'] = df['month'].astype(str)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['month'],
            y=df['amount'],
            mode='lines+markers',
            name='Burn',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=8, color='#e74c3c')
        ))
        fig.update_layout(
            title=f'Burn Trend ({currency})',
            xaxis_title='Month',
            yaxis_title=f'Amount ({currency})',
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Area chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['month'],
            y=df['amount'],
            mode='lines',
            name='Burn',
            fill='tozeroy',
            line=dict(color='#3498db', width=2),
            fillcolor='rgba(52, 152, 219, 0.3)'
        ))
        fig.update_layout(
            title=f'Cumulative Burn ({currency})',
            xaxis_title='Month',
            yaxis_title=f'Amount ({currency})',
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)


def render_gauge_metric(value: float, title: str, min_val: float = 0, max_val: float = 24, 
                        unit: str = "months"):
    """
    Render a gauge chart for a single metric.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#2ecc71" if value > 12 else "#f39c12" if value > 6 else "#e74c3c"},
            'steps': [
                {'range': [0, 6], 'color': "rgba(231, 76, 60, 0.2)"},
                {'range': [6, 12], 'color': "rgba(243, 156, 18, 0.2)"},
                {'range': [12, 24], 'color': "rgba(46, 204, 113, 0.2)"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': 6
            }
        }
    ))
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)