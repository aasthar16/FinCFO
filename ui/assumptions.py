"""
Assumptions ledger UI component using Streamlit.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any


def render_assumptions(assumptions: List[Dict[str, Any]]):
    """
    Render the assumptions ledger with expandable entries.
    """
    if not assumptions:
        st.info("📋 No assumptions logged yet.")
        return
    
    st.markdown("### 📋 Assumptions Ledger")
    st.caption("All assumptions made during financial analysis")
    
    # Group by source
    grouped = {}
    for a in assumptions:
        source = a.get('source', 'unknown')
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(a)
    
    # Display by source
    for source, items in grouped.items():
        with st.expander(f"📌 {source.upper()} ({len(items)} assumptions)", expanded=False):
            for idx, assumption in enumerate(items):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{assumption.get('parameter', 'N/A')}**")
                    st.caption(assumption.get('rationale', ''))
                
                with col2:
                    st.markdown(f"Value: **{assumption.get('value', 'N/A')}**")
                
                with col3:
                    confidence = assumption.get('confidence', 0)
                    if confidence:
                        color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                        st.markdown(f"Confidence: **<span style='color:{color}'>{confidence*100:.0f}%</span>**", 
                                   unsafe_allow_html=True)
                    else:
                        st.caption("No confidence")
                
                if idx < len(items) - 1:
                    st.divider()


def render_assumptions_dashboard(assumptions: List[Dict[str, Any]]):
    """
    Render a complete assumptions dashboard with charts.
    """
    if not assumptions:
        st.info("📋 No assumptions logged yet. Start analyzing finances to generate assumptions.")
        return
    
    st.markdown("### 📊 Assumptions Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Assumptions by source
        source_counts = {}
        for a in assumptions:
            source = a.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        if source_counts:
            fig = go.Figure(data=[go.Bar(
                x=list(source_counts.keys()),
                y=list(source_counts.values()),
                marker_color='#3498db',
                text=list(source_counts.values()),
                textposition='outside'
            )])
            fig.update_layout(
                title='Assumptions by Source',
                xaxis_title='Source',
                yaxis_title='Count',
                height=250,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence distribution
        confidences = [a.get('confidence', 0) for a in assumptions if a.get('confidence') is not None]
        if confidences:
            fig = go.Figure(data=[go.Histogram(
                x=[c * 100 for c in confidences],
                nbinsx=10,
                marker_color='#2ecc71',
                name='Confidence'
            )])
            fig.update_layout(
                title='Confidence Distribution',
                xaxis_title='Confidence (%)',
                yaxis_title='Count',
                height=250,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Full ledger
    st.divider()
    render_assumptions(assumptions)


def render_assumption_card(assumption: Dict[str, Any]):
    """
    Render a single assumption as a card.
    """
    confidence = assumption.get('confidence', 0)
    color = "#27ae60" if confidence > 0.7 else "#f39c12" if confidence > 0.4 else "#e74c3c"
    
    st.markdown(f"""
    <div style="border: 1px solid #e0e0e0; 
                border-radius: 0.5rem; 
                padding: 0.8rem; 
                margin: 0.5rem 0;
                background: #fafafa;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <strong>{assumption.get('parameter', 'N/A')}</strong>
                <span style="color: #666; margin-left: 0.5rem;">= {assumption.get('value', 'N/A')}</span>
            </div>
            <div style="color: {color};">
                {confidence*100:.0f}% confidence
            </div>
        </div>
        <div style="font-size: 0.85rem; color: #666; margin-top: 0.3rem;">
            {assumption.get('rationale', '')}
        </div>
        <div style="font-size: 0.75rem; color: #999; margin-top: 0.3rem;">
            Source: {assumption.get('source', 'unknown')}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_assumptions_timeline(assumptions: List[Dict[str, Any]]):
    """
    Render assumptions over time.
    """
    if not assumptions:
        return
    
    # Create timeline data
    timeline_data = []
    for a in assumptions:
        if 'timestamp' in a:
            timeline_data.append({
                'timestamp': pd.to_datetime(a['timestamp']),
                'parameter': a.get('parameter', 'N/A'),
                'value': a.get('value', 'N/A'),
                'source': a.get('source', 'unknown')
            })
    
    if not timeline_data:
        return
    
    df = pd.DataFrame(timeline_data)
    df = df.sort_values('timestamp')
    
    fig = go.Figure()
    
    for source in df['source'].unique():
        source_df = df[df['source'] == source]
        fig.add_trace(go.Scatter(
            x=source_df['timestamp'],
            y=[1] * len(source_df),
            mode='markers',
            name=source,
            marker=dict(size=15, symbol='circle'),
            text=source_df['parameter'] + ' = ' + source_df['value'],
            hoverinfo='text'
        ))
    
    fig.update_layout(
        title='Assumptions Timeline',
        xaxis_title='Time',
        yaxis_title='',
        height=200,
        showlegend=True,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_yaxis(showticklabels=False)
    
    st.plotly_chart(fig, use_container_width=True)