"""
Interactive web dashboard for utility pole health monitoring and maintenance planning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Utility Pole Health Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-high {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .status-excellent { color: #4caf50; font-weight: bold; }
    .status-good { color: #8bc34a; font-weight: bold; }
    .status-fair { color: #ffc107; font-weight: bold; }
    .status-poor { color: #ff9800; font-weight: bold; }
    .status-critical { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


class PoleDashboard:
    """Main dashboard class for utility pole health monitoring."""
    
    def __init__(self):
        self.colors = {
            'critical': '#d32f2f',
            'high': '#f57c00', 
            'medium': '#fbc02d',
            'low': '#388e3c',
            'excellent': '#1976d2'
        }
    
    @st.cache_data
    def load_data(_self, assessment_file: str, schedule_file: str, soil_file: str = None):
        """Load assessment data with caching."""
        try:
            # Load assessment results
            assessment_df = pd.read_csv(assessment_file)
            
            # Load maintenance schedule
            schedule_df = pd.read_csv(schedule_file)
            
            # Load soil history if available
            soil_history = None
            if soil_file and os.path.exists(soil_file):
                soil_history = pd.read_csv(soil_file)
                soil_history['sample_date'] = pd.to_datetime(soil_history['sample_date'])
            
            return assessment_df, schedule_df, soil_history
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, None
    
    def render_header(self):
        """Render dashboard header."""
        st.markdown('<h1 class="main-header"> Utility Pole Health Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Poles", 
                value=len(st.session_state.assessment_df),
                delta=None
            )
        
        with col2:
            urgent_count = len(st.session_state.assessment_df[
                st.session_state.assessment_df['requires_immediate_attention'] == True
            ])
            st.metric(
                label="Urgent Attention", 
                value=urgent_count,
                delta=f"{urgent_count/len(st.session_state.assessment_df)*100:.1f}%" if len(st.session_state.assessment_df) > 0 else "0%"
            )
        
        with col3:
            avg_health = st.session_state.assessment_df['overall_health_score'].mean()
            st.metric(
                label="Average Health Score", 
                value=f"{avg_health:.1f}",
                delta=f"{'Good' if avg_health > 70 else 'Needs Attention'}"
            )
        
        with col4:
            total_cost = st.session_state.schedule_df['estimated_cost'].sum()
            st.metric(
                label="Total Maintenance Cost", 
                value=f"${total_cost:,.0f}",
                delta=None
            )
    
    def render_sidebar(self):
        """Render sidebar controls."""
        st.sidebar.title(" Dashboard Controls")
        
        # Data filters
        st.sidebar.header("Filters")
        
        # Pole type filter
        pole_types = ['All'] + list(st.session_state.assessment_df['pole_type'].unique())
        selected_pole_types = st.sidebar.multiselect(
            "Pole Types", 
            pole_types, 
            default=['All']
        )
        
        # Priority filter
        priorities = ['All'] + list(st.session_state.assessment_df['maintenance_priority'].unique())
        selected_priorities = st.sidebar.multiselect(
            "Maintenance Priorities", 
            priorities, 
            default=['All']
        )
        
        # Health score range
        health_range = st.sidebar.slider(
            "Health Score Range", 
            0, 100, (0, 100)
        )
        
        # Apply filters
        filtered_df = st.session_state.assessment_df.copy()
        
        if 'All' not in selected_pole_types:
            filtered_df = filtered_df[filtered_df['pole_type'].isin(selected_pole_types)]
        
        if 'All' not in selected_priorities:
            filtered_df = filtered_df[filtered_df['maintenance_priority'].isin(selected_priorities)]
        
        filtered_df = filtered_df[
            (filtered_df['overall_health_score'] >= health_range[0]) &
            (filtered_df['overall_health_score'] <= health_range[1])
        ]
        
        st.session_state.filtered_df = filtered_df
        
        # Data refresh
        st.sidebar.header("Data Management")
        if st.sidebar.button(" Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Export options
        st.sidebar.header("Export Options")
        if st.sidebar.button(" Generate Report"):
            self.generate_pdf_report()
        
        if st.sidebar.button(" Export Data"):
            self.export_data()
    
    def render_overview_tab(self):
        """Render overview tab content."""
        st.header(" Fleet Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Health score distribution
            fig_health = px.histogram(
                st.session_state.filtered_df, 
                x='overall_health_score',
                nbins=20,
                title="Health Score Distribution",
                color_discrete_sequence=['#1f77b4']
            )
            fig_health.add_vline(
                x=st.session_state.filtered_df['overall_health_score'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {st.session_state.filtered_df['overall_health_score'].mean():.1f}"
            )
            st.plotly_chart(fig_health, use_container_width=True)
        
        with col2:
            # Priority breakdown
            priority_counts = st.session_state.filtered_df['maintenance_priority'].value_counts()
            fig_priority = px.pie(
                values=priority_counts.values,
                names=priority_counts.index,
                title="Maintenance Priority Breakdown",
                color_discrete_map=self.colors
            )
            st.plotly_chart(fig_priority, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Pole type analysis
            type_health = st.session_state.filtered_df.groupby('pole_type')['overall_health_score'].agg(['mean', 'std']).reset_index()
            fig_type = px.bar(
                type_health, 
                x='pole_type', 
                y='mean',
                error_y='std',
                title="Average Health Score by Pole Type",
                color='mean',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_type, use_container_width=True)
        
        with col4:
            # Risk factors radar chart
            risk_columns = ['moisture_risk', 'erosion_risk', 'chemical_corrosion_risk', 'bearing_capacity_risk']
            available_risks = [col for col in risk_columns if col in st.session_state.filtered_df.columns]
            
            if available_risks:
                avg_risks = st.session_state.filtered_df[available_risks].mean()
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=avg_risks.values,
                    theta=[col.replace('_risk', '').replace('_', ' ').title() for col in avg_risks.index],
                    fill='toself',
                    name='Average Risk'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    title="Average Risk Factors"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
    
    def render_map_tab(self):
        """Render interactive map tab."""
        st.header(" Pole Location Map")
        
        if 'latitude' in st.session_state.filtered_df.columns and 'longitude' in st.session_state.filtered_df.columns:
            # Calculate map center
            center_lat = st.session_state.filtered_df['latitude'].mean()
            center_lon = st.session_state.filtered_df['longitude'].mean()
            
            # Create folium map
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            
            # Add markers
            for idx, row in st.session_state.filtered_df.iterrows():
                # Color by priority
                priority = row.get('maintenance_priority', 'unknown')
                color = {
                    'critical': 'red',
                    'high': 'orange', 
                    'medium': 'yellow',
                    'low': 'green',
                    'unknown': 'gray'
                }.get(priority, 'gray')
                
                # Create popup
                popup_content = f"""
                <b>Pole ID:</b> {row.get('pole_id', 'Unknown')}<br>
                <b>Health Score:</b> {row.get('overall_health_score', 'N/A'):.1f}/100<br>
                <b>Priority:</b> {priority.title()}<br>
                <b>Pole Type:</b> {row.get('pole_type', 'Unknown')}<br>
                <b>Age:</b> {row.get('age_years', 'Unknown'):.1f} years<br>
                <b>Safety Risk:</b> {row.get('safety_risk', 0):.3f}<br>
                <b>Estimated Cost:</b> ${row.get('estimated_cost', 0):,.0f}
                """
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=8,
                    popup=folium.Popup(popup_content, max_width=300),
                    color='black',
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(m)
            
            # Display map
            map_data = st_folium(m, width=1200, height=600)
            
            # Show selected pole details
            if map_data['last_object_clicked_popup']:
                st.subheader("Selected Pole Details")
                # Parse popup content to show pole details
                st.info("Click on a pole marker to see detailed information")
        else:
            st.warning("Geographic coordinates not available for mapping")
    
    def render_maintenance_tab(self):
        """Render maintenance planning tab."""
        st.header(" Maintenance Planning")
        
        # Maintenance timeline
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Maintenance Schedule")
            
            # Filter schedule data
            schedule_filtered = st.session_state.schedule_df[
                st.session_state.schedule_df['pole_id'].isin(st.session_state.filtered_df['pole_id'])
            ]
            
            # Convert dates
            if 'recommended_action_date' in schedule_filtered.columns:
                schedule_filtered['recommended_action_date'] = pd.to_datetime(schedule_filtered['recommended_action_date'])
                
                # Timeline chart
                fig_timeline = px.timeline(
                    schedule_filtered,
                    x_start='recommended_action_date',
                    x_end='recommended_action_date',
                    y='pole_id',
                    color='maintenance_priority',
                    title="Maintenance Timeline",
                    color_discrete_map=self.colors
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Maintenance table
            st.subheader("Detailed Schedule")
            display_columns = ['pole_id', 'maintenance_priority', 'health_score', 
                             'recommended_actions', 'estimated_cost', 'days_until_action']
            available_columns = [col for col in display_columns if col in schedule_filtered.columns]
            
            if available_columns:
                st.dataframe(
                    schedule_filtered[available_columns].sort_values('days_until_action'),
                    use_container_width=True
                )
        
        with col2:
            st.subheader("Cost Analysis")
            
            # Cost by priority
            if 'estimated_cost' in schedule_filtered.columns and 'maintenance_priority' in schedule_filtered.columns:
                cost_by_priority = schedule_filtered.groupby('maintenance_priority')['estimated_cost'].sum()
                
                fig_cost = px.bar(
                    x=cost_by_priority.index,
                    y=cost_by_priority.values,
                    title="Cost by Priority",
                    color=cost_by_priority.index,
                    color_discrete_map=self.colors
                )
                st.plotly_chart(fig_cost, use_container_width=True)
            
            # Budget analysis
            st.subheader("Budget Planning")
            total_cost = schedule_filtered['estimated_cost'].sum() if 'estimated_cost' in schedule_filtered.columns else 0
            
            budget = st.number_input("Available Budget ($)", value=10000, step=1000)
            
            if total_cost > 0:
                budget_utilization = min(budget / total_cost * 100, 100)
                st.metric(
                    "Budget Utilization",
                    f"{budget_utilization:.1f}%",
                    f"${budget - total_cost:,.0f} {'surplus' if budget > total_cost else 'deficit'}"
                )
                
                # Affordable poles
                schedule_sorted = schedule_filtered.sort_values('maintenance_priority')
                cumulative_cost = 0
                affordable_count = 0
                
                for _, row in schedule_sorted.iterrows():
                    cost = row.get('estimated_cost', 0)
                    if cumulative_cost + cost <= budget:
                        cumulative_cost += cost
                        affordable_count += 1
                    else:
                        break
                
                st.info(f"With current budget, you can maintain {affordable_count} out of {len(schedule_filtered)} poles")
    
    def render_analytics_tab(self):
        """Render advanced analytics tab."""
        st.header(" Advanced Analytics")
        
        # Time series analysis if soil history is available
        if st.session_state.soil_history is not None:
            st.subheader("Soil Condition Trends")
            
            # Select pole for analysis
            pole_ids = st.session_state.soil_history['pole_id'].unique()
            selected_pole = st.selectbox("Select Pole for Trend Analysis", pole_ids)
            
            if selected_pole:
                pole_soil_data = st.session_state.soil_history[
                    st.session_state.soil_history['pole_id'] == selected_pole
                ]
                
                # Multi-parameter trend chart
                fig_trends = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=['Moisture Content', 'pH Level', 'Electrical Conductivity', 'Bearing Capacity'],
                    vertical_spacing=0.1
                )
                
                if 'moisture_content' in pole_soil_data.columns:
                    fig_trends.add_trace(
                        go.Scatter(x=pole_soil_data['sample_date'], y=pole_soil_data['moisture_content'],
                                 mode='lines+markers', name='Moisture'),
                        row=1, col=1
                    )
                
                if 'ph' in pole_soil_data.columns:
                    fig_trends.add_trace(
                        go.Scatter(x=pole_soil_data['sample_date'], y=pole_soil_data['ph'],
                                 mode='lines+markers', name='pH'),
                        row=1, col=2
                    )
                
                if 'electrical_conductivity' in pole_soil_data.columns:
                    fig_trends.add_trace(
                        go.Scatter(x=pole_soil_data['sample_date'], y=pole_soil_data['electrical_conductivity'],
                                 mode='lines+markers', name='EC'),
                        row=2, col=1
                    )
                
                if 'bearing_capacity' in pole_soil_data.columns:
                    fig_trends.add_trace(
                        go.Scatter(x=pole_soil_data['sample_date'], y=pole_soil_data['bearing_capacity'],
                                 mode='lines+markers', name='Bearing Capacity'),
                        row=2, col=2
                    )
                
                fig_trends.update_layout(height=600, title_text=f"Soil Conditions for Pole {selected_pole}")
                st.plotly_chart(fig_trends, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Risk Factor Correlations")
        
        numeric_columns = st.session_state.filtered_df.select_dtypes(include=[np.number]).columns
        correlation_matrix = st.session_state.filtered_df[numeric_columns].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            title="Risk Factor Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Predictive insights
        st.subheader("Predictive Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Health score prediction based on age
            if 'age_years' in st.session_state.filtered_df.columns:
                fig_age_health = px.scatter(
                    st.session_state.filtered_df,
                    x='age_years',
                    y='overall_health_score',
                    color='maintenance_priority',
                    title="Health vs Age Relationship",
                    trendline="ols",
                    color_discrete_map=self.colors
                )
                st.plotly_chart(fig_age_health, use_container_width=True)
        
        with col2:
            # Risk score distribution
            if 'composite_risk' in st.session_state.filtered_df.columns:
                fig_risk_dist = px.histogram(
                    st.session_state.filtered_df,
                    x='composite_risk',
                    title="Composite Risk Distribution",
                    nbins=20
                )
                st.plotly_chart(fig_risk_dist, use_container_width=True)
    
    def generate_pdf_report(self):
        """Generate PDF report (placeholder)."""
        st.success("Report generation feature coming soon!")
    
    def export_data(self):
        """Export filtered data."""
        csv = st.session_state.filtered_df.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name=f"pole_assessment_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def main():
    """Main dashboard application."""
    dashboard = PoleDashboard()
    
    # Initialize session state
    if 'assessment_df' not in st.session_state:
        # Try to load default data
        assessment_file = 'Output/pole_health_assessment.csv'
        schedule_file = 'Output/maintenance_schedule.csv'
        soil_file = 'Input/sample_soil_data.csv'
        
        if os.path.exists(assessment_file) and os.path.exists(schedule_file):
            assessment_df, schedule_df, soil_history = dashboard.load_data(
                assessment_file, schedule_file, soil_file
            )
            
            if assessment_df is not None:
                st.session_state.assessment_df = assessment_df
                st.session_state.schedule_df = schedule_df
                st.session_state.soil_history = soil_history
                st.session_state.filtered_df = assessment_df.copy()
            else:
                st.error("Failed to load data. Please run the assessment first.")
                st.stop()
        else:
            st.error("""
            No assessment data found. Please run the pole assessment first:
            
            ```bash
            python main.py --create-sample-data
            python main.py --poles Input/sample_poles.csv --soil Input/sample_soil_data.csv
            ```
            """)
            st.stop()
    
    # Render dashboard
    dashboard.render_header()
    dashboard.render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([" Overview", " Map", " Maintenance", " Analytics"])
    
    with tab1:
        dashboard.render_overview_tab()
    
    with tab2:
        dashboard.render_map_tab()
    
    with tab3:
        dashboard.render_maintenance_tab()
    
    with tab4:
        dashboard.render_analytics_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Utility Pole Health Dashboard** | Built with Streamlit | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
