"""
Interactive web dashboard for utility pole health monitoring and maintenance planning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import signalplot
import folium
from streamlit_folium import st_folium
import os
from datetime import datetime, timedelta
import logging

# Apply SignalPlot minimalist defaults
signalplot.apply()

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
            fig_health, ax_health = plt.subplots(figsize=(8, 5))
            ax_health.hist(st.session_state.filtered_df['overall_health_score'], 
                          bins=20, alpha=0.7, color='#1f77b4', edgecolor='black')
            mean_score = st.session_state.filtered_df['overall_health_score'].mean()
            ax_health.axvline(mean_score, color='red', linestyle='--', 
                            label=f'Mean: {mean_score:.1f}')
            ax_health.set_xlabel('Health Score')
            ax_health.set_ylabel('Frequency')
            ax_health.set_title('Health Score Distribution')
            ax_health.legend()
            ax_health.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_health)
            plt.close(fig_health)
        
        with col2:
            # Priority breakdown
            priority_counts = st.session_state.filtered_df['maintenance_priority'].value_counts()
            fig_priority, ax_priority = plt.subplots(figsize=(8, 5))
            colors_list = [self.colors.get(p, 'gray') for p in priority_counts.index]
            ax_priority.pie(priority_counts.values, labels=priority_counts.index, 
                          autopct='%1.1f%%', colors=colors_list, startangle=90)
            ax_priority.set_title('Maintenance Priority Breakdown')
            plt.tight_layout()
            st.pyplot(fig_priority)
            plt.close(fig_priority)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Pole type analysis
            type_health = st.session_state.filtered_df.groupby('pole_type')['overall_health_score'].agg(['mean', 'std']).reset_index()
            fig_type, ax_type = plt.subplots(figsize=(8, 5))
            bars = ax_type.bar(type_health['pole_type'], type_health['mean'], 
                              yerr=type_health['std'], capsize=5, alpha=0.7)
            # Color bars by mean health score
            for bar, mean_val in zip(bars, type_health['mean']):
                if mean_val < 50:
                    bar.set_color('#d32f2f')  # red
                elif mean_val < 70:
                    bar.set_color('#fbc02d')  # yellow
                else:
                    bar.set_color('#388e3c')  # green
            ax_type.set_xlabel('Pole Type')
            ax_type.set_ylabel('Average Health Score')
            ax_type.set_title('Average Health Score by Pole Type')
            ax_type.tick_params(axis='x', rotation=45)
            ax_type.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig_type)
            plt.close(fig_type)
        
        with col4:
            # Risk factors bar chart (replacing radar chart)
            risk_columns = ['moisture_risk', 'erosion_risk', 'chemical_corrosion_risk', 'bearing_capacity_risk']
            available_risks = [col for col in risk_columns if col in st.session_state.filtered_df.columns]
            
            if available_risks:
                avg_risks = st.session_state.filtered_df[available_risks].mean()
                fig_risk, ax_risk = plt.subplots(figsize=(8, 5))
                risk_labels = [col.replace('_risk', '').replace('_', ' ').title() for col in avg_risks.index]
                bars = ax_risk.barh(risk_labels, avg_risks.values, alpha=0.7)
                # Color by risk level
                for bar, val in zip(bars, avg_risks.values):
                    if val < 0.33:
                        bar.set_color('#388e3c')  # green
                    elif val < 0.66:
                        bar.set_color('#fbc02d')  # yellow
                    else:
                        bar.set_color('#d32f2f')  # red
                ax_risk.set_xlabel('Average Risk Score')
                ax_risk.set_title('Average Risk Factors')
                ax_risk.set_xlim(0, 1)
                ax_risk.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig_risk)
                plt.close(fig_risk)
    
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
                
                # Timeline chart - bar chart showing days until action
                fig_timeline, ax_timeline = plt.subplots(figsize=(10, max(6, len(schedule_filtered) * 0.3)))
                schedule_sorted = schedule_filtered.sort_values('recommended_action_date')
                y_pos = range(len(schedule_sorted))
                colors_list = [self.colors.get(p, 'gray') for p in schedule_sorted['maintenance_priority']]
                ax_timeline.barh(y_pos, schedule_sorted['days_until_action'] if 'days_until_action' in schedule_sorted.columns else [30]*len(schedule_sorted),
                                color=colors_list, alpha=0.7)
                ax_timeline.set_yticks(y_pos)
                ax_timeline.set_yticklabels(schedule_sorted['pole_id'])
                ax_timeline.set_xlabel('Days Until Action')
                ax_timeline.set_title('Maintenance Timeline')
                ax_timeline.grid(True, alpha=0.3, axis='x')
                plt.tight_layout()
                st.pyplot(fig_timeline)
                plt.close(fig_timeline)
            
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
                
                fig_cost, ax_cost = plt.subplots(figsize=(6, 5))
                colors_list = [self.colors.get(p, 'gray') for p in cost_by_priority.index]
                ax_cost.bar(cost_by_priority.index, cost_by_priority.values, color=colors_list, alpha=0.7)
                ax_cost.set_xlabel('Priority Level')
                ax_cost.set_ylabel('Total Cost ($)')
                ax_cost.set_title('Cost by Priority')
                ax_cost.tick_params(axis='x', rotation=45)
                ax_cost.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig_cost)
                plt.close(fig_cost)
            
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
                fig_trends, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig_trends.suptitle(f'Soil Conditions for Pole {selected_pole}', fontsize=16, fontweight='bold')
                
                if 'moisture_content' in pole_soil_data.columns:
                    axes[0, 0].plot(pole_soil_data['sample_date'], pole_soil_data['moisture_content'],
                                   marker='o', linestyle='-', label='Moisture')
                    axes[0, 0].set_title('Moisture Content')
                    axes[0, 0].set_ylabel('Moisture (m³/m³)')
                    axes[0, 0].grid(True, alpha=0.3)
                    axes[0, 0].tick_params(axis='x', rotation=45)
                
                if 'ph' in pole_soil_data.columns:
                    axes[0, 1].plot(pole_soil_data['sample_date'], pole_soil_data['ph'],
                                   marker='s', linestyle='-', label='pH', color='orange')
                    axes[0, 1].set_title('pH Level')
                    axes[0, 1].set_ylabel('pH')
                    axes[0, 1].grid(True, alpha=0.3)
                    axes[0, 1].tick_params(axis='x', rotation=45)
                
                if 'electrical_conductivity' in pole_soil_data.columns:
                    axes[1, 0].plot(pole_soil_data['sample_date'], pole_soil_data['electrical_conductivity'],
                                   marker='^', linestyle='-', label='EC', color='green')
                    axes[1, 0].set_title('Electrical Conductivity')
                    axes[1, 0].set_ylabel('EC (dS/m)')
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].tick_params(axis='x', rotation=45)
                
                if 'bearing_capacity' in pole_soil_data.columns:
                    axes[1, 1].plot(pole_soil_data['sample_date'], pole_soil_data['bearing_capacity'],
                                   marker='d', linestyle='-', label='Bearing Capacity', color='red')
                    axes[1, 1].set_title('Bearing Capacity')
                    axes[1, 1].set_ylabel('Bearing Capacity (kPa)')
                    axes[1, 1].grid(True, alpha=0.3)
                    axes[1, 1].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig_trends)
                plt.close(fig_trends)
        
        # Correlation analysis
        st.subheader("Risk Factor Correlations")
        
        numeric_columns = st.session_state.filtered_df.select_dtypes(include=[np.number]).columns
        correlation_matrix = st.session_state.filtered_df[numeric_columns].corr()
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax_corr)
        ax_corr.set_title('Risk Factor Correlation Matrix')
        plt.tight_layout()
        st.pyplot(fig_corr)
        plt.close(fig_corr)
        
        # Predictive insights
        st.subheader("Predictive Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Health score prediction based on age
            if 'age_years' in st.session_state.filtered_df.columns:
                fig_age_health, ax_age_health = plt.subplots(figsize=(8, 6))
                for priority in st.session_state.filtered_df['maintenance_priority'].unique():
                    data = st.session_state.filtered_df[
                        st.session_state.filtered_df['maintenance_priority'] == priority
                    ]
                    ax_age_health.scatter(data['age_years'], data['overall_health_score'],
                                        label=priority, color=self.colors.get(priority, 'gray'),
                                        alpha=0.6, s=50)
                # Add trend line
                z = np.polyfit(st.session_state.filtered_df['age_years'].dropna(),
                              st.session_state.filtered_df['overall_health_score'].dropna(), 1)
                p = np.poly1d(z)
                ax_age_health.plot(st.session_state.filtered_df['age_years'], 
                                 p(st.session_state.filtered_df['age_years']), 
                                 "r--", alpha=0.8, label='Trend')
                ax_age_health.set_xlabel('Age (years)')
                ax_age_health.set_ylabel('Health Score')
                ax_age_health.set_title('Health vs Age Relationship')
                ax_age_health.legend()
                ax_age_health.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_age_health)
                plt.close(fig_age_health)
        
        with col2:
            # Risk score distribution
            if 'composite_risk' in st.session_state.filtered_df.columns:
                fig_risk_dist, ax_risk_dist = plt.subplots(figsize=(8, 6))
                ax_risk_dist.hist(st.session_state.filtered_df['composite_risk'], 
                                 bins=20, alpha=0.7, color='orange', edgecolor='black')
                ax_risk_dist.set_xlabel('Composite Risk Score')
                ax_risk_dist.set_ylabel('Frequency')
                ax_risk_dist.set_title('Composite Risk Distribution')
                ax_risk_dist.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                st.pyplot(fig_risk_dist)
                plt.close(fig_risk_dist)
    
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
