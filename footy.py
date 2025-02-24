import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from itertools import combinations
from tqdm import tqdm
from thefuzz import fuzz
from thefuzz import process

def get_data_info(file):
    """
    Read file and return all column names and unique player names.
    Works with both uploaded files and file paths.
    """
    df = pd.read_csv(file)
    df.columns = df.columns.str.upper()  # This line makes all columns uppercase
    all_columns = df.columns.tolist()
    player_names = df['PLAYER NAME'].unique().tolist()
    return df, all_columns, player_names

def get_drill_time(df):
    """Calculate drill duration in minutes"""
    # Find start time column - check both naming conventions
    start_col = next((col for col in df.columns if col in ['DRILL START TIME', 'START TIME','SPLIT START TIME']), None)
    end_col = next((col for col in df.columns if col in ['DRILL END TIME', 'END TIME','SPLIT END TIME']), None)
    
    if not start_col or not end_col:
        raise ValueError("Could not find start/end time columns")
    
    # Check the format of time values
    sample_start = str(df[start_col].iloc[0])
    
    # Handle Excel serial date numbers (like 45708.798)
    if sample_start.replace('.', '', 1).isdigit():
        # Convert Excel serial numbers to datetime
        df['start time'] = pd.TimedeltaIndex(df[start_col] % 1 * 86400, unit='s') + pd.Timestamp('1900-01-01')
        df['end time'] = pd.TimedeltaIndex(df[end_col] % 1 * 86400, unit='s') + pd.Timestamp('1900-01-01')
    else:
        # Regular datetime parsing for time strings (like '9:43:15')
        df['start time'] = pd.to_datetime(df[start_col])
        df['end time'] = pd.to_datetime(df[end_col])
    
    return (df['end time'] - df['start time']).dt.total_seconds() / 60

def get_drill_names(df):
    """Extract and process drill names with fuzzy matching and uppercase standardization"""
    # Find drill title column - check both naming conventions
    drill_col = next((col for col in df.columns if col.upper() in ['DRILL TITLE', 'DRILL NAME']), None)
    
    if not drill_col:
        raise ValueError("Could not find drill title column. Available columns: " + ", ".join(df.columns))
    
    # Convert drill titles to uppercase
    df[drill_col] = df[drill_col].str.upper()
    
    # Split and process drill titles
    drill_title = df[drill_col].str.split(':', expand=True)
    
    # Check how many columns we actually got after splitting
    num_cols = len(drill_title.columns)
    
    if num_cols < 5:
        # If we have fewer than 5 columns, just use all available columns
        drill_names = drill_title.iloc[:, :num_cols].apply(
            lambda row: ':'.join(row.dropna().values.astype(str)), axis=1
        )
    else:
        # Original logic for 5 or more columns
        drill_title = drill_title[~drill_title[4].isna()]
        drill_names = drill_title.iloc[:, :4].apply(
            lambda row: ':'.join(row.values.astype(str)), axis=1
        )
    
    unique_drills = list(set(drill_names))
    
    # Perform fuzzy matching to combine similar drill names
    final_drills = []
    processed_drills = set()
    
    for drill in unique_drills:
        if drill not in processed_drills:
            # Find all similar drills with 96% or higher similarity
            matches = process.extract(drill, unique_drills, scorer=fuzz.ratio, limit=None)
            similar_drills = [match[0] for match in matches if match[1] >= 96]
            
            # Add all similar drills to processed set
            processed_drills.update(similar_drills)
            
            # Use the first occurrence as the standardized name
            if similar_drills:
                final_drills.append(similar_drills[0])
    
    # Create a mapping dictionary for similar drills
    drill_mapping = {}
    for drill in unique_drills:
        best_match = process.extractOne(drill, final_drills, scorer=fuzz.ratio)
        drill_mapping[drill] = best_match[0]
    
    # Apply the mapping to standardize drill names
    return drill_names.map(drill_mapping)

def get_target_metric(df, selected_drills, metrics, drill_times):
    """Calculate metrics per minute for selected drills"""
    # Convert selected drills to uppercase for consistency
    selected_drills = [drill.upper() for drill in selected_drills]
    
    results = []
    
    for drill, time_multiplier in zip(selected_drills, drill_times):
        drill_data = df[df['Drill title 2'].str.upper() == drill]
        
        if drill_data.empty:
            continue
        
        for player in drill_data['PLAYER NAME'].unique():
            player_data = drill_data[drill_data['PLAYER NAME'] == player]
            player_metrics = {
                "PLAYER NAME": player,
            }
            
            for metric in metrics:
                if metric in player_data.columns:
                    per_minute = player_data[metric].mean() / player_data['time'].mean()
                    player_metrics[metric] = per_minute * time_multiplier
            
            results.append(player_metrics)
    
    if results:
        final_df = pd.DataFrame(results)
        return final_df.groupby(['PLAYER NAME']).sum()
    
    return pd.DataFrame()

def create_metric_scatter_plots(final_data, selected_metrics):
    """Create scatter plots for all combinations of selected metrics"""
    # Get all possible pairs of metrics
    metric_pairs = list(combinations(selected_metrics, 2))
    
    for metric1, metric2 in metric_pairs:
        final_data_reset = final_data.reset_index()
        fig = px.scatter(
            final_data_reset,
            x=metric1,
            y=metric2,
            color='PLAYER NAME',
            title=f'{metric1} vs {metric2} by Player'
        )
        
        # Add average lines
        fig.add_hline(
            y=final_data[metric2].mean(),
            line=dict(color='black', width=3)
        )
        fig.add_vline(
            x=final_data[metric1].mean(),
            line=dict(color='black', width=3)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display statistics
        st.write(f'{metric1} vs {metric2} Statistics')
        col1, col2 = st.columns(2)
        with col1:
            st.write(f'Average {metric1}:', round(final_data[metric1].mean()))
            st.write(f'Max {metric1}:', round(final_data[metric1].max()))
        with col2:
            st.write(f'Average {metric2}:', round(final_data[metric2].mean()))
            st.write(f'Max {metric2}:', round(final_data[metric2].max()))

# Streamlit UI
st.title('Soccer Pre-Training Analysis')

# File upload
uploaded_file = st.file_uploader("Upload GPS file", type=['gps', 'csv'])

if uploaded_file is not None:
    # Load data
    df, columns, player_names = get_data_info(uploaded_file)
    
    # Display columns
    st.subheader("Available Columns")
    st.write(columns)
    
    # Select metrics
    default_metrics = [
        "TOTAL DISTANCE",
        "HIGH SPEED RUNNING ABSOLUTE",
        "DISTANCE Z6 ABSOLUTE",
        "ACCELERATIONS",
        "DECELERATIONS"
    ]
    selected_metrics = st.multiselect(
        "Select metrics to analyze",
        columns,
        default=[metric for metric in default_metrics if metric in columns]
    )
    
    # Select players
    selected_players = st.multiselect(
        "Select players to analyze",
        player_names
    )
    
    if selected_players:
        # Filter data for selected players
        df_filtered = df[df['PLAYER NAME'].isin(selected_players)]
        
        # Calculate drill times
        df_filtered['time'] = get_drill_time(df_filtered)
        
        # Process drill names
        df_filtered['Drill title 2'] = get_drill_names(df_filtered)
        
        # Select drills
        force_check = st.sidebar.checkbox("Show only drills with all selected players")
        
        if force_check:
            drills_with_all_players = (
                df_filtered.groupby('Drill title 2')['PLAYER NAME']
                .apply(lambda x: set(x) == set(selected_players))
                .reset_index()
            )
            valid_drills = drills_with_all_players[
                drills_with_all_players['PLAYER NAME'] == True
            ]['Drill title 2']
            drill_names = st.sidebar.multiselect(
                "Select Drill",
                valid_drills,
                [],
                disabled=False
            )
        else:
            drills = get_drill_names(df_filtered)
            drill_names = st.sidebar.multiselect(
                "Select Drill",
                drills.unique(),
                [],
                disabled=False
            )
        
        # Set drill durations
        drill_times = []
        if drill_names:
            st.subheader("Set Duration for Each Drill (minutes)")
            cols = st.columns(min(2, len(drill_names)))
            for idx, drill in enumerate(drill_names):
                with cols[idx % 2]:
                    parts = drill.split(":")
                    drill_part = parts[1] if len(parts) > 1 else parts[0]
                    drill_time = st.slider(
                        label=f'{drill_part}',
                        min_value=2,
                        max_value=30,
                        value=2,
                        key=f"time_{idx}"
                    )
                    drill_times.append(drill_time)
            
            # Calculate and display results
            if len(drill_names) > 0:
                final_data = get_target_metric(
                    df_filtered,
                    drill_names,
                    selected_metrics,
                    drill_times
                )
                
                if final_data.empty:
                    st.write("No data found for selected drills")
                else:
                    st.subheader("Analysis Results")
                    st.table(final_data)
                    
                    # Create visualizations
                    for metric in selected_metrics:
                        st.write(f"{metric} by Player")
                        
                        # Prepare data with rankings
                        chart_data = final_data.reset_index().sort_values(metric, ascending=False)
                        
                        # Create a color column based on ranking
                        chart_data['color'] = 'middle'  # Default color
                        chart_data.loc[chart_data.head(3).index, 'color'] = 'top'  # Top 3
                        chart_data.loc[chart_data.tail(3).index, 'color'] = 'bottom'  # Bottom 3
                        
                        # Create the chart with colored bars
                        chart = alt.Chart(
                            chart_data,
                            height=400,
                            width=800
                        ).mark_bar().encode(
                            x=alt.X('PLAYER NAME', sort=None),
                            y=metric,
                            color=alt.Color(
                                'color:N',
                                scale=alt.Scale(
                                    domain=['top', 'middle', 'bottom'],
                                    range=['green', 'gray', 'red']
                                ),
                                legend=None
                            )
                        )
                        
                        # Add value labels on top of bars
                        text = chart.mark_text(
                            align='center',
                            baseline='bottom',
                            dy=-5  # Adjust this value to control label position above bars
                        ).encode(
                            text=alt.Text(metric, format='.1f')
                        )
                        
                        # Combine bar chart and labels
                        final_chart = (chart + text)
                        
                        st.altair_chart(final_chart)
                    
                    # Scatter plot for metrics
                    if len(selected_metrics) >= 2:
                        create_metric_scatter_plots(final_data, selected_metrics)
