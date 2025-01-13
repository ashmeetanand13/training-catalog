import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from itertools import combinations
from tqdm import tqdm
import pandas as pd
from thefuzz import fuzz
from thefuzz import process

def get_data_info(file):
    """
    Read file and return all column names and unique player names.
    Works with both uploaded files and file paths.
    """
    df = pd.read_csv(file)
    all_columns = df.columns.tolist()
    player_names = df['PLAYER FIRST NAME'].unique().tolist()
    return df, all_columns, player_names

def get_drill_time(df):
    """Calculate drill duration in minutes"""
    df['start time'] = pd.to_datetime(df['DRILL START TIME'])
    df['end time'] = pd.to_datetime(df['DRILL END TIME'])
    return (df['end time'] - df['start time']).dt.total_seconds()/60

def get_drill_names(df):
    """Extract and process drill names with fuzzy matching and uppercase standardization"""
    # Convert drill titles to uppercase
    df['DRILL TITLE'] = df['DRILL TITLE'].str.upper()
    
    # Split and process drill titles
    drill_title = df['DRILL TITLE'].str.split(':', expand=True).iloc[:,:5]
    drill_title = drill_title[~drill_title[4].isna()]
    
    # Join the first 4 columns and get unique drill names
    drill_names = drill_title.iloc[:,:4].apply(lambda row: ':'.join(row.values.astype(str)), axis=1)
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
    # """Calculate metrics per minute for selected drills"""
    # Convert selected drills to uppercase for consistency
    selected_drills = [drill.upper() for drill in selected_drills]
    
    results = []
    
    for drill, time_multiplier in zip(selected_drills, drill_times):
        drill_data = df[df['Drill title 2'].str.upper() == drill]
        
        if drill_data.empty:
            continue
            
        for player in drill_data['PLAYER FIRST NAME'].unique():
            player_data = drill_data[drill_data['PLAYER FIRST NAME'] == player]
            player_metrics = {
                "PLAYER FIRST NAME": player,
            }
            
            for metric in metrics:
                if metric in player_data.columns:
                    per_minute = player_data[metric].mean() / player_data['time'].mean()
                    player_metrics[metric] = per_minute * time_multiplier
            
            results.append(player_metrics)
    
    if results:
        final_df = pd.DataFrame(results)
        return final_df.groupby(['PLAYER FIRST NAME']).sum()
    
    return pd.DataFrame()

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
    default_metrics = ["TOTAL DISTANCE", "HIGH SPEED RUNNING ABSOLUTE", "DISTANCE Z6 ABSOLUTE", 
                      "ACCELERATIONS", "DECELERATIONS"]
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
        df_filtered = df[df['PLAYER FIRST NAME'].isin(selected_players)]
        
        # Calculate drill times
        df_filtered['time'] = get_drill_time(df_filtered)
        
        # Process drill names
        df_filtered['Drill title 2'] = get_drill_names(df_filtered)
        
        # Select drills
        force_check = st.sidebar.checkbox("Show only drills with all selected players")
        
        if force_check:
            drills_with_all_players = (
                df_filtered.groupby('Drill title 2')['PLAYER FIRST NAME']
                .apply(lambda x: set(x) == set(selected_players))
                .reset_index()
            )
            valid_drills = drills_with_all_players[drills_with_all_players['PLAYER FIRST NAME'] == True]['Drill title 2']
            drill_names = st.sidebar.multiselect(
                "Select Drill", valid_drills, [], disabled=False
            )
        else:
            drills = get_drill_names(df_filtered)
            drill_names = st.sidebar.multiselect(
                "Select Drill", drills.unique(), [], disabled=False
            )
        
        # Set drill durations
        drill_times = []
        if drill_names:
            st.subheader("Set Duration for Each Drill (minutes)")
            cols = st.columns(min(2, len(drill_names)))
            for idx, drill in enumerate(drill_names):
                with cols[idx % 2]:
                    drill_time = st.slider(
                        f'{drill.split(":")[1]}',
                        2, 30, 2,
                        key=f"time_{idx}"
                    )
                drill_times.append(drill_time)
        
        # Calculate and display results
        if len(drill_names) > 0:
            final_data = get_target_metric(df_filtered, drill_names, selected_metrics, drill_times)
            
            if final_data.empty:
                st.write("No data found for selected drills")
            else:
                st.subheader("Analysis Results")
                st.table(final_data)
                
                # Create visualizations
                for metric in selected_metrics:
                    st.write(f"{metric} by Player")
                    chart = alt.Chart(
                        final_data.reset_index().sort_values(metric, ascending=False),
                        height=400,
                        width=800
                    ).mark_bar().encode(
                        x=alt.X('PLAYER FIRST NAME', sort=None),
                        y=metric
                    )
                    st.altair_chart(chart)
                
                # Scatter plot for HID vs VHID
                if 'HIGH SPEED RUNNING ABSOLUTE' in selected_metrics and 'DISTANCE Z6 ABSOLUTE' in selected_metrics:
                    final_data_reset = final_data.reset_index()
                    fig = px.scatter(
                        final_data_reset,
                        x='HIGH SPEED RUNNING ABSOLUTE',
                        y='DISTANCE Z6 ABSOLUTE',
                        color='PLAYER FIRST NAME',
                        title='HID vs VHID by Player'
                    )
                    
                    # Add average lines
                    fig.add_hline(
                        y=final_data['DISTANCE Z6 ABSOLUTE'].mean(),
                        line=dict(color='black', width=3)
                    )
                    fig.add_vline(
                        x=final_data['HIGH SPEED RUNNING ABSOLUTE'].mean(),
                        line=dict(color='black', width=3)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display statistics
                    st.write('HID vs VHID Statistics')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write('Average HID:', round(final_data['HIGH SPEED RUNNING ABSOLUTE'].mean()))
                        st.write('Max HID:', round(final_data['HIGH SPEED RUNNING ABSOLUTE'].max()))
                    with col2:
                        st.write('Average VHID:', round(final_data['DISTANCE Z6 ABSOLUTE'].mean()))
                        st.write('Max VHID:', round(final_data['DISTANCE Z6 ABSOLUTE'].max()))


                        # Scatter plot for Acceleration vs Deceleration
                if 'ACCELERATIONS' in selected_metrics and 'DECELERATIONS' in selected_metrics:
                    final_data_reset = final_data.reset_index()
                    fig2 = px.scatter(
                        final_data_reset,
                        x='ACCELERATIONS',
                        y='DECELERATIONS',
                        color='PLAYER FIRST NAME',
                        title='ACCELERATION vs DECELERATION by Player'
                    )
                    
                    # Add average lines
                    fig2.add_hline(
                        y=final_data['DECELERATIONS'].mean(),
                        line=dict(color='black', width=3)
                    )
                    fig2.add_vline(
                        x=final_data['ACCELERATIONS'].mean(),
                        line=dict(color='black', width=3)
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Display statistics
                    st.write('ACCELERATION vs DECELERATION Statistics')
                    col3, col4 = st.columns(2)
                    with col3:
                        st.write('Average ACCELERATION:', round(final_data['ACCELERATIONS'].mean()))
                        st.write('Max ACCELERATION:', round(final_data['ACCELERATIONS'].max()))
                    with col4:
                        st.write('Average DECELERATION:', round(final_data['DECELERATIONS'].mean()))
                        st.write('Max DECELERATION:', round(final_data['DECELERATIONS'].max()))
