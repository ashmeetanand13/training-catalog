import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from itertools import combinations
from thefuzz import fuzz
from thefuzz import process

def get_data_info(file):
    """Read file and return all column names and unique player names."""
    
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    df = None
    for encoding in encodings_to_try:
        try:
            if hasattr(file, 'seek'):
                file.seek(0)
            df = pd.read_csv(file, encoding=encoding)
            st.success(f"✅ Successfully read file with {encoding} encoding")
            break
        except:
            continue
    
    if df is None:
        st.error("❌ Failed to read the CSV file")
        return None, [], []
    
    # Clean column names
    df.columns = df.columns.astype(str).str.strip().str.upper()
    df.columns = df.columns.str.replace(r'[^\w\s]', '_', regex=True)
    all_columns = df.columns.tolist()
    
    # Find player column
    player_cols = ['PLAYER_NAME', 'PLAYER', 'NAME', 'PLAYERNAME', 'FULL_NAME', 'ATHLETE_NAME']
    player_col = None
    
    for col in player_cols:
        if col in df.columns:
            player_col = col
            break
    
    # Try partial matches
    if player_col is None:
        for col in all_columns:
            if 'PLAYER' in col or 'NAME' in col:
                player_col = col
                break
    
    if player_col is None:
        st.error(f"Could not find player column. Available: {all_columns}")
        return df, all_columns, []
    
    # Standardize player column
    if player_col != 'PLAYER_NAME':
        df = df.rename(columns={player_col: 'PLAYER_NAME'})
    
    # Clean and get player names
    df['PLAYER_NAME'] = df['PLAYER_NAME'].astype(str).str.strip()
    player_names = df['PLAYER_NAME'].unique().tolist()
    player_names = [name for name in player_names if name and name != 'nan' and name.strip()]
    
    return df, all_columns, player_names

def parse_drill_name_format(drill_text):
    """Parse drill name formats - semicolon and colon"""
    drill_text = str(drill_text).strip()
    
    # Handle semicolon format: T=WU;V=Prepractice;N=24
    if ';' in drill_text and '=' in drill_text:
        pairs = {}
        for pair in drill_text.split(';'):
            if '=' in pair:
                key, value = pair.split('=', 1)
                pairs[key.strip()] = value.strip()
        
        drill_parts = []
        if 'T' in pairs and pairs['T']:
            drill_parts.append(f"Type:{pairs['T']}")
        if 'V' in pairs and pairs['V']:
            drill_parts.append(f"Variation:{pairs['V']}")
        if 'N' in pairs and pairs['N']:
            drill_parts.append(f"N:{pairs['N']}")
        
        return ':'.join(drill_parts) if drill_parts else drill_text
    
    # Handle colon format: Drill:Type:Phase
    elif ':' in drill_text:
        parts = drill_text.split(':')
        return ':'.join(parts[:4])
    
    return drill_text

def get_drill_time(df):
    """Calculate drill duration in minutes"""
    
    start_patterns = ['START_TIME', 'DRILL_START_TIME', 'SPLIT_START_TIME', 'PERIOD_START_TIME','Start Time']
    end_patterns = ['END_TIME', 'DRILL_END_TIME', 'SPLIT_END_TIME', 'PERIOD_END_TIME','End Time']
    
    start_col = None
    end_col = None
    
    for pattern in start_patterns:
        if pattern in df.columns:
            start_col = pattern
            break
    
    for pattern in end_patterns:
        if pattern in df.columns:
            end_col = pattern
            break
    
    if not start_col or not end_col:
        st.error("Could not find start/end time columns")
        st.write("Available columns:", df.columns.tolist())
        raise ValueError("Could not find start/end time columns")
    
    try:
        sample_start = str(df[start_col].iloc[0])
        
        if sample_start.replace('.', '', 1).isdigit():
            # Excel serial numbers
            df['start_time'] = pd.TimedeltaIndex(df[start_col] % 1 * 86400, unit='s') + pd.Timestamp('1900-01-01')
            df['end_time'] = pd.TimedeltaIndex(df[end_col] % 1 * 86400, unit='s') + pd.Timestamp('1900-01-01')
        else:
            # Regular datetime
            df['start_time'] = pd.to_datetime(df[start_col])
            df['end_time'] = pd.to_datetime(df[end_col])
        
        return (df['end_time'] - df['start_time']).dt.total_seconds() / 60
    
    except Exception as e:
        st.error(f"Error processing time columns: {str(e)}")
        raise

def get_drill_names(df):
    """Extract and process drill names"""
    
    drill_patterns = ['DRILL_TITLE', 'DRILL_NAME', 'SPLIT_NAME', 'PERIOD_NAME']
    drill_col = None
    
    for pattern in drill_patterns:
        if pattern in df.columns:
            drill_col = pattern
            break
    
    if not drill_col:
        st.error("Could not find drill column")
        st.write("Available columns:", df.columns.tolist())
        raise ValueError("Could not find drill column")
    
    # Clean and process drill names
    df[drill_col] = df[drill_col].astype(str).str.upper().str.strip()
    drill_names = df[drill_col].apply(parse_drill_name_format)
    
    # Fuzzy matching to group similar drills
    unique_drills = list(set(drill_names))
    final_drills = []
    processed_drills = set()
    
    for drill in unique_drills:
        if drill not in processed_drills:
            matches = process.extract(drill, unique_drills, scorer=fuzz.ratio, limit=None)
            similar_drills = [match[0] for match in matches if match[1] >= 90]
            processed_drills.update(similar_drills)
            if similar_drills:
                final_drills.append(similar_drills[0])
    
    # Create mapping
    drill_mapping = {}
    for drill in unique_drills:
        best_match = process.extractOne(drill, final_drills, scorer=fuzz.ratio)
        drill_mapping[drill] = best_match[0]
    
    return drill_names.map(drill_mapping)

def get_target_metric(df, selected_drills, metrics, drill_times):
    """Calculate metrics per minute for selected drills"""
    
    selected_drills = [drill.upper() for drill in selected_drills]
    results = []
    
    for drill, time_multiplier in zip(selected_drills, drill_times):
        drill_data = df[df['Drill_title_2'].str.upper() == drill]
        
        if drill_data.empty:
            continue
        
        for player in drill_data['PLAYER_NAME'].unique():
            player_data = drill_data[drill_data['PLAYER_NAME'] == player]
            player_metrics = {"PLAYER_NAME": player}
            
            for metric in metrics:
                if metric in player_data.columns:
                    metric_values = player_data[metric].dropna()
                    time_values = player_data['time'].dropna()
                    
                    if len(metric_values) > 0 and len(time_values) > 0:
                        per_minute = metric_values.mean() / time_values.mean()
                        player_metrics[metric] = per_minute * time_multiplier
                    else:
                        player_metrics[metric] = 0
            
            results.append(player_metrics)
    
    if results:
        final_df = pd.DataFrame(results)
        return final_df.groupby(['PLAYER_NAME']).sum()
    
    return pd.DataFrame()

def create_metric_scatter_plots(final_data, selected_metrics):
    """Create scatter plots for all combinations of selected metrics"""
    
    metric_pairs = list(combinations(selected_metrics, 2))
    
    for metric1, metric2 in metric_pairs:
        final_data_reset = final_data.reset_index()
        fig = px.scatter(
            final_data_reset,
            x=metric1,
            y=metric2,
            color='PLAYER_NAME',
            title=f'{metric1} vs {metric2} by Player'
        )
        
        fig.add_hline(y=final_data[metric2].mean(), line=dict(color='black', width=3))
        fig.add_vline(x=final_data[metric1].mean(), line=dict(color='black', width=3))
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f'Average {metric1}:', round(final_data[metric1].mean(), 2))
            st.write(f'Max {metric1}:', round(final_data[metric1].max(), 2))
        with col2:
            st.write(f'Average {metric2}:', round(final_data[metric2].mean(), 2))
            st.write(f'Max {metric2}:', round(final_data[metric2].max(), 2))

# Streamlit UI
st.title('Soccer Pre-Training Analysis')
st.write("Upload your GPS data file (CSV format) to analyze player performance metrics.")

uploaded_file = st.file_uploader("Upload GPS file", type=['csv'], help="Supports CSV files with various encodings")

if uploaded_file is not None:
    try:
        # Load data
        df, columns, player_names = get_data_info(uploaded_file)
        
        if df is None:
            st.stop()
        
        st.success(f"File loaded successfully! Found {len(df)} rows and {len(columns)} columns.")
        
        with st.expander("Available Columns"):
            st.write(columns)
        
        # Select metrics
        default_metrics = ["TOTAL_DISTANCE", "HIGH_SPEED_RUNNING_ABSOLUTE", "DISTANCE_Z6_ABSOLUTE", "ACCELERATIONS", "DECELERATIONS"]
        available_defaults = [metric for metric in default_metrics if metric in columns]
        
        selected_metrics = st.multiselect("Select metrics to analyze", columns, default=available_defaults)
        
        if not selected_metrics:
            st.warning("Please select at least one metric to analyze.")
            st.stop()
        
        # Select players
        if not player_names:
            st.error("No player names found in the data.")
            st.stop()
            
        selected_players = st.multiselect("Select players to analyze", player_names)
        
        if not selected_players:
            st.warning("Please select at least one player to analyze.")
            st.stop()
        
        # Filter data for selected players
        df_filtered = df[df['PLAYER_NAME'].isin(selected_players)].copy()
        
        if df_filtered.empty:
            st.error("No data found for selected players.")
            st.stop()
        
        # Calculate drill times
        try:
            df_filtered['time'] = get_drill_time(df_filtered)
        except Exception as e:
            st.error(f"Could not calculate drill times: {str(e)}")
            st.stop()
        
        # Process drill names
        try:
            df_filtered['Drill_title_2'] = get_drill_names(df_filtered)
        except Exception as e:
            st.error(f"Could not process drill names: {str(e)}")
            st.stop()
        
        # Select drills
        force_check = st.sidebar.checkbox("Show only drills with all selected players")
        
        if force_check:
            drills_with_all_players = (
                df_filtered.groupby('Drill_title_2')['PLAYER_NAME']
                .apply(lambda x: set(x) == set(selected_players))
                .reset_index()
            )
            valid_drills = drills_with_all_players[drills_with_all_players['PLAYER_NAME'] == True]['Drill_title_2']
            
            if len(valid_drills) == 0:
                st.warning("No drills found with all selected players. Try unchecking the filter.")
                st.stop()
                
            drill_names = st.sidebar.multiselect("Select Drill", valid_drills, [])
        else:
            drills = get_drill_names(df_filtered)
            unique_drills = drills.unique()
            drill_names = st.sidebar.multiselect("Select Drill", unique_drills, [])
        
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
                final_data = get_target_metric(df_filtered, drill_names, selected_metrics, drill_times)
                
                if final_data.empty:
                    st.write("No data found for selected drills")
                else:
                    st.subheader("Analysis Results")
                    st.dataframe(final_data, use_container_width=True)
                    
                    # Create visualizations
                    for metric in selected_metrics:
                        if metric in final_data.columns:
                            st.write(f"**{metric} by Player**")
                            
                            chart_data = final_data.reset_index().sort_values(metric, ascending=False)
                            
                            # Ensure proper column naming
                            if 'index' in chart_data.columns:
                                chart_data = chart_data.rename(columns={'index': 'PLAYER_NAME'})
                            
                            # Color coding
                            chart_data['color'] = 'middle'
                            if len(chart_data) >= 3:
                                chart_data.loc[chart_data.head(3).index, 'color'] = 'top'
                                chart_data.loc[chart_data.tail(3).index, 'color'] = 'bottom'
                            
                            # Create chart
                            chart = alt.Chart(chart_data, height=400, width=800).mark_bar().encode(
                                x=alt.X('PLAYER_NAME:N', sort=None),
                                y=alt.Y(f'{metric}:Q'),
                                color=alt.Color('color:N', scale=alt.Scale(domain=['top', 'middle', 'bottom'], range=['green', 'gray', 'red']), legend=None)
                            )
                            
                            text = chart.mark_text(align='center', baseline='bottom', dy=-5).encode(
                                text=alt.Text(f'{metric}:Q', format='.1f')
                            )
                            
                            st.altair_chart((chart + text), use_container_width=True)
                    
                    # Scatter plots
                    if len(selected_metrics) >= 2:
                        st.subheader("Metric Correlations")
                        create_metric_scatter_plots(final_data, selected_metrics)
        else:
            st.info("Please select at least one drill to analyze.")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        
        with st.expander("Debug Information"):
            st.write("Error details:", str(e))
            if 'df' in locals():
                st.write("Data shape:", df.shape if df is not None else "No data loaded")
                st.write("Columns:", df.columns.tolist() if df is not None else "No columns")

else:
    st.info("Please upload a CSV file to begin analysis.")
    st.write("""
    **Expected file format:**
    - CSV file with player performance data
    - Required columns: Player Name, Drill/Split information, Start/End times
    - Common metrics: Total Distance, Accelerations, Decelerations, etc.
    """)
