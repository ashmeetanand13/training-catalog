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
    Handles various encoding issues robustly.
    """
    
    # List of common encodings to try in order of preference
    encodings_to_try = [
        'utf-8', 
        'latin-1', 
        'cp1252', 
        'iso-8859-1', 
        'utf-16',
        'ascii'
    ]
    
    df = None
    successful_encoding = None
    
    # Try each encoding
    for encoding in encodings_to_try:
        try:
            # Reset file pointer if it's a file-like object (Streamlit uploader)
            if hasattr(file, 'seek'):
                file.seek(0)
            
            # Try to read with current encoding
            df = pd.read_csv(file, encoding=encoding)
            successful_encoding = encoding
            st.success(f"✅ Successfully read file with {encoding} encoding")
            break
            
        except (UnicodeDecodeError, UnicodeError):
            st.warning(f"❌ Failed with {encoding} encoding, trying next...")
            continue
        except Exception as e:
            error_msg = str(e).lower()
            if any(word in error_msg for word in ['codec', 'decode', 'unicode', 'encoding']):
                st.warning(f"❌ Encoding issue with {encoding}, trying next...")
                continue
            else:
                st.error(f"Non-encoding error with {encoding}: {str(e)}")
                break
    
    # If standard encodings failed, try with error handling strategies
    if df is None:
        st.warning("🔄 Standard encodings failed, trying with error handling...")
        
        error_strategies = [
            ('utf-8', 'replace'),
            ('latin-1', 'ignore'),
            ('cp1252', 'replace'),
            ('utf-8', 'ignore')
        ]
        
        for encoding, error_handling in error_strategies:
            try:
                if hasattr(file, 'seek'):
                    file.seek(0)
                
                df = pd.read_csv(file, encoding=encoding, errors=error_handling)
                st.info(f"✅ Read file with {encoding} encoding and '{error_handling}' error handling")
                break
                
            except Exception as e:
                continue
    
    # Final fallback: try to read as binary and convert
    if df is None:
        st.warning("🔄 Trying binary read approach...")
        try:
            if hasattr(file, 'seek'):
                file.seek(0)
            
            # Read file content as bytes
            if hasattr(file, 'read'):
                content = file.read()
                if isinstance(content, bytes):
                    # Try to decode bytes to string
                    for encoding in ['utf-8', 'latin-1', 'cp1252']:
                        try:
                            decoded_content = content.decode(encoding, errors='replace')
                            # Create a StringIO object from the decoded content
                            from io import StringIO
                            string_io = StringIO(decoded_content)
                            df = pd.read_csv(string_io)
                            st.info(f"✅ Successfully read using binary approach with {encoding}")
                            break
                        except Exception as e:
                            continue
            
        except Exception as e:
            st.error(f"Binary read approach failed: {str(e)}")
    
    if df is None or df.empty:
        st.error("❌ Failed to read the CSV file with any method")
        st.write("**Troubleshooting tips:**")
        st.write("- Ensure the file is a valid CSV")
        st.write("- Try opening the file in a text editor and saving it with UTF-8 encoding")
        st.write("- Check if the file has any special characters")
        return None, [], []
    
    # Clean up the dataframe
    try:
        # Clean up column names
        df.columns = df.columns.astype(str)  # Ensure all column names are strings
        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace
        df.columns = df.columns.str.replace('\ufeff', '', regex=False)  # Remove BOM
        df.columns = df.columns.str.replace('\x00', '', regex=False)  # Remove null characters
        df.columns = df.columns.str.replace(r'[^\w\s]', '_', regex=True)  # Replace special chars with underscore
        df.columns = df.columns.str.upper()  # Make all columns uppercase
        
        all_columns = df.columns.tolist()
        
        # Find the player name column (handle various possible names)
        player_col = None
        possible_player_cols = ['PLAYER_NAME', 'PLAYER', 'NAME', 'PLAYER NAME']
        
        for col in possible_player_cols:
            if col in df.columns:
                player_col = col
                break
        
        if player_col is None:
            st.error(f"Could not find player name column. Available columns: {', '.join(all_columns)}")
            st.write("Please ensure your CSV has a column named one of: " + ", ".join(possible_player_cols))
            return df, all_columns, []
        
        # Standardize the player column name to 'PLAYER_NAME' for consistency
        if player_col != 'PLAYER_NAME':
            df = df.rename(columns={player_col: 'PLAYER_NAME'})
        
        # Clean player names and get unique values
        df['PLAYER_NAME'] = df['PLAYER_NAME'].astype(str).str.strip()
        df['PLAYER_NAME'] = df['PLAYER_NAME'].str.replace('\x00', '', regex=False)  # Remove null characters
        player_names = df['PLAYER_NAME'].unique().tolist()
        
        # Remove empty or invalid player names
        player_names = [name for name in player_names if name and name != 'nan' and name.strip()]
        
        return df, all_columns, player_names
        
    except Exception as e:
        st.error(f"Error cleaning dataframe: {str(e)}")
        return None, [], []

def parse_drill_name_format(drill_text):
    """
    Parse different drill name formats:
    1. Semicolon-separated key-value pairs: "T=WU;V=Prepractice;N=24;A=;I=1of1;NT="
    2. Colon-separated traditional format: "Drill:Type:Phase:etc"
    """
    drill_text = str(drill_text).strip()
    
    # Check if it's the semicolon key-value format
    if ';' in drill_text and '=' in drill_text:
        # Parse key-value pairs
        pairs = {}
        for pair in drill_text.split(';'):
            if '=' in pair:
                key, value = pair.split('=', 1)
                pairs[key.strip()] = value.strip()
        
        # Create a meaningful drill name from the key-value pairs
        drill_parts = []
        
        # Common key mappings for meaningful names
        if 'T' in pairs and pairs['T']:  # Type
            drill_parts.append(f"Type:{pairs['T']}")
        if 'V' in pairs and pairs['V']:  # Version/Variation
            drill_parts.append(f"Variation:{pairs['V']}")
        if 'N' in pairs and pairs['N']:  # Number
            drill_parts.append(f"N:{pairs['N']}")
        if 'A' in pairs and pairs['A']:  # Activity
            drill_parts.append(f"Activity:{pairs['A']}")
        if 'I' in pairs and pairs['I']:  # Instance
            drill_parts.append(f"Instance:{pairs['I']}")
        
        # If no meaningful parts found, use the first few non-empty values
        if not drill_parts:
            for key, value in pairs.items():
                if value:  # Only include non-empty values
                    drill_parts.append(f"{key}:{value}")
                if len(drill_parts) >= 3:  # Limit to 3 parts
                    break
        
        return ':'.join(drill_parts) if drill_parts else drill_text
    
    # Traditional colon-separated format
    elif ':' in drill_text:
        parts = drill_text.split(':')
        # Take first 4 parts or all parts if less than 4
        return ':'.join(parts[:4])
    
    # Single word or phrase - return as is
    else:
        return drill_text

def get_drill_time(df):
    """Calculate drill duration in minutes"""
    # Find start time column - check both naming conventions
    start_col = next((col for col in df.columns if col in ['DRILL_START_TIME', 'START_TIME','SPLIT_START_TIME','Start Time']), None)
    end_col = next((col for col in df.columns if col in ['DRILL_END_TIME', 'END_TIME','SPLIT_END_TIME','End Time']), None)
    
    if not start_col or not end_col:
        st.error("Could not find start/end time columns")
        st.write("Available columns:", df.columns.tolist())
        raise ValueError("Could not find start/end time columns")
    
    try:
        # Check the format of time values
        sample_start = str(df[start_col].iloc[0])
        
        # Handle Excel serial date numbers (like 45708.798)
        if sample_start.replace('.', '', 1).isdigit():
            # Convert Excel serial numbers to datetime
            df['start_time'] = pd.TimedeltaIndex(df[start_col] % 1 * 86400, unit='s') + pd.Timestamp('1900-01-01')
            df['end_time'] = pd.TimedeltaIndex(df[end_col] % 1 * 86400, unit='s') + pd.Timestamp('1900-01-01')
        else:
            # Regular datetime parsing for time strings (like '9:43:15')
            df['start_time'] = pd.to_datetime(df[start_col])
            df['end_time'] = pd.to_datetime(df[end_col])
        
        return (df['end_time'] - df['start_time']).dt.total_seconds() / 60
    
    except Exception as e:
        st.error(f"Error processing time columns: {str(e)}")
        st.write("Sample start time value:", sample_start)
        raise

def get_drill_names(df):
    """Extract and process drill names with fuzzy matching and uppercase standardization"""
    # Find drill title column - check various naming conventions
    possible_drill_cols = [
        'DRILL_TITLE', 'DRILL_NAME', 'SPLIT_NAME', 'PERIOD_NAME'
    ]
    
    drill_col = None
    for col in df.columns:
        if col.upper() in possible_drill_cols:
            drill_col = col
            break
    
    if not drill_col:
        st.error("Could not find drill title column")
        st.write("Available columns:", df.columns.tolist())
        st.write("Expected column names:", possible_drill_cols)
        raise ValueError("Could not find drill title column. Available columns: " + ", ".join(df.columns))
    
    try:
        # Convert drill titles to uppercase and clean them
        df[drill_col] = df[drill_col].astype(str).str.upper().str.strip()
        df[drill_col] = df[drill_col].str.replace('\x00', '', regex=False)  # Remove null characters
        
        st.info(f"Found drill column: {drill_col}")
        st.info(f"Sample drill name: {df[drill_col].iloc[0]}")
        
        # Parse drill names using the new format-aware function
        drill_names = df[drill_col].apply(parse_drill_name_format)
        
        # Get unique drills for display
        unique_drills = list(set(drill_names))
        
        st.info(f"Found {len(unique_drills)} unique drill patterns")
        
        # Perform fuzzy matching to combine similar drill names
        final_drills = []
        processed_drills = set()
        
        for drill in unique_drills:
            if drill not in processed_drills:
                # Find all similar drills with 90% or higher similarity (lowered threshold for more variety)
                matches = process.extract(drill, unique_drills, scorer=fuzz.ratio, limit=None)
                similar_drills = [match[0] for match in matches if match[1] >= 90]
                
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
        standardized_names = drill_names.map(drill_mapping)
        
        # Show some examples for debugging (optional)
        if st.checkbox("Show drill name processing examples", value=False):
            with st.expander("Drill Name Processing Examples"):
                st.write("Sample original vs processed:")
                for i in range(min(3, len(df))):
                    orig = df[drill_col].iloc[i]
                    processed = standardized_names.iloc[i]
                    st.write(f"Original: {orig}")
                    st.write(f"Processed: {processed}")
                    st.write("---")
        
        return standardized_names
    
    except Exception as e:
        st.error(f"Error processing drill names: {str(e)}")
        st.write("Sample drill data:", df[drill_col].head().tolist() if drill_col else "No drill column found")
        raise

def get_target_metric(df, selected_drills, metrics, drill_times):
    """Calculate metrics per minute for selected drills"""
    # Convert selected drills to uppercase for consistency
    selected_drills = [drill.upper() for drill in selected_drills]
    
    results = []
    
    try:
        for drill, time_multiplier in zip(selected_drills, drill_times):
            drill_data = df[df['Drill_title_2'].str.upper() == drill]
            
            if drill_data.empty:
                continue
            
            for player in drill_data['PLAYER_NAME'].unique():
                player_data = drill_data[drill_data['PLAYER_NAME'] == player]
                player_metrics = {
                    "PLAYER_NAME": player,
                }
                
                for metric in metrics:
                    if metric in player_data.columns:
                        # Handle potential NaN values
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
    
    except Exception as e:
        st.error(f"Error calculating target metrics: {str(e)}")
        return pd.DataFrame()

def create_metric_scatter_plots(final_data, selected_metrics):
    """Create scatter plots for all combinations of selected metrics"""
    try:
        # Get all possible pairs of metrics
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
                st.write(f'Average {metric1}:', round(final_data[metric1].mean(), 2))
                st.write(f'Max {metric1}:', round(final_data[metric1].max(), 2))
            with col2:
                st.write(f'Average {metric2}:', round(final_data[metric2].mean(), 2))
                st.write(f'Max {metric2}:', round(final_data[metric2].max(), 2))
    
    except Exception as e:
        st.error(f"Error creating scatter plots: {str(e)}")

# Streamlit UI
st.title('Soccer Pre-Training Analysis')
st.write("Upload your GPS data file (CSV format) to analyze player performance metrics.")

# File upload
uploaded_file = st.file_uploader("Upload GPS file", type=['csv', 'gps'], help="Supports CSV files with various encodings")

if uploaded_file is not None:
    try:
        # Load data
        df, columns, player_names = get_data_info(uploaded_file)
        
        if df is None:
            st.stop()
        
        # Display file info
        st.success(f"File loaded successfully! Found {len(df)} rows and {len(columns)} columns.")
        
        # Display columns in an expandable section
        with st.expander("Available Columns"):
            st.write(columns)
        
        # Select metrics
        default_metrics = [
            "TOTAL_DISTANCE",
            "HIGH_SPEED_RUNNING_ABSOLUTE", 
            "DISTANCE_Z6_ABSOLUTE",
            "ACCELERATIONS",
            "DECELERATIONS"
        ]
        
        available_defaults = [metric for metric in default_metrics if metric in columns]
        
        selected_metrics = st.multiselect(
            "Select metrics to analyze",
            columns,
            default=available_defaults,
            help="Choose the performance metrics you want to analyze"
        )
        
        if not selected_metrics:
            st.warning("Please select at least one metric to analyze.")
            st.stop()
        
        # Select players
        if not player_names:
            st.error("No player names found in the data.")
            st.stop()
            
        selected_players = st.multiselect(
            "Select players to analyze",
            player_names,
            help="Choose which players to include in the analysis"
        )
        
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
            valid_drills = drills_with_all_players[
                drills_with_all_players['PLAYER_NAME'] == True
            ]['Drill_title_2']
            
            if len(valid_drills) == 0:
                st.warning("No drills found with all selected players. Try unchecking the filter.")
                st.stop()
                
            drill_names = st.sidebar.multiselect(
                "Select Drill",
                valid_drills,
                [],
                disabled=False
            )
        else:
            drills = get_drill_names(df_filtered)
            unique_drills = drills.unique()
            
            drill_names = st.sidebar.multiselect(
                "Select Drill",
                unique_drills,
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
                    st.dataframe(final_data, use_container_width=True)
                    
                    # Create visualizations
                    for metric in selected_metrics:
                        if metric in final_data.columns:
                            st.write(f"**{metric} by Player**")
                            
                            # Prepare data with rankings - ensure proper column names
                            chart_data = final_data.reset_index()
                            chart_data = chart_data.sort_values(metric, ascending=False)
                            
                            # Ensure the player name column is properly named
                            if 'index' in chart_data.columns:
                                chart_data = chart_data.rename(columns={'index': 'PLAYER_NAME'})
                            
                            # Create a color column based on ranking
                            chart_data['color'] = 'middle'  # Default color
                            if len(chart_data) >= 3:
                                chart_data.loc[chart_data.head(3).index, 'color'] = 'top'  # Top 3
                                chart_data.loc[chart_data.tail(3).index, 'color'] = 'bottom'  # Bottom 3
                            
                            # Create the chart with colored bars
                            chart = alt.Chart(
                                chart_data,
                                height=400,
                                width=800
                            ).mark_bar().encode(
                                x=alt.X('PLAYER_NAME:N', sort=None),
                                y=alt.Y(f'{metric}:Q'),
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
                                dy=-5
                            ).encode(
                                text=alt.Text(f'{metric}:Q', format='.1f')
                            )
                            
                            # Combine bar chart and labels
                            final_chart = (chart + text)
                            
                            st.altair_chart(final_chart, use_container_width=True)
                    
                    # Scatter plot for metrics
                    if len(selected_metrics) >= 2:
                        st.subheader("Metric Correlations")
                        create_metric_scatter_plots(final_data, selected_metrics)
        else:
            st.info("Please select at least one drill to analyze.")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Please check your file format and try again.")
        
        # Debug information
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
