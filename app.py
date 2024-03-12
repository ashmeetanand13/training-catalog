#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from fuzzywuzzy import fuzz
import streamlit as st
import altair as alt
import pandas as pd
import plotly.express as px

# import api_db

# MATCH_ID = '3753628'
# SQL_SELECT_1 = f"SELECT * FROM STS.Fit_ABF"

# # Establish connection to DB. Automatically uses values defined in res/config.json
# con = api_db.get_connection_to_db()
# cursor = con.cursor()

# # Read results from select statement directly into a DataFrame
# df = pd.read_sql(SQL_SELECT_1, con)

# # I usually make sure all of the column names are capitalized.
# df.columns = map(str.upper, df.columns)

st.title('Sounders FC - Catalog')
# Create a text element and let the reader know the data is loading.
# data_load_state = st.text('Loading data...')



#function to get the data from the csv for the drill catalog 
#making a first team player list so you know what players you want to select and use. 
#@st.cache(allow_output_mutation=True)
@st.cache_data

def get_data(file_name, player_names_file="FIRST_TEAM.txt"):
    
    df = pd.read_csv(file_name, index_col = 0)
    
    first_team = []
    
#with open, opens at the file and then closes it when its done using it, so you dont have to manually close it.     
    with open ("FIRST_TEAM.txt" , "r") as file:
        out = file.read()
        first_team = out.split("\n")
    
#to get rid of all the first year data as the drill names are in different format to what we use now
    df = df[df["DRILL_DATE"]>'2022-01-01']
    
    df.DRILL_TITLE = df.DRILL_TITLE.str.upper()
    
    return df[df.PLAYER_FIRST_NAME.isin(first_team)]


# In[2]:


#function to get the time from adam and use it everywhere in the data as we will need minutes values, 
#to find per minute values for all the metrics.
def get_drill_time(df):
    assert df.ADAM.size > 0
    
    drill_time = df.ADAM.str.split(" " , expand = True )[1]
    #time delta funtion to get the seconds vaules and divide it with 60 for minutes. 
    drill_time_mins = pd.to_timedelta(drill_time).dt.seconds / 60
    
    return drill_time_mins


# In[78]:


# total = 0

# for drill in tqdm.tqdm_notebook(f_drill_lib.DRILL_TITLE_UPPER.unique()):
#     for drill2 in f_drill_lib.DRILL_TITLE_UPPER.unique():
#         ratio = fuzz.ratio(drill, drill2)
#         if  ratio >= 98 and ratio < 100:
# #             print(drill)
# #             print(drill2)
# #             print(f"Ratio: {ratio}")
#             total += 1
    
# total



#spliting the drill title to based on the column as we need all the values and not the last empty ones (Block drills)
def get_drill_names(df, player_list, force_check=False):

    assert df.DRILL_TITLE.size > 0
    
    drill_titles = df.DRILL_TITLE.str.split(':', expand=True).iloc[:,:5]

    #deleting the drills that have no reps and then only using the drills that have values. 
    drill_titles = drill_titles[~drill_titles[4].isna()]
    #using the lambda to joning the rows after we are done seprating and eleminatin the ones we dont need.
    df['DRILL_TITLE_2'] =  drill_titles.iloc[:,:4].apply(lambda row: ':'.join(row.values.astype(str)), axis=1)
    #only return the unique values of the drills we will be using. 



    # player_list = ['Nicolas','Alex', 'Yeimar']
    filter_df = df[df.PLAYER_FIRST_NAME.isin(player_list)]

    if force_check:
        filter_df = filter_df.groupby('DRILL_TITLE_2').PLAYER_FIRST_NAME.apply(lambda x: ','.join(x)).str.split(",")
        filter_df = filter_df[filter_df.apply(lambda x: all(elem in x  for elem in player_list))].reset_index()


    return filter_df.DRILL_TITLE_2.unique()



#get the data frame and then get the drill name as string. 
def get_drill_data(df: pd.DataFrame, drill_name: str):
#regex is used to eleminate the + or / as they have a differnet function. 
    drills_to_test = df[df.DRILL_TITLE.str.contains(drill_name,case = False, regex=False)].DRILL_TITLE
#group the data by drill names. 
    grouped = pd.DataFrame()
#make a set of drills to include.
    drills_to_include  = set()
# making a set of the drills i would like to include. 
    minratio = 90

    for drill_1 in (drills_to_test.unique()):
            ratio = fuzz.ratio(drill_1.upper(), drill_name.upper())
#             if ratio < minratio:
#                 print(drill_1)
            if ratio >= minratio and ratio < 100:
                drills_to_include.add(drill_1)
    
    
    return df[df.DRILL_TITLE.isin(drills_to_include)]
     
    



# target_time = 5
# metrics_list = arr
#get the new time for the drills as what time you would like your players to play. 
def get_target_metric(df, drill_names, metrics_list, target_times , player_list):
  
    drills = []
    
#using player name as index as it will not show player names and just values. 
    for drill_name, time  in zip(drill_names, target_times):
        
        target_df = pd.DataFrame()

        drill_data = get_drill_data(df,drill_name)

        drill_data = drill_data[drill_data.PLAYER_FIRST_NAME.isin(player_list)].set_index("PLAYER_FIRST_NAME")

        for i in metrics_list:
            target_df[i] = (drill_data[i] / get_drill_time(drill_data)) * time
            
        drills.append(target_df.groupby("PLAYER_FIRST_NAME").mean())


    final_df = pd.concat(drills).groupby("PLAYER_FIRST_NAME").sum() if len(drills) > 0 else []

    return final_df



df = get_data("ABF.csv")
arr = ["TOTAL_DISTANCE", "HIGH_SPEED_RUNNING_ABSOLUTE", "DISTANCE_Z6_ABSOLUTE", "ACCELERATIONS", "DECELERATIONS"]
arr2 = ["TOTAL_DISTANCE", "HID", "VHID", "ACCELERATIONS", "DECELERATIONS"]
training_drills = []


options = st.sidebar.multiselect(
     'Player list', 
     df.PLAYER_FIRST_NAME.sort_values().unique().tolist(),
     df.PLAYER_FIRST_NAME.sort_values().unique().tolist()
)



# Notify the reader that the data was successfully loaded.
# data_load_state.text('')

force_check = st.sidebar.checkbox("Show only drill with everyone")

drills = get_drill_names(df,options, force_check) if len(options) > 0 else []

drill_names =  st.sidebar.multiselect("Select Drill", drills , [],  disabled=False)

drill_times = []

for drill in drill_names:
    drill_time = st.slider(f'Select time for {drill}', 2, 30, 2)
    drill_times.append(drill_time)


if len(drill_names ) > 0:  
    final_data = get_target_metric(df, drill_names, arr, drill_times, options)
    final_data = final_data.rename(columns={'HIGH_SPEED_RUNNING_ABSOLUTE': 'HID','DISTANCE_Z6_ABSOLUTE': 'VHID'})
 
    st.table(final_data)
    
    for metric in arr2:
        st.write(alt.Chart(final_data.reset_index().sort_values(metric, ascending = False), height = 400,width = 800).mark_bar().encode(
        x=alt.X('PLAYER_FIRST_NAME', sort=None), y = metric))
else:
    st.text("Please select a drill to continue")


# %%
st.sidebar.selectbox('Select_Player', df.PLAYER_FIRST_NAME.sort_values().unique() )

fig = px.pie(df.PLAYER_FIRST_NAME, values='HID', names='drill_names')
if st.sidebar.selectbox('I:',['f']) == 'f': b()