import streamlit as st
import pandas as pd
import os
import plotly.express as px  # Plotly ko interactive charts ke liye import karein
from datetime import datetime, timedelta

# --- Configuration ---
LOG_FILE = 'emotion_log.csv'

# Page Configuration
st.set_page_config(
    page_title="EduSense Classroom Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading Function (Real-time Refresh) ---
# Har 2 seconds mein data refresh hoga
@st.cache_data(ttl=2) 
def load_data():
    if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
        return pd.DataFrame()
    
    # Error handling ke saath CSV padhein
    try:
        df = pd.read_csv(LOG_FILE)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame()

# --- Sidebar Filters ---
st.sidebar.header("📊 Filter Settings")

# Timeframe Filter
time_filter = st.sidebar.select_slider(
    'Show data for last:',
    options=['Full Session', '60 Minutes', '30 Minutes', '15 Minutes', '5 Minutes'],
    value='Full Session'
)

# --- Main Dashboard Layout ---
st.title("👨‍🏫 EduSense: Real-time Student Engagement")
st.markdown("Monitor student mood to optimize teaching strategies.")

df = load_data()

if df.empty:
    st.warning("No data logged yet. Please run 'test_webcam.py' in a separate terminal to start collecting data.")
else:
    # --- Apply Time Filter ---
    filtered_df = df.copy()
    if time_filter != 'Full Session':
        minutes = int(time_filter.split(' ')[0])
        time_limit = datetime.now() - timedelta(minutes=minutes)
        filtered_df = df[df['Timestamp'] >= time_limit]

    if filtered_df.empty:
        st.info(f"No data available in the last {time_filter}. Showing full session data.")
        filtered_df = df
        
    # --- 1. Key Metrics ---
    st.header("⚡ Live Insights")
    
    # Averages calculate karein
    avg_happy = filtered_df['Happy_Percent'].mean()
    avg_sad = filtered_df['Sad_Percent'].mean()
    avg_neutral = filtered_df['Neutral_Percent'].mean()
    
    # Dominant Mood find karein
    avg_moods = {'Happy': avg_happy, 'Sad': avg_sad, 'Neutral': avg_neutral}
    dominant_mood = max(avg_moods, key=avg_moods.get)
    
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Average Happiness", f"{avg_happy:.1f}%")
    col2.metric("Average Sadness", f"{avg_sad:.1f}%")
    col3.metric("Average Neutral", f"{avg_neutral:.1f}%")
    col4.metric("Dominant Mood", dominant_mood)

    st.markdown("---")

    # --- 2. Actionable Recommendation ---
    st.header("💡 Key Recommendation")

    if avg_sad > 20:
        st.error("ACTION NEEDED: Sadness/Frustration level is high. Consider taking a break or asking for direct feedback.")
    elif avg_happy > avg_sad and avg_happy > avg_neutral:
        st.success("SUCCESS: Engagement is high and students appear happy. Continue the current teaching pace.")
    elif avg_neutral > 35:
        st.warning("ATTENTION: Neutral/Focus level is high. Break the monotony with an interactive question or activity.")
    else:
        st.info("Engagement is balanced. Monitoring is ongoing.")

    st.markdown("---")

    # --- 3. Interactive Trend Chart (Plotly) ---
    st.header(f"📈 Emotion Trend ({time_filter})")
    st.caption("Hover to see exact percentages and zoom in on struggling moments.")
    
    df_plot = filtered_df[['Timestamp', 'Happy_Percent', 'Sad_Percent', 'Neutral_Percent']]

    fig = px.line(df_plot, x='Timestamp', y=['Happy_Percent', 'Sad_Percent', 'Neutral_Percent'],
                  labels={'value': 'Percentage (%)', 'Timestamp': 'Time of Day'},
                  title="Class Emotion Distribution Over Time",
                  height=450)

    # Frustration Zone (Red Area)
    # Sadness 20% se upar ho toh yahaan fill karein
    fig.add_hrect(y0=20, y1=100, line_width=0, fillcolor="red", opacity=0.1, 
                  annotation_text="Frustration Zone (Sad > 20%)", 
                  annotation_position="top left")
    
    # Layout customize karein
    fig.update_layout(legend_title_text='Emotion')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- 4. Raw Data (Hidden) ---
    with st.expander("Show Raw Data Log"):
        st.dataframe(filtered_df.sort_values(by='Timestamp', ascending=False), use_container_width=True)