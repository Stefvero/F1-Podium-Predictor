import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import plotly.express as px

def run():  
    st.title("F1 Podium Prediction App")

    st.subheader("This page consists of Exploratory Data Analysis (EDA) of F1 dataset")

    data = mpimg.imread('./src/f1.jpg')
    st.image(data, caption = "Formula 1")

    df = pd.read_csv('./src/f1df.csv')
    st.dataframe(df)

    # Barplot
    st.write('### Barplot of Podium Probability by Grid Position')
    grid_podium = (df.groupby('grid_position')['podium'].agg(['mean', 'count']).reset_index()
                   .rename(columns={'mean': 'podium_rate', 'count': 'races'}))

    fig, ax = plt.subplots(figsize = (9, 4))
    ax.bar(grid_podium['grid_position'], grid_podium['podium_rate'] * 100)
    ax.set_xlabel("Starting Grid Position")
    ax.set_ylabel("Podium Probability (%)")
    ax.set_title("Podium Probability by Grid Position")
    plt.tight_layout()
    st.pyplot(fig)

    st.write('### Driver age and Podiums')
    # Age and Podium Barplot
    fig, ax = plt.subplots(figsize = (9, 4))
    colors = {0: '#FF0000', 1: "#2BFF00"}
    for label in [0, 1]:
        ax.hist(df[df['podium'] == label]['driver_age'].dropna(),bins = 40,label = f'Podium={label}', color = colors[label], edgecolor = 'black')
    ax.set_xlabel('Driver Age')
    ax.set_ylabel('Count')
    ax.set_title('Driver Age Distribution: Podium vs Not Podium')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.write('### Correlation heatmap of numerical columns')
    num_cols = ['grid_position', 'quali_position', 'round', 'year', 'driver_points_before', 'driver_position_before', 'driver_wins_before', 'driver_age', 'constructor_position_before', 'driver_historical_dnf_rate', 'pit_stops', 'podium']
    corrnum = df[num_cols].corr()
    fig, ax = plt.subplots(figsize = (12, 9))
    sns.heatmap(corrnum, annot = True, fmt = '.2f',cmap = 'YlOrRd', ax = ax, vmin = -1, vmax = 1)
    ax.set_title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    st.pyplot(fig)

if __name__ == "__main__":
    run()