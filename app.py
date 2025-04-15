# app.py
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.embed import components
from streamlit.components.v1 import html

st.set_page_config(page_title="Cosmetic Moisturizers t-SNE", layout="wide")
st.title("💄 Cosmetic Moisturizers (Dry Skin) - t-SNE Analysis")

# Load dataset
url = 'https://raw.githubusercontent.com/engineers-planet/Machine-Learning_Deep-learning_Free-Download/381de74cb080305f43ffb710db13f3e6f5ce54e0/Cosmetic%20Product%20Analysis/datasets/cosmetics.csv'
df = pd.read_csv(url)

# Filter dry-skin moisturizers
moisturizers_dry = df[(df['Label'] == "Moisturizer") & (df["Dry"] == 1)].copy()
moisturizers_dry.reset_index(drop=True, inplace=True)

# Tokenize ingredients
ingredient_idx = {}
corpus = []
idx = 0
for i in range(len(moisturizers_dry)):
    ingredients = moisturizers_dry['Ingredients'][i].lower().split(', ')
    corpus.append(ingredients)
    for ingredient in ingredients:
        if ingredient not in ingredient_idx:
            ingredient_idx[ingredient] = idx
            idx += 1

# One-hot encoding
M = len(corpus)
N = len(ingredient_idx)
A = np.zeros((M, N))
for i, tokens in enumerate(corpus):
    for token in tokens:
        A[i, ingredient_idx[token]] = 1

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(A)
moisturizers_dry['X'] = tsne_results[:, 0]
moisturizers_dry['Y'] = tsne_results[:, 1]

# Create Bokeh plot
source = ColumnDataSource(moisturizers_dry)
plot = figure(
    title='t-SNE Visualization of Dry Skin Moisturizers',
    x_axis_label='T-SNE 1',
    y_axis_label='T-SNE 2',
    width=800,
    height=500,
    tools="pan,wheel_zoom,reset"
)
plot.scatter(x='X', y='Y', source=source, size=10, color='#FF7373', alpha=0.8)

hover = HoverTool(tooltips=[
    ("Name", "@Name"),
    ("Brand", "@Brand"),
    ("Price", "$@Price"),
    ("Rank", "@Rank")
])
plot.add_tools(hover)

# Embed plot into Streamlit
script, div = components(plot)
html(f"{script}{div}", height=550)

st.markdown("---")
st.subheader("📋 Sample of Data")
st.dataframe(moisturizers_dry[['Name', 'Brand', 'Price', 'Rank']].head())
