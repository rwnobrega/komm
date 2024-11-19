import streamlit as st

# local import
from utils import show_about

st.set_page_config(page_title="Komm Demo", layout="wide")

st.title("Komm Demo")

st.markdown("""
    This interactive demo showcases various features of the [Komm library](https://komm.dev).

    Please select a page from the sidebar to get started.
    """)

show_about()
