import streamlit as st

# Local import
from utils import show_about

st.set_page_config(page_title="Komm Demo", layout="wide")


st.title("Komm Demo")

st.markdown(
    """
    This interactive demo showcases various features of the Komm library, a toolkit for analysis and simulation of analog and digital communication systems.
    """
)

st.header("Available demos")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Binary sequences")
    st.markdown(
        """
        Explore different types of binary sequences commonly used in communications:
        - Barker sequencesfrom st_pages import Page, add_page_title, show_pages
        - Walsh-Hadamard sequences
        - Linear-feedback shift register (LFSR) sequences
        """
    )

    st.subheader("2. Constellations")
    st.markdown(
        """
        Interactive visualization of digital modulation constellations:
        - Phase-shift keying (PSK)
        - Quadrature amplitude modulation (QAM)
        """
    )

with col2:
    st.subheader("3. Pulse formatting")
    st.markdown(
        """
        Visualize various pulse shaping techniques:
        - Sinc pulse
        - Raised cosine pulse
        - Gaussian pulse
        """
    )


show_about()
