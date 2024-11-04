import inspect

import streamlit as st


def show_about():
    st.sidebar.markdown(
        """
        ### About Komm

        Komm is an open-source library for Python 3 providing tools for analysis and simulation of analog and digital communication systems.

        [![PyPI page](https://badge.fury.io/py/komm.svg)](https://pypi.org/project/komm/)
        [![Documentation](https://img.shields.io/badge/docs-komm.dev-blue)](https://komm.dev/)
        [![GitHub](https://img.shields.io/badge/github-rwnobrega%2Fkomm-black)](https://github.com/rwnobrega/komm)
        """
    )


def show_code(func):
    source = inspect.getsource(func)
    lines = source.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("def "):
            source = "\n".join(lines[i:])
            break

    st.write("")
    with st.expander("Show code"):
        st.code(source, language="python")


def show_documentation(title, stub):
    st.markdown(
        f"### {title} <a href='https://komm.dev/ref/{stub}' style='text-decoration: none;'><span style='font-size: 0.7em; margin-left: 0.1em'>🌐</span></a>",
        unsafe_allow_html=True,
    )
