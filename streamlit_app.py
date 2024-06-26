import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

n_steps = st.slider("Number of time steps", 1000, 10000, 1100)
seed = st.slider("Random Seed", 1, 300, 31)



def plot_polar(n_steps,seed):
    indices=np.linspace(0,1,n_steps)
    del_theta=1/10                    ##step size for angle
    del_r=1                           ##step size for radius

    r=np.zeros(n_steps)
    theta=np.zeros(n_steps)
    Z= np.zeros(n_steps,dtype='complex128')
    Z[0]=0
    theta=0
    r=0
    np.random.seed(seed)
    # Generate random choices for k_r and k_theta for all steps at once
    k_r = np.random.choice(a=np.arange(3)-1, p=[0.25, 0.25, 0.5], size=n_steps-1)
    k_theta = np.random.choice(a=np.arange(3)-1, p=[0.25, 0.25, 0.5], size=n_steps-1)
    r_values = np.cumsum(np.concatenate(([r], del_r * k_r)))
    theta_values = np.cumsum(np.concatenate(([theta], del_theta * k_theta)))
    Z = r_values * np.exp(1j * theta_values)
    # If the first value of Z should be 0, adjust as needed
    Z[0] = 1 * r * np.exp(1j * theta)

    df = pd.DataFrame({
        "x": Z.real,
        "y": Z.imag,
        "idx": indices,
        "rand": np.random.randn(n_steps),
    })

    return df

df=plot_polar(n_steps,seed)

st.altair_chart(alt.Chart(df, height=700, width=700)
    .mark_point(filled=True)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        color=alt.Color("idx", legend=None, scale=alt.Scale()),
        size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
    ))
