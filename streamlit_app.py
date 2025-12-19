import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Quantization SNR Demo",
    page_icon="🎚️",
    layout="wide"
)

# --- Constants ---
A = 1.0  # full-scale width used to define Delta

# --- Sidebar: Controls ---
st.sidebar.header("Configuration")

# Streamlit re-runs the script on every widget interaction,
# so we define inputs directly here.
N = st.sidebar.slider("Bits (N)", min_value=1, max_value=16, value=8, step=1)
c = st.sidebar.slider("Backoff c (Amplitude = A/c)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
f = st.sidebar.slider("Sine frequency [Hz]", min_value=0.1, max_value=2000.0, value=5.0, step=0.1)

# --- Logic / Computation ---
# (Logic copied from original script)
N_safe = int(max(1, min(24, N)))
c_safe = float(max(1e-6, c))
f_safe = float(max(0.1, min(2000.0, f)))

# Time axis and sine input with backoff c
t = np.linspace(0.0, 1.0, 4000, endpoint=False)
x = (A / (2.0 * c_safe)) * np.sin(2.0 * np.pi * f_safe * t)

# Uniform mid-tread quantizer
Delta = A / (2 ** N_safe)  # step size
xq = Delta * np.floor(x / Delta + 0.5)
e = x - xq

# SNR (measured) and theory
e_power = np.mean(e ** 2) if np.mean(e ** 2) > 0 else 1e-18
snr_db = 10.0 * np.log10(np.mean(x ** 2) / e_power) if np.mean(x ** 2) > 0 else -np.inf
theory_db = 6.02 * N_safe - 20.0 * np.log10(c_safe)

# --- Main Layout ---
st.title("🎚️ Quantization SNR — Mid-tread Uniform Quantizer")
st.markdown(
    "Adjust the **bits (N)**, **backoff (c)**, and **frequency** in the sidebar to see "
    "time-domain quantization, error histograms, and SNR calculations."
)

st.divider()

# Display Metrics using Streamlit's native metric component
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Bits (N)", value=N_safe)
with col2:
    st.metric(label="Step Size (Delta)", value=f"{Delta:.6f}")
with col3:
    st.metric(label="Measured SNR", value=f"{snr_db:.2f} dB")
with col4:
    st.metric(label="Theory SNR", value=f"{theory_db:.2f} dB", delta=f"{snr_db - theory_db:.2f} diff")

st.divider()

# --- Plotting ---
# We use st.columns to arrange plots side-by-side or stacked.
# Given the detailed nature, stacked (vertical) often works best,
# or side-by-side on wide screens.

# Plot 1: Time-domain
fig1, ax1 = plt.subplots(figsize=(8, 3.5))
n_zoom = 400
ax1.plot(t[:n_zoom], x[:n_zoom], label="x(t): input", lw=1.6)
ax1.step(t[:n_zoom], xq[:n_zoom], where="mid", label="Q(x): quantized", lw=1.0)
ax1.set_title(f"Time-domain quantization (First {n_zoom} samples)")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Amplitude")
ax1.grid(alpha=0.3)
ax1.legend(loc="upper right")
fig1.tight_layout()

# Plot 2: Error histogram
fig2, ax2 = plt.subplots(figsize=(8, 3.5))
ax2.hist(e, bins=60, density=True, color="gray", edgecolor="none")
ax2.axvline(+Delta / 2.0, color="r", ls="--", lw=1)
ax2.axvline(-Delta / 2.0, color="r", ls="--", lw=1)
ax2.set_title("Quantization error distribution e = x - Q(x)")
ax2.set_xlabel("Error")
ax2.set_ylabel("Probability density")
ax2.text(0.02, 0.95, f"Delta = {Delta:.6f}\nBounds ±Delta/2", transform=ax2.transAxes, va="top")
ax2.grid(alpha=0.2)
fig2.tight_layout()

# Render plots
left_plot, right_plot = st.columns(2)
with left_plot:
    st.pyplot(fig1)
with right_plot:
    st.pyplot(fig2)

# --- Interpretation / Details ---
with st.expander("ℹ️ Interpretation & Details", expanded=True):
    st.markdown(f"""
    * **Levels L:** {2**N_safe}
    * **Backoff c:** {c_safe:.4f}
    * **Theoretical Formula:** $$SNR \\approx 6.02 N - 20 \\log_{{10}}(c)$$
    
    **Key Takeaways:**
    1.  Increasing **N** lowers $\\Delta$ and increases SNR (~6 dB/bit at full scale).
    2.  Increasing **c** (more backoff from full-scale) reduces SNR.
    """)
