import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# --- Page Config ---
st.set_page_config(
    page_title="Quantization Lab",
    page_icon="🎛️",
    layout="wide"
)

# --- CSS for clean embedding ---
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    /* Make metrics stand out */
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# st.title("🎛️ Quantization Signal-to-Noise Ratio (SNR)")
st.markdown("""###🎛️ Quantization Signal-to-Noise Ratio (SNR)""")
st.markdown("""
Explore how **Bit Depth**, **Signal Amplitude**, and **Quantizer Type** affect the quality of a digital signal.
""")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. ADC Settings")
    N = st.slider("Bit Depth (N)", 2, 16, 4, help="Number of bits determines the number of discrete levels.")
    quant_type = st.radio("Quantizer Type", ["Mid-Tread", "Mid-Rise"], help="Mid-Tread has a zero level. Mid-Rise does not.")
    
    st.divider()
    
    st.header("2. Signal Settings")
    sig_type = st.selectbox("Signal Shape", ["Sine Wave", "Uniform Random", "Gaussian"])
    c = st.slider("Backoff Factor (c)", 0.1, 10.0, 1.0, 0.1, help="Higher c = smaller signal. c < 1 causes clipping.")
    
    # Advanced settings in expander
    with st.expander("Advanced"):
        f_hz = st.number_input("Frequency (Hz)", 1.0, 1000.0, 5.0)
        fs = 10000  # Fixed high sampling rate for simulation

# --- Logic & Computation ---

# 1. Setup Time and Range
duration = 1.0
t = np.linspace(0, duration, int(fs*duration), endpoint=False)
A_full_scale = 1.0 # The ADC range is -0.5 to +0.5 (Width A=1.0)

# 2. Generate Signal (Amplitude = A_full_scale / (2*c))
amp = A_full_scale / (2 * c)

if sig_type == "Sine Wave":
    x_clean = amp * np.sin(2 * np.pi * f_hz * t)
    # Theoretical SNR for Sine: 6.02N + 1.76 dB (minus backoff)
    theory_constant = 1.76 
elif sig_type == "Uniform Random":
    x_clean = (np.random.rand(len(t)) - 0.5) * (1.0/c) # Scaled uniform
    # Theoretical SNR for Uniform: 6.02N dB (minus backoff)
    theory_constant = 0.0
else: # Gaussian
    x_clean = np.random.normal(0, amp/3, len(t)) # approx fits in 3-sigma
    theory_constant = None # Hard to predict exact PDF match

# 3. Quantization Logic
L = 2**N
Delta = A_full_scale / L

# Normalize to step size
x_scaled = x_clean / Delta

if quant_type == "Mid-Tread":
    # Round to nearest integer (includes 0)
    x_int = np.floor(x_scaled + 0.5)
else:
    # Mid-Rise: Round to nearest half-integer (no 0)
    x_int = np.floor(x_scaled) + 0.5

# Re-scale and Simulate ADC Clipping (Saturation)
# ADC cannot output values outside its range
max_idx = (L // 2) - 1 if quant_type == "Mid-Tread" else (L // 2)
min_idx = -(L // 2)

x_int_clipped = np.clip(x_int, min_idx, max_idx)
x_quant = x_int_clipped * Delta

# 4. Error & SNR Calculation
error = x_clean - x_quant
signal_power = np.mean(x_clean**2)
noise_power = np.mean(error**2)

if noise_power > 0:
    measured_snr = 10 * np.log10(signal_power / noise_power)
else:
    measured_snr = 999.9 # Infinite SNR (perfect match)

# Theoretical SNR (valid only if not clipping)
if c >= 1.0 and theory_constant is not None:
    theory_snr = (6.02 * N) + theory_constant - (20 * np.log10(c))
else:
    theory_snr = None # Theory breaks down during clipping

# --- Dashboard Layout ---

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Step Size (Δ)", f"{Delta:.4f}", f"{L} Levels")
col2.metric("Signal Power", f"{signal_power:.4f}")
col3.metric("Measured SNR", f"{measured_snr:.2f} dB")
if theory_snr:
    col4.metric("Theoretical SNR", f"{theory_snr:.2f} dB", f"{measured_snr - theory_snr:.2f} diff")
else:
    col4.metric("Theoretical SNR", "N/A (Clipping/PDF)")

# Plots
tab1, tab2 = st.tabs(["Time Domain & Histogram", "Frequency Spectrum"])

with tab1:
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Time Domain Plot (Zoomed)
        fig_time, ax_time = plt.subplots(figsize=(10, 5))
        zoom_samples = 200
        ax_time.plot(t[:zoom_samples], x_clean[:zoom_samples], label="Analog Input", color='tab:blue', alpha=0.6, linewidth=2)
        ax_time.step(t[:zoom_samples], x_quant[:zoom_samples], label="Digital Output", color='tab:red', where='mid')
        ax_time.set_title("Quantization Steps (Zoomed)", fontsize=14)
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        st.pyplot(fig_time)

    with c2:
        # Error Histogram
        fig_hist, ax_hist = plt.subplots(figsize=(5, 5))
        ax_hist.hist(error, bins=50, density=True, color='gray', alpha=0.7)
        ax_hist.axvline(Delta/2, color='red', linestyle='--', label='+Δ/2')
        ax_hist.axvline(-Delta/2, color='red', linestyle='--', label='-Δ/2')
        ax_hist.set_title("Error Distribution (PDF)", fontsize=14)
        ax_hist.set_xlabel("Quantization Error")
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        st.pyplot(fig_hist)

with tab2:
    # Frequency Spectrum (FFT)
    fig_fft, ax_fft = plt.subplots(figsize=(12, 5))
    
    N_fft = len(x_clean)
    yf_signal = fft(x_clean)
    yf_noise = fft(error)
    xf = fftfreq(N_fft, 1/fs)[:N_fft//2]
    
    # Normalize
    mag_signal = 2.0/N_fft * np.abs(yf_signal[0:N_fft//2])
    mag_noise = 2.0/N_fft * np.abs(yf_noise[0:N_fft//2])
    
    ax_fft.plot(xf, 20*np.log10(mag_signal+1e-12), label="Signal", alpha=0.7)
    ax_fft.plot(xf, 20*np.log10(mag_noise+1e-12), label="Quantization Noise Floor", alpha=0.7, color='red', linewidth=0.5)
    
    ax_fft.set_title("Frequency Spectrum (Notice the Noise Floor drop as N increases)", fontsize=14)
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude (dB)")
    ax_fft.legend(loc='upper right')
    ax_fft.grid(True, alpha=0.3)
    ax_fft.set_ylim(-120, 10)
    st.pyplot(fig_fft)

# --- Educational Expander ---
with st.expander("📚 Why doesn't the Theory match the Measurement exactly?"):
    st.markdown(r"""
    The "Rule of Thumb" formula ($SNR \approx 6.02N$) assumes the signal is **Uniformly Distributed** across the quantization steps.
    
    1.  **Sine Waves:** A sine wave spends more time at the peaks than near zero. This shape actually results in a theoretical SNR of **$6.02N + 1.76$ dB**.
    2.  **Clipping:** If you lower the **Backoff (c)** below 1.0, the signal exceeds the ADC's range. This introduces massive distortion, causing the SNR to plummet far below the theoretical prediction.
    3.  **Backoff:** If you increase **c**, you aren't using the full range of bits. Every factor of 2 in backoff costs you roughly **1 bit** (6 dB) of resolution.
    """)





# import streamlit as st
# import numpy as np
# import matplotlib.pyplot as plt

# # --- Page Config: Wide mode by default ---
# st.set_page_config(
#     page_title="Quantization SNR",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# # --- CSS: Remove padding & UI chrome for embedding ---
# st.markdown("""
#     <style>
#         /* Remove top padding to sit flush in an iframe */
#         .block-container {
#             padding-top: 1rem;
#             padding-bottom: 1rem;
#             padding-left: 2rem;
#             padding-right: 2rem;
#         }
#         /* Hide the Streamlit main menu and footer */
#         #MainMenu {visibility: hidden;}
#         footer {visibility: hidden;}
#         header {visibility: hidden;}
#     </style>
# """, unsafe_allow_html=True)

# # --- Constants ---
# A = 1.0

# # --- Controls (Top Row) ---
# # We use a container with a border to group controls visually
# with st.container(border=True):
#     st.markdown("**⚙️ Signal Settings**")
#     col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    
#     with col_ctrl1:
#         N = st.slider("Bits (N)", 1, 16, 8, 1)
#     with col_ctrl2:
#         c = st.slider("Backoff (c)", 0.01, 10.0, 1.0, 0.01, help="Amplitude = A/c")
#     with col_ctrl3:
#         f = st.slider("Frequency (Hz)", 0.1, 2000.0, 5.0, 0.1)

# # --- Computation ---
# N_safe = int(max(1, min(24, N)))
# c_safe = float(max(1e-6, c))
# f_safe = float(max(0.1, min(2000.0, f)))

# t = np.linspace(0.0, 1.0, 4000, endpoint=False)
# x = (A / (2.0 * c_safe)) * np.sin(2.0 * np.pi * f_safe * t)

# Delta = A / (2 ** N_safe)
# xq = Delta * np.floor(x / Delta + 0.5)
# e = x - xq

# e_power = np.mean(e ** 2) if np.mean(e ** 2) > 0 else 1e-18
# snr_db = 10.0 * np.log10(np.mean(x ** 2) / e_power) if np.mean(x ** 2) > 0 else -np.inf
# theory_db = 6.02 * N_safe - 20.0 * np.log10(c_safe)

# # --- Metrics Row ---
# m1, m2, m3, m4 = st.columns(4)
# m1.metric("Res / Step Size", f"{Delta:.5f}")
# m2.metric("Levels", f"{2**N_safe}")
# m3.metric("Measured SNR", f"{snr_db:.2f} dB")
# m4.metric("Theory SNR", f"{theory_db:.2f} dB", delta=f"{snr_db-theory_db:.2f} diff")

# # --- Plotting Setup ---
# # Set style to dark_background or default depending on your site, 
# # but here we use standard with transparent background for blending.
# plt.rcParams.update({'figure.autolayout': True})

# def style_plot(fig, ax):
#     # Make plot transparent to blend with embedding site background
#     fig.patch.set_alpha(0)
#     ax.patch.set_alpha(0)
#     # Style tweaks
#     ax.grid(True, alpha=0.3, linestyle='--')
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)

# # Plot 1: Time
# fig1, ax1 = plt.subplots(figsize=(6, 3))
# n_zoom = 400
# ax1.plot(t[:n_zoom], x[:n_zoom], label="Input", lw=1.5, color="#00a8cc")
# ax1.step(t[:n_zoom], xq[:n_zoom], where="mid", label="Quantized", lw=1.2, color="#ff4b4b")
# ax1.set_title(f"Time Domain (Zoom)", fontsize=10, loc='left', color="gray")
# ax1.legend(loc="upper right", frameon=False, fontsize='small')
# style_plot(fig1, ax1)

# # Plot 2: Histogram
# fig2, ax2 = plt.subplots(figsize=(6, 3))
# ax2.hist(e, bins=60, density=True, color="#555555", alpha=0.7)
# ax2.axvline(+Delta/2, color="#ff4b4b", ls=":", lw=1.5)
# ax2.axvline(-Delta/2, color="#ff4b4b", ls=":", lw=1.5)
# ax2.set_title("Error Histogram", fontsize=10, loc='left', color="gray")
# ax2.set_yticks([]) # Hide y-ticks for cleaner look
# style_plot(fig2, ax2)

# # --- Visual Layout ---
# p1, p2 = st.columns(2)
# with p1:
#     st.pyplot(fig1, use_container_width=True)
# with p2:
#     st.pyplot(fig2, use_container_width=True)




# # import streamlit as st
# # import numpy as np
# # import matplotlib.pyplot as plt

# # # --- Page Configuration ---
# # st.set_page_config(
# #     page_title="Quantization SNR Demo",
# #     page_icon="🎚️",
# #     layout="wide"
# # )

# # # --- Constants ---
# # A = 1.0  # full-scale width used to define Delta

# # # --- Sidebar: Controls ---
# # st.sidebar.header("Configuration")

# # # Streamlit re-runs the script on every widget interaction,
# # # so we define inputs directly here.
# # N = st.sidebar.slider("Bits (N)", min_value=1, max_value=16, value=8, step=1)
# # c = st.sidebar.slider("Backoff c (Amplitude = A/c)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
# # f = st.sidebar.slider("Sine frequency [Hz]", min_value=0.1, max_value=2000.0, value=5.0, step=0.1)

# # # --- Logic / Computation ---
# # # (Logic copied from original script)
# # N_safe = int(max(1, min(24, N)))
# # c_safe = float(max(1e-6, c))
# # f_safe = float(max(0.1, min(2000.0, f)))

# # # Time axis and sine input with backoff c
# # t = np.linspace(0.0, 1.0, 4000, endpoint=False)
# # x = (A / (2.0 * c_safe)) * np.sin(2.0 * np.pi * f_safe * t)

# # # Uniform mid-tread quantizer
# # Delta = A / (2 ** N_safe)  # step size
# # xq = Delta * np.floor(x / Delta + 0.5)
# # e = x - xq

# # # SNR (measured) and theory
# # e_power = np.mean(e ** 2) if np.mean(e ** 2) > 0 else 1e-18
# # snr_db = 10.0 * np.log10(np.mean(x ** 2) / e_power) if np.mean(x ** 2) > 0 else -np.inf
# # theory_db = 6.02 * N_safe - 20.0 * np.log10(c_safe)

# # # --- Main Layout ---
# # st.title("🎚️ Quantization SNR — Mid-tread Uniform Quantizer")
# # st.markdown(
# #     "Adjust the **bits (N)**, **backoff (c)**, and **frequency** in the sidebar to see "
# #     "time-domain quantization, error histograms, and SNR calculations."
# # )

# # st.divider()

# # # Display Metrics using Streamlit's native metric component
# # col1, col2, col3, col4 = st.columns(4)
# # with col1:
# #     st.metric(label="Bits (N)", value=N_safe)
# # with col2:
# #     st.metric(label="Step Size (Delta)", value=f"{Delta:.6f}")
# # with col3:
# #     st.metric(label="Measured SNR", value=f"{snr_db:.2f} dB")
# # with col4:
# #     st.metric(label="Theory SNR", value=f"{theory_db:.2f} dB", delta=f"{snr_db - theory_db:.2f} diff")

# # st.divider()

# # # --- Plotting ---
# # # We use st.columns to arrange plots side-by-side or stacked.
# # # Given the detailed nature, stacked (vertical) often works best,
# # # or side-by-side on wide screens.

# # # Plot 1: Time-domain
# # fig1, ax1 = plt.subplots(figsize=(8, 3.5))
# # n_zoom = 400
# # ax1.plot(t[:n_zoom], x[:n_zoom], label="x(t): input", lw=1.6)
# # ax1.step(t[:n_zoom], xq[:n_zoom], where="mid", label="Q(x): quantized", lw=1.0)
# # ax1.set_title(f"Time-domain quantization (First {n_zoom} samples)")
# # ax1.set_xlabel("Time [s]")
# # ax1.set_ylabel("Amplitude")
# # ax1.grid(alpha=0.3)
# # ax1.legend(loc="upper right")
# # fig1.tight_layout()

# # # Plot 2: Error histogram
# # fig2, ax2 = plt.subplots(figsize=(8, 3.5))
# # ax2.hist(e, bins=60, density=True, color="gray", edgecolor="none")
# # ax2.axvline(+Delta / 2.0, color="r", ls="--", lw=1)
# # ax2.axvline(-Delta / 2.0, color="r", ls="--", lw=1)
# # ax2.set_title("Quantization error distribution e = x - Q(x)")
# # ax2.set_xlabel("Error")
# # ax2.set_ylabel("Probability density")
# # ax2.text(0.02, 0.95, f"Delta = {Delta:.6f}\nBounds ±Delta/2", transform=ax2.transAxes, va="top")
# # ax2.grid(alpha=0.2)
# # fig2.tight_layout()

# # # Render plots
# # left_plot, right_plot = st.columns(2)
# # with left_plot:
# #     st.pyplot(fig1)
# # with right_plot:
# #     st.pyplot(fig2)

# # # --- Interpretation / Details ---
# # with st.expander("ℹ️ Interpretation & Details", expanded=True):
# #     st.markdown(f"""
# #     * **Levels L:** {2**N_safe}
# #     * **Backoff c:** {c_safe:.4f}
# #     * **Theoretical Formula:** $$SNR \\approx 6.02 N - 20 \\log_{{10}}(c)$$
    
# #     **Key Takeaways:**
# #     1.  Increasing **N** lowers $\\Delta$ and increases SNR (~6 dB/bit at full scale).
# #     2.  Increasing **c** (more backoff from full-scale) reduces SNR.
# #     """)
