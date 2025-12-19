import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless for Spaces
import matplotlib.pyplot as plt
import gradio as gr

A = 1.0  # full-scale width used to define Delta

def run_demo(N=8, c=1.0, f=5.0):
    # Safety guards
    N = int(max(1, min(24, N)))
    c = float(max(1e-6, c))
    f = float(max(0.1, min(2000.0, f)))

    # Time axis and sine input with backoff c
    t = np.linspace(0.0, 1.0, 4000, endpoint=False)
    x = (A / (2.0 * c)) * np.sin(2.0 * np.pi * f * t)

    # Uniform mid-tread quantizer
    Delta = A / (2 ** N)  # step size
    xq = Delta * np.floor(x / Delta + 0.5)
    e = x - xq

    # SNR (measured) and theory
    # guard against divide-by-zero when e ≈ 0
    e_power = np.mean(e ** 2) if np.mean(e ** 2) > 0 else 1e-18
    snr_db = 10.0 * np.log10(np.mean(x ** 2) / e_power) if np.mean(x ** 2) > 0 else -np.inf
    theory_db = 6.02 * N - 20.0 * np.log10(c)

    # --- Plot 1: time-domain (zoom first 400 samples) ---
    fig1, ax1 = plt.subplots(figsize=(7, 3))
    n_zoom = 400
    ax1.plot(t[:n_zoom], x[:n_zoom], label="x(t): input", lw=1.6)
    ax1.step(t[:n_zoom], xq[:n_zoom], where="mid", label="Q(x): quantized", lw=1.0)
    ax1.set_title(f"Time-domain quantization (N={N}, c={c:.2f}, f={f:.1f} Hz)")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")
    fig1.tight_layout()

    # --- Plot 2: error histogram with ±Delta/2 markers ---
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.hist(e, bins=60, density=True, color="gray", edgecolor="none")
    ax2.axvline(+Delta / 2.0, color="r", ls="--", lw=1)
    ax2.axvline(-Delta / 2.0, color="r", ls="--", lw=1)
    ax2.set_title("Quantization error e = x - Q(x)")
    ax2.set_xlabel("Error")
    ax2.set_ylabel("Probability density")
    ax2.text(0.02, 0.95, f"Delta = {Delta:.6f}\nBounds ±Delta/2", transform=ax2.transAxes, va="top")
    ax2.grid(alpha=0.2)
    fig2.tight_layout()

    # Readout panel
    readout = (
        f"Bits N          : {N}\n"
        f"Backoff c       : {c:.4f}  (Amplitude = A/c)\n"
        f"Levels L        : {2**N}\n"
        f"Step size Delta : {Delta:.8f}\n"
        f"Measured SNR    : {snr_db:.2f} dB\n"
        f"Theory SNR      : {theory_db:.2f} dB  (≈ 6.02·N − 20·log10(c))\n\n"
        "Interpretation:\n"
        "- Increasing N lowers Delta and increases SNR (~6 dB/bit at full scale).\n"
        "- Increasing c (more backoff from full-scale) reduces SNR by 20·log10(c) dB."
    )

    return readout, fig1, fig2

with gr.Blocks(title="Quantization SNR — Mid-tread Uniform Quantizer") as demo:
    gr.Markdown("## 🎚️ Quantization SNR — Mid-tread Uniform Quantizer")
    gr.Markdown(
        "Adjust the **bits (N)**, **backoff (c)** (how much of full-scale the signal uses), "
        "and **frequency** to see time-domain quantization, error histogram, and SNR."
    )

    with gr.Row():
        with gr.Column(scale=1):
            N = gr.Slider(1, 16, value=8, step=1, label="Bits (N)")
            c = gr.Slider(0.01, 10.0, value=1.0, step=0.01, label="Backoff c (Amplitude = A/c)")
            f = gr.Slider(0.1, 2000.0, value=5.0, step=0.1, label="Sine frequency [Hz]")
            run = gr.Button("Update")
        with gr.Column(scale=2):
            readouts = gr.Textbox(label="Readouts", lines=10)
            plot_time = gr.Plot(label="Time-domain (zoom)")
            plot_err = gr.Plot(label="Error histogram")

    # initial run + reactive updates
    run.click(run_demo, inputs=[N, c, f], outputs=[readouts, plot_time, plot_err])
    for w in (N, c, f):
        w.change(run_demo, inputs=[N, c, f], outputs=[readouts, plot_time, plot_err])

demo.launch()