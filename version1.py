import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import wave
import io

# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Fourier Audio Lab",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Fourier Audio Lab")
st.markdown(
    "### Upload audio → Fourier analysis → Noise filtering → Clean audio"
)

# --------------------------------------------------
# WAV READING
# --------------------------------------------------

def read_wav(file):
    with wave.open(file, "rb") as wav:

        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.getnframes()

        raw_data = wav.readframes(frames)

    # Convert audio bytes to numbers
    if sample_width == 1:
        audio = np.frombuffer(
            raw_data,
            dtype=np.uint8
        ).astype(np.float32)

        audio = audio - 128

    elif sample_width == 2:
        audio = np.frombuffer(
            raw_data,
            dtype=np.int16
        ).astype(np.float32)

    elif sample_width == 4:
        audio = np.frombuffer(
            raw_data,
            dtype=np.int32
        ).astype(np.float32)

    else:
        st.error("Unsupported WAV format.")
        return None, None

    # Stereo → Mono
    if channels > 1:
        audio = audio.reshape(-1, channels)
        audio = np.mean(audio, axis=1)

    # Normalize
    max_value = np.max(np.abs(audio))

    if max_value != 0:
        audio = audio / max_value

    return audio, sample_rate


# --------------------------------------------------
# CREATE WAV FOR PLAYBACK
# --------------------------------------------------

def create_wav(audio, sample_rate):

    audio = np.clip(audio, -1, 1)

    audio_int = (audio * 32767).astype(np.int16)

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        wav.writeframes(audio_int.tobytes())

    return buffer.getvalue()


# --------------------------------------------------
# DEMO AUDIO
# --------------------------------------------------

def create_demo_audio():

    sample_rate = 44100

    duration = 5

    t = np.linspace(
        0,
        duration,
        int(sample_rate * duration),
        endpoint=False
    )

    # Low frequency component
    bass = 0.5 * np.sin(2 * np.pi * 120 * t)

    # Mid frequency component
    voice = 0.35 * np.sin(2 * np.pi * 440 * t)

    # High frequency noise
    noise = 0.15 * np.random.randn(len(t))

    audio = bass + voice + noise

    audio = audio / np.max(np.abs(audio))

    return audio, sample_rate


# --------------------------------------------------
# AUDIO INPUT
# --------------------------------------------------

st.subheader("🎙️ 1. Choose Your Audio")

uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav"]
)

use_demo = st.button("🎵 Use Demo Audio")

audio = None
sample_rate = None

if uploaded_file is not None:

    audio, sample_rate = read_wav(uploaded_file)

elif use_demo:

    audio, sample_rate = create_demo_audio()


# --------------------------------------------------
# PROCESS AUDIO
# --------------------------------------------------

if audio is not None:

    st.success("Audio loaded successfully!")

    # ----------------------------------------------
    # ORIGINAL AUDIO
    # ----------------------------------------------

    st.subheader("▶️ Original Audio")

    original_wav = create_wav(
        audio,
        sample_rate
    )

    st.audio(
        original_wav,
        format="audio/wav"
    )

    # ----------------------------------------------
    # TIME DOMAIN
    # ----------------------------------------------

    st.subheader("📈 2. Time-Domain Waveform")

    time = np.arange(len(audio)) / sample_rate

    fig, ax = plt.subplots(
        figsize=(12, 3)
    )

    ax.plot(time, audio)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")

    ax.set_title(
        "Original Audio Waveform"
    )

    ax.grid(True)

    st.pyplot(fig)

    # ----------------------------------------------
    # FOURIER TRANSFORM
    # ----------------------------------------------

    st.subheader("🔬 3. Fourier Analysis")

    st.write(
        "The audio is converted from the time domain "
        "into its frequency components using the FFT."
    )

    fft_result = np.fft.rfft(audio)

    frequencies = np.fft.rfftfreq(
        len(audio),
        1 / sample_rate
    )

    magnitude = np.abs(fft_result)

    # ----------------------------------------------
    # FREQUENCY SPECTRUM
    # ----------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    ax.plot(
        frequencies,
        magnitude
    )

    ax.set_xlim(0, 10000)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

    ax.set_title(
        "Frequency Spectrum"
    )

    ax.grid(True)

    st.pyplot(fig)

    # ----------------------------------------------
    # FILTER
    # ----------------------------------------------

    st.subheader("🔧 4. Noise Filter")

    cutoff = st.slider(
        "Choose High-Frequency Cutoff",
        min_value=500,
        max_value=10000,
        value=3000,
        step=100
    )

    st.write(
        f"Frequencies above **{cutoff} Hz** "
        "will be attenuated."
    )

    # Frequency mask
    filter_mask = frequencies <= cutoff

    filtered_fft = fft_result * filter_mask

    # ----------------------------------------------
    # INVERSE FOURIER TRANSFORM
    # ----------------------------------------------

    filtered_audio = np.fft.irfft(
        filtered_fft,
        n=len(audio)
    )

    # Normalize
    max_value = np.max(
        np.abs(filtered_audio)
    )

    if max_value != 0:
        filtered_audio = (
            filtered_audio / max_value
        )

    # ----------------------------------------------
    # FILTER WEIGHT GRAPH
    # ----------------------------------------------

    st.subheader("⚙️ Fourier Filter Weights")

    fig, ax = plt.subplots(
        figsize=(12, 3)
    )

    weights = filter_mask.astype(float)

    ax.plot(
        frequencies,
        weights
    )

    ax.set_xlim(0, 10000)

    ax.set_ylim(-0.1, 1.1)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Weight W(f)")

    ax.set_title(
        "Frequency Weighting Function"
    )

    ax.grid(True)

    st.pyplot(fig)

    # ----------------------------------------------
    # FILTERED AUDIO
    # ----------------------------------------------

    st.subheader("✨ 5. Cleaned Audio")

    filtered_wav = create_wav(
        filtered_audio,
        sample_rate
    )

    st.audio(
        filtered_wav,
        format="audio/wav"
    )

    # ----------------------------------------------
    # BEFORE / AFTER SPECTRUM
    # ----------------------------------------------

    st.subheader(
        "📊 6. Before vs After Filtering"
    )

    fig, ax = plt.subplots(
        figsize=(12, 4)
    )

    filtered_magnitude = np.abs(
        filtered_fft
    )

    ax.plot(
        frequencies,
        magnitude,
        label="Original"
    )

    ax.plot(
        frequencies,
        filtered_magnitude,
        label="Filtered"
    )

    ax.set_xlim(0, 10000)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")

    ax.set_title(
        "Fourier Spectrum Comparison"
    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)

    # ----------------------------------------------
    # MATHEMATICAL BACKGROUND
    # ----------------------------------------------

    st.subheader(
        "🧮 7. Mathematical Background"
    )

    st.latex(
        r"""
        f(x)=\frac{a_0}{2}
        +\sum_{n=1}^{\infty}
        \left[
        a_n\cos(nx)+b_n\sin(nx)
        \right]
        """
    )

    st.write(
        "Fourier analysis represents a signal as "
        "a combination of sinusoidal components."
    )

    st.latex(
        r"""
        X[k]=
        \sum_{n=0}^{N-1}
        x[n]e^{-j2\pi kn/N}
        """
    )

    st.write(
        "For digital audio, the Fast Fourier Transform "
        "(FFT) efficiently calculates these frequency "
        "components."
    )

    st.latex(
        r"""
        X_{\mathrm{filtered}}[k]
        =
        W[k]X[k]
        """
    )

    st.write(
        "Here W[k] acts as the frequency weight. "
        "Frequencies below the cutoff are kept, "
        "while higher frequencies are attenuated."
    )

    # ----------------------------------------------
    # STATISTICS
    # ----------------------------------------------

    st.subheader("📋 8. Analysis")

    original_energy = np.mean(
        audio ** 2
    )

    filtered_energy = np.mean(
        filtered_audio ** 2
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Sample Rate",
            f"{sample_rate} Hz"
        )

    with col2:
        st.metric(
            "Cutoff Frequency",
            f"{cutoff} Hz"
        )

    with col3:
        st.metric(
            "Original Energy",
            f"{original_energy:.4f}"
        )

else:

    st.info(
        "👆 Upload a WAV file or click "
        "**Use Demo Audio** to start."
    )