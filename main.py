import speech_recognition as srec
import speech_recognition as srec
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly, butter, sosfiltfilt
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100
SAMPLE_WIDTH = 2
DTYPE = np.int16
NAME_ORIGINAL_WAV = f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].wav"
NAME_ORIGINAL_RAW = f"./Sounds/Sound_{SAMPLE_RATE}[Hz]_{SAMPLE_WIDTH}[byte].raw"
NAME_RESAMPLED_WAV = "./Sounds/Sound_4000[Hz]_2[byte].wav"
NAME_RESAMPLED_RAW = "./Sounds/Sound_4000[Hz]_2[byte].raw"
NAME_FILTERED_WAV = "./Sounds/Filtered_4000[Hz]_2[byte].wav"
NAME_FILTERED_RAW = "./Sounds/Filtered_4000[Hz]_2[byte].raw"

def sound_recoder(rec, mic):
    with mic as source:
        print("Говоріть...")
        audio = rec.listen(source)

    wav_data = audio.get_wav_data(
        convert_rate=SAMPLE_RATE,
        convert_width=SAMPLE_WIDTH
    )

    raw_data = audio.get_raw_data(
        convert_rate=SAMPLE_RATE,
        convert_width=SAMPLE_WIDTH
    )

    with open(NAME_ORIGINAL_WAV, "wb") as f:
        f.write(wav_data)

    with open(NAME_ORIGINAL_RAW, "wb") as f:
        f.write(raw_data)

    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    fs_target = 4000
    g = gcd(fs_original, fs_target)
    up = fs_target // g
    down = fs_original // g
    data_resampled = resample_poly(data, up, down)
    sf.write(NAME_RESAMPLED_WAV, data_resampled, fs_target)

    with open(NAME_ORIGINAL_RAW, "rb") as f:
        raw_bytes = f.read()
    signal = np.frombuffer(raw_bytes, dtype=DTYPE)
    signal_float = signal.astype(np.float32) / 32768.0
    g = gcd(fs_original, fs_target)
    up = fs_target // g
    down = fs_original // g
    resampled = resample_poly(signal_float, up, down)
    resampled_int16 = np.int16(resampled * 32767)
    with open(NAME_RESAMPLED_RAW, "wb") as f:
        f.write(resampled_int16.tobytes())

    data, fs_original = sf.read(NAME_ORIGINAL_WAV)
    if len(data.shape) > 1:
        data = data[:, 0]
    cutoff = 4000
    order = 6
    sos = butter(order, cutoff, btype='low', fs=SAMPLE_RATE, output='sos')
    filtered = sosfiltfilt(sos, data)
    sf.write(NAME_FILTERED_WAV, filtered, SAMPLE_RATE)

    with open(NAME_ORIGINAL_RAW, "rb") as f:
        raw_bytes = f.read()
    signal = np.frombuffer(raw_bytes, dtype=DTYPE)
    signal_float = signal.astype(np.float32) / 32768.0
    sos = butter(order, cutoff, btype='low', fs=SAMPLE_RATE, output='sos')
    filtered_raw = sosfiltfilt(sos, signal_float)
    filtered_int16 = np.int16(filtered_raw * 32767)
    with open(NAME_FILTERED_RAW, "wb") as f:
        f.write(filtered_int16.tobytes())

    data, fs = sf.read(NAME_ORIGINAL_WAV)
    time = np.arange(len(data)) / fs
    plt.figure(figsize=(12, 6))
    plt.plot(time, data, label=f"Оригінал (fs={SAMPLE_RATE} Гц)")
    data, fs = sf.read(NAME_RESAMPLED_WAV)
    time = np.arange(len(data)) / fs
    plt.plot(time, data, label=f"Ресемпл (fs={fs} Гц)")
    data, fs = sf.read(NAME_FILTERED_WAV)
    time = np.arange(len(data)) / fs
    plt.plot(time, data, label=f"Фільтрований (LPF {cutoff} Гц)")
    plt.title("Порівняння сигналів у часовій області")
    plt.xlabel("Час (мс)")
    plt.ylabel("Амплітуда")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Sounds/Графік порівняння мовних сигналів з різними частотами дискретизації.png", dpi=600)
    plt.show()
    

def recognize_speech(rec, mic):
    with mic as source:
        rec.adjust_for_ambient_noise(source)
        audio = rec.listen(source)
    result = {"Текст": None}
    result["Текст"] = rec.recognize_google(audio, show_all=False, language="uk-UA")
    return result

if __name__ == "__main__":
    recognizer = srec.Recognizer()
    microphone = srec.Microphone()

    sound_recoder(recognizer, microphone)
