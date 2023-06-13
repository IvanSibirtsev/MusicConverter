import math
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
from src import pyin


def show(y, times, f0):
    db_spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(db_spectrogram, x_axis='time', y_axis='log', ax=ax)
    ax.set(title='Fundamental frequency estimation')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
    ax.legend(loc='upper right')
    plt.show()


def merge(array: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged = []
    delta = 3
    index = 0
    prev_pitch = array[0][0]
    prev_time = array[0][1]
    while index < len(array):
        pitch, time = array[index]
        if time <= 0 or math.isnan(pitch):
            index += 1
            continue

        if abs(pitch - prev_pitch) <= delta and len(merged) > 1:
            merged[-1] = (prev_pitch, time + prev_time)
            prev_time = time + prev_time
        else:
            merged.append((pitch, time))
            prev_time = time
            prev_pitch = pitch

        index += 1

    return merged


def filtering(array: list[tuple[float, float]]) -> list[tuple[float, float]]:
    filtered = []
    index = 0
    while index < len(array):
        pitch, time = array[index]
        if time < 20 or pitch == 0:
            index += 1
            continue
        filtered.append((pitch, time * 2))
        index += 1
    return filtered


def main():
    path = sys.argv[1]
    audio, _ = librosa.load(path)
    f0, times, _ = pyin.pyin(audio, min_frequency=librosa.note_to_hz('C2'), max_frequency=librosa.note_to_hz('C7'))

    array = []
    for tup in zip(f0, times):
        array.append(tup)

    merged = merge(array)
    filtered = filtering(merged)
    strings = []

    for tup in filtered:
        strings.append(f"tone(13, {int(tup[0])}, {int(tup[1])})")

    with open("music.txt", "w") as music:
        music.write("\n".join(strings))


if __name__ == "__main__":
    main()
