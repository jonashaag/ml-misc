"""Label a folder of .wav files in order to train a boolean classifier.
Creates a foo.label file for each foo.wav file in the folder.
"""

import sys
import os
import librosa
import glob
import random
import pyloudnorm
import json
from pathlib import Path

import numpy as np
from audiostream import get_output
from audiostream.sources.thread import ThreadSource


class MonoAmplitudeSource(ThreadSource):
    """A data source for float32 mono binary data, as loaded by libROSA/soundfile."""

    def __init__(self, stream, data, *args, **kwargs):
        super().__init__(stream, *args, **kwargs)
        self.chunksize = kwargs.get("chunksize", 64)
        self.data = data
        self.cursor = 0

    def get_bytes(self):
        chunk = self.data[self.cursor : self.cursor + self.chunksize]
        self.cursor += self.chunksize

        if not isinstance(chunk, np.ndarray):
            chunk = np.array(chunk)
        assert len(chunk.shape) == 1 and chunk.dtype in (
            np.dtype("float32"),
            np.dtype("float64"),
        )

        # Convert to 16 bit format.
        return (chunk * 2 ** 15).astype("int16").tobytes()


class MySource(MonoAmplitudeSource):
    def set_data(self, data):
        self.data = data
        self.cursor = 0


def getsample():
    DS_DIR = Path(sys.argv[1]).expanduser()
    chunks = list(DS_DIR.glob("*.wav"))
    random.shuffle(chunks)
    for chunk in chunks:
        if not labelfile(chunk).exists():
            break
    else:
        raise RuntimeError("No more chunks")
    return (
        chunk,
        pyloudnorm.normalize.peak(
            librosa.core.load(chunk, sr=audio_sr, res_type="kaiser_fast")[0],
            -1,
        ),
    )


def wait_keyboard(expected_keys):
    from pynput import keyboard

    with keyboard.Events() as events:
        while True:
            event = events.get()
            if isinstance(event, keyboard.Events.Press):
                continue
            name = str(getattr(event.key, "name", event.key))
            if name in expected_keys:
                return name
            else:
                print("Unexpected input", name)


def labelfile(f):
    return f.with_suffix(".label")


# ---


import threading, queue

audio_sr = 44100
stream = get_output(channels=1, rate=audio_sr, buffersize=1024)
source = MySource(stream, [])

cur_f = getsample()
source.set_data(cur_f[1])
source.start()
prev_samples = []

randsample_q = queue.Queue(maxsize=2)


def queuer():
    while True:
        randsample_q.put(getsample())


queuer_t = threading.Thread(target=queuer)
queuer_t.daemon = True
queuer_t.start()

while True:
    source.set_data(cur_f[1])
    print("Labeling", labelfile(cur_f[0]))
    key = wait_keyboard(["left", "right", "up", "down", "space"])
    if key in ("left", "right", "down"):
        # Left = no, Right = yes, Down = skip this file
        print("Writing to", labelfile(cur_f[0]))
        labelfile(cur_f[0]).open("w").write(
            {"right": "yes", "left": "no", "down": "skip"}[key]
        )
        prev_samples.append(cur_f)
        cur_f = randsample_q.get()
    else:  # up
        # Up = Undo last label
        cur_f = prev_samples.pop()
        os.unlink(labelfile(cur_f[0]))

source.stop()
