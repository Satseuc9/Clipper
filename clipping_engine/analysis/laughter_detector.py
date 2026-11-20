"""
Sound-event detection with YAMNet (AudioSet classes) via the LiteRT runtime.

YAMNet takes 0.975 s windows of 16 kHz mono audio and returns 521 class
scores. We slide it with a 0.48 s hop and group classes into the reactions
that matter for clips. Measured on ESC-50 clips: laughter scores 0.15-0.59
on laughing recordings and ~0 on clapping, rain, typing, dogs, footsteps and
speech (one crying-baby clip reached 0.33), which is why laughter alone is
never enough to select a moment.

Limitations: YAMNet cannot count people, so "multiple people laughing" is
approximated by sustained laughter together with crowd/babble classes.
"""

import zipfile

import numpy as np

from clipping_engine.analysis.model_files import model_path

RATE = 16_000
WINDOW = 15_600          # samples per YAMNet frame (0.975 s)
HOP = 7_680              # 0.48 s

GROUPS = {
    "laughter": ["Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"],
    "scream": ["Screaming", "Shout", "Yell", "Bellow", "Children shouting", "Whoop"],
    "cheer": ["Cheering", "Applause", "Clapping"],
    "crowd": ["Crowd", "Hubbub, speech noise, speech babble", "Chatter"],
    "gasp": ["Gasp"],
    "cry": ["Crying, sobbing", "Wail, moan", "Groan"],
    "speech": ["Speech", "Conversation", "Narration, monologue"],
    "music": ["Music"],
    "silence": ["Silence"],
}
# Classes never counted as "unusual" sounds (they are the normal stream background).
COMMON = set(sum(GROUPS.values(), [])) | {
    "Video game music", "Sound effect", "Inside, small room", "Inside, large room or hall", "Television",
    "White noise", "Pink noise", "Static", "Noise", "Hum", "Mains hum", "Buzz", "Distortion",
    "Environmental noise", "Echo", "Reverberation", "Radio", "Background music",
}

_model = None


class YamNet:
    def __init__(self):
        from ai_edge_litert.interpreter import Interpreter
        path = model_path("yamnet.tflite")
        self.labels = zipfile.ZipFile(path).read("yamnet_label_list.txt").decode().splitlines()
        self.interpreter = Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self._in = self.interpreter.get_input_details()[0]["index"]
        self._out = self.interpreter.get_output_details()[0]["index"]
        self.group_index = {g: [self.labels.index(n) for n in names if n in self.labels]
                            for g, names in GROUPS.items()}
        self.common_index = np.array([i for i, n in enumerate(self.labels) if n in COMMON])

    def scores(self, audio):
        """(frame_centre_times, scores[frames, 521]) for a 16 kHz mono float32 array."""
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < WINDOW:
            audio = np.pad(audio, (0, WINDOW - len(audio)))
        starts = range(0, len(audio) - WINDOW + 1, HOP)
        out = np.empty((len(starts), len(self.labels)), dtype=np.float32)
        for k, s in enumerate(starts):
            self.interpreter.set_tensor(self._in, audio[s:s + WINDOW])
            self.interpreter.invoke()
            out[k] = self.interpreter.get_tensor(self._out)[0]
        times = (np.arange(len(out)) * HOP + WINDOW / 2) / RATE
        return times, out

    def grouped(self, scores):
        """{group: per-frame score} using the max over the group's classes."""
        return {g: scores[:, idx].max(axis=1) if idx else np.zeros(len(scores))
                for g, idx in self.group_index.items()}

    def unusual(self, scores, min_score=0.4, max_share=0.03):
        """
        Per-frame score of sound classes that are rare in this window and not
        normal stream background (speech, music, game audio). Returns
        (scores, names of the rare classes).
        """
        mask = np.ones(scores.shape[1], dtype=bool)
        mask[self.common_index] = False
        share = (scores[:, mask] >= min_score).mean(axis=0)
        rare = share <= max_share
        sub = scores[:, mask][:, rare]
        names = [self.labels[i] for i in np.flatnonzero(mask)[rare]]
        return (sub.max(axis=1) if sub.size else np.zeros(len(scores), np.float32)), names


def get_model():
    """One YAMNet instance per process (worker processes load it lazily)."""
    global _model
    if _model is None:
        _model = YamNet()
    return _model
