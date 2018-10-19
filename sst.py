#!/usr/bin/env python3

from deepspeech import Model
import numpy as np
import pyaudio
from threading import Thread
from array import array

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.50

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 2.10

# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 15

def do_the_work(inp, rate):
    print(ds.stt(speech_input, rate))

# load model
ds = Model('models/output_graph.pbmm', N_FEATURES, N_CONTEXT, 'models/alphabet.txt', BEAM_WIDTH)
ds.enableDecoderWithLM('models/alphabet.txt', 'models/lm.binary', 'models/trie', LM_WEIGHT, VALID_WORD_COUNT_WEIGHT)

# initialize recording
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)


print("Start talking...")
while True:
    # record audio
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        data_chunk = array('h', data)
        vol = max(data_chunk)
        if (vol >= 500):
            frames.append(data)
    speech_input = np.frombuffer(b''.join(frames), np.int16)

    do_the_work(speech_input, RATE)
    t = Thread(target=do_the_work, args=(speech_input, RATE))
    t.start()


# close stuff
# stream.stop_stream()
# stream.close()
# audio.terminate()
