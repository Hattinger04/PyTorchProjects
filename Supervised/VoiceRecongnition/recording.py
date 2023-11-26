import pyaudio
import numpy as np
import os
import numpy as np
import game as g
import tensorflow as tf

from audio_converting import preprocess_audiobuffer


FRAMES_PER_BUFFER = 1600
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
limit = 50
p = pyaudio.PyAudio()
commands = ["down","go","left", "right", "stop", "up" ]
os.chdir(r"C:\Users\s8gre\Documents\Schule\KerasProjects\VoiceRecongnition")
loaded_model = tf.saved_model.load(".\AI\data\save")


def record_audio():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAMES_PER_BUFFER
    )

    while True:
        data = stream.read(RATE)
        audio_data = np.fromstring(data, dtype=np.int16)
        if np.abs(audio_data).mean() > limit:
            buffer = np.frombuffer(data, dtype=np.int16)
            command, percentage = predict_mic(buffer)
            if percentage > 0.95:
                g.move_turtle(command)
                if command == "stop": 
                    break
        else:
            print("Not talking")
   
    stream.stop_stream()
    stream.close()
    p.terminate()

def predict_mic(audio):
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction["predictions"], axis=1)
    command = commands[label_pred[0]]
    percentage = prediction["predictions"][0][label_pred[0]]
    return command, percentage

record_audio()