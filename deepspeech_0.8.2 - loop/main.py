import deepspeech
import datetime
import numpy as np
import pyaudio




#model path
model_file_path = 'C:/Users/Ankan/Desktop/deepspeech_0.8.2/models/deepspeech-0.8.2-models.pbmm'
model = deepspeech.Model(model_file_path)

#model scorer path
scorer_file_path = 'C:/Users/Ankan/Desktop/deepspeech_0.8.2/models/deepspeech-0.8.2-models.scorer'
model.enableExternalScorer(scorer_file_path)


# hyperparameter tuning for beam search decoder
lm_alpha = 0.75
lm_beta = 1.85
model.setScorerAlphaBeta(lm_alpha, lm_beta)
beam_width = 500
model.setBeamWidth(beam_width)


#Creating model stream
ds_stream = model.createStream()




chunk = 2048
# sample format
FORMAT = pyaudio.paInt16
# mono, change to 2 if you want stereo
channels = 1
# 16000 samples per second
sample_rate = 16000
record_seconds = 4
# initialize PyAudio object
p = pyaudio.PyAudio()
# open stream object as input & output
stream = p.open(format=FORMAT,
        channels=channels,
        rate=sample_rate,
        input=True,
        output=True,
        frames_per_buffer=chunk)


t0=datetime.datetime.now()
try:
    while True:
        text=""
        print("Listening........")
        for i in range(int((16000 / chunk) * record_seconds)):
            data = stream.read(chunk) 
            data32 = np.frombuffer(data, dtype=np.int16)
            ds_stream.feedAudioContent(data32)
            text = ds_stream.intermediateDecode()
            # if you want to hear your voice while recording
            # stream.write(data)
            
        text = ds_stream.finishStream()
        print(text)
        ds_stream = model.createStream()

except KeyboardInterrupt:
    pass

t1=datetime.datetime.now()
print("Session duration----->",t1-t0)
# stop and close stream
stream.stop_stream()
stream.close()
# terminate pyaudio object
p.terminate()



