import sys
import numpy as np
import librosa
import torch
import laion_clap

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt() # download the default pretrained checkpoint.

def embed_files(audio_file_list):
    audio_embeddings = model.get_audio_embedding_from_filelist(x = audio_file_list, use_tensor=False)

if __name__ == '__main__':
    embeddings = embed_files(sys.argv[1:])
    print(embeddings)
