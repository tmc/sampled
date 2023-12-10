import os
import sys
import numpy as np
import librosa
import torch
import laion_clap
import urllib.parse

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt() # download the default pretrained checkpoint.


def embed_files(audio_file_list):
    print("file list:", audio_file_list)
    audio_embeddings = model.get_audio_embedding_from_filelist(x = audio_file_list, use_tensor=False)
    return audio_embeddings

if __name__ == '__main__':
    files = sys.argv[1:]
    files = [f for f in files if os.path.exists(f)]
    embeddings = embed_files(files)
    # append files to data/samples.csv
    csv_file = 'data/samples.csv'
    with open(csv_file,'a') as fd:
        for i, f in enumerate(files):
            e = list(embeddings[i])
            # print on one line:
            # 'replace ~/Downloads/ with http://localhost:8080/'
            f = str(e).replace('~/Downloads/','http://localhost:8080/')
            # urlencode f
            f = urllib.parse.quote_plus(f)
            fd.write(f'{i},{f},"{e}"\n')
