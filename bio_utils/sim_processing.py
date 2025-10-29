import os
import sys
import traceback

import numpy as np
import soundfile as sf
import torch
import shutil
import wespeaker
import seaborn
import matplotlib.pyplot as plt
from numpy.linalg import norm
from tqdm import tqdm
from torchaudio.transforms import Resample
from transformers import AutoFeatureExtractor, WavLMForXVector
from bio_utils.speaker_verification.models.ecapa_tdnn import ECAPA_TDNN_SMALL
from espnet2.bin.spk_inference import Speech2Embedding


TEMP_FOLDER = ''

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()

uni_speech_model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None).cuda().eval()
# path to downloaded checkpoint
checkpoint_path = ''
state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
uni_speech_model.load_state_dict(state_dict['model'], strict=False)

wespeaker_path1 = ''
wespeaker_model = wespeaker.load_model_local(wespeaker_path1)
wespeaker_model.set_device('cuda:0')

espnet_model = Speech2Embedding.from_pretrained(
        model_tag="espnet/voxcelebs12_rawnet3", device='cuda:0')

if os.path.exists(TEMP_FOLDER):
    shutil.rmtree(TEMP_FOLDER)
os.makedirs(TEMP_FOLDER)


def process_wav(wav_path):
    wav_data, sr = sf.read(wav_path)
    inputs = feature_extractor(
        [wav_data], sampling_rate=sr, return_tensors="pt", padding=True, savedir=TEMP_FOLDER
    )
    with torch.no_grad():
        embeddings = model(**inputs).embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
    return embeddings.numpy()


def process_wav_espnet(wav_path):
    wav, sr = sf.read(wav_path)

    return espnet_model(wav).cpu().numpy()


def process_wav_uni_speech(wav_path, **kwargs):
    wav, sr = sf.read(wav_path)

    wav = torch.from_numpy(wav).unsqueeze(0).float()
    resample = Resample(orig_freq=sr, new_freq=16000)
    wav = resample(wav)
    wav = wav.cuda()

    with torch.no_grad():
        emb = uni_speech_model(wav)

    return emb.cpu().numpy()


def process_wav_wespeaker(wav_path, **kwargs):
    embedding = wespeaker_model.extract_embedding(wav_path)

    return embedding.cpu().numpy()


def process_protocol(enroll_wav_dir, test_wav_dir, proto_path, ext='wav'):
    enroll_models = {}
    test_models = {}
    all_scores = []

    with open(proto_path) as in_file:
        lines = in_file.readlines()
        for line in tqdm(lines):
            try:
                model_id, test_id, lab = line.strip().split()
                if lab == 'imp':
                    continue
                if model_id not in enroll_models:
                    in_wav_path = os.path.join(enroll_wav_dir, model_id + '.{}'.format(ext))
                    enroll_models[model_id] = process_wav_espnet(in_wav_path)
                model_emb = enroll_models[model_id]

                if test_id not in test_models:
                    in_wav_path = os.path.join(test_wav_dir, test_id + '.{}'.format(ext))
                    emb = process_wav_espnet(in_wav_path)
                    test_models[test_id] = emb
                test_emb = test_models[test_id]
                all_scores.append(np.dot(model_emb, test_emb.T) / (norm(model_emb) * norm(test_emb)))
            except KeyboardInterrupt:
                sys.exit()
                pass
            except:
                print(line)
                traceback.print_exc()
                continue
    mean_val = np.mean(all_scores)
    print(mean_val)

    return mean_val


def process_trials(dir1, dir2, proto_path, emb_func):

    all_scores = []

    mis_dir = './low_score_pairs_new11'
    os.makedirs(mis_dir, exist_ok=True)
    conf_dir = './high_score_pairs_new11'
    os.makedirs(conf_dir, exist_ok=True)

    with open(proto_path) as in_file:
        lines = in_file.readlines()
        for line in tqdm(lines):
            try:
                id1, id2 = line.strip().split()

                in_wav_path = os.path.join(dir1, 'origid_{}__refid_{}.wav'.format(id1.split('.')[0],
                                                                                  id2.split('.')[0]))
                model_emb = emb_func(in_wav_path)


                in_wav_path = os.path.join(dir2, id2)
                test_emb = emb_func(in_wav_path)

                score = np.dot(model_emb, test_emb.T) / (norm(model_emb) * norm(test_emb))
                all_scores.append(score)

            except KeyboardInterrupt:
                sys.exit()
                pass
            except:
                print(line)
                traceback.print_exc()
                continue
    mean_val = np.mean(all_scores)
    seaborn.displot(all_scores)
    os.makedirs('./test_pics_scores/', exist_ok=True)
    plt.savefig('./test_pics_scores/wespeaker_no_norm_model_v2.png')

    return mean_val
