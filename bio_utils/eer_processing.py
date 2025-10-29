import os
import sys
import traceback
import uuid
import shutil
import torch
import numpy as np
import soundfile as sf
import seaborn
import sklearn.metrics
from matplotlib import pyplot as plt
from numpy.linalg import norm
from tqdm import tqdm


SPEECH_BRAIN_TEMP_FOLDER = ''

if os.path.exists(SPEECH_BRAIN_TEMP_FOLDER):
    shutil.rmtree(SPEECH_BRAIN_TEMP_FOLDER)
os.makedirs(SPEECH_BRAIN_TEMP_FOLDER)


def process_protocol(enroll_wav_dir, test_wav_dir, proto_path, speaker_engine, use_vad, save_path=None,
                     save_plot=False, enroll_min_dur=None, enroll_max_dur=None, ex='wav',
                     mean_lst=None, std_lst=None):
    utt2model = {}
    pred_lst = []
    label_lst = []

    use_score_norm = False
    if use_score_norm:
        utt2type = {}
        f_path = ''
        with open(f_path) as in_file:
            for line in in_file:
                utt, _t = line.strip('\n').split()
                utt2type[utt] = _t

    with open(proto_path) as in_file:
        lines = in_file.readlines()
        for line in tqdm(lines):
            try:
                model_id, test_id, lab = line.strip().split()
                model_id = os.path.splitext(model_id)[0]
                test_id = os.path.splitext(test_id)[0]
                if model_id not in utt2model:
                    in_wav_path = os.path.join(enroll_wav_dir, model_id + '.{}'.format(ex))
                    utt2model[model_id] = process_wav(in_wav_path, speaker_engine, use_vad, enroll_min_dur,
                                                      enroll_max_dur)
                model_emb = utt2model[model_id]

                if test_id not in utt2model:
                    in_wav_path = os.path.join(test_wav_dir, test_id + '.{}'.format(ex))
                    emb = process_wav(in_wav_path, speaker_engine, use_vad)
                    utt2model[test_id] = emb
                test_emb = utt2model[test_id]

                cos_sim = np.dot(model_emb, test_emb.T) / (norm(model_emb) * norm(test_emb))
            except KeyboardInterrupt:
                sys.exit()
                pass
            except:
                traceback.print_exc()
                continue

            if lab == 'imp':
                label_lst.append(0)
            elif lab == 'tar':
                label_lst.append(1)
            else:
                raise ValueError
            score = cos_sim
            if use_score_norm:
                model_type = utt2type[model_id.split('_')[-1]]
                test_type = utt2type[test_id.split('_')[-1]]

                if model_type == 'whisper' and test_type == 'whisper':
                    mean = mean_lst[1]
                    std = std_lst[1]
                elif model_type == 'speech' and test_type == 'speech':
                    mean = mean_lst[0]
                    std = std_lst[0]
                else:
                    mean = mean_lst[-1]
                    std = std_lst[-1]
                score = (score - mean) / std

            pred_lst.append(np.array(score).squeeze())

    protocol_name = os.path.splitext(os.path.basename(proto_path))[0]

    if save_plot:
        pred_lst = np.array(pred_lst)
        label_lst = np.array(label_lst)
        seaborn.distplot(pred_lst[label_lst == 1], hist=False, rug=True, label='target')
        seaborn.distplot(pred_lst[label_lst == 0], hist=False, rug=True, label='impostor')
        plt.legend()
        plt.savefig(os.path.join(save_path, protocol_name + '_norm.png'))


    EER = compute_eer(label=label_lst,
                      pred=pred_lst)
    print('{} EER: {}'.format(protocol_name, EER))
    print('mean: {}\nstd: {}'.format(np.mean(pred_lst), np.std(pred_lst)))
    return EER * 100


def process_wav(in_wav_path, speaker_engine, use_vad, enroll_min_dur=None, enroll_max_dur=None):
    temp_file_path = './{}.wav'.format(str(uuid.uuid4()))

    processed_wav_path = in_wav_path

    emb = speaker_engine(processed_wav_path, enroll_min_dur=enroll_min_dur, enroll_max_dur=enroll_max_dur)

    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    return emb


def build_embedding_ecapa(wav_path, speaker_model, **kwargs):
    wav_name = os.path.basename(wav_path)
    cache_file = os.path.join(SPEECH_BRAIN_TEMP_FOLDER, wav_name)
    if os.path.exists(cache_file):
        os.remove(cache_file)
    waveform_x = speaker_model.load_audio(path=wav_path, savedir=SPEECH_BRAIN_TEMP_FOLDER)
    batch_x = waveform_x.unsqueeze(0)
    emb1 = speaker_model.encode_batch(batch_x, normalize=True)[0, :, :].cpu().numpy()

    return emb1


def build_embedding_nemo(wav_path, speaker_model, **kwargs):
    data, rate = sf.read(wav_path)

    enroll_min_dur = kwargs['enroll_min_dur']
    enroll_max_dur = kwargs['enroll_max_dur']

    if enroll_min_dur is not None and enroll_max_dur is not None:
        actual_len = data.shape[0]
        if actual_len / rate > enroll_max_dur:
            random_len = np.random.randint(rate * enroll_min_dur, rate * enroll_max_dur)
            random_start = np.random.randint(0, actual_len - random_len)
            data = data[random_start: random_start + random_len]

    audio_signal, audio_signal_len = (
        torch.tensor([data.astype(np.float32)], device=speaker_model.device),
        torch.tensor([data.shape[0]], device=speaker_model.device),
    )
    speaker_model.freeze()
    _, emb1 = speaker_model(input_signal=audio_signal, input_signal_length=audio_signal_len)
    emb1 = emb1.cpu().numpy()

    return emb1


def compute_eer(label, pred, positive_label=1):
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr

    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = (eer_1 + eer_2) / 2

    return eer
