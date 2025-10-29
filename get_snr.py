import math
import json
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path


from asr_utils.model import ASRHubert
from asr_utils.utils import get_speech_segments


SAMPLE_RATE = 16000
FRAMES_PER_MS = SAMPLE_RATE / 1000

SNR_PAUSE_MIN_DUR = FRAMES_PER_MS * 100


def get_snr(waveform, sample_rate, speech_segments):
    """Function to calculate SNR value for given waveform according to given speech segments.
    It returns SNR value in dB.

    :param waveform:         input waveform to process
    :type waveform:          np.ndarray
    :param sample_rate:      sample rate
    :type sample_rate:       int
    :param speech_segments:  found speech segments, list of tuples (st, end, label)
    :type speech_segments:   list

    :return float
    """
    speech = np.array([])
    non_speech = np.array([])

    st = 0

    vis_segments = []

    for segment in speech_segments:
        end = math.ceil(segment[0] * sample_rate)
        if (end - st) > SNR_PAUSE_MIN_DUR:
            vis_segments.append(((st, end), False))
            non_speech = np.concatenate((non_speech, waveform[st: end]))
        else:
            # discussing, maybe these slices must be totally discarded
            vis_segments.append(((st, end), True))
            speech = np.concatenate((speech, waveform[st: end]))
        s_st = math.floor(segment[0] * sample_rate)
        s_end = math.ceil(segment[1] * sample_rate)
        vis_segments.append(((s_st, s_end), True))
        speech = np.concatenate((speech, waveform[s_st: s_end]))

        st = s_end

    non_speech = np.concatenate((non_speech, waveform[st:]))
    vis_segments.append(((st, len(waveform)), False))

    p_signal = np.sum(speech ** 2) / len(speech)
    p_noise = np.sum(non_speech ** 2) / len(non_speech)

    return 10 * np.log10(p_signal / (p_noise + 1e-8))


def parse_args():
    parser = argparse.ArgumentParser(description='snr util')

    parser.add_argument(
        '--test_dir_path',
        type=str,
        help='Path to directory with test wav files',
    )
    parser.add_argument(
        '--transcript_json_path',
        type=str,
        help='Path to json with transcripts',
        default=''
    )
    parser.add_argument(
        '--cuda_num',
        type=int,
        default=0,
        help='device id'
    )

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    _test_wav_dir = args.test_dir_path
    _transcript_json_path = args.transcript_json_path
    _cuda_num = args.cuda_num

    _asr_model = ASRHubert(device='cuda:{}'.format(_cuda_num))
    _transcript_dict = {}

    if _transcript_json_path == '':
        use_hyp = True
    else:
        use_hyp = False
        _transcript_dict = json.load(open(_transcript_json_path))

    snr_lst = list()

    path = Path(_test_wav_dir)
    for p in path.rglob("*"):
        utt = p.stem

        _waveform, sample_rate = sf.read(p)
        emissions, hyp_transcript = _asr_model(_waveform.squeeze())

        if use_hyp:
            txt = hyp_transcript[0]
        else:
            txt = _transcript_dict[utt]

        speech_segments = get_speech_segments(emissions=emissions,
                                              transcript=txt.replace(' ', '|'),
                                              dictionary=_asr_model.get_vocab())

        snr = get_snr(waveform=_waveform,
                      sample_rate=sample_rate,
                      speech_segments=speech_segments)
        snr_lst.append(snr)

    print('Mean SNR: {}'.format(np.mean(snr_lst)))
