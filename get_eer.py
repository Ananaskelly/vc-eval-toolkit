import os
import shutil
import argparse
import logging

from functools import partial
import nemo.collections.asr as nemo_asr
from speechbrain.pretrained import SpeakerRecognition

from bio_utils.eer_processing import process_protocol, build_embedding_ecapa, build_embedding_nemo

logging.disable(logging.CRITICAL)

SPEECH_BRAIN_CACHE_DIR = '../pretrained_models/spkrec-ecapa-voxceleb'


SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(description='eer util')
    parser.add_argument(
        '--enroll_dir_path',
        type=str,
        help='Path to directory with enroll wav files',
    )
    parser.add_argument(
        '--test_dir_path',
        type=str,
        help='Path to directory with test wav files',
    )
    parser.add_argument(
        '--protocol_path',
        type=str,
        help='Path to protocol',
    )
    parser.add_argument(
        '--speaker_model',
        type=str,
        help='speaker model: NEMO, ECAPA',
        default='NEMO'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./test_pics',
        help='Path to store output data',
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

    _enroll_wav_dir = args.enroll_dir_path
    _test_wav_dir = args.test_dir_path
    _proto_path = args.protocol_path
    _save_path = args.save_path
    _model = args.speaker_model

    os.makedirs(_save_path, exist_ok=True)

    os.makedirs(SPEECH_BRAIN_CACHE_DIR, exist_ok=True)
    if _model == 'ECAPA':
        speaker_model = SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                                                        savedir=SPEECH_BRAIN_CACHE_DIR,
                                                        run_opts={'device': 'cuda:{}'.format(args.cuda_num)})
        _speaker_engine = partial(build_embedding_ecapa, speaker_model=speaker_model)

    elif _model == 'NEMO':
        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
        _speaker_engine = partial(build_embedding_nemo, speaker_model=speaker_model)

    else:
        raise ValueError

    process_protocol(enroll_wav_dir=_enroll_wav_dir,
                     test_wav_dir=_test_wav_dir,
                     proto_path=_proto_path,
                     save_path=_save_path,
                     speaker_engine=_speaker_engine,
                     use_vad=False)
