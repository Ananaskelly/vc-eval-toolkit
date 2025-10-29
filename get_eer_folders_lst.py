import os
import shutil
from functools import partial

import nemo.collections.asr as nemo_asr

from speechbrain.pretrained import SpeakerRecognition

from bio_utils.eer_processing import process_protocol, build_embedding_ecapa, build_embedding_nemo
from bio_utils.sim_processing import process_protocol as process_protocol_wav_lm_sim

SPEECH_BRAIN_CACHE_DIR = '../pretrained_models/spkrec-ecapa-voxceleb'
SPEECH_BRAIN_TEMP_FOLDER = '/media/data2/ananaskelly/work_dir/speechbrain_folder/ecapa_tdnn'


if __name__ == '__main__':

    CUDA_NUM = 0
    MODEL = 'NEMO'
    GET_SIMILARITY_SCORE = True
    GET_EER = True

    SAVE_PATH = './test_pics_mean'
    os.makedirs(SAVE_PATH, exist_ok=True)

    TEST_WAV_DIR_ROOT = ''
    STORAGE_ROOT = ''
    EXP_NAME_LST = []
    PROTO_ROOT = '../protocols'

    os.makedirs(SPEECH_BRAIN_CACHE_DIR, exist_ok=True)
    if MODEL == 'ECAPA':
        speaker_model = SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
                                                        savedir=SPEECH_BRAIN_CACHE_DIR,
                                                        run_opts={'device': 'cuda:{}'.format(CUDA_NUM)})
        _speaker_engine = partial(build_embedding_ecapa, speaker_model=speaker_model)

    elif MODEL == 'NEMO':
        speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large').eval()
        _speaker_engine = partial(build_embedding_nemo, speaker_model=speaker_model)
    else:
        raise ValueError

    if os.path.exists(SPEECH_BRAIN_TEMP_FOLDER):
        shutil.rmtree(SPEECH_BRAIN_TEMP_FOLDER)
    os.makedirs(SPEECH_BRAIN_TEMP_FOLDER)

    proto_mapping = {
        'chains': 'chains/chains_protocol.txt',
        'chains2vctk': 'chains/chains_whsp_vctk_enroll_protocol.txt',
        'cv': 'common_voice/common_voice_protocol.txt',
        'libri_test_clean2common_voice': 'libri_test_clean/libri_test_clean_common_voice_enroll_protocol.txt',
        'common_voice2common_voice': 'common_voice/common_voice2common_voice_protocol_v2.txt'

    }
    enrolls_mapping = {
        'chains': 'CHAINS/chains_enrolls/',
        'chains2vctk': 'VCTK/vctk_enrolls/',
        'cv': 'COMMON_VOICE/common_voice_enrolls',
        'es': 'ENG_SPONTAN/eng_spontan_enrolls/',
        'libri_test_clean2common_voice': 'COMMON_VOICE/common_voice_enrolls',
        'common_voice2common_voice': 'COMMON_VOICE/common_voice_enrolls'
    }

    for EXP_NAME in EXP_NAME_LST:
        results = {}

        for test_case in list(enrolls_mapping.keys()):
            results[test_case] = {}
            test_wav_dir = os.path.join(TEST_WAV_DIR_ROOT, '{}_{}'.format(EXP_NAME, test_case))
            if not os.path.exists(test_wav_dir):
                continue
            proto_path = os.path.join(PROTO_ROOT, proto_mapping[test_case])
            if GET_EER:
                eer_val = process_protocol(enroll_wav_dir=os.path.join(STORAGE_ROOT, enrolls_mapping[test_case]),
                                           test_wav_dir=test_wav_dir,
                                           proto_path=proto_path,
                                           save_path=SAVE_PATH,
                                           speaker_engine=_speaker_engine,
                                           use_vad=False,
                                           # enroll_min_dur=5,
                                           # enroll_max_dur=10
                                           )
                results[test_case]['eer_val'] = eer_val

            if GET_SIMILARITY_SCORE:
                sim_val = process_protocol_wav_lm_sim(enroll_wav_dir=os.path.join(STORAGE_ROOT, enrolls_mapping[test_case]),
                                                      test_wav_dir=test_wav_dir,
                                                      proto_path=proto_path)
                results[test_case]['sim_val'] = sim_val
        print('{}'.format(EXP_NAME))
        for k, v in results.items():
            print('{} results:'.format(k))
            for _k, _v in v.items():
                print('     {}: {}'.format(_k, _v))
