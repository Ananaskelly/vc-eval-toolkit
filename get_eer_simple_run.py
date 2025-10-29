import os
import shutil
import logging

from functools import partial
import nemo.collections.asr as nemo_asr
# from speechbrain.pretrained import SpeakerRecognition

from bio_utils.eer_processing import process_protocol, build_embedding_ecapa, build_embedding_nemo
# from bio_utils.sim_processing import process_wav_wespeaker, process_wav_uni_speech, process_wav_wespeaker2
from bio_utils.voxblink2.model import Voxblink2Resnet34
logging.disable(logging.CRITICAL)

SPEECH_BRAIN_CACHE_DIR = '../pretrained_models/spkrec-ecapa-voxceleb'
SPEECH_BRAIN_TEMP_FOLDER = '/media/data2/ananaskelly/work_dir/speechbrain_folder/ecapa_tdnn'


SAMPLE_RATE = 16000


if __name__ == '__main__':

    save_path = './test_pics_comp'
    mean_score_n2n = 0.0
    std_score_n2n = 1.0
    mean_score_w2w = 0.0
    std_score_w2w = 1.0
    mean_score_n2w = 0.0
    std_score_n2w = 1.0
    # for _model in ['ECAPA', 'NEMO', 'wavlm_tdnn', 'wespeaker']:
    for _model in ['NEMO']:
        _cuda_num = 1
        os.makedirs(save_path, exist_ok=True)

        os.makedirs(SPEECH_BRAIN_CACHE_DIR, exist_ok=True)
        # if _model == 'ECAPA':
        #     speaker_model = SpeakerRecognition.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb',
        #                                                     savedir=SPEECH_BRAIN_CACHE_DIR,
        #                                                     run_opts={'device': 'cuda:{}'.format(_cuda_num)})
        #     _speaker_engine = partial(build_embedding_ecapa, speaker_model=speaker_model)
        #     mean_score_n2n = 0.11
        #     std_score_n2n = 0.13
        #     mean_score_w2w = 0.23
        #     std_score_w2w = 0.12
        #     mean_score_n2w = 0.03
        #     std_score_n2w = 0.09

        if _model == 'NEMO':
            # speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name='titanet_large')
            speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from('/media/data2/avdeeva/whisper_exploration/speaker_recognition/checkpoints/model_v1.pth')
            _speaker_engine = partial(build_embedding_nemo, speaker_model=speaker_model)
            mean_score_n2n = 0.10
            std_score_n2n = 0.15
            mean_score_w2w = 0.25
            std_score_w2w = 0.15
            mean_score_n2w = 0.02
            std_score_n2w = 0.10

        # elif _model == 'wavlm_tdnn':
        #     _speaker_engine = process_wav_uni_speech
        #     mean_score_n2n = 0.13
        #     std_score_n2n = 0.14
        #     mean_score_w2w = 0.21
        #     std_score_w2w = 0.13
        #     mean_score_n2w = 0.07
        #     std_score_n2w = 0.10
        #
        # elif _model == 'wespeaker':
        #     _speaker_engine = process_wav_wespeaker
        #     mean_score_n2n = 0.12
        #     std_score_n2n = 0.14
        #     mean_score_w2w = 0.24
        #     std_score_w2w = 0.14
        #     mean_score_n2w = 0.08
        #     std_score_n2w = 0.10

        # elif _model == 'wespeaker_sam':
        #     _speaker_engine = process_wav_wespeaker2
        # elif _model == 'voxblink2':
        #     speaker_voxblink2_model = Voxblink2Resnet34(device='cuda:{}'.format(_cuda_num),
        #                                                 return_frame_level=False)
        #     _speaker_engine = speaker_voxblink2_model
        else:
            raise ValueError

        if os.path.exists(SPEECH_BRAIN_TEMP_FOLDER):
            shutil.rmtree(SPEECH_BRAIN_TEMP_FOLDER)
        os.makedirs(SPEECH_BRAIN_TEMP_FOLDER)
        print('start processing with {}'.format(_model))

        mean_lst = [mean_score_n2n, mean_score_w2w, mean_score_n2w]
        std_lst = [std_score_n2n, std_score_w2w, std_score_n2w]
        process_protocol(enroll_wav_dir='/media/data2/avdeeva/datasets/speechOcean_meta/meta_for_eer_test/'
                                        'whisper_enrolls',
                         test_wav_dir='/media/data2/avdeeva/datasets/speechOcean_meta/meta_for_eer_test/'
                                      'whisper_tests',
                         proto_path='/media/data2/avdeeva/datasets/speechOcean_meta/meta_for_eer_test/protocols/'
                                    'whisper2whisper.txt',
                         save_path=save_path,
                         speaker_engine=_speaker_engine,
                         use_vad=False,
                         save_plot=False,
                         ex='wav')
