import os
#os.environ['CUDA_VISIBLE_DEVICES'] =  '1'
from train_lyft import run_train,run_evaluate_TTA


if __name__ == "__main__":
    config_path = '../submit.pp250.config'
    pretrained_path = None
    model_dir = '/media/wenjing/hdd/trainning_files_lyft_final/pp250/'
    run_train(config_path=config_path, model_dir=model_dir, multi_gpu=True, resume=True, pretrained_path=pretrained_path)

    run_evaluate_TTA(config_path=config_path,model_dir=model_dir)
