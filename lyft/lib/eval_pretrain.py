import os
#each eval may need 24 hours, you may want to use one gpu for each eval, modify this
os.environ['CUDA_VISIBLE_DEVICES'] =  '1'
from train_lyft import run_evaluate_TTA

# since we the evaluation shufft the point cloud each time, the score might be slightly different (say 0.002) each time
# the lyft3d_pred_merge_mutli.csv generated in the outputs fold evaluate to 0.218 (Public Board) and 0.214 (Private Board)
# which is around 0.002 point lower than our score on the leaderboard

if __name__ == "__main__":
    model_type = 'fhd100_v1'

    if model_type == 'pp100': #public score 0.191 you may need 24GB mem
        config_path = '../configs/submit.pp100.config'
        pretrained_path = '../pretrained_models/voxelnet-340000-pp100.tckpt'
        result_name = '../outputs/lyft3d_pred_pp100.csv'
        use_train = True
        split = False
    elif model_type == 'pp125': #public score 0.189 you need 24GB mem
        config_path = '../configs/submit.pp125.config'
        pretrained_path = '../pretrained_models/voxelnet-374500-pp125.tckpt'
        result_name = '../outputs/lyft3d_pred_pp125.csv'
        use_train = True
        split = False
    elif model_type == 'pp250': #public score 0.193 you need 24 GB RAM
        config_path = '../configs/submit.pp250.config'
        pretrained_path = '../pretrained_models/voxelnet-305500-pp250.tckpt'
        result_name = '../outputs/lyft3d_pred_pp250.csv'
        use_train = False
        split = True
    elif model_type == 'fhd100_v0':  #public score 0.187  #very time consuming, take ~36 hours
        config_path = '../configs/submit.fhd100.v0.config'
        pretrained_path = '../pretrained_models/voxelnet-250000-fhd100-v0.tckpt'
        result_name = '../outputs/lyft3d_pred_fhd100_v0.csv'
        use_train = False
        split = False
    elif model_type == 'fhd100_v1': #public score 0.191  #take ~30 hours
        config_path = '../configs/submit.fhd100.v1.config'
        pretrained_path = '../pretrained_models/voxelnet-180000-fhd100-v1.tckpt'
        result_name = '../outputs/lyft3d_pred_fhd100_v1.csv'
        use_train = False
        split = False
    elif model_type == 'fhd125_v0': #public score 0.193 #~25 hours
        config_path = '../configs/submit.fhd125.v0.config'
        pretrained_path = '../pretrained_models/voxelnet-70000-fhd125-v0.tckpt'
        result_name = '../outputs/lyft3d_pred_fhd125_v0.csv'
        use_train = False
        split = False
    elif model_type == 'fhd125_v1': #public score 0.197 ~25hours
        config_path = '../configs/submit.fhd125.v1.config'
        pretrained_path = '../pretrained_models/voxelnet-110000-fhd125-v1.tckpt'
        result_name = '../outputs/lyft3d_pred_fhd125_v1.csv'
        use_train = False
        split = False



    run_evaluate_TTA(config_path=config_path,model_dir=None,result_name=result_name,ckpt_path=pretrained_path,debug = False,split=split,use_train=use_train)
