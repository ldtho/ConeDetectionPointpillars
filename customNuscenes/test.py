config_path = '/kaggle/code/ConeDetectionPointpillars/customNuscenes/configs/cones_pp_v4.config'
# config_path = '/kaggle/code/ConeDetectionPointpillars/customNuscenes/configs/cones_pp_initial_v2.config'
# config_path = '/kaggle/code/ConeDetectionPointpillars/customNuscenes/configs/cones_pp_initialak.config'
model_dir = f'/kaggle/code/ConeDetectionPointpillars/customNuscenes/outputs/cones_pp_v4'
result_path = None
create_folder = True
display_step = 25
# pretrained_path="/kaggle/code/ConeDetectionPointpillars/customNuscenes/outputs/1611755918.3152874/29-1-2021_9:53/voxelnet-5850.tckpt"
pretrained_path = None
multi_gpu = False
measure_time = False
resume = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = str(Path(model_dir).resolve())
if create_folder:
    if Path(model_dir).exists():
        model_dir = torchplus.train.create_folder(model_dir)
model_dir = Path(model_dir)

if not resume and model_dir.exists():
    raise ValueError("model dir exists and you don't specify resume")

model_dir.mkdir(parents=True, exist_ok=True)
if result_path is None:
    result_path = model_dir / 'results'
config_file_bkp = "pipeline.config"

if isinstance(config_path, str):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
else:
    config = config_path
    proto_str = text_format.MessageToString(config, indent=2)

with (model_dir / config_file_bkp).open('w') as f:
    f.write(proto_str)
# Read config file
input_cfg = config.train_input_reader
eval_input_cfg = config.eval_input_reader
model_cfg = config.model.second  # model's config
train_cfg = config.train_config  # training config
