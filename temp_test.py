import os, time
import importlib
from collections import namedtuple

import env
from models import ACModel, Discriminator

import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str,
    help="Configure file used for training. Please refer to files in `config` folder.")
parser.add_argument("--ckpt", type=str, default=None,
    help="Checkpoint directory or file for training or evaluation.")
parser.add_argument("--test", action="store_true", default=False,
    help="Run visual evaluation.")
parser.add_argument('--headless', action='store_true',
    help='Run headless without creating a viewer window')
parser.add_argument("--seed", type=int, default=42,
    help="Random seed.")
parser.add_argument("--device", type=int, default=0,
    help="ID of the target GPU device for model running.")
parser.add_argument("--pretrained", type=str, default=None,
    help="Use pretrained checkpoint of discriminator")
parser.add_argument("--resume", type=str, default=None,
    help="resume with existing checkpoint")

parser.add_argument("--upper", type=str, default=None,
    help="upper discriminator")
parser.add_argument("--lower", type=str, default=None,
    help="lower discriminator")

settings = parser.parse_args()

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(settings.seed)
np.random.seed(settings.seed)
random.seed(settings.seed)
torch.manual_seed(settings.seed)
torch.cuda.manual_seed(settings.seed)
torch.cuda.manual_seed_all(settings.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


FPS = 30
FRAMESKIP = 2
CONTROL_MODE = "position"
HORIZON = 8
NUM_ENVS = 512
BATCH_SIZE = 256 #HORIZON*NUM_ENVS//16
OPT_EPOCHS = 5
ACTOR_LR = 5e-6
CRITIC_LR = 1e-4
GAMMA = 0.95
GAMMA_LAMBDA = GAMMA * 0.95

TRAINING_PARAMS = dict(
    max_epochs = 10000,
    save_interval = None,
    terminate_reward = -1
)
def logger(obs, rews, info):
    buffer = dict(r=[])
    buffer_disc = {
    name: dict(fake=[], seq_len=[]) for name in env.discriminators.keys()  # Dict[str, DiscriminatorConfig]
    }

    has_goal_reward = env.rew_dim > 0
    if has_goal_reward:
        buffer["r"].append(rews)
    
    multi_critics = env.reward_weights is not None
    if multi_critics:
        rewards = torch.zeros(1, len(env.discriminators)+env.rew_dim)                      # [num_envs X 8, reward 개수]
    else:
        rewards = torch.zeros(1, len(env.discriminators))
    
    fakes = info["disc_obs"]
    disc_seq_len = info["disc_seq_len"]

    for name, fake in fakes.items():
        buffer_disc[name]["fake"].append(fake)
        buffer_disc[name]["seq_len"].append(disc_seq_len[name])

    with torch.no_grad():
        # 1. Reward related to discriminators
        disc_data_raw = []
        for name, data in buffer_disc.items():              # data: fake, real, seq_len
            disc = model.discriminators[name]   
            fake = torch.cat(data["fake"])                  # [N * HORIZON, 2/5, 56/49] / len(data["fake"]) = HORIZON
            seq_len = torch.cat(data["seq_len"])            # [N * HORIZON]
            end_frame = seq_len - 1
            disc_data_raw.append((name, disc, fake, end_frame))
        
        for name, disc, ob, seq_end_frame in disc_data_raw:
            r = (disc(ob, seq_end_frame).clamp_(-1, 1).mean(-1, keepdim=True)) # clamp shape: [num_envs X 8, 32: ensemble]   / r.shape: [num_envs X 8, 1]
            if rewards is None:
                rewards = r
            else:
                rewards[:, env.discriminators[name].id] = r.squeeze_(-1)    # id: 0 / 1

        # 2. Reward related to goal      
        if has_goal_reward:
            rewards_task = torch.cat(buffer["r"])                           # [num_envs X 8, 2] / buffer["r"]: [8, 512, 2]
            if rewards is None:
                rewards = rewards_task
            else:
                rewards[:, -rewards_task.size(-1):] = rewards_task          # 마지막 reward 들 (개수만큼)
        
        else:
            rewards_task = None

        rewards = rewards.mean(0).cpu().tolist()                                        # [num_reward]
        print("Reward: {}".format("/".join(list(map("{:.4f}".format, rewards)))))
    return rewards

def test(env, model):
    model.eval()
    env.reset()
    rewards_tot = 0
    count = 0
    while not env.request_quit:
        obs, info = env.reset_done()
        seq_len = info["ob_seq_lens"]
        actions = model.act(obs, seq_len-1)
        obs, rews, _, info = env.step(actions)                                 # apply_actions -> do_simulation -> refresh_tensors -> observe()

        reward = logger(obs, rews, info)
        
        if info["terminate"].item() is False:
            rewards_tot += reward[0]
            count +=1
        else:
            rewards_tot += reward[0]
            print("\n------\nLength: {:d}, avg episode reward: {:.4f}\n-----".format(count, rewards_tot/count))
            rewards_tot = 0
            count = 0

if __name__ == "__main__":
    if settings.test:
        num_envs = 1
    else:
        num_envs = NUM_ENVS
        if settings.ckpt:
            # pretrained model을 사용하지 않을 때, ckpt가 이미 있는지 확인
            if not settings.pretrained:
                if os.path.isfile(settings.ckpt) or os.path.exists(os.path.join(settings.ckpt, "ckpt")):
                    raise ValueError("Checkpoint folder {} exists. Add `--test` option to run test with an existing checkpoint file".format(settings.ckpt))
            import shutil, sys
            os.makedirs(settings.ckpt, exist_ok=True)
            shutil.copy(settings.config, settings.ckpt)
            with open(os.path.join(settings.ckpt, "command_{}.txt".format(time.time())), "w") as f:
                f.write(" ".join(sys.argv))

    if os.path.splitext(settings.config)[-1] in [".npy", ".json", ".yaml"]:
        config = object()
        config.env_params = dict(
            motion_file = settings.config
        )
    else:
        spec = importlib.util.spec_from_file_location("config", settings.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

    # if headless
    if settings.headless:
        config.env_params['graphics_device'] = -1
    
    if hasattr(config, "training_params"):
        TRAINING_PARAMS.update(config.training_params)
    if not TRAINING_PARAMS["save_interval"]:
        TRAINING_PARAMS["save_interval"] = TRAINING_PARAMS["max_epochs"]
    print("TRAINING_PARAMS: ", TRAINING_PARAMS)
    training_params = namedtuple('x', TRAINING_PARAMS.keys())(*TRAINING_PARAMS.values())
    if hasattr(config, "discriminators"):
        discriminators = {
            name: env.DiscriminatorConfig(**prop)
            for name, prop in config.discriminators.items()
        }
    else:
        discriminators = {"_/full": env.DiscriminatorConfig()}
    if hasattr(config, "env_cls"):
        env_cls = getattr(env, config.env_cls)
    else:
        env_cls = env.ICCGANHumanoid
    print("env_cls: ", env_cls, config.env_params)
    env = env_cls(num_envs, FPS, FRAMESKIP,
        control_mode=CONTROL_MODE,
        discriminators=discriminators,
        compute_device=settings.device, 
        **config.env_params
    )
    if settings.test:
        env.episode_length = 500000

    value_dim = len(env.discriminators)+env.rew_dim
    model = ACModel(env.state_dim, env.act_dim, env.goal_dim, value_dim)
    discriminators = torch.nn.ModuleDict({
        name: Discriminator(dim) for name, dim in env.disc_dim.items()
    })
    device = torch.device(settings.device)
    model.to(device)
    discriminators.to(device)
    model.discriminators = discriminators

    if settings.test:
        if settings.pretrained is not None or settings.resume is not None:
            raise ValueError("This is test time. You can't use arguments of pretrained or resume")
        
        if settings.ckpt is not None and os.path.exists(settings.ckpt):
            if os.path.isdir(settings.ckpt):
                ckpt = os.path.join(settings.ckpt, "ckpt")
            else:
                ckpt = settings.ckpt
                settings.ckpt = os.path.dirname(ckpt)
            if os.path.exists(ckpt):
                # upper, lower 정보가 따로 따로 들어오면 actor/value만 update!
                if settings.upper is not None and settings.lower is not None:
                    print("Load model from {}".format(ckpt))
                    state_dict = torch.load(ckpt, map_location=torch.device(settings.device))
                    model_dict = model.state_dict()                                                        # current model
                    not_disc_key = []
                    # filter discriminator related keys
                    for k in model_dict.keys():
                        if ('discriminators' in k) == False: 
                            not_disc_key.append(k)
                    pretrained_dict = {k:v for k, v in state_dict["model"].items() if k in not_disc_key}
                    model_dict.update(pretrained_dict)
                    # 4. load the new state dict
                    model.load_state_dict(model_dict)
                # upper, lower 정보가 안들어오면 모델 전체 update!
                else:
                    print("Load model from {}".format(ckpt))
                    state_dict = torch.load(ckpt, map_location=torch.device(settings.device))
                    model.load_state_dict(state_dict["model"])

        # use pretrained discriminators
        if settings.upper:
            upper_ckpt = settings.upper
            up_state_dict = torch.load(upper_ckpt, map_location=torch.device(settings.device))
            model_dict = model.state_dict()                                                        # current model
            # filter discriminator related keys
            upper_disc_key = []
            for k in model_dict.keys():
                if 'discriminators.punch/upper' in k: 
                    upper_disc_key.append(k)
            print("upper_disc_key: ", upper_disc_key)
            # 2. filter out unnecessary keys
            pretrained_dict = {k:v for k, v in up_state_dict["model"].items() if k in upper_disc_key}
            # 3. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 4. load the new state dict
            model.load_state_dict(model_dict)
            pass

        if settings.lower:
            lower_ckpt = settings.lower
            low_state_dict = torch.load(lower_ckpt, map_location=torch.device(settings.device))
            
            model_dict = model.state_dict()                                                        # current model
            # filter discriminator related keys
            lower_disc_key = []
            for k in model_dict.keys():
                if 'discriminators.walk_in_place/lower' in k: 
                    lower_disc_key.append(k)
            print("lower_disc_key: ", lower_disc_key)
            # 2. filter out unnecessary keys
            pretrained_dict = {k:v for k, v in low_state_dict["model"].items() if k in lower_disc_key}
            # 3. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 4. load the new state dict
            model.load_state_dict(model_dict)

        env.render()
        test(env, model)

    # train 일 때
    else:
        # 저장할 ckpt 폴더가 있어야 한다
        if settings.ckpt is not None and os.path.exists(settings.ckpt):
            if os.path.isdir(settings.ckpt):
                ckpt = os.path.join(settings.ckpt, "ckpt")  
            else:
                ckpt = settings.ckpt
                settings.ckpt = os.path.dirname(ckpt)              
            
            # resume 시키려면 
            if settings.resume:
                if settings.resume is not None and os.path.isfile(settings.resume) and os.path.exists(settings.resume):
                    resume_ckpt = settings.resume
                    state_dict = torch.load(resume_ckpt, map_location=torch.device(settings.device))      # loaded model
                    model_dict = model.state_dict()                                                        # current model
                    model.load_state_dict(state_dict['model'])
                    print("Resuming training with checkpoint: {}".format(resume_ckpt))
                else:
                    raise ValueError("Please correctly type checkpoint path to resume training")

            # pretrained model of discriminator 사용
            if settings.pretrained:
                pretrained_ckpt = settings.pretrained
                if os.path.exists(pretrained_ckpt):
                    print("Load PRETRAINED discriminator model from {}".format(pretrained_ckpt))
                    state_dict = torch.load(pretrained_ckpt, map_location=torch.device(settings.device))   # pretrained model
                    model_dict = model.state_dict()                                                        # current model
                    # 1. filter keys w.r.t. discriminators
                    discriminator_key = []
                    for k in model_dict.keys():
                        if 'discriminators' in k: 
                            discriminator_key.append(k)
                    # 2. filter out unnecessary keys
                    pretrained_dict = {k:v for k, v in state_dict["model"].items() if k in discriminator_key}
                    # 3. overwrite entries in the existing state dict
                    model_dict.update(pretrained_dict) 
                    # 4. load the new state dict
                    model.load_state_dict(model_dict)
                    model_dict = model.state_dict()                                             # updated model
                else:
                    raise ValueError("Please correctly type checkpoint path to use pretrained model for training")

            # train(env, model, settings.ckpt, training_params)
        else:
            raise ValueError("Please correctly type checkpoint directory path to save model")
            
