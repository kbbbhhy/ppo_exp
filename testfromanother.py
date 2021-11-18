import os
os.environ['OMP_NUM_THREADS']='1'
import argparse
import torch
from src.env import create_train_env
from src.model import PPO
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT,RIGHT_ONLY
import torch.nn.functional as F

def get_args():
    parser=argparse.ArgumentParser("Implementation for trained model to test in different level")
    parser.add_argument("--world",type=int,default=1)
    parser.add_argument("--stage",type=int,default=1)
    parser.add_argument("--trained_model_world",type=int,default=1)
    parser.add_argument("--trained_model_stage",type=int,default=1)
    parser.add_argument("--trained_model_iter",type=str,default="")
    parser.add_argument("--action_type",type=str,default="simple")
    parser.add_argument("--saved_path",type=str,default="trained_models")
    parser.add_argument("--output_path",type=str,default="output")
    args=parser.parse_args()
    return args

def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if opt.action_type=='right':
        actions=RIGHT_ONLY
    elif opt.action_type=="simple":
        actions=SIMPLE_MOVEMENT
    else:
        actions=COMPLEX_MOVEMENT
    env=create_train_env(opt.world,opt.stage,actions,f"{opt.output_path}/video_{opt.world}_{opt.stage}_{opt.trained_model_iter}.mp4")
    model=PPO(env.observation_space.shape[0],len(actions))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(f"{opt.saved_path}/ppo_super_mario_bros_{opt.trained_model_world}_{opt.trained_model_stage}_{opt.trained_model_iter}"))
        model.cuda()
    else:
        model.load_state_dict(torch.load(f"{opt.saved_path}/ppo_super_mario_bros_{opt.trained_model_world}_{opt.trained_model_stage}_{opt.trained_model_iter}",map_location=lambda storage,loc:storage))
    model.eval()
    state=torch.from_numpy(env.reset())
    while True:
        if torch.cuda.is_available():
            state=state.cuda()
        logits,value=model(state)
        policy=F.softmax(logits,dim=1)
        action=torch.argmax(policy).item()
        state,reward,done,info=env.step(action)
        state=torch.from_numpy(state)
        print("not get")
        if info["flag_get"]:
            print(f"World {opt.world} stage {opt.stage} completed.")
            break

if __name__=="__main__":
    opt=get_args()
    test(opt)
