import argparse
import os, shutil

import wandb
# wandb.login(key='913841cb22c908099db4951c258f4242c1d1b7aa')
os.environ['WANDB_API_KEY'] = '913841cb22c908099db4951c258f4242c1d1b7aa'
os.environ['WANDB_SILENT'] = 'true'

from tqdm import tqdm

import random
import numpy as np
import torch

import gymnasium as gym
from PPO import PPO


def evaluate_policy(env, model, turns=3):
    total_scores = 0

    for j in range(turns):
        state, info = env.reset()

        end = False
        while not end:
            # Take deterministic actions at test time
            action, action_logprobe = model.select_action(state, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            end = (done or truncated)

            total_scores += reward
            state = obs 

    return int(total_scores // turns)
    

def render_environment(env, model, args):
    run = wandb.init(
        project="RL-Project-2",
        name="Evaluation Rendering", 
        config={
            "checkpoint": args.checkpoint,
        }
    )

    try:
        print(f'Render Evaluation -> Env:{args.env}')

        total_steps = 0
        while True:
            reward = evaluate_policy(env, model, turns=1)
            total_steps += 1

            wandb.log({
                "reward": reward,
            })
            
            print(f'Reward Episode - {total_steps}: {reward}')

    except (AssertionError, KeyboardInterrupt) as err:
        print(f'Exiting Render Mode ...')

    finally:
        run.finish()

    return


def _set_reproducable(seed=0):
    print(f'Random Seed: {seed}')

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def main(args):
    if not args.random_train:
        _set_reproducable(args.seed)

    env = gym.make(args.env, render_mode = "human" if args.render else None)
    eval_env = gym.make(args.env)
    

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    args.max_e_steps = env._max_episode_steps
    
    print('Env:', args.env,'  state_dim:', args.state_dim,'  action_dim:', args.action_dim,'   Random Seed:', args.seed, '  max_e_steps:', args.max_e_steps)
    print('\n')
    
    model = PPO(**vars(args))

    if args.checkpoint:
        model.load(args.env, args.checkpoint)

    if not os.path.exists('./model'): 
        os.mkdir('./model')

    if args.render:
        render_environment(env, model, args)

        env.close()
        eval_env.close()
        return
    
    run = wandb.init(
        project="RL-Project-2",
        config={
            "env": args.env,
            "seed": args.seed,
            "max_train_steps": args.max_train_steps,
            "log_step": args.log_step,
            "save_step": args.save_step,
            "eval_step": args.eval_step,
            "T_horizon": args.T_horizon,
            "T_horizon": args.T_horizon,
            "K_epochs": args.K_epochs,
            "gamma": args.gamma,
            "lambda": args.lambd,
            "epsilon": args.epsilon,
            "hidden_size": args.hidden_size,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "entropy_coef": args.entropy_coef,
            "entropy_coef_decay": args.entropy_coef_decay,
        }
    )

    traj_length, total_steps = 0, 0
    env_seed = args.seed
    
    try:
        pbar = tqdm(total = args.max_train_steps)
        while total_steps < args.max_train_steps:
            state, info = env.reset(seed=env_seed)  # Do not use args.seed directly, as it will overfit to that particlar seed
            env_seed += 1

            end = False
            while not end:
                # Iteracting with Environment
                action, action_logprob = model.select_action(state, deterministic=False) # use stochastic when training
                obs, reward, done, truncated, info = env.step(action)

                end = (done or truncated)
                    
                wandb.log({
                    "Interact/reward": reward,
                    "Interact/total_steps": total_steps,
                })

                model.store_transition(state, action, reward, obs, action_logprob, done, end, idx = traj_length)
                state = obs 

                traj_length += 1
                total_steps += 1

                # Train
                if traj_length % args.T_horizon == 0:
                    model.train()
                    traj_length = 0

                # Evaluate
                if total_steps % args.eval_step == 0:
                    reward = evaluate_policy(eval_env, model, turns=3) # evaluate the policy for 3 times, and get averaged result
                    
                    wandb.log({
                        "Eval/reward": reward,
                        "Eval/total_steps": total_steps,
                        "Eval/eval_steps": total_steps // args.eval_step,
                    })

                    print('EnvName:', args.env, 'seed:', args.seed, 'steps: {}k'.format(int(total_steps/1000)), 'reward:', reward)

                # Save 
                if total_steps % args.save_step == 0:
                    model.save(args.env, total_steps)

            pbar.update(1)
    
    except (AssertionError, KeyboardInterrupt) as err:
        print(f'Exiting Training ... : {total_steps} steps')

    finally:
        env.close()
        eval_env.close()
        run.finish()
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default=None, help='Set Device type')
    
    parser.add_argument('--random_train', help='Set device type', action='store_true')
    parser.add_argument('--seed', type=int, default=0, help='Seed for Randomness')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Gym Environment with Discrete Action Space')
    parser.add_argument('--render', help='Rendering Mode', action='store_true')
    parser.add_argument('--checkpoint', type=int, default=None, help='Path to pretrained model')

    parser.add_argument('--max_train_steps', type=int, default=5e7, help='Max training steps')
    parser.add_argument('--log_step', type=int, default=10, help='Log Step')
    parser.add_argument('--save_step', type=int, default=1e5, help='Model saving interval, in steps.')
    parser.add_argument('--eval_step', type=int, default=5e3, help='Model evaluating interval, in steps.')

    parser.add_argument('--T_horizon', type=int, default=2048, help='length of long trajectory')
    parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
    parser.add_argument('--epsilon', type=float, default=0.2, help='PPO Clipping value')

    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')

    parser.add_argument('--entropy_coef', type=float, default=0, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
    # parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
    # parser.add_argument('--advantage_normalization', type=str2bool, default=False, help='Advantage normalization')

    args = parser.parse_args()
    
    if not args.device:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    main(args)
    
