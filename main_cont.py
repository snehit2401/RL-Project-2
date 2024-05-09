import argparse
import os, shutil
from datetime import datetime

import wandb
# wandb.login(key='913841cb22c908099db4951c258f4242c1d1b7aa')
os.environ['WANDB_API_KEY'] = '913841cb22c908099db4951c258f4242c1d1b7aa'
os.environ['WANDB_SILENT'] = 'true'

from tqdm import tqdm

import random
import numpy as np
import torch

import gymnasium as gym
from PPO_cont import PPO


def Action_adapter(a,max_action):
	#from [0,1] to [-max,max]
	return  2*(a-0.5)*max_action

def Reward_adapter(r):
	# For BipedalWalker
    # if r <= -100: r = -1

	# For Pendulum-v0
    r = (r + 8) / 8
    return r


def evaluate_policy(env, agent, max_action, turns):
	total_scores = 0
	for j in range(turns):
		s, info = env.reset()
		done = False
		while not done:
			a, logprob_a = agent.select_action(s, deterministic=True) # Take deterministic actions when evaluation
			act = Action_adapter(a, max_action)  # [0,1] to [-max,max]
			s_next, r, dw, tr, info = env.step(act)
			done = (dw or tr)

			total_scores += r
			s = s_next

	return total_scores/turns


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
            reward = evaluate_policy(env, model, args.max_action, turns=1)
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
    torch.cuda.manual_seed_all(seed) # safe to call when cuda not available

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def main(args):
    # Build Env
    env = gym.make(args.env, render_mode = "human" if args.render else None)
    eval_env = gym.make(args.env)

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    args.max_steps = env._max_episode_steps
    print('Env:',args.env,'  state_dim:',args.state_dim,'  action_dim:',args.action_dim,
          '  max_a:',args.max_action,'  min_a:',env.action_space.low[0], 'max_steps', args.max_steps)

    # Seed Everything
    _set_reproducable(args.seed)

    if not os.path.exists('./checkpoints'): 
        os.mkdir('./checkpoints')
   
    model = PPO(**vars(args))

    if args.checkpoint:
        model.load(args.env, args.checkpoint)

    if args.render:
        render_environment(env, model, args)

        env.close()
        eval_env.close()
        return

    run = wandb.init(
        project="RL-Project-2",
        config={
            "env": args.env,
        }
    )

    traj_length, total_steps, train_steps = 0, 0, 0
    env_seed = args.seed

    try:
        pbar = tqdm(total = args.Max_train_steps)
        while total_steps < args.Max_train_steps:
            s, info = env.reset(seed=env_seed) # Do not use args.seed directly, or it can overfit to args.seed
            env_seed += 1
            done = False
            curr_steps = total_steps

            '''Interact & trian'''
            while not done:
                '''Interact with Env'''
                a, logprob_a = model.select_action(s, deterministic=False) # use stochastic when training
                act = Action_adapter(a, args.max_action) #[0,1] to [-max,max]
                s_next, r, dw, tr, info = env.step(act) # dw: dead&win; tr: truncated
                r = Reward_adapter(r)
                done = (dw or tr)
                
                if total_steps % self.log_interval == 0:
                    wandb.log({
                        "Interact/reward": r,
                        "Interact/total_steps": total_steps,
                    })

                '''Store the current transition'''
                model.put_data(s, a, r, s_next, logprob_a, done, dw, idx = traj_length)
                s = s_next

                traj_length += 1
                total_steps += 1

                '''Update if its time'''
                if traj_length % args.T_horizon == 0:
                    train_steps = model.train(total_steps, train_steps)
                    traj_length = 0

                '''Record & log'''
                if total_steps % args.eval_interval == 0:
                    score = evaluate_policy(eval_env, model, args.max_action, turns=3) # evaluate the policy for 3 times, and get averaged result
                    
                    print('EnvName:',args.env,'seed:',args.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', score)

                    wandb.log({
                        "Eval/reward": score,
                        "Eval/total_steps": total_steps,
                        "Eval/eval_steps": total_steps // args.eval_interval,
                    })

                '''Save model'''
                if total_steps % args.save_interval==0:
                    model.save(args.env, int(total_steps/1000))

            pbar.update(total_steps - curr_steps)

    except (AssertionError, KeyboardInterrupt) as err:
        print(f'Exiting Training ... : {total_steps} steps')

    finally:
        pbar.close()
        env.close()
        eval_env.close()
        run.finish()

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dvc', type=str, default=None, help='running device: cuda or cpu')
    parser.add_argument('--env', type=str, default='Pendulum-v1', help='Gym Environment')

    parser.add_argument('--render', help='Rendering Mode', action='store_true')
    parser.add_argument('--checkpoint', type=int, default=None, help='Path to pretrained model')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
    parser.add_argument('--Max_train_steps', type=int, default=int(5e7), help='Max training steps')
    parser.add_argument('--save_interval', type=int, default=int(1e5), help='Model saving interval, in steps.')
    parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')
    parser.add_argument('--log_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
    parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
    parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
    parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
    parser.add_argument('--net_width', type=int, default=150, help='Hidden net width')
    parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
    parser.add_argument('--c_lr', type=float, default=2e-4, help='Learning rate of critic')
    parser.add_argument('--l2_reg', type=float, default=1e-3, help='L2 regulization coefficient for Critic')
    parser.add_argument('--a_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of actor')
    parser.add_argument('--c_optim_batch_size', type=int, default=64, help='lenth of sliced trajectory of critic')
    parser.add_argument('--entropy_coef', type=float, default=1e-3, help='Entropy coefficient of Actor')
    parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')

    args = parser.parse_args()
    if args.dvc is None:
        args.dvc = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    print(args)

    main(args)





