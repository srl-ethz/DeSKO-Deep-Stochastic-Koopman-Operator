import gym
import numpy as np
import pandas as pd
from utils import generate_reference_for_trunk
from gym.envs import mujoco
SEED = None


reference_ = pd.read_csv('halfcheetah_states.csv')
reference = reference_.values



VARIANT = {


    # 'env_name': 'GRN',
    'env_name': 'GRN_observation_noise',
    # 'env_name': 'GRN_process_noise',
    #  'env_name': 'HalfCheetahEnv_cost',
    # 'env_name': 'cartpole_cost',
    # 'env_name': 'cartpole_random',    #with process noise
    # 'env_name': 'cartpole_observation_noise',
    # 'env_name': 'trunk_arm_sim',

    #training prams
    #  'alg_name': 'DKO',  # (Deep Koopman Operator)
     'alg_name': 'DeSKO',   # (Deep Stochastic Koopman Operator)
    #  'alg_name': 'MLP',   # (Feedforward NN)

    'additional_description': '',

    'train_model': True,
    # 'train_model': False,
    # 'continue_training':True,
    'continue_training': False,
    # 'eval_control': True,
    'eval_control': False,
    # 'store_hyperparameter':True,
    'store_hyperparameter': False,  # store hyperparameters even while evaluation
    'save_frequency': 5,

    # 'import_saved_data': True,
    'import_saved_data': False,

    # 'continue_data_collection': True,
    'continue_data_collection': False,
    'collect_data_with_controller': True,
    # 'collect_data_with_controller': False,


    # 'evaluation_form': 'dynamic',
    # 'evaluation_form': 'impulse',
    'evaluation_form': 'constant_impulse',    #choose magnitude in EVAL_PARAMS
    # 'evaluation_form': 'various_disturbance',
    #'evaluation_form': 'param_variation',
    # 'evaluation_form': 'trained_disturber',

    'num_of_trials': 4,  # number of random seeds
    'eval_list': [
        'DeSKO',
    ],
    'trials_for_eval': [str(i) for i in range(0, 10)],
}

VARIANT['log_path']='/'.join(['./log', VARIANT['env_name'], VARIANT['alg_name'] + VARIANT['additional_description']])

ENV_PARAMS = {
    'cartpole_cost': {
        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': True,

        ### MPC params
        'reference': np.array([0,0,0,0], dtype=np.float32),
        'Q': np.diag([1., .1, 10., 0.01]),
        'R': np.diag([0.1]),
        'end_weight':100.,
        'control_horizon': 6,
        'MPC_pred_horizon': 16,
        'apply_state_constraints': False,
        'apply_action_constraints': True,
    },

    'cartpole_random': {
        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,

        ### MPC params
        'reference': np.array([0, 0, 0, 0], dtype=np.float32),
        'Q': np.diag([1., .1, 10., 0.01]),
        'R': np.diag([0.1]),
        'end_weight': 100.,
        'control_horizon': 6,
        'MPC_pred_horizon': 16,
        'apply_state_constraints': False,
        'apply_action_constraints': True,
    },

    'cartpole_observation_noise': {

        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        # 'eval_render': True,
        'eval_render': False,

        ### MPC params
        'reference': np.array([0, 0, 0, 0], dtype=np.float32),
        'Q': np.diag([1., .1, 10., 0.01]),
        'R': np.diag([0.1]),
        'end_weight': 100.,
        'control_horizon': 6,
        'MPC_pred_horizon': 16,
        'apply_state_constraints': False,
        'apply_action_constraints': True,
    },



    'trunk_arm_sim': {
        'max_ep_steps': 250,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'eval_render': False,
        # 'eval_render': False,

        ### MPC params
        'reference': generate_reference_for_trunk(2),
        'Q': np.diag(np.concatenate([1.*np.ones([4], dtype=np.float32), 0*np.ones([4], dtype=np.float32)])),
        'R': np.diag(0.1*np.ones([6])),
        'end_weight': 50.,
        'control_horizon': 6,
        'MPC_pred_horizon': 16,
        'apply_state_constraints': False,
        'apply_action_constraints': True,
    },


    'GRN': {
        'max_ep_steps': 400,
        'max_global_steps': int(1e5),
        'max_episodes': int(1e5),
        'disturbance dim': 2,
        'eval_render': False,

        ### MPC params
        'reference': np.array([0, 0, 0, 6, 0, 0], dtype=np.float32),
        'Q': np.diag([0., 0.,  0., 1., 0., 0.]),
        'R': np.diag(0.1*np.ones([3])),
        'end_weight': 100.,
        'control_horizon': 6,
        'MPC_pred_horizon': 16,
        'apply_state_constraints': False,
        'apply_action_constraints': True,

    },

    'GRN_observation_noise': {
        'max_ep_steps': 400,
        'max_global_steps': int(1e5),
        'max_episodes': int(1e5),
        'disturbance dim': 2,
        'eval_render': False,

        ### MPC params
        'reference': np.array([0, 0, 0, 6, 0, 0], dtype=np.float32),
        'Q': np.diag([0., 0., 0., 1., 0., 0.]),
        'R': np.diag(0.1 * np.ones([3])),
        'end_weight': 100.,
        'control_horizon': 6,
        'MPC_pred_horizon': 16,
        'apply_state_constraints': False,
        'apply_action_constraints': True,

    },

    'GRN_process_noise': {
        'max_ep_steps': 400,
        'max_global_steps': int(1e5),
        'max_episodes': int(1e5),
        'disturbance dim': 2,
        'eval_render': False,

        ### MPC params
        'reference': np.array([0, 0, 0, 6, 0, 0], dtype=np.float32),
        'Q': np.diag([0., 0., 0., 1., 0., 0.]),
        'R': np.diag(0.1 * np.ones([3])),
        'end_weight': 100.,
        'control_horizon': 6,
        'MPC_pred_horizon': 16,
        'apply_state_constraints': False,
        'apply_action_constraints': True,

    },


    'HalfCheetahEnv_cost': {
        'max_ep_steps': 500,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 6,
        'eval_render': True,


        'reference': reference,
        'Q': np.diag([2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2, 1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1]),
        'R': np.diag(0.01 * np.ones([6])),
        'end_weight': 100.,
        'control_horizon': 6,
        'MPC_pred_horizon': 16,
        'apply_state_constraints': False,
        'apply_action_constraints': True,
    },
}

ALG_PARAMS = {
    'DKO': {
        'controller_name': 'MPC',
        'iter_of_data_collection':3,
        'learning_rate': 1e-4,
        # 'encoder_struct': [256, 128, 64],
        'encoder_struct': [128, 100, 80],
        'latent_dim': 10,
        'pred_horizon': 16,
        'l2_regularizer': 0.01,
        'val_frac': 0.1,
        'batch_size': 128,
        'num_epochs': 400,
        'decay_rate': 0.9,
        'decay_steps': 10,
        'total_data_size': 40000,
        'further_collect_data_size': 1000,
        'segment_of_test': 8,
        'n_subseq': 220,  # number of subsequences to divide each sequence into
        'store_last_n_paths': 10,  # number of trajectories for evaluation during training
        'start_of_trial': 0,
        'history_horizon': 0,
    },


    'DeSKO': {
        # 'controller_name': 'Stochastic_MPC_with_observation',
        'controller_name': 'Stochastic_MPC_with_observation_v2',
        # 'controller_name': 'Stochastic_MPC_with_motion_planning',    #use it for HalfCheetah environment
        'n_of_random_seeds': 40,
        'iter_of_data_collection': 3,
        'learning_rate': 1e-3,
        'decay_rate': 0.9,
        'decay_steps': 10,
        'activation': 'relu',
        # 'activation': 'elu',
        # 'encoder_struct': [128, 100, 80],
        # 'encoder_struct': [256, 128, 64],
        'encoder_struct': [256, 100, 80],
        'latent_dim': 20,
        'pred_horizon': 16,
        'alpha': .1,
        'target_entropy': -20.,
        'l2_regularizer': 0.1,
        'val_frac': 0.1,
        'batch_size': 128,
        'num_epochs': 400,       #400
        'total_data_size': 40000,
        'further_collect_data_size': 1000,
        'segment_of_test': 8,
        'n_subseq': 220,  # number of subsequences to divide each sequence into
        'store_last_n_paths': 10,  # number of trajectories for evaluation during training
        'start_of_trial': 0,
        'history_horizon': 0,
    },

    'MLP': {
        'controller_name': 'MPC',
        'iter_of_data_collection': 3,
        'learning_rate': 1e-4,
        'encoder_struct': [256, 128, 64],
        # 'encoder_struct': [128, 100, 80],
        'pred_horizon': 16,
        'latent_dim': 0,
        'l2_regularizer': 0.01,
        'val_frac': 0.1,
        'batch_size': 128,
        'num_epochs': 400,
        'decay_rate': 0.9,
        'decay_steps': 10,
        'total_data_size': 40000,
        'further_collect_data_size': 1000,
        'segment_of_test': 8,
        'n_subseq': 220,  # number of subsequences to divide each sequence into
        'store_last_n_paths': 10,  # number of trajectories for evaluation during training
        'start_of_trial': 0,
        'history_horizon': 0,
    },
}


EVAL_PARAMS = {
    ### param_variation is only applicable to the cartpole environment

    'param_variation': {
        'param_variables': {
            'mass_of_pole': np.arange(0.05, 0.55, 0.05),  # default is 0.1
            'length_of_pole': np.arange(0.1, 2.1, 0.1),  # default is 0.5
            'mass_of_cart': np.arange(0.1, 2.1, 0.1),    # default is 1.0
            # 'gravity': np.arange(9, 10.1, 0.1),  # 0.1
        },
        'grid_eval': True,
        # 'grid_eval': False,
        'grid_eval_param': ['length_of_pole', 'mass_of_cart'],
        'num_of_paths': 2,   # number of path for evaluation
    },

    'impulse': {
        # 'magnitude_range': np.arange(150, 160, 5),
        # 'magnitude_range': np.arange(90, 155, 5),
        # 'magnitude_range': np.arange(10000, 40000, 5000),
        'magnitude_range': np.arange(0.1, 0.5, .05),
        'num_of_paths': 5,   # number of path for evaluation
        'impulse_instant': 200,
    },

    'constant_impulse': {
        # 'magnitude_range': np.arange(120, 125, 5),
        # 'magnitude_range': np.arange(10000, 40000, 5000), ## trunk arm
        'magnitude_range': np.arange(80, 155, 5),       # cartpole
        # 'magnitude_range': np.arange(80, 155, 5),
        # 'magnitude_range': np.arange(0.2, 2.2, .2),
        # 'magnitude_range': np.arange(0.1, 0.5, .05),   #for GRN (oscillator)
        # 'magnitude_range': np.arange(0.1, 1.0, .05),
        'num_of_paths': 20,   # number of path for evaluation
        'impulse_instant': 20,
    },
    'various_disturbance': {
        'form': ['sin', 'tri_wave'][0],
        'period_list': np.arange(2, 11, 1),
        # 'magnitude': np.array([1, 1, 1, 1, 1, 1]),
        'magnitude': np.array([10000]),
        # 'grid_eval': False,
        'num_of_paths': 5,   # number of path for evaluation
    },


    'dynamic': {
        'eval_additional_description': '',
        'num_of_paths': 10,   # number of path for evaluation
        'plot_average': True,
        # 'plot_average': False,
        'directly_show': True,
        'dimension_of_interest': 3,
    },
}

for key in ENV_PARAMS[VARIANT['env_name']].keys():
    VARIANT[key] = ENV_PARAMS[VARIANT['env_name']][key]
for key in ALG_PARAMS[VARIANT['alg_name']].keys():
    VARIANT[key] = ALG_PARAMS[VARIANT['alg_name']][key]
for key in EVAL_PARAMS[VARIANT['evaluation_form']].keys():
    VARIANT[key] = EVAL_PARAMS[VARIANT['evaluation_form']][key]
# VARIANT['eval_params']=EVAL_PARAMS[VARIANT['evaluation_form']]



def get_env_from_name(args):
    name = args['env_name']
    if name == 'cartpole_cost':
        from envs.ENV_V1 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_random':
        from envs.ENV_V2 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'cartpole_observation_noise':
        from envs.ENV_V3 import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'trunk_arm_sim':
        from envs.trunk_arm_simulator import TrunkArmSimulator as dreamer
        if 'reference' in args.keys():
            env = dreamer(args['reference'])
        else:
            env = dreamer()
        env = env.unwrapped
    elif name == 'GRN':
        from envs.oscillator import oscillator as env
        env = env(args['reference'])
        env = env.unwrapped
    elif name == 'GRN_observation_noise':
        from envs.oscillator_observation_noise import oscillator as env
        env = env()
        env = env.unwrapped
    elif name == 'GRN_process_noise':
        from envs.oscillator_process_noise import oscillator as env
        env = env()
        env = env.unwrapped
    elif name == 'HalfCheetahEnv_cost':
        #from mujoco.half_cheetah import mujoco.HalfCheetahEnv as env
        from envs.half_cheetah_cost import HalfCheetahEnv_cost as env
        env = env(args['reference'])
        #env = dreamer()
        env = env.unwrapped

    else:
        env = gym.make(name)
        env = env.unwrapped

    env.seed(SEED)
    return env


def get_model(name):

    if name == 'DKO':
        from DKO import Koopman as build_func

    elif name == 'DeSKO':
        from DeSKO import Koopman as build_func
    elif name == 'MLP':
        from MLP import MLP as build_func

    return build_func

def get_controller(model, args):
    if args['controller_name'] == 'MPC':
        from controller import MPC as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'MPC_with_motion_planning':
        from controller import MPC_with_motion_planning as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Time_varying_MPC':
        from controller import Time_varying_MPC as build_func
        controller = build_func(model, args)

    elif args['controller_name'] == 'Stochastic_MPC':
        from controller import Stochastic_MPC as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_v2':
        from controller import Stochastic_MPC_v2 as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_v3':
        from controller import Stochastic_MPC_v3 as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_with_observation':
        from controller import Stochastic_MPC_with_observation as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_with_observation_v2':
        from controller import Stochastic_MPC_with_observation_v2 as build_func
        controller = build_func(model, args)
    elif args['controller_name'] == 'Stochastic_MPC_with_motion_planning':
        from controller import Stochastic_MPC_with_motion_planning as build_func
        controller = build_func(model, args)

    else:
        print('controller does not exist')
        raise NotImplementedError
    return controller

def store_hyperparameters(path, args):
    np.save(path + "/hyperparameters.npy", args)


def restore_hyperparameters(path):
    args = np.load(path + "/hyperparameters.npy", allow_pickle=True).item()
    return args
