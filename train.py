from replay_memory import ReplayMemory
from variant import *
from utils import visualize_predictions

import logger
from robustness_eval import *
import time

def main():
    args = VARIANT
    root_dir = args['log_path']
    env = get_env_from_name(args)
    args['state_dim'] = env.observation_space.shape[0]
    args['act_dim'] = env.action_space.shape[0]
    args['s_bound_low'] = env.observation_space.low
    args['s_bound_high'] = env.observation_space.high
    args['a_bound_low'] = env.action_space.low
    args['a_bound_high'] = env.action_space.high
    os.makedirs(root_dir, exist_ok=True)
    if args['train_model']:
        store_hyperparameters(root_dir, args)
    for i in range(args['start_of_trial'], args['start_of_trial'] + args['num_of_trials']):


        args['log_path'] = root_dir + '/' + str(i)
        print('logging to ' + args['log_path'])

        model = train(args, env)
        if args['eval_control']:
            args['log_path'] = root_dir
            if args['store_hyperparameter']:
                store_hyperparameters(root_dir, args)
            controller = get_controller(model, args)
            controller._build_controller()
            controller.check_controllability()

            if args['evaluation_form'] == 'dynamic':
                dynamic(controller, env, args, args)
            elif args['evaluation_form'] == 'constant_impulse':
                constant_impulse(controller, env, args)
            # simple_validation(controller, env, args)
        tf.reset_default_graph()


def train(args, env):

    build_func = get_model(args['alg_name'])
    model = build_func(args)
    if args['train_model'] is False:
        if args['env_name'] == 'linear_sys':
            model.A_result = env.A.T
            model.B_result = env.B.T
        else:
            success = model.restore(args['log_path'])
            if not success:
                print(args['log_path'] + ' does not exist')
                raise NotImplementedError
        return model
    if args['continue_training']:
        success = model.restore(args['log_path'])
        if not success:
            print(args['log_path'] + ' does not exist')
            raise NotImplementedError

    logger.configure(dir=args['log_path'], format_strs=['csv'])
    # Generate data
    [shift, scale, shift_u, scale_u] = model.get_shift_and_scale()
    # Generate training data
    replay_memory = ReplayMemory(args, shift, scale, shift_u, scale_u, env, predict_evolution=True)
    model.set_shift_and_scale(replay_memory)


    # Define counting variables
    count = 0
    count_decay = 0
    decay_epochs = []

    # Initialize variable to track validation score over time
    old_score = 1e20


    lr = args['learning_rate']

    for e in range(args['num_epochs']):


        # Initialize loss
        loss = 0.0
        val_loss = 0.0
        loss_count = 0
        b = 0
        replay_memory.reset_batchptr_train()

        # Loop over batches
        while b < replay_memory.n_batches_train:
            start = time.time()

            # Get inputs
            batch_dict = replay_memory.next_batch_train()
            out = model.learn(batch_dict, lr, args)
            b += 1



        model.store_Koopman_operator(replay_memory)
        # Evaluate loss on validation set
        score = model.calc_val_loss(replay_memory)
        [logger.logkv(key, out[key]) for key in out.keys()]
        # logger.logkv('train_loss', loss)
        logger.logkv('epoch', e)
        logger.logkv('validation_loss', score)
        logger.logkv('learning_rate', lr)

        logger.dumpkvs()
        string_to_print = [args['alg_name'] + args['additional_description'], '|']
        string_to_print.extend(['epoch:', str(e), '|'])
        [string_to_print.extend([key, ':', str(round(out[key], 2)), '|']) for key in out.keys()]
        string_to_print.extend(['validation_loss:', str(round(score, 2)), '|'])
        string_to_print.extend(['learning_rate:', str(round(lr, 4)), '|'])
        print(''.join(string_to_print))
        # print('Validation Loss: {0:f}'.format(score))

        # Set learning rate
        if (old_score - score) < -0.01 and e >= 8:
            count_decay += 1
            decay_epochs.append(e)
            # if len(decay_epochs) >= 3 and np.sum(np.diff(decay_epochs)[-2:]) == 2:
            #     break

            # lr = args['learning_rate'] * (args['decay_rate'] ** count_decay)
            # print('setting learning rate to ', lr)
        ## stair decay
        if (e + 1) % args['decay_steps'] == 0:
            lr = lr * args['decay_rate']
        # ## constant decay
        # frac = 1.0 - e / args['num_epochs']
        # lr = args['learning_rate'] * frac
        # print('setting learning rate to ', lr)

        old_score = score
        if e % args['save_frequency'] == 0:

            model.save_result(args['log_path'], verbose=False )
            # print("model saved to {}".format(args['log_path']))
            visualize_predictions(args, model, replay_memory, env, e)

    return model



if __name__ == '__main__':
    main()