import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_predictions(args, model, replay_memory, env, e=0):
    """Plot predictions for a system against true time evolution
    Args:
        args: Various arguments and specifications
        sess: TensorFlow session
        net: Neural network dynamics model
        replay_memory: Object containing training/validation data
        env: Simulation environment
        e: Current training epoch
    """
    # Get inputs (test trajectory that is twice the size of a standard sequence)
    # x = np.zeros((args['batch_size'], args['pred_horizon'], args['state_dim']), dtype=np.float32)
    # u = np.zeros((args['batch_size'], args['pred_horizon'] - 1, args['act_dim']), dtype=np.float32)
    x = replay_memory.x_test
    u = replay_memory.u_test
    furture_states = []
    t = args['history_horizon']
    plot_x_tick = range(x.shape[0])

    while t + args['pred_horizon'] < x.shape[0]:
        pred_trajectory = [x[t]]
        input = [x[t - i] for i in range(args['history_horizon']+1)]
        input.reverse()
        pred_trajectory.extend(model.make_prediction(input, u[t:t + args['pred_horizon']-1], args))
        furture_states.append([
            plot_x_tick[t:t + args['pred_horizon']],
            np.array(pred_trajectory)])
        t += args['segment_of_test']

    # preds = preds[1:]
    #
    # # Find mean, max, and min of predictions
    # pred_mean = np.mean(preds, axis=0)
    # pred_std = np.std(preds, axis=0)
    # pred_min = np.amin(preds, axis=0)
    # pred_max = np.amax(preds, axis=0)

    # diffs = np.linalg.norm(
    #     (preds[:, :args['pred_horizon']] - sess.run(net.shift)) / sess.run(net.scale) - x[0, :args['pred_horizon']], axis=(1, 2))
    # best_pred = np.argmin(diffs)
    # worst_pred = np.argmax(diffs)
    #
    # # Plot different quantities
    # x = x * sess.run(net.scale) + sess.run(net.shift)
    #
    # # # Find indices for random predicted trajectories to plot
    # ind0 = best_pred
    # ind1 = worst_pred
    #
    # Plot values
    plt.close()
    # plt.figure(figsize=(9, 6))
    f, axs = plt.subplots(args['state_dim']+args['act_dim'], sharex=True, figsize=(15, 15))
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')


    for i in range(args['state_dim']):
        axs[i].plot(plot_x_tick, x[:, i], 'k')
        for obj in furture_states:
            axs[i].plot(obj[0], obj[1][:, i], 'r')
        # axs[i].fill_between(range(1, 2 * args['pred_horizon']), pred_min[:, i], pred_max[:, i], facecolor='blue', alpha=0.5)
        # axs[i].set_ylim([np.amin(x[0, :, i]) - 0.2, np.amax(x[0, :, i]) + 0.2])3

    for i in range(args['act_dim']):
        row = i + args['state_dim']
        axs[row].plot(plot_x_tick[:-1], u[:, i], 'b')

    plt.xlabel('Time Step')
    # plt.xlim([1, 2 * args['pred_horizon'] - 1])
    os.makedirs(args['log_path']+'/predictions', exist_ok=True)
    plt.savefig(args['log_path']+'/predictions/predictions_' + str(e) + '.png')


def mlp(input, sizes, activation, output_activation=None, name="", regularizer = None, reuse=None):
    """Creates a multi-layered perceptron using Tensorflow.

    Args:
        sizes (list): The size of each of the layers.

        activation (function): The activation function used for the
            hidden layers.

        output_activation (function, optional): The activation function used for the
            output layers. Defaults to tf.keras.activations.linear.

        regularizer (function, optional): Regularizer used to prevent overfitting

        name (st, optional): A nameprefix that is added before the layer name. Defaults
            to an empty string.

    Returns:
        output( Tensor): Output of the multi-layer perceptron
    """
    if reuse is None:
        trainable = True
    else:
        trainable = False

    with tf.variable_scope(name, reuse=reuse):
        # Create model
        for j in range(len(sizes) - 1):
            input = input if j == 0 else output
            act = activation if j < len(sizes) - 2 else output_activation
            output = tf.layers.dense(
                    input,
                    sizes[j + 1],
                    activation=act,
                    name=name + "/l{}".format(j + 1),
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer,
                    trainable=trainable
            )
            # output = tf.layers.dropout(output, rate=0.3)
    return output


def generate_reference_for_trunk(num_of_segment):
    threshold = [np.pi, np.pi/4]
    reference = []
    for n in range(num_of_segment):
        # theta = np.random.uniform(0, 1, 1) * threshold[0]
        # phi = np.random.uniform(-1, 1, 1) * threshold[1]
        theta = 1. * threshold[0]
        phi = 1. * threshold[1]
        reference.append(np.squeeze(np.array([theta * np.cos(phi), theta * np.sin(phi)])))

    reference.append(np.zeros([2*num_of_segment], dtype=np.float32))
    return np.concatenate(reference, axis=0)