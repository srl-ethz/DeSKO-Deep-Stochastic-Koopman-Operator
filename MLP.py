import tensorflow as tf

import numpy as np
import os
import tensorflow_probability as tfp


class MLP(object):



    def __init__(
        self,
        args,
        **kwargs
    ):
        """
        Args:
            latent_dim (int): Dimension of the observation space.

            act_dim (int): Dimension of the action space.

            hidden_sizes (list): Sizes of the hidden layers.

            activation (function): The hidden layer activation function.

            output_activation (function, optional): The activation function used for
                the output layers. Defaults to tf.keras.activations.linear.

            name (str, optional): The Lyapunov critic name. Defaults to
                "lyapunov_critic".
        """
        self.sess = tf.Session()
        self.diagnotics = {}
        self.opt_list = []
        self.pred_horizon = args['pred_horizon']
        self.x_input = tf.placeholder(tf.float32, [None, args['pred_horizon'], args['state_dim']], 'x')
        self.a_input = tf.placeholder(tf.float32, [None, args['pred_horizon']-1, args['act_dim']], 'a')

        self.shift = tf.Variable(np.zeros(args['state_dim']), trainable=False, name="state_shift", dtype=tf.float32)
        self.scale = tf.Variable(np.zeros(args['state_dim']), trainable=False, name="state_scale", dtype=tf.float32)
        self.shift_u = tf.Variable(np.zeros(args['act_dim']), trainable=False, name="action_shift", dtype=tf.float32)
        self.scale_u = tf.Variable(np.zeros(args['act_dim']), trainable=False, name="action_scale", dtype=tf.float32)


        self.lr = tf.placeholder(tf.float32, None, 'learning_rate')
        self.l2_reg = tf.contrib.layers.l1_regularizer(args['l2_regularizer'])
        # self.l1_reg = tf.contrib.layers.l1_regularizer(args['l2_regularizer'])


        self._create_encoder(args)

        self._create_forward_recursive(args)
        self._create_optimizer(args)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()


    def _create_encoder(self, args):
        self.sizes = [args['state_dim']+args['act_dim']]+args['encoder_struct'] + [args['state_dim']]
        input = tf.concat([self.x_input[:,0], self.a_input[:,0]],axis=1)
        self.encoder, self.weights, self.biases = mlp(input,
                           self.sizes, tf.nn.relu, name='encoder',
                           regularizer=self.l2_reg)

    def _forward_pred(self, input):

        for i in range(0, len(self.sizes) - 2):
            input = tf.nn.leaky_relu(tf.add(tf.matmul(input, self.weights[i]), self.biases[i]))

        output = tf.add(tf.matmul(input, self.weights[-1]), self.biases[-1])
        return output

    def _create_forward_recursive(self,args):

        def model_predict_cond(pred_time_step, max_iter, state, forward_preds):
            return tf.less(pred_time_step, max_iter)

        def model_predict_iteration(pred_time_step, max_iter, state, forward_preds):
            input = tf.concat([state, self.a_input[:, pred_time_step]],axis=1)
            next_state = self._forward_pred(input)

            forward_preds = forward_preds.write(pred_time_step, next_state)

            return pred_time_step + 1, max_iter, next_state, forward_preds

        i = tf.constant(0)
        horizon = tf.constant(self.pred_horizon-1)
        forward_preds = tf.TensorArray(dtype=tf.float32, size=self.pred_horizon, dynamic_size=True)
        pred_time_step,_, final_state, self.forward_preds = \
            tf.while_loop(cond=model_predict_cond, body=model_predict_iteration,
                          loop_vars=[i, horizon, self.x_input[:,0], forward_preds],)
                          # shape_invariants=[i.get_shape(), horizon.get_shape(), self.x_input[:,0].get_shape(), tf.TensorShape([None])])

        self.forward_preds = [tf.expand_dims(self.forward_preds.read(i, name=None),1)for i in range(self.pred_horizon-1)]
        self.forward_preds = tf.concat(self.forward_preds,axis=1)
    def _create_optimizer(self, args):
        Y = self.x_input[:, 1:]
        forward_pred_loss = tf.losses.mean_squared_error(labels=Y, predictions=self.forward_preds)

        val_x = self.x_input[:, 1:]
        val_y = self.forward_preds
        self.val_loss = tf.losses.mean_squared_error(labels=val_x, predictions=val_y)

        self.loss =forward_pred_loss
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.diagnotics.update({'loss': self.loss})
        self.opt_list.append(self.train)

    def calc_val_loss(self, replay_memory):

        batch_dict = replay_memory.get_all_val_data()
        x = batch_dict['states']
        u = batch_dict['inputs']

        # Construct inputs for network
        feed_in = {}
        feed_in[self.x_input] = x
        feed_in[self.a_input] = u

        # Find loss
        feed_out = self.val_loss
        loss = self.sess.run(feed_out, feed_in)

        return loss

    def learn(self, batch_dict, lr, args):
        x = batch_dict['states']
        a = batch_dict['inputs']

        # Construct inputs for network
        feed_in = {}
        feed_in[self.x_input] = x
        feed_in[self.a_input] = a
        feed_in[self.lr] = lr

        self.sess.run(self.opt_list, feed_in)

        diagnotics = self.sess.run([self.diagnotics[key] for key in self.diagnotics.keys()], feed_in)
        output = {}
        [output.update({key: value}) for (key, value) in zip(self.diagnotics.keys(), diagnotics)]

        return output

    def _create_prediction_model(self, args):

        self.x_t = tf.placeholder(tf.float32, [None, args['state_dim']], 'x_t')
        self.a_t = tf.placeholder(tf.float32, [None, args['pred_horizon']-1, args['act_dim']], 'a_t')
        self.shifted_x_t = (self.x_t - self.shift) / self.scale
        self.shifted_a_t = (self.a_t - self.shift_u) / self.scale_u
        self.encoder_t = mlp(self.shifted_x_t,
                             args['encoder_struct'] + [args['latent_dim']], tf.nn.relu, name='encoder', reuse=True)
        self.phi_t = tf.concat([self.shifted_x_t, self.encoder_t], axis=1)

        forward_pred = []
        phi_t = self.phi_t
        for t in range(args['pred_horizon'] - 1):
            u = self.shifted_a_t[:, t]
            phi_t = tf.matmul(phi_t, self.A_tensor) + tf.matmul(u, self.B_tensor)
            forward_pred.append(phi_t)
        self.future_states = tf.stack(forward_pred, axis=1)[:, :, :args['state_dim']]

    def store_Koopman_operator(self, replay_memory):
        return

    def make_prediction(self, x_t, u, args):
        feed_dict = {}

        x = np.tile(x_t, (self.pred_horizon,1))
        feed_dict[self.x_input] = [x]
        feed_dict[self.a_input] = [u]
        [future_states] = self.sess.run(self.forward_preds, feed_dict)

        return future_states

    def encode(self, x):

        return

    def get_shift_and_scale(self):

        return self.sess.run([self.shift, self.scale, self.shift_u, self.scale_u])

    def set_shift_and_scale(self, replay_memory):

        operates = [
            tf.assign(self.shift, replay_memory.shift_x),
            tf.assign(self.scale, replay_memory.scale_x),
            tf.assign(self.shift_u, replay_memory.shift_u),
            tf.assign(self.scale_u, replay_memory.scale_u)
        ]
        self.sess.run(operates)


    def save_result(self, path, verbose = True):

        os.makedirs(path + "/model", exist_ok=True)

        save_path = self.saver.save(self.sess, path + "/model/model.ckpt")


        # save_path = self.saver.save(self.sess, "\\log\\model.ckpt")

        if verbose is True:
            print("Save to path: ", save_path)

    def restore(self, path):
        model_file = tf.train.latest_checkpoint(path+'/model/')
        if model_file is None:
            success_load = False
            return success_load
        self.saver.restore(self.sess, model_file)
        success_load = True

        return success_load

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
    weights = []
    biases = []

    # construct the neural networks
    with tf.variable_scope(name):
        for i in range(1, len(sizes)):
            weights.append(tf.Variable(tf.random_normal([sizes[i - 1], sizes[i]], stddev=0.1),
                                            name='weights_' + str(i - 1), dtype=tf.float32))
            biases.append(tf.Variable(tf.random_normal([sizes[i]], stddev=0.1),
                                           name='biases_' + str(i - 1), dtype=tf.float32))

    x = input
    for i in range(0, len(sizes) - 2):
        x = activation(tf.add(tf.matmul(x, weights[i]), biases[i]))

    output = tf.add(tf.matmul(x, weights[-1]), biases[-1])

    return output, weights, biases

