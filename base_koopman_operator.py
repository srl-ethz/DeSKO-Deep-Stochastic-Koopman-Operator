import tensorflow as tf
from utils import mlp
import numpy as np
import os
import tensorflow_probability as tfp

SCALE_DIAG_MIN_MAX = (-20, 2)
"""This version uses stochastic koopman operator"""
class base_Koopman(object):
    """Koopman.

    Attributes:
        A (tf.Variable): Weights of the Koopman operator
        B (tf.Variable): Weights of the Koopman operator
    """

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


        self.loss_weight = tf.placeholder(tf.float32, [args['state_dim']], 'loss_weight')
        self.loss_weight_num = np.sqrt(np.diagonal(args['Q']))


        self.lr = tf.placeholder(tf.float32, None, 'learning_rate')
        self.l2_reg = tf.contrib.layers.l1_regularizer(args['l2_regularizer'])
        # self.l1_reg = tf.contrib.layers.l1_regularizer(args['l2_regularizer'])

        self._create_koopman_result_holder(args)

        self._create_encoder(args)
        self._create_koopman_operator(args)
        self._create_forward_pred(args)

        self._create_backward_pred(args)
        self._create_optimizer(args)
        self._create_prediction_model(args)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _create_koopman_result_holder(self, args):
        self.A_result = np.zeros([args['state_dim'] + args['latent_dim'], args['state_dim'] + args['latent_dim']])

        self.A_tensor = tf.Variable(self.A_result,
                                    trainable=False, name="A_tensor", dtype=tf.float32)

        self.B_result = np.zeros([args['act_dim'], args['state_dim'] + args['latent_dim']])
        self.B_tensor = tf.Variable(self.B_result, trainable=False, name="B_tensor", dtype=tf.float32)

    def _create_encoder(self, args):

        pass
        # # epsilon = tf.random_normal([tf.shape(self.mean)[0], args['pred_horizon'], args['latent_dim']])
        # self.stochastic_latent = epsilon * self.sigma + self.mean

    def _create_koopman_operator(self, args):
        """
        Create the Koopman operators
        :param args:
        :return:
        """
        pass


    def _create_forward_pred(self, args):
        """
        Iteratively predict future state with the Koopman operator
        :param args(list):
        :return: forward_pred(Tensor): forward predictions
        """
        pass

    def _create_backward_pred(self, args):
        """
        Iteratively predict the past states with the Koopman operator
        :param args:
        :return:
        """
        pass

    def _create_optimizer(self, args):

        pass

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
        feed_in[self.loss_weight] = self.loss_weight_num

        self.sess.run(self.opt_list, feed_in)

        diagnotics = self.sess.run([self.diagnotics[key] for key in self.diagnotics.keys()], feed_in)
        output = {}
        [output.update({key: value}) for (key, value) in zip(self.diagnotics.keys(), diagnotics)]

        return output

    def store_Koopman_operator(self, replay_memory):
        batch_dict = replay_memory.get_all_train_data()
        x = batch_dict['states']
        a = batch_dict['inputs']

        feed_in = {}
        feed_in[self.x_input] = x
        feed_in[self.a_input] = a

        # Find loss and perform training operation
        feed_out = [self.A, self.B, tf.assign(self.A_tensor, self.A), tf.assign(self.B_tensor, self.B)]
        out = self.sess.run(feed_out, feed_in)
        self.A_result = out[0]
        self.B_result = out[1]

    def make_prediction(self, x_t, u, args):
        feed_dict = {}
        future_states = []
        feed_dict[self.shifted_x_t] = x_t
        feed_dict[self.shifted_a_t] = [u]
        [future_states] = self.sess.run(self.future_states, feed_dict)

        return future_states

    def encode(self, x):

        pass

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
        feed_out = [self.A_tensor, self.B_tensor]
        out = self.sess.run(feed_out, {})
        self.A_result = out[0]
        self.B_result = out[1]
        success_load = True

        return success_load