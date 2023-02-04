import tensorflow as tf
from utils import mlp
import numpy as np
import math
import os
import tensorflow_probability as tfp

from base_koopman_operator import base_Koopman

SCALE_DIAG_MIN_MAX = (-20, 2)
"""This version uses stochastic koopman operator with observation matrix"""
class Koopman(base_Koopman):
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

        if args['target_entropy'] is None:
            self.target_entropy = -args['latent_dim']  # lower bound of the entropy
        else:
            self.target_entropy = args['target_entropy']

        self.log_alpha = tf.get_variable('alpha', None, tf.float32, initializer=tf.log(args['alpha']))  # Entropy Temperature
        self.alpha = tf.exp(self.log_alpha)

        super(Koopman, self).__init__(args)

    def _create_koopman_result_holder(self, args):
        self.A_result = np.zeros([args['latent_dim'], args['latent_dim']])

        self.A_tensor = tf.Variable(self.A_result,
                                    trainable=False, name="A_tensor", dtype=tf.float32)

        self.B_result = np.zeros([args['act_dim'], args['latent_dim']])
        self.B_tensor = tf.Variable(self.B_result, trainable=False, name="B_tensor", dtype=tf.float32)

        self.C_result = np.zeros([args['latent_dim'], args['state_dim']])
        self.C_tensor = tf.Variable(self.C_result, trainable=False, name="C_tensor", dtype=tf.float32)

    def _create_encoder(self, args):
        # if args['activation'] == 'relu':
        #     activation = tf.nn.relu
        # elif args['activation'] == 'elu':
        #     activation = tf.nn.elu
        # else:
        #     print(args['activation']+' is not implemented as a activation function')
        #     raise KeyError
        activation = tf.nn.relu
        self.mean = mlp(self.x_input,
                           args['encoder_struct'] + [args['latent_dim']], activation, name='mean', regularizer=self.l2_reg)

        log_sigma = mlp(self.x_input,
                        args['encoder_struct'] + [args['latent_dim']], activation, name='sigma', regularizer=self.l2_reg)

        log_sigma = tf.clip_by_value(log_sigma, *SCALE_DIAG_MIN_MAX)
        self.sigma = tf.exp(log_sigma)

        base_distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(args['latent_dim']),
                                                                     scale_diag=tf.ones(args['latent_dim']))


        epsilon = base_distribution.sample([tf.shape(self.mean)[0], args['pred_horizon']])

        bijector = tfp.bijectors.Affine(shift=self.mean, scale_diag=self.sigma)
        self.stochastic_latent = bijector.forward(epsilon)

        # # epsilon = tf.random_normal([tf.shape(self.mean)[0], args['pred_horizon'], args['latent_dim']])
        # self.stochastic_latent = epsilon * self.sigma + self.mean

    def _create_koopman_operator(self, args):
        """
        Create the Koopman operators
        :param args:
        :return:
        """
        with tf.variable_scope('koopman', regularizer=self.l2_reg):
            self.A = tf.get_variable('A', shape=[args['latent_dim'],
                                                 args['latent_dim']])
            self.B = tf.get_variable('B', shape=[args['act_dim'], args['latent_dim']])
            self.C = tf.get_variable('C', shape=[args['latent_dim'], args['state_dim']])
            # self.A_inv = tf.get_variable('A_inv', shape=[args['state_dim'] + args['latent_dim'],
            #                                              args['state_dim'] + args['latent_dim']])
        return


    def _create_forward_pred(self, args):
        """
        Iteratively predict future state with the Koopman operator
        :param args(list):
        :return: forward_pred(Tensor): forward predictions
        """
        forward_pred = []
        x_mean_forward_pred = []
        mean_forward_pred = []
        sigma_forward_pred = []

        phi_t = self.stochastic_latent[:, 0]
        mean_t = self.mean[:, 0]
        sigma_t = self.sigma[:, 0]

        for t in range(args['pred_horizon']-1):
            phi_t = tf.matmul(phi_t, self.A) + tf.matmul(self.a_input[:, t], self.B)
            x_t = tf.matmul(phi_t, self.C)
            mean_t = tf.matmul(mean_t, self.A) + tf.matmul(self.a_input[:, t], self.B)
            sigma_t = tf.matmul(sigma_t, self.A) + tf.matmul(self.a_input[:, t], self.B)
            x_mean_t = tf.matmul(mean_t, self.C)
            forward_pred.append(x_t)
            mean_forward_pred.append(mean_t)
            x_mean_forward_pred.append(x_mean_t)
            sigma_forward_pred.append(sigma_t)

        self.forward_pred = tf.stack(forward_pred, axis=1)
        self.x_mean_forward_pred = tf.stack(x_mean_forward_pred, axis=1)
        self.mean_forward_pred = tf.stack(mean_forward_pred, axis=1)
        self.sigma_forward_pred = tf.stack(sigma_forward_pred, axis=1)
        return

    def _create_backward_pred(self, args):
        """
        Iteratively predict the past states with the Koopman operator
        :param args:
        :return:
        """
        # backward_pred = []
        # mean_backward_pred = []
        # sigma_backward_pred = []
        #
        # phi_t = tf.concat([self.x_input[:, -1], self.stochastic_latent[:, -1]], axis=1)
        # mean_t = tf.concat([self.x_input[:, 0], self.mean[:, 0]], axis=1)
        # sigma_t = tf.concat([self.x_input[:, 0], self.sigma[:, 0]], axis=1)
        # for t in range(args['pred_horizon'] - 1, 0, -1):
        #     phi_t = tf.matmul(phi_t - tf.matmul(self.a_input[:, t-1], self.B), self.A_inv)
        #     mean_t = tf.matmul(mean_t - tf.matmul(self.a_input[:, t - 1], self.B), self.A_inv)
        #     sigma_t = tf.matmul(sigma_t - tf.matmul(self.a_input[:, t - 1], self.B), self.A_inv)
        #     backward_pred.append(phi_t)
        #     mean_backward_pred.append(mean_t)
        #     sigma_backward_pred.append(sigma_t)
        #
        # backward_pred = tf.stack(backward_pred, axis=1)
        # mean_backward_pred = tf.stack(mean_backward_pred, axis=1)
        # sigma_backward_pred = tf.stack(sigma_backward_pred, axis=1)
        # self.backward_pred = tf.reverse(backward_pred, [1])
        # self.mean_backward_pred = tf.reverse(mean_backward_pred, [1])
        # self.sigma_backward_pred = tf.reverse(sigma_backward_pred, [1])

        return

    def _create_optimizer(self, args):
        mean = tf.reshape(self.mean, [-1, args['latent_dim']])
        sigma = tf.reshape(self.sigma, [-1, args['latent_dim']])
        sample = tf.reshape(self.stochastic_latent, [-1, args['latent_dim']])
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
        self.entropy = - tf.reduce_mean(dist.log_prob(sample, name='entropy'))

        self.alpha_loss = alpha_loss = self.log_alpha * tf.stop_gradient(self.entropy - self.target_entropy)
        self.alpha_train = tf.train.AdamOptimizer(self.lr).minimize(alpha_loss, var_list=self.log_alpha)



        forward_pred_loss = tf.losses.mean_squared_error(labels=tf.stop_gradient(self.mean[:, 1:]), predictions=self.mean_forward_pred[:, :])\
                            + tf.losses.mean_squared_error(labels=tf.stop_gradient(self.sigma[:, 1:]), predictions=self.sigma_forward_pred[:, :])

        self.reconstruct_loss = reconstruct_loss = tf.losses.mean_squared_error(labels=self.x_input[:, 1:],
                                                        predictions=self.forward_pred[:, :])
        reconstruct_pred_T_loss = tf.reduce_mean(tf.square(self.x_input[:, -1,]- self.forward_pred[:, -1]))
        # val_x = self.x_input[:, 1:]* self.scale + self.shift
        # val_y = self.x_mean_forward_pred[:, :] * self.scale + self.shift
        # self.val_loss = tf.abs(tf.reduce_mean((val_x - val_y)/(tf.abs(val_y)+1e-10)))

        val_x = self.x_input[:, 1:]
        val_y = self.x_mean_forward_pred[:, :]
        self.val_loss = tf.losses.mean_squared_error(labels=val_x, predictions=val_y)


        self.loss =1 * forward_pred_loss + 10 * reconstruct_loss+ 0 * reconstruct_pred_T_loss - tf.stop_gradient(self.alpha) * self.entropy #+ weighted_reconstruct_loss
        #+ tf.reduce_sum(tf.losses.get_regularization_losses()) #+ weighted_reconstruct_loss + weighted_reconstruct_T_loss

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=params)

        grad_norm = []
        self.grads = []
        for grad in tf.gradients(self.loss, params):
            if grad is not None:
                self.grads.append(grad)
                grad_norm.append(tf.norm(grad))
        grad_norm = tf.reduce_max(grad_norm)

        self.diagnotics.update({
            'loss': self.loss,
            'lagrange multiplier': self.alpha,
            'entropy': self.entropy,
            'gradient': grad_norm
        })
        self.opt_list.extend([self.train, self.alpha_train])

    def _create_prediction_model(self, args):

        self.x_t = tf.placeholder(tf.float32, [None, args['state_dim']], 'x_t')
        self.a_t = tf.placeholder(tf.float32, [None, args['pred_horizon']-1, args['act_dim']], 'a_t')
        self.shifted_x_t = (self.x_t - self.shift)/self.scale
        self.shifted_a_t = (self.a_t - self.shift_u) / self.scale_u
        self.mean_t = mlp(self.shifted_x_t,
                           args['encoder_struct'] + [args['latent_dim']], tf.nn.relu, name='mean', reuse=True)
        self.sigma_t = mlp(self.shifted_x_t,
                           args['encoder_struct'] + [args['latent_dim']], tf.nn.relu, name='sigma', reuse=True)


        forward_pred = []
        phi_t = self.mean_t
        for t in range(args['pred_horizon'] - 1):
            u = self.shifted_a_t[:, t]
            phi_t = tf.matmul(phi_t, self.A) + tf.matmul(u, self.B)
            x_t = tf.matmul(phi_t, self.C)
            forward_pred.append(x_t)
        self.future_states = tf.stack(forward_pred, axis=1)[:, :]

    def store_Koopman_operator(self, replay_memory):
        batch_dict = replay_memory.get_all_train_data()
        x = batch_dict['states']
        a = batch_dict['inputs']

        feed_in = {}
        feed_in[self.x_input] = x
        feed_in[self.a_input] = a

        # Find loss and perform training operation
        feed_out = [self.A, self.B, self.C, tf.assign(self.A_tensor, self.A), tf.assign(self.B_tensor, self.B), tf.assign(self.C_tensor, self.C),]
        out = self.sess.run(feed_out, feed_in)
        self.A_result = out[0]
        self.B_result = out[1]
        self.C_result = out[2]

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
        #print(x[])

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
        for key in output.keys():
            if math.isnan(output[key]):
                print('NaN appears')
                raise ValueError
        return output



    def encode(self, x):
        feed_dict = {}

        feed_dict[self.x_t] = x
        [mean, sigma] = self.sess.run([self.mean_t, self.sigma_t], feed_dict)

        return mean, sigma

    def restore(self, path):
        model_file = tf.train.latest_checkpoint(path+'/model/')
        if model_file is None:
            success_load = False
            return success_load
        self.saver.restore(self.sess, model_file)
        feed_out = [self.A_tensor, self.B_tensor, self.C_tensor]
        out = self.sess.run(feed_out, {})
        self.A_result = out[0]
        self.B_result = out[1]
        self.C_result = out[2]
        success_load = True

        return success_load









