import tensorflow as tf
from utils import mlp
import numpy as np
import os
from base_koopman_operator import base_Koopman


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
        super(Koopman, self).__init__(args)


    def _create_encoder(self, args):
        self.encoder = mlp(self.x_input,
                           args['encoder_struct'] + [args['latent_dim']], tf.nn.relu, name='encoder',
                           regularizer=self.l2_reg)

    def _create_koopman_operator(self, args):
        """
        Create the Koopman operators
        :param args:
        :return:
        """
        X = tf.concat([self.x_input[:, 0], self.encoder[:, 0], self.a_input[:, 0]], axis=1)
        Y = tf.concat([self.x_input[:, 1], self.encoder[:, 1]], axis=1)
        K = tf.matrix_solve_ls(X, Y, l2_regularizer=args['l2_regularizer'])
        self.A = K[:args['state_dim'] + args['latent_dim'], :]
        self.B = K[-args['act_dim']:, :]
        # self.A_inv = tf.matrix_solve_ls(Y - tf.matmul(self.a_input[:, 0], self.B),
        #                                 X[:, :args['state_dim'] + args['latent_dim']],
        #                                 l2_regularizer=args['l2_regularizer'])
        return

    def _create_forward_pred(self, args):
        """
        Iteratively predict future state with the Koopman operator
        :param args(list):
        :return: forward_pred(Tensor): forward predictions
        """
        forward_pred = []
        phi_t = tf.concat([self.x_input[:, 0], self.encoder[:, 0]], axis=1)
        for t in range(args['pred_horizon']-1):
            phi_t = tf.matmul(phi_t, self.A) + tf.matmul(self.a_input[:, t], self.B)
            forward_pred.append(phi_t)
        self.forward_pred = tf.stack(forward_pred, axis=1)

        return

    def _create_backward_pred(self, args):
        """
        Iteratively predict the past states with the Koopman operator
        :param args:
        :return:
        """
        # backward_pred = []
        # phi_t = tf.concat([self.x_input[:, -1], self.encoder[:, -1]], axis=1)
        # for t in range(args['pred_horizon'] - 1, 0, -1):
        #     phi_t = tf.matmul(phi_t - tf.matmul(self.a_input[:, t-1], self.B), self.A_inv)
        #     backward_pred.append(phi_t)
        #
        # backward_pred = tf.stack(backward_pred, axis=1)
        # self.backward_pred = tf.reverse(backward_pred, [1])

        return

    def _create_optimizer(self, args):
        Y = tf.concat([self.x_input, self.encoder], axis=2)
        forward_pred_loss = tf.losses.mean_squared_error(labels=Y[:, 1:], predictions=self.forward_pred)
        self.reconstruct_loss = reconstruct_loss = tf.losses.mean_squared_error(labels=Y[:, 1:, :args['state_dim']],
                                                        predictions=self.forward_pred[:, :, :args['state_dim']])

        val_x = self.x_input[:, 1:]
        val_y = self.forward_pred[:, :, :args['state_dim']]
        self.val_loss = tf.losses.mean_squared_error(labels=val_x, predictions=val_y)

        reconstruct_pred_T_loss = tf.reduce_mean(tf.square(Y[:, -1, :args['state_dim']] \
                                                      - self.forward_pred[:, -1, :args['state_dim']]))
        forward_pred_T_loss = tf.reduce_mean(tf.square(self.forward_pred[:, -1] \
                                                         - Y[:, - 1]))
        weighted_reconstruct_loss = tf.losses.mean_squared_error(labels=Y[:, 1:, :args['state_dim']]* self.loss_weight,
                                                                                predictions=self.forward_pred[:, :,
                                                                                            :args['state_dim']]*self.loss_weight)

        weighted_reconstruct_T_loss = tf.losses.mean_squared_error(labels=Y[:, -1, :args['state_dim']]* self.loss_weight,
                                                                                predictions=self.forward_pred[:, -1,
                                                                                            :args['state_dim']]*self.loss_weight)
        # backward_pred_loss = tf.losses.mean_squared_error(labels=Y[:, :-1], predictions=self.backward_pred)
        self.loss =1* forward_pred_loss + 1 * forward_pred_T_loss + 10 * reconstruct_loss + \
                   10 * reconstruct_pred_T_loss #+ weighted_reconstruct_loss + weighted_reconstruct_T_loss
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.diagnotics.update({'loss': self.loss})
        self.opt_list.append(self.train)




    def encode(self, x):
        feed_dict = {}

        feed_dict[self.x_t] = x
        [phi_t] = self.sess.run(self.phi_t, feed_dict)

        return phi_t















