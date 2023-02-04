import numpy as np
from cvxpy import *
import mosek
from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import block_diag
import time
def dlqr(A, B, Q, R):
    '''
    dlqr solves the discrete time LQR controller
    Inputs:     A, B: system dynamics
    Outputs:    K: optimal feedback matrix
    '''


    P = solve_discrete_are(A, B, Q, R)
    K = - np.dot(np.linalg.inv(R + np.dot(B.T, np.dot(P, B))), np.dot(B.T, np.dot(P, A)))

    return K


class base_MPC(object):

    def __init__(self, model, args):


        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['MPC_pred_horizon']
        self.a_dim = args['act_dim']
        self.LQT_gamma = 0.5  # for cost discount

        self.end_weight = args['end_weight']
        self.args = args
        self.model = model

        self._build_matrices(args)


    def _build_matrices(self, args):

        self.latent_dim = args['latent_dim'] + args['state_dim']
        self.state_dim = args['state_dim']
        self.Q = self.args['Q']
        self.R = self.args['R']

        self.A_holder = Parameter((self.latent_dim, self.latent_dim,))
        self.B_holder = Parameter((self.latent_dim, self.args['act_dim'],))
        self.K_holder = Parameter((self.args['act_dim'], self.latent_dim,))
        self.P_holder = Parameter((self.latent_dim, self.latent_dim,))

        self.u_s_holder = Parameter((self.args['act_dim'],))
        self.ref = Parameter(self.state_dim)

    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.get_shift_and_scale()
        self.reference = (self.args['reference'] - self.shift) / self.scale
        self.A = self.model.A_result.T
        self.B = self.model.B_result.T
        self.C = np.hstack((np.eye(self.state_dim), np.zeros([self.state_dim, self.args['latent_dim']])))

        self._shift_and_scale_bounds(self.args)
        self._set_LQR_controller()

        self._create_set_point_u_prob()        #
        self._create_prob(self.args)

    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_lowh'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                self.a_bound_low = (args['a_bound_low'] - self.shift_u) / self.scale_u
            else:
                self.a_bound_low = None
                self.a_bound_high = None

    def _set_LQR_controller(self):

        A_1 = np.sqrt(self.LQT_gamma) * block_diag(self.A, np.eye(self.state_dim))
        B_1 = np.sqrt(self.LQT_gamma) * np.vstack((self.B, np.zeros([self.state_dim, self.a_dim])))
        C_1 = np.hstack((self.C, -np.eye(self.state_dim)))

        self.CQC = CQC = np.dot(C_1.T, np.dot(self.Q, C_1))
        self.K = dlqr(A_1, B_1, CQC, self.R)
        self.CPC = self._get_trm_cost(A_1, B_1, CQC, self.R, self.K)

    def _create_set_point_u_prob(self):

        self.u_s_var = Variable((self.a_dim))
        phi_s = Variable((self.latent_dim - self.state_dim))
        x_s = hstack([self.ref, phi_s])
        X_s = hstack([x_s, self.ref])

        constraint = [self.C @ x_s == self.C @ (self.A @ x_s + self.B @ self.K @ X_s + self.B @ self.u_s_var)]
        objective = quad_form(self.u_s_var, np.eye(self.a_dim))

        self._set_point_prob = Problem(Minimize(objective), constraint)

    def _get_set_point_u(self, reference):

        self.ref.value = reference
        try:
            self._set_point_prob.solve(solver=MOSEK, warm_start=True,
                            mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
        except cvxpy.error.SolverError:
            self._set_point_prob.solve(solver=MOSEK, warm_start=True,
                            mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}, verbose=True)

        if self._set_point_prob.status is 'optimal':
            self.u_s_holder.value = self.u_s_var.value
        else:
            print('no suitable set point control input')
            self.u_s_holder.value = np.zeros([self.a_dim])


    def _get_trm_cost(self, A, B, Q, R, K):
        '''
        get_trm_cost returns the matrix P associated with the terminal cost
        Outputs:    P: the matrix associated with the terminal cost in the objective
        '''

        A_lyap = (A + np.dot(B, K)).T
        Q_lyap = Q + np.dot(K.T, np.dot(R, K))

        P = solve_discrete_lyapunov(A_lyap, Q_lyap)

        return P


    def _create_prob(self, args):
        self.u = Variable((self.a_dim, self.control_horizon))
        self.x = Variable((self.latent_dim, self.pred_horizon+1))
        self.x_init = Parameter(self.latent_dim)

        objective = 0.
        constraints = [self.x[:, 0] == self.x_init]
        for k in range(self.pred_horizon):
            X = vstack([reshape(self.x[:, k], (self.latent_dim, 1)), reshape(self.ref, (self.state_dim,1))])[:,0]
            k_u = k if k <= self.control_horizon-1 else self.control_horizon-1
            objective += quad_form(X, self.CQC) + quad_form(self.u[:, k_u]-self.u_s_holder, self.R)
            constraints += [self.x[:, k + 1] == self.A_holder @ self.x[:, k] + self.B_holder @self.K @ X + self.B_holder @ self.u[:, k_u]]
            if args['apply_state_constraints']:
                constraints += [self.s_bound_low <= self.x[:self.state_dim, k],
                                self.x[:self.state_dim, k] <= self.s_bound_high]
            if args['apply_action_constraints'] and k <self.control_horizon:
                constraints += [self.a_bound_low <= self.K @ X + self.u[:, k], self.K @ X + self.u[:, k] <= self.a_bound_high]
        X = vstack([reshape(self.x[:, -1], (self.latent_dim,1)), reshape(self.ref, (self.state_dim,1))])[:,0]
        objective += quad_form(X, self.CPC)
        self.prob = Problem(Minimize(objective), constraints)

    def choose_action(self, x_0, reference, *args):
        pass

    def reset(self):
        pass

    def update_reference(self, reference):
        self.reference = reference
        self._get_set_point_u(self.reference)        #halfcheetah

    def check_controllability(self):

        gamma = [self.model.B_result.T]
        A = self.model.A_result.T
        for d in range(self.latent_dim-1):

            gamma.append(np.matmul(A, gamma[d]))

        gamma = np.concatenate(np.array(gamma), axis=1)
        rank = np.linalg.matrix_rank(gamma)
        print('rank of controllability matrix is ' + str(rank) +'/'+ str(self.latent_dim))

    def restore(self, path):
        success = self.model.restore(path)
        self._build_controller()
        return success



class MPC(base_MPC):
    def __init__(self, model, args):

        super(MPC, self).__init__(model, args)

    def choose_action(self, x_0, reference, *args):
        if hasattr(self, 'last_state'):
            if len(self.last_state)<1:
                u = np.random.uniform(self.a_bound_low, self.a_bound_high)
                self.last_state = x_0
                return u
            else:
                phi_0 = self.model.encode([[self.last_state], [x_0]])
                self.last_state = x_0
        else:
            phi_0 = self.model.encode([x_0])
        self.x_init.value = phi_0
        self._get_set_point_u(reference)      ####
        self.A_holder.value = self.model.A_result.T
        self.B_holder.value = self.model.B_result.T
        self.prob.solve(solver=MOSEK, warm_start=True, mosek_params={mosek.iparam.intpnt_solve_form:mosek.solveform.dual})
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            X = np.concatenate([phi_0, self.reference])      #self.reference[0,:]
            u = np.dot(self.K, X) + self.u[:, 0].value
            u = u * self.scale_u + self.shift_u
        else:
            print("Error: Cannot solve mpc..")
            X = np.concatenate([phi_0, self.reference])      #self.reference[0,:]
            u = np.dot(self.K, X)
            u = u * self.scale_u + self.shift_u

        return u

    def reset(self):
        if hasattr(self, 'last_state'):
            self.last_state = []



    def simple_choose_action(self, phi_0, *args):

        self.x_init.value = phi_0
        self.prob.solve(solver=OSQP, warm_start=True)
        self.prob.solve()
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            u = self.u[:,0].value
        else:
            print("Error: Cannot solve mpc..")
            u = None

        return u



class MPC_with_motion_planning(base_MPC):
    def __init__(self, model, args):

        super(MPC_with_motion_planning, self).__init__(model, args)

    def _build_matrices(self, args):

        self.latent_dim = args['latent_dim'] + args['state_dim']
        self.state_dim = args['state_dim']
        self.Q = self.args['Q']
        self.R = self.args['R']

        self.A_holder = Parameter((self.latent_dim, self.latent_dim,))
        self.B_holder = Parameter((self.latent_dim, self.args['act_dim'],))
        self.K_holder = Parameter((self.args['act_dim'], self.latent_dim,))
        self.P_holder = Parameter((self.latent_dim, self.latent_dim,))

        self.u_s_holder = Parameter((self.args['act_dim'],))
        self.ref = Parameter((args['pred_horizon'],self.state_dim, ))

    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.get_shift_and_scale()
        self.reference = np.zeros((999, 18))
        for i in range(984):
            self.reference[i, :] = (self.args['reference'][i, :] - self.shift) / self.scale

        self.A = self.model.A_result.T
        self.B = self.model.B_result.T
        self.C = np.hstack((np.eye(self.state_dim), np.zeros([self.state_dim, self.args['latent_dim']])))

        self._shift_and_scale_bounds(self.args)
        self._set_LQR_controller()

        #self._create_set_point_u_prob()        #
        self._create_prob(self.args)

    def _create_prob(self, args):
        self.u = Variable((self.a_dim, self.control_horizon))
        self.x = Variable((self.latent_dim, self.pred_horizon+1))
        self.x_init = Parameter(self.latent_dim)

        objective = 0.
        constraints = [self.x[:, 0] == self.x_init]
        for k in range(self.pred_horizon):
            X = vstack([reshape(self.x[:, k], (self.latent_dim, 1)), reshape(self.ref[k,:], (self.state_dim,1))])[:,0]
            k_u = k if k <= self.control_horizon-1 else self.control_horizon-1
            objective += quad_form(X, self.CQC) + quad_form(self.u[:, k_u], self.R)
            constraints += [self.x[:, k + 1] == self.A_holder @ self.x[:, k] + self.B_holder @self.K @ X + self.B_holder @ self.u[:, k_u]]
            if args['apply_state_constraints']:
                constraints += [self.s_bound_low <= self.x[:self.state_dim, k],
                                self.x[:self.state_dim, k] <= self.s_bound_high]
            if args['apply_action_constraints'] and k <self.control_horizon:
                constraints += [self.a_bound_low <= self.K @ X + self.u[:, k], self.K @ X + self.u[:, k] <= self.a_bound_high]
        X = vstack([reshape(self.x[:, -1], (self.latent_dim,1)), reshape(self.ref[k,:], (self.state_dim,1))])[:,0]
        objective += quad_form(X, self.CPC)
        self.prob = Problem(Minimize(objective), constraints)

    def choose_action(self, x_0, reference, *args):
        if hasattr(self, 'last_state'):
            if len(self.last_state)<1:
                u = np.random.uniform(self.a_bound_low, self.a_bound_high)
                self.last_state = x_0
                return u
            else:
                phi_0 = self.model.encode([[self.last_state], [x_0]])
                self.last_state = x_0
        else:
            phi_0 = self.model.encode([x_0])
        self.x_init.value = phi_0
        #self._get_set_point_u(reference)      ####
        self.A_holder.value = self.model.A_result.T
        self.B_holder.value = self.model.B_result.T

        self.ref.value = reference

        self.prob.solve(solver=MOSEK, warm_start=True, mosek_params={mosek.iparam.intpnt_solve_form:mosek.solveform.dual})


        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            X = np.concatenate([phi_0, self.reference[0,:]])      #self.reference[0,:]
            u = np.dot(self.K, X) + self.u[:, 0].value
            u = u * self.scale_u + self.shift_u
        else:
            print("Error: Cannot solve mpc..")
            X = np.concatenate([phi_0, self.reference[0,:]])      #self.reference[0,:]
            u = np.dot(self.K, X)
            u = u * self.scale_u + self.shift_u

        return u

    def reset(self):
        if hasattr(self, 'last_state'):
            self.last_state = []



    def simple_choose_action(self, phi_0, *args):

        self.x_init.value = phi_0
        self.prob.solve(solver=OSQP, warm_start=True)
        self.prob.solve()
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            u = self.u[:,0].value
        else:
            print("Error: Cannot solve mpc..")
            u = None

        return u



class pure_MPC(object):
    def __init__(self, model, args):
        self.horizon = args['control_horizon']
        self.a_dim = 1

        self.latent_dim = 2
        self.state_dim = 2
        if args['apply_state_constraints']:
            self.s_bound_low = args['s_bound_low']
            self.s_bound_high = args['s_bound_high']
        else:
            self.s_bound_low = None
            self.s_bound_high = None
        if args['apply_action_constraints']:
            self.a_bound_low = args['a_bound_low']
            self.a_bound_high = args['a_bound_high']
        else:
            self.a_bound_low = None
            self.a_bound_high = None


        # self.A = np.transpose(model.A_result)
        # self.B = np.transpose(model.B_result)
        self.A = np.array([
            [-0.3672, 0.7038],
            [-1.8462, 2.0094]])
        self.B = np.array([[-1], [1]])
        self.Q = args['Q']
        self.R = args['R']


        self._create_prob(args)


    def _create_prob(self, args):
        # Define problem
        self.u = Variable((self.a_dim, self.horizon))
        self.x = Variable((self.latent_dim, self.horizon + 1))
        self.x_init = Parameter(self.latent_dim)
        objective = 0
        constraints = [self.x[:, 0] == self.x_init]
        for k in range(self.horizon):
            objective += quad_form(self.x[:self.state_dim, k], self.Q) + quad_form(self.u[:, k], self.R)
            constraints += [self.x[:, k + 1] == self.A @ self.x[:, k] + self.B @ self.u[:, k]]
            if args['apply_state_constraints']:
                constraints += [self.s_bound_low <= self.x[:self.state_dim, k], self.x[:self.state_dim, k] <= self.s_bound_high]
            if args['apply_action_constraints']:
                constraints += [self.a_bound_low <= self.u[:, k], self.u[:, k] <= self.a_bound_high]
        objective += 100 * quad_form(self.x[:self.state_dim, -1], self.Q)
        self.prob = Problem(Minimize(objective), constraints)

    def check_controllability(self):

        gamma = [self.B]
        for d in range(self.latent_dim-1):

            gamma.append(np.matmul(self.A, gamma[d]))

        gamma = np.squeeze(np.array(gamma))
        rank = np.linalg.matrix_rank(gamma)
        print(rank)

    def simple_choose_action(self, phi_0, *args):

        self.x_init.value = phi_0
        self.prob.solve(solver=OSQP, warm_start=True)
        self.prob.solve()
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            u = self.u[:,0].value * self.scale_u + self.shift_u
        else:
            print("Error: Cannot solve mpc..")
            u = None

        return u

    def restore(self, path):
        return True


    def linear_predict(self,x, u):

        x = np.matmul(self.A, x) + np.matmul(self.B, u)

        return x


class Time_varying_MPC(base_MPC):

    def __init__(self, model, args):

        super(Time_varying_MPC, self).__init__(model, args)
        self._create_prob(args)

    def choose_action(self, x_0, *args):

        phi_0, A, B = self.model.encode([x_0])
        self.x_init.value = phi_0[0]
        self.A.value = A[0].T
        self.B.value = B[0].T
        # self.prob.solve(solver=OSQP, warm_start=True)
        self.prob.solve(solver=MOSEK, warm_start=True,
                        mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
        # self.prob.solve()
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            u = self.u[:, 0].value * self.scale_u + self.shift_u
        else:
            print("Error: Cannot solve mpc..")
            u = None

        return u

    def check_controllability(self):
        print('rank is changing according to states')


class Stochastic_MPC(base_MPC):

    def __init__(self, model, args):
        self.n_of_random_seeds = args['n_of_random_seeds']
        super(Stochastic_MPC, self).__init__(model, args)


    def _create_prob(self, args):
        self.u = Variable((self.a_dim, self.control_horizon))
        self.x = Variable((self.latent_dim, self.pred_horizon + 1))
        self.ref = Parameter(self.state_dim)

        self.x_t = Parameter((self.state_dim, 1))
        self.mean_t = Parameter((self.latent_dim-self.state_dim, 1))
        self.sigma_t = Parameter((self.latent_dim-self.state_dim, 1))
        self.epsilon = Parameter((1, self.n_of_random_seeds))

        phi_t = self.mean_t + matmul(kron(self.epsilon, self.sigma_t), np.ones([self.n_of_random_seeds,1]))/self.n_of_random_seeds
        phi_t = vstack([self.x_t, phi_t])
        objective = 0.
        constraints = [self.x[:, 0] == phi_t[:, 0] ]
        for k in range(self.pred_horizon):
            X = vstack([reshape(self.x[:, k], (self.latent_dim,1)), reshape(self.ref, (self.state_dim,1))])[:,0]
            k_u = k if k <= self.control_horizon - 1 else self.control_horizon - 1
            objective += quad_form(X, self.CQC) + quad_form(self.u[:, k_u]-self.u_s, self.R)
            constraints += [self.x[:, k + 1] == self.A_holder @ self.x[:, k] + self.B_holder @self.K @ X +  self.B_holder @ self.u[:, k_u]]
            if args['apply_state_constraints']:
                constraints += [self.s_bound_low <= self.x[:self.state_dim, k],
                                self.x[:self.state_dim, k] <= self.s_bound_high]
            if args['apply_action_constraints'] and k < self.control_horizon:
                constraints += [self.a_bound_low <= self.K @ X + self.u[:, k],
                                self.K @ X + self.u[:, k] <= self.a_bound_high]
        X = vstack([reshape(self.x[:, -1], (self.latent_dim, 1)), reshape(self.ref, (self.state_dim,1))])[:,0]
        objective += quad_form(X, self.CPC)
        self.prob = Problem(Minimize(objective), constraints)


    def choose_action(self, x_0, *args):

        [mean, sigma] = self.model.encode([x_0])
        epsilon = np.random.normal(0.,1.,[1, self.n_of_random_seeds])
        self.mean_t.value = np.expand_dims(mean[0], 1)
        self.sigma_t.value = np.expand_dims(sigma[0], 1)
        self.epsilon.value = epsilon
        self.x_t.value = np.expand_dims(x_0, 1)
        self.ref.value = self.reference
        self.A_holder.value = self.A
        self.B_holder.value = self.B
        # self.prob.solve(solver=OSQP, warm_start=True)
        # t1 = time.time()
        try:
            self.prob.solve(solver=MOSEK, warm_start=True,
                            mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
        except cvxpy.error.SolverError:
            self.prob.solve(solver=MOSEK, warm_start=True,
                            mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}, verbose = True)
        # t2 = time.time()
        # print(str(t2-t1))
        # self.prob.solve()
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            X = np.concatenate([self.x[:, 0].value, self.reference])
            # X = np.concatenate([x_0, mean[0], self.reference])
            u = np.dot(self.K, X) #+ self.u[:, 0].value
            u = u * self.scale_u + self.shift_u
        else:
            print('Not optimal')
            raise cvxpy.error.SolverError

        return u



class Stochastic_MPC_v2(base_MPC):

    def __init__(self, model, args):
        self.n_of_random_seeds = args['n_of_random_seeds']
        super(Stochastic_MPC_v2, self).__init__(model, args)


    def _create_prob(self, args):
        self.u = Variable((self.a_dim, self.control_horizon))
        self.x = Variable((self.latent_dim * self.n_of_random_seeds, self.pred_horizon + 1))
        self.mean_t = Parameter((self.latent_dim-self.state_dim, 1))
        self.sigma_t = Parameter((self.latent_dim-self.state_dim, 1))
        self.x_t = Parameter((self.state_dim, 1))
        self.epsilon = Parameter((1, self.n_of_random_seeds))
        unit_diag_matrix = np.eye(self.n_of_random_seeds)
        unit_vector = np.ones([1, self.n_of_random_seeds])
        full_state_Q = np.pad(self.Q, ((0,self.latent_dim-self.state_dim),(0,self.latent_dim-self.state_dim)),'constant')
        extended_A = kron(unit_diag_matrix, self.A)
        extended_Q = np.kron(unit_diag_matrix, full_state_Q)
        extended_B = kron(unit_diag_matrix, self.B)
        extende_u = kron(unit_vector.transpose(), self.u)
        phi_t = kron(unit_vector, vstack([self.x_t, self.mean_t])) + vstack(
            [kron(unit_vector, self.x_t), kron(self.epsilon, self.sigma_t)])
        phi_t = reshape(phi_t, (self.n_of_random_seeds * self.latent_dim, 1))
        objective = 0.
        constraints = [self.x[:, 0] == phi_t[:,0]]
        for k in range(self.pred_horizon):
            k_u = k if k <= self.control_horizon - 1 else self.control_horizon - 1

            objective += quad_form(self.x[:, k], extended_Q)/self.n_of_random_seeds + quad_form(self.u[:, k_u], self.R)
            constraints += [self.x[:, k + 1] == extended_A @ self.x[:, k] + extended_B @ extende_u[:, k_u]]
            if args['apply_state_constraints']:
                constraints += [self.s_bound_low <= self.x[:self.state_dim, k],
                                self.x[:self.state_dim, k] <= self.s_bound_high]
            if args['apply_action_constraints'] and k < self.control_horizon:
                constraints += [self.a_bound_low <= self.u[:, k], self.u[:, k] <= self.a_bound_high]
        objective += self.end_weight * quad_form(self.x[:, -1], extended_Q)/self.n_of_random_seeds
        self.prob = Problem(Minimize(objective), constraints)

    def choose_action(self, x_0, *args):

        [mean, sigma] = self.model.encode([x_0])
        epsilon = np.random.normal(0.,1.,[1, self.n_of_random_seeds])
        self.mean_t.value = np.expand_dims(mean[0], 1)
        self.sigma_t.value = np.expand_dims(sigma[0], 1)
        self.epsilon.value = epsilon
        self.x_t.value = np.expand_dims(x_0, 1)
        self.A.value = self.model.A_result.T
        self.B.value = self.model.B_result.T
        # self.prob.solve(solver=OSQP, warm_start=True)
        self.prob.solve(solver=MOSEK, warm_start=True,
                        mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
        # self.prob.solve()
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            u = self.u[:, 0].value * self.scale_u + self.shift_u
        else:
            print("Error: Cannot solve mpc..")
            u = None

        return u

class Stochastic_MPC_v3(base_MPC):

    def __init__(self, model, args):
        self.n_of_random_seeds = args['n_of_random_seeds']
        super(Stochastic_MPC_v3, self).__init__(model, args)


    def _create_prob(self, args):
        self.u = Variable((self.a_dim, self.control_horizon))
        self.mean = Variable((self.latent_dim , self.pred_horizon + 1))
        self.sigma = Variable((self.latent_dim, self.pred_horizon + 1))
        self.x_t = Parameter((self.state_dim, 1))
        self.mean_t = Parameter((self.latent_dim-self.state_dim, 1))
        self.sigma_t = Parameter((self.latent_dim-self.state_dim, 1))
        mean_t = vstack([self.x_t, self.mean_t])
        sigma_t = vstack([np.zeros(self.x_t.shape), self.sigma_t])

        objective = 0.
        constraints = [self.mean[:, 0] == mean_t[:, 0]]
        constraints += [self.sigma[:, 0] == sigma_t[:, 0]]
        for k in range(self.pred_horizon):
            X_mu = vstack([reshape(self.mean[:, k], (self.latent_dim, 1)), reshape(self.ref, (self.state_dim, 1))])[:, 0]
            k_u = k if k <= self.control_horizon - 1 else self.control_horizon - 1
            objective += quad_form(self.mean[:self.state_dim, k], self.Q) + quad_form(self.sigma[:self.state_dim, k], self.Q) + quad_form(self.u[:, k_u], self.R)
            # objective += quad_form(self.mean[:self.state_dim, k], self.Q) + quad_form(self.sigma[:, k],
            #                                                                           np.eye(self.latent_dim)) + quad_form(
            #     self.u[:, k_u], self.R)
            constraints += [self.mean[:, k + 1] == self.A @ self.mean[:, k] + self.B @ self.u[:, k_u]]
            constraints += [self.sigma[:, k + 1] == self.A @ self.sigma[:, k] + self.B @ self.u[:, k_u]]
            if args['apply_state_constraints']:
                constraints += [self.s_bound_low <= self.x[:self.state_dim, k],
                                self.x[:self.state_dim, k] <= self.s_bound_high]
            if args['apply_action_constraints'] and k < self.control_horizon:
                constraints += [self.a_bound_low <= self.u[:, k], self.u[:, k] <= self.a_bound_high]
        objective += self.end_weight * quad_form(self.mean[:self.state_dim, -1], self.Q)
        self.prob = Problem(Minimize(objective), constraints)


    def choose_action(self, x_0, *args):

        [mean, sigma] = self.model.encode([x_0])

        self.mean_t.value = np.expand_dims(mean[0], 1)
        self.sigma_t.value = np.expand_dims(sigma[0], 1)
        self.x_t.value = np.expand_dims(x_0, 1)
        self.A.value = self.model.A_result.T
        self.B.value = self.model.B_result.T
        # self.prob.solve(solver=OSQP, warm_start=True)
        self.prob.solve(solver=MOSEK, warm_start=True,
                        mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
        # self.prob.solve()
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            u = self.u[:, 0].value * self.scale_u + self.shift_u
        else:
            print("Error: Cannot solve mpc..")
            u = None

        return u

def test():
    A = np.array([[-1., 2.], [2.2, 1.7]])
    B = np.array([[2.], [1.6]])
    Q = np.eye(1)
    gamma = 0.8
    R = np.eye(1)

    C = np.array([[1., 2.]])

    A_1 = np.sqrt(gamma) * block_diag(A, np.eye(1))
    B_1 = np.sqrt(gamma) * np.vstack((B, np.zeros([1, 1])))
    C_1 = np.hstack((C, -np.eye(1)))
    # CQC = np.vstack((np.hstack((np.dot(C.T, np.dot(Q, C)), -np.dot(C.T, Q))), np.hstack((-np.dot(Q, C), Q))))
    CQC = np.dot(C_1.T, np.dot(Q, C_1))
    K = dlqr(A_1, B_1, CQC, R)
    print('no bug')

class Stochastic_MPC_with_observation(base_MPC):

    def __init__(self, model, args):
        self.n_of_random_seeds = args['n_of_random_seeds']
        super(Stochastic_MPC_with_observation, self).__init__(model, args)

    def _build_matrices(self, args):

        self.latent_dim = args['latent_dim']
        self.state_dim = args['state_dim']
        self.Q = self.args['Q']
        self.R = self.args['R']

        self.A_holder = Parameter((self.latent_dim, self.latent_dim,))
        self.B_holder = Parameter((self.latent_dim, self.args['act_dim'],))

        self.K_holder = Parameter((self.args['act_dim'], self.latent_dim,))
        self.P_holder = Parameter((self.latent_dim, self.latent_dim,))

        self.u_s_holder = Parameter((self.args['act_dim'],))
        self.ref = Parameter(self.state_dim)

    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.get_shift_and_scale()
        self.reference = (self.args['reference'] - self.shift) / self.scale
        self.A = self.model.A_result.T
        self.B = self.model.B_result.T
        self.C = self.model.C_result.T

        self._shift_and_scale_bounds(self.args)
        self._set_LQR_controller()
        self._create_set_point_u_prob()
        self._get_set_point_u(self.reference)
        self._create_prob(self.args)

    def _create_set_point_u_prob(self):

        self.u_s_var = Variable((self.a_dim))
        phi_s = Variable((self.latent_dim))
        X_s = hstack([phi_s, self.ref])

        constraint = [self.C @ phi_s == self.C @ (self.A @ phi_s + self.B @ self.K @ X_s + self.B @ self.u_s_var)]
        constraint += [self.C @ phi_s == self.ref]
        objective = quad_form(self.u_s_var, np.eye(self.a_dim))

        self._set_point_prob = Problem(Minimize(objective), constraint)

    def _create_prob(self, args):
        # self.K = np.zeros([self.a_dim, self.latent_dim + self.state_dim])
        self.u = Variable((self.a_dim, self.control_horizon))
        self.x = Variable((self.latent_dim, self.pred_horizon + 1))
        self.phi_t = Parameter((self.latent_dim, 1))
        # self.mean_t = Parameter((self.latent_dim, 1))
        # self.sigma_t = Parameter((self.latent_dim, 1))
        # self.epsilon = Parameter((1, self.n_of_random_seeds))
        #
        # phi_t = self.mean_t + matmul(kron(self.epsilon, self.sigma_t), np.ones([self.n_of_random_seeds,1]))/self.n_of_random_seeds

        objective = 0.
        constraints = [self.x[:, 0] == self.phi_t[:, 0] ]
        for k in range(self.pred_horizon):
            X = vstack([reshape(self.x[:, k], (self.latent_dim,1)), reshape(self.ref, (self.state_dim,1))])[:,0]
            k_u = k if k <= self.control_horizon - 1 else self.control_horizon - 1
            objective += quad_form(X, self.CQC) + quad_form(self.u[:, k_u]-self.u_s_holder, self.R)
            constraints += [self.x[:, k + 1] == self.A @ self.x[:, k] + self.B @self.K @ X + self.B @ self.u[:, k_u]]
            if args['apply_state_constraints']:
                constraints += [self.s_bound_low <= self.x[:self.state_dim, k],
                                self.x[:self.state_dim, k] <= self.s_bound_high]
            if args['apply_action_constraints'] and k < self.control_horizon:
                constraints += [self.a_bound_low <= self.K @ X + self.u[:, k],
                                self.K @ X + self.u[:, k] <= self.a_bound_high]
        X = vstack([reshape(self.x[:, -1], (self.latent_dim, 1)), reshape(self.ref, (self.state_dim,1))])[:,0]
        objective += quad_form(X, self.CPC)
        self.prob = Problem(Minimize(objective), constraints)


    def choose_action(self, x_0, reference, *args):

        [mean, sigma] = self.model.encode([x_0])
        reconstruct_x = self.C.dot(mean[0])*self.scale + self.shift
        epsilon = np.random.normal(0.,.1, [1, self.n_of_random_seeds])
        # self.mean_t.value = np.expand_dims(mean[0], 1)
        # self.sigma_t.value = np.expand_dims(sigma[0], 1)
        # self.epsilon.value = epsilon
        phi_t = np.expand_dims(mean[0], 1) + np.kron(epsilon, np.expand_dims(sigma[0], 1)).dot(np.ones([self.n_of_random_seeds,1]))/self.n_of_random_seeds
        # self.phi_t.value = np.expand_dims(mean[0], 1)
        self.phi_t.value = phi_t
        # self.prob.solve(solver=OSQP, warm_start=True)
        # t1 = time.time()
        try:
            self.prob.solve(solver=MOSEK, warm_start=True,
                            mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
        except cvxpy.error.SolverError:
            self.prob.solve(solver=MOSEK, warm_start=True,
                            mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}, verbose=True)
        # t2 = time.time()
        # print(str(t2-t1))
        # self.prob.solve()
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            X = np.concatenate([mean[0], self.reference])
            # X = np.concatenate([phi_t[:,0], self.reference])
            u = np.dot(self.K, X) + self.u[:, 0].value
            u = u * self.scale_u + self.shift_u
        else:
            X = np.concatenate([mean[0], self.reference])
            # X = np.concatenate([phi_t[:,0], self.reference])
            u = np.dot(self.K, X)
            u = u * self.scale_u + self.shift_u
            print('Not optimal')
            # raise cvxpy.error.SolverError

        return u

class Stochastic_MPC_with_observation_v2(Stochastic_MPC_with_observation):
    def __init__(self, model, args):

        super(Stochastic_MPC_with_observation_v2, self).__init__(model, args)

    def _create_set_point_u_prob(self):

        self.u_s_var = Variable((self.a_dim))
        phi_s = Variable((self.latent_dim))

        constraint = [self.C @ phi_s == self.C @ (self.A @ phi_s + self.B @ self.u_s_var)]
        constraint += [self.C @ phi_s == self.ref]
        objective = quad_form(self.u_s_var, np.eye(self.a_dim))

        self._set_point_prob = Problem(Minimize(objective), constraint)

    def _create_prob(self, args):

        self.u = Variable((self.a_dim, self.control_horizon))
        self.mean = Variable((self.latent_dim, self.pred_horizon + 1))
        self.sigma = Variable((self.latent_dim, self.pred_horizon + 1))



        self.mean_t = Parameter((self.latent_dim, 1))
        self.sigma_t = Parameter((self.latent_dim, 1))


        objective = 0.
        constraints = [self.mean[:, 0] == self.mean_t[:, 0]]
        constraints += [self.sigma[:, 0] == self.sigma_t[:, 0]]
        for k in range(self.pred_horizon):
            X_mean = vstack([reshape(self.mean[:, k], (self.latent_dim,1)), reshape(self.ref, (self.state_dim,1))])[:,0]
            X_sigma = vstack([reshape(self.sigma[:, k], (self.latent_dim, 1)), np.zeros([self.state_dim, 1])])[:,
                     0]
            k_u = k if k <= self.control_horizon - 1 else self.control_horizon - 1
            objective += quad_form(X_mean, self.CQC) + quad_form(self.u[:, k_u]-self.u_s_holder, self.R)
            constraints += [self.mean[:, k + 1] == self.A @ self.mean[:, k] + self.B @ self.K @ (X_sigma) + self.B @ self.u[:, k_u]]
            constraints += [self.sigma[:, k + 1] == self.A @ self.sigma[:,k] + self.B @ self.K @ (X_sigma)+ self.B @ self.u[:, k_u]]

            if args['apply_state_constraints']:
                constraints += [self.s_bound_low <= self.x[:self.state_dim, k],
                                self.x[:self.state_dim, k] <= self.s_bound_high]
            if args['apply_action_constraints'] and k < self.control_horizon:
                constraints += [self.a_bound_low <= self.K @ (X_sigma) + self.u[:, k],
                                self.K @ (X_sigma) + self.u[:, k] <= self.a_bound_high]
        X_mean = vstack([reshape(self.mean[:, -1], (self.latent_dim, 1)), reshape(self.ref, (self.state_dim, 1))])[:, 0]

        objective += quad_form(X_mean, self.CQC)
        self.prob = Problem(Minimize(objective), constraints)

    def choose_action(self, x_0, reference, *args):
        t1 = time.time()
        [mean, sigma] = self.model.encode([x_0])


        self.mean_t.value = np.expand_dims(mean[0], 1)
        self.sigma_t.value = np.expand_dims(sigma[0], 1)

        # self.prob.solve(solver=OSQP, warm_start=True)

        try:
            self.prob.solve(solver=MOSEK, warm_start=True,
                           mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
        except cvxpy.error.SolverError:
            self.prob.solve(solver=MOSEK, warm_start=True,
                            mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}, verbose=True)

        # self.prob.solve()
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            #X = np.concatenate([mean[0], self.reference])
            X = np.concatenate([sigma[0], np.zeros_like(self.reference)])
            u = np.dot(self.K, X) + self.u[:, 0].value
            u = u * self.scale_u + self.shift_u
        else:
            X = np.concatenate([mean[0], self.reference])
            # X = np.concatenate([sigma[0], np.zeros_like(self.reference)])
            u = np.dot(self.K, X)
            u = u * self.scale_u + self.shift_u
            print('Not optimal')
            # raise cvxpy.error.SolverError
        t2 = time.time()
        # print(str(t2 - t1))
        self._store_result()
        return u

    def _store_result(self):
        reorganize_value(self.u)
        reorganize_value(self.mean)
        reorganize_value(self.sigma)

class Stochastic_MPC_with_motion_planning(base_MPC):

    def __init__(self, model, args):
        self.n_of_random_seeds = args['n_of_random_seeds']
        super(Stochastic_MPC_with_motion_planning, self).__init__(model, args)

    def _build_matrices(self, args):

        self.latent_dim = args['latent_dim']
        self.state_dim = args['state_dim']
        self.Q = self.args['Q']
        self.R = self.args['R']

        self.A_holder = Parameter((self.latent_dim, self.latent_dim,))
        self.B_holder = Parameter((self.latent_dim, self.args['act_dim'],))

        self.K_holder = Parameter((self.args['act_dim'], self.latent_dim,))
        self.P_holder = Parameter((self.latent_dim, self.latent_dim,))

        self.u_s_holder = Parameter((self.args['act_dim'],))
        self.ref = Parameter((args['pred_horizon'],self.state_dim, ))

    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.get_shift_and_scale()
        self.reference = np.zeros((999,18))
        for i in range(983):
            self.reference[i,:]=(self.args['reference'][i,:] - self.shift) / self.scale
        #self.reference = (self.args['reference'] - self.shift) / self.scale                                                                                                         ######
        self.A = self.model.A_result.T
        self.B = self.model.B_result.T
        self.C = self.model.C_result.T

        self._shift_and_scale_bounds(self.args)
        self._set_LQR_controller()
        #self._create_set_point_u_prob()        #####
        #self._get_set_point_u(self.reference)                                                                                                                                       ######
        self._create_prob(self.args)

    def _create_set_point_u_prob(self):

        self.u_s_var = Variable((self.a_dim))
        phi_s = Variable((self.latent_dim))
        X_s = hstack([phi_s, self.ref])

        constraint = [self.C @ phi_s == self.C @ (self.A @ phi_s + self.B @ self.K @ X_s + self.B @ self.u_s_var)]
        constraint += [self.C @ phi_s == self.ref]
        objective = quad_form(self.u_s_var, np.eye(self.a_dim))

        self._set_point_prob = Problem(Minimize(objective), constraint)

    def _create_prob(self, args):
        # self.K = np.zeros([self.a_dim, self.latent_dim + self.state_dim])
        self.u = Variable((self.a_dim, self.control_horizon))
        self.x = Variable((self.latent_dim, self.pred_horizon + 1))
        self.phi_t = Parameter((self.latent_dim, 1))
        # self.mean_t = Parameter((self.latent_dim, 1))
        # self.sigma_t = Parameter((self.latent_dim, 1))
        # self.epsilon = Parameter((1, self.n_of_random_seeds))
        #
        # phi_t = self.mean_t + matmul(kron(self.epsilon, self.sigma_t), np.ones([self.n_of_random_seeds,1]))/self.n_of_random_seeds

        objective = 0.
        constraints = [self.x[:, 0] == self.phi_t[:, 0] ]
        for k in range(self.pred_horizon):
            X = vstack([reshape(self.x[:, k], (self.latent_dim,1)), reshape(self.ref[k,:], (self.state_dim,1))])[:,0]
            k_u = k if k <= self.control_horizon - 1 else self.control_horizon - 1
            objective += quad_form(X, self.CQC) + quad_form(self.u[:, k_u], self.R)
            constraints += [self.x[:, k + 1] == self.A @ self.x[:, k] + self.B @self.K @ X + self.B @ self.u[:, k_u]]
            if args['apply_state_constraints']:
                constraints += [self.s_bound_low <= self.x[:self.state_dim, k],
                                self.x[:self.state_dim, k] <= self.s_bound_high]
            if args['apply_action_constraints'] and k < self.control_horizon:
                constraints += [self.a_bound_low <= self.K @ X + self.u[:, k],
                                self.K @ X + self.u[:, k] <= self.a_bound_high]
        X = vstack([reshape(self.x[:, -1], (self.latent_dim, 1)), reshape(self.ref[k,:], (self.state_dim,1))])[:,0]
        objective += quad_form(X, self.CPC)
        self.prob = Problem(Minimize(objective), constraints)


    def choose_action(self, x_0, reference, *args):

        [mean, sigma] = self.model.encode([x_0])
        reconstruct_x = self.C.dot(mean[0])*self.scale + self.shift
        epsilon = np.random.normal(0.,.1, [1, self.n_of_random_seeds])
        # self.mean_t.value = np.expand_dims(mean[0], 1)
        # self.sigma_t.value = np.expand_dims(sigma[0], 1)
        # self.epsilon.value = epsilon
        phi_t = np.expand_dims(mean[0], 1) + np.kron(epsilon, np.expand_dims(sigma[0], 1)).dot(np.ones([self.n_of_random_seeds,1]))/self.n_of_random_seeds
        # self.phi_t.value = np.expand_dims(mean[0], 1)
        self.phi_t.value = phi_t
        self.ref.value = reference
        # self.prob.solve(solver=OSQP, warm_start=True)
        # t1 = time.time()
        try:
            self.prob.solve(solver=MOSEK, warm_start=True,
                            mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual})
        except cvxpy.error.SolverError:
            self.prob.solve(solver=MOSEK, warm_start=True,
                            mosek_params={mosek.iparam.intpnt_solve_form: mosek.solveform.dual}, verbose=True)
        # t2 = time.time()
        # print(str(t2-t1))
        # self.prob.solve()
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            X = np.concatenate([mean[0], self.reference[0,:]])
            # X = np.concatenate([phi_t[:,0], self.reference])
            u = np.dot(self.K, X) + self.u[:, 0].value
            u = u * self.scale_u + self.shift_u
        else:
            X = np.concatenate([mean[0], self.reference[0,:]])
            # X = np.concatenate([phi_t[:,0], self.reference])
            u = np.dot(self.K, X)
            u = u * self.scale_u + self.shift_u
            print('Not optimal')
            # raise cvxpy.error.SolverError

        return u



def reorganize_value(variable):
    val = variable.value
    val[:, :-1] = variable.value[:, 1:]
    variable.project_and_assign(val)

if __name__ == '__main__':
    test()