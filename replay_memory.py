import math
import numpy as np
import random
import progressbar
import os

# Class to load and preprocess data
class ReplayMemory():
    def __init__(self, args, shift, scale, shift_u, scale_u, env, predict_evolution=False):
        """Constructs object to hold and update training/validation data.
        Args:
            args: Various arguments and specifications
            shift: Shift of state values for normalization
            scale: Scaling of state values for normalization
            shift_u: Shift of action values for normalization
            scale_u: Scaling of action values for normalization
            env: Simulation environment
            net: Neural network dynamics model
            sess: TensorFlow session
            predict_evolution: Whether to predict how system will evolve in time
        """
        self.batch_size = args['batch_size']
        self.seq_length = args['pred_horizon']
        self.shift_x = shift
        self.scale_x = scale
        self.shift_u = shift_u
        self.scale_u = scale_u
        self.env = env
        self.total_steps = 0

        print('validation fraction: ', args['val_frac'])


        if args['import_saved_data'] or args['continue_data_collection']:
            self._restore_data('./data/' + args['env_name'])
            self._process_data(args)

            print('creating splits...')
            self._create_split(args)
            self._determine_shift_and_scale(args)
        else:
            print("generating data...")
            self._generate_data(args)
            self._process_data(args)

            print('creating splits...')
            self._create_split(args)

            print('shifting/scaling data...')
            self._shift_scale(args)

    def _generate_data(self, args):
        """Load data from environment
        Args:
            args: Various arguments and specifications
        """

        # Initialize array to hold states and actions
        x = []
        u = []

        # Define progress bar
        bar = progressbar.ProgressBar(maxval=args['total_data_size']).start()
        length_list = []
        done_list = []
        # Loop through episodes
        while True:
            # Define arrays to hold observed states and actions in each trial
            x_trial = np.zeros((args['max_ep_steps'], args['state_dim']), dtype=np.float32)
            u_trial = np.zeros((args['max_ep_steps']-1, args['act_dim']), dtype=np.float32)

            # Reset environment and simulate with random actions
            x_trial[0] = self.env.reset()
            for t in range(1, args['max_ep_steps']):
                action = self.env.action_space.sample()  
                u_trial[t-1] = action
                step_info = self.env.step(action)
                x_trial[t] = np.squeeze(step_info[0])

                if step_info[3]['data_collection_done']:
                    break
            self.total_steps += t
            done_list.append(step_info[3]['data_collection_done'])
            length_list.append(t)
            j = 0
            while j + self.seq_length < len(x_trial):
                x.append(x_trial[j:j + self.seq_length])
                u.append(u_trial[j:j + self.seq_length-1])
                j+=1

            if len(x) >= args['total_data_size']:
                break
            bar.update(len(x))
        bar.finish()

        # Generate test scenario
        self.x_test = []
        self.u_test = []
        self.x_test.append(self.env.reset())
        for t in range(1, args['max_ep_steps']):
            action = self.env.action_space.sample()
            self.u_test.append(action)
            step_info = self.env.step(action)
            self.x_test.append(np.squeeze(step_info[0]))
            if step_info[3]['data_collection_done']:
                break

        x = np.array(x)
        u = np.array(u)
        # Reshape and trim data sets
        self.x = x.reshape(-1, self.seq_length, args['state_dim'])
        self.u = u.reshape(-1, self.seq_length-1, args['act_dim'])
        len_x = int(np.floor(len(self.x)/args['batch_size'])*args['batch_size'])
        self.x = self.x[:len_x]
        self.u = self.u[:len_x]

    def _process_data(self, args):
        """Create batch dicts and shuffle data
        Args:
            args: Various arguments and specifications
        """
        # Create batch_dict
        self.batch_dict = {}

        # Print tensor shapes
        print('states: ', self.x.shape)
        print('inputs: ', self.u.shape)
            
        self.batch_dict['states'] = np.zeros((args['batch_size'], self.seq_length, args['state_dim']))
        self.batch_dict['inputs'] = np.zeros((args['batch_size'], self.seq_length-1, args['act_dim']))

        # Shuffle data before splitting into train/val
        print('shuffling...')
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.u = self.u[p]

    def _create_split(self, args):
        """Divide data into training/validation sets
        Args:
            args: Various arguments and specifications
        """
        # Compute number of batches
        self.n_batches = len(self.x)//args['batch_size']
        self.n_batches_val = int(math.floor(args['val_frac'] * self.n_batches))
        self.n_batches_train = self.n_batches - self.n_batches_val

        print('num training batches: ', self.n_batches_train)
        print('num validation batches: ', self.n_batches_val)

        # Divide into train and validation datasets
        self.x_val = self.x[self.n_batches_train*args['batch_size']:]
        self.u_val = self.u[self.n_batches_train*args['batch_size']:]
        self.x = self.x[:self.n_batches_train*args['batch_size']]
        self.u = self.u[:self.n_batches_train*args['batch_size']]

        # Set batch pointer for training and validation sets
        self.reset_batchptr_train()
        self.reset_batchptr_val()

    def _shift_scale(self, args):
        """Shift and scale data to be zero-mean, unit variance
        Args:
            args: Various arguments and specifications
        """
        # Find means and std if not initialized to anything
        if np.sum(self.scale_x) == 0.0:
            self._determine_shift_and_scale(args)

        # Shift and scale values for test sequence
        self.x_test = (self.x_test - self.shift_x)/self.scale_x
        self.u_test = (self.u_test - self.shift_u)/self.scale_u

    def _determine_shift_and_scale(self, args):
        self.shift_x = np.mean(self.x[:self.n_batches_train], axis=(0, 1))
        self.scale_x = np.std(self.x[:self.n_batches_train], axis=(0, 1))
        self.shift_u = np.mean(self.u[:self.n_batches_train], axis=(0, 1))
        self.scale_u = np.std(self.u[:self.n_batches_train], axis=(0, 1))

        # Remove very small scale values
        self.scale_x[self.scale_x < 1e-6] = 1.0

        # Set u norm params to be 0, 1 for pendulum environment
        if args['env_name'] == 'Pendulum-v0':
            self.shift_u = np.zeros_like(self.shift_u)
            self.scale_u = np.ones_like(self.scale_u)

    def update_data(self, x_new, u_new, val_frac):
        """Update training/validation data
        Args:
            x_new: New state values
            u_new: New control inputs
            val_frac: Fraction of new data to include in validation set
        """
        # First permute data
        p = np.random.permutation(len(x_new))
        x_new = x_new[p]
        u_new = u_new[p]


        # Divide new data into training and validation components
        n_seq_val = max(int(math.floor(val_frac * len(x_new))), 1)
        n_seq_train = len(x_new) - n_seq_val
        x_new_val = x_new[n_seq_train:]
        u_new_val = u_new[n_seq_train:]
        x_new = x_new[:n_seq_train]
        u_new = u_new[:n_seq_train]

        # Now update training and validation data
        self.x = np.concatenate((x_new, self.x), axis=0)
        self.u = np.concatenate((u_new, self.u), axis=0)
        self.x_val = np.concatenate((x_new_val, self.x_val), axis=0)
        self.u_val = np.concatenate((u_new_val, self.u_val), axis=0)

        # Update sizes of train and val sets
        self.n_batches_train = len(self.x)//self.batch_size
        self.n_batches_val = len(self.x_val)//self.batch_size

    def next_batch_train(self):
        """Sample a new batch from training data
        Args:
            None
        Returns:
            batch_dict: Batch of training data
        """
        # Extract next batch
        batch_index = self.batch_permuation_train[self.batchptr_train*self.batch_size:(self.batchptr_train+1)*self.batch_size]
        self.batch_dict['states'] = (self.x[batch_index] - self.shift_x)/self.scale_x
        self.batch_dict['inputs'] = (self.u[batch_index] - self.shift_u)/self.scale_u

        # Update pointer
        self.batchptr_train += 1
        return self.batch_dict

    def random_sample(self):

        batch_index = np.random.choice(self.x.shape[0],
                                   size=self.batch_size, replace=False)

        self.batch_dict['states'] = (self.x[batch_index] - self.shift_x) / self.scale_x
        self.batch_dict['inputs'] = (self.u[batch_index] - self.shift_u) / self.scale_u


        return self.batch_dict

    def reset_batchptr_train(self):
        """Reset pointer to first batch in training set
        Args:
            None
        """
        self.batch_permuation_train = np.random.permutation(len(self.x))
        self.batchptr_train = 0

    def next_batch_val(self):
        """Sample a new batch from validation data
        Args:
            None
        Returns:
            batch_dict: Batch of validation data
        """
        # Extract next validation batch
        batch_index = range(self.batchptr_val*self.batch_size,(self.batchptr_val+1)*self.batch_size)
        self.batch_dict['states'] = (self.x_val[batch_index] - self.shift_x)/self.scale_x
        self.batch_dict['inputs'] = (self.u_val[batch_index] - self.shift_u)/self.scale_u

        # Update pointer
        self.batchptr_val += 1
        return self.batch_dict

    def get_all_val_data(self):
        self.batch_dict['states'] = (self.x_val - self.shift_x) / self.scale_x
        self.batch_dict['inputs'] = (self.u_val - self.shift_u) / self.scale_u
        return self.batch_dict

    def get_all_train_data(self):
        # Extract next batch
        self.batch_dict['states'] = (self.x - self.shift_x)/self.scale_x
        self.batch_dict['inputs'] = (self.u - self.shift_u)/self.scale_u
        return self.batch_dict

    def save_data(self, path):
        os.makedirs(path, exist_ok=True)
        np.save(path + '/x.npy', self.x)
        np.save(path + '/u.npy', self.u)
        np.save(path + '/x_test.npy', self.x_test)
        np.save(path + '/u_test.npy', self.u_test)
        np.save(path + '/x_val.npy', self.x_val)
        np.save(path + '/u_val.npy', self.u_val)

    def _restore_data(self, path):
        self.x = np.load(path + '/x.npy')
        self.u = np.load(path + '/u.npy')
        self.x_val = np.load(path + '/x_val.npy')
        self.u_val = np.load(path + '/u_val.npy')
        self.x_test = np.load(path + '/x_test.npy')
        self.u_test = np.load(path + '/u_test.npy')

    def _generate_data_with_controller(self, controller, args):
        # Initialize array to hold states and actions
        x = []
        u = []

        # Define progress bar
        bar = progressbar.ProgressBar(maxval=args['total_data_size']).start()

        # Loop through episodes
        while True:
            # Define arrays to hold observed states and actions in each trial
            x_trial = np.zeros((args['max_ep_steps'], args['state_dim']), dtype=np.float32)
            u_trial = np.zeros((args['max_ep_steps'] - 1, args['act_dim']), dtype=np.float32)

            # Reset environment and simulate with random actions
            x_trial[0] = self.env.reset()
            controller.update_reference(self.env.reference)
            for t in range(1, args['max_ep_steps']):
                action = controller.choose_action(x_trial[t-1], self.env.reference)
                u_trial[t - 1] = action
                step_info = self.env.step(action)
                x_trial[t] = np.squeeze(step_info[0])
                if step_info[3]['data_collection_done']:
                    break
            j = 0
            while j + self.seq_length < len(x_trial):
                x.append(x_trial[j:j + self.seq_length])
                u.append(u_trial[j:j + self.seq_length - 1])
                j += 1

            if len(x) >= args['total_data_size']:
                break
            bar.update(len(x))
        bar.finish()

        # Generate test scenario
        x_trial = np.zeros((args['max_ep_steps'], args['state_dim']), dtype=np.float32)
        u_trial = np.zeros((args['max_ep_steps'] - 1, args['act_dim']), dtype=np.float32)
        # Reset environment and simulate with random actions
        x_trial[0] = self.env.reset()
        controller.update_reference(self.env.reference)
        for t in range(1, args['max_ep_steps']):
            action = controller.choose_action(x_trial[t - 1], self.env.reference)
            u_trial[t - 1] = action
            step_info = self.env.step(action)
            x_trial[t] = np.squeeze(step_info[0])
            if step_info[3]['data_collection_done']:
                break

        self.x_test = x_trial
        self.u_test = u_trial

        x = np.array(x)
        u = np.array(u)
        # Reshape and trim data sets
        x = x.reshape(-1, self.seq_length, args['state_dim'])
        u = u.reshape(-1, self.seq_length - 1, args['act_dim'])
        self.x = np.concatenate([self.x, x], axis=0)
        self.u = np.concatenate([self.u, u], axis=0)
        len_x = int(np.floor(len(self.x) / args['batch_size']) * args['batch_size'])
        self.x = self.x[:len_x]
        self.u = self.u[:len_x]

    def update_and_process_data_with_controller(self, controller, args):

        self._generate_data_with_controller(controller, args)

        self._process_data(args)

        print('creating splits...')
        self._create_split(args)

        print('shifting/scaling data...')
        self._shift_scale(args)

    def reset_batchptr_val(self):
        """Reset pointer to first batch in validation set
        Args:
            None
        """
        self.batchptr_val = 0

