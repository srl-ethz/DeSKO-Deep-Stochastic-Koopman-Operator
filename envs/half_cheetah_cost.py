import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import error, spaces
import pandas as pd


reference_ = pd.read_csv('halfcheetah_states.csv')
reference =reference_.values


means = np.zeros((4,18))

for i in range(18):
    means[0,i] = np.mean(reference[1:,i])
    means[1,i] = np.var(reference[1:,i])
    means[2, i] = np.min(reference[1:, i])
    means[3, i] = np.max(reference[1:, i])


means[1,17] = 1
means[3,17] = 1





def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'), dtype=np.float32)
        high = np.full(observation.shape, float('inf'), dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space

class HalfCheetahEnv_cost(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,des_v=None):

        if des_v is None:
            self.des_v = np.array([10, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
        else:
            self.des_v = des_v
        #v=args['reference']
        #self.des_v = args['reference']
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)



    def step(self, action, disturbance_noise = np.zeros([6])):
        xposbefore = self.sim.data.qpos[0]
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(action + disturbance_noise, self.frame_skip)

        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        cost_ctrl = 0.1 * np.square(action).sum()
        v = (xposafter - xposbefore)/self.dt
        #run_cost = np.square(v-self.des_v[0])
        # reward_run = xposafter
        #reward = run_cost #+ cost_ctrl
        reward = ob[8]
        # if abs(ob[2]) > np.pi/2:
        #     done = True
        # else:
        done = False


        l_rewards = 0.

        # print(xposafter)
        #return np.concatenate(ob, v), reward, done, dict(data_collection_done= done,reference=self.des_v, state_of_interest=v)
        return ob, reward, done, dict(data_collection_done= done, reference=self.des_v, state_of_interest=v)

    def step_halfcheetah(self, action, reference, disturbance_noise = np.zeros([6])):
        xposbefore = self.sim.data.qpos[0]
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(action + disturbance_noise, self.frame_skip)

        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        cost_ctrl = 0.1 * np.square(action).sum()
        v = (xposafter - xposbefore)/self.dt
        #run_cost = np.square(v-self.des_v[0])
        # reward_run = xposafter
        #reward = run_cost #+ cost_ctrl

        ##standartize reference and ob
        # reference_stand = np.zeros(reference.shape)
        # ob_stand = np.zeros(ob.shape)
        # for i in range(reference.shape[0]):
        #     reference_stand[i] = (reference[i]-means[0,i])/means[1,i]
        #     ob_stand[i] = (ob[i]-means[0,i])/means[1,i]
        #
        #
        # reward = np.square(np.subtract(reference_stand, ob_stand)).mean()
        #

        ##normalize data between 0 and 1
        # reference_stand = np.zeros(reference.shape)
        # ob_stand = np.zeros(ob.shape)
        # for i in range(reference.shape[0]):
        #     reference_stand[i] = (reference[i]-means[2,i])/(means[3,i]-means[2,i])
        #     ob_stand[i] = (ob[i]-means[2,i])/(means[3,i]-means[2,i])
        #
        #
        # reward = np.square(np.subtract(reference_stand, ob_stand)).mean()

        ##tracking error of joint angles
        # reward = np.square(np.subtract(reference[1:7], ob[1:7])).mean()

        ## reward = velocity in x direction
        reward = ob[8]

        print(reward)

        # if abs(ob[2]) > np.pi/2:
        #     done = True
        # else:
        done = False


        l_rewards = 0.

        # print(xposafter)
        #return np.concatenate(ob, v), reward, done, dict(data_collection_done= done,reference=self.des_v, state_of_interest=v)
        return ob, reward, done, dict(data_collection_done=done, reference=self.des_v, state_of_interest=v)

    def _get_obs(self):
        zero=np.array([0])
        return np.concatenate([
            #self.sim.data.qpos.flat[1:],
            #np.sqrt(np.absolute((self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt)),
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            zero
            #(self.sim.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt
            #self.get_body_com("torso").flat
        ])

    def _set_observation_space(self, observation):
        #low = np.full(observation.shape+1, -float('inf'), dtype=np.float32)     #new ob space
        low = np.full(observation.shape, -10000., dtype=np.float32)
        #high = np.full(observation.shape+1, float('inf'), dtype=np.float32)	    #new ob space 
        high = np.full(observation.shape, 10000., dtype=np.float32)
        space = spaces.Box(low, high, dtype=observation.dtype)
        self.observation_space = convert_observation_to_space(observation)

        return self.observation_space

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
