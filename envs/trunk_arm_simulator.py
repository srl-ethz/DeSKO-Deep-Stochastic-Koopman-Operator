
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from softtrunk_pybind_module import Simulator, State, SoftTrunkModel


class TrunkArmSimulator(gym.Env):
    """

    """

    def __init__(self, desired_pos=None):
        self.stm = SoftTrunkModel()
        state = State()
        # set initial state

        # set pressure (use A matrix to figure out number of chambers)
        self.stm.updateState(state)
        _, _, _, _, _, A, _ = self.stm.getModel()
        self.Ts = 0.01
        max_pressure = 600*100
        min_pressure = 0
        high = np.pi/6 * np.ones(2*state.q.shape[0])
        action_bound_high = max_pressure * np.ones(A.shape[1])
        action_bound_low = min_pressure * np.ones(A.shape[1])
        self.action_space = spaces.Box(low=action_bound_low, high=action_bound_high, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        if desired_pos is None:
            self.desired_pos = generate_reference_for_trunk(2)
            self.fixed_ref = False
        else:
            self.desired_pos = desired_pos
            self.fixed_ref = True
        self.seed()
        self.viewer = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, impulse=np.zeros([6])):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = list(action + impulse)
        self.sim.simulate(action)
        state = self.sim.getState()
        observation = np.concatenate([state.q, state.dq])
        done = False
        cost = self.cost(state)
        return observation, cost, done, dict(reference=self.desired_pos, data_collection_done=False, )

    def reset(self):
        if self.fixed_ref is False:
            self.desired_pos = generate_reference_for_trunk(2)
        state = State()

        self.sim = Simulator(self.stm, 0.01, 1, state)
        observation = np.concatenate([state.q, state.dq])

        return observation

    def cost(self, state):
        Q = np.concatenate([np.ones(4), np.zeros(4)])
        cost = Q * np.linalg.norm(self.desired_pos-np.concatenate([state.q, state.dq]))
        return cost

    def close(self):
        return

def extract_position_from_frame(frame):
    pos = frame[:3, 3]
    return pos

def generate_reference_for_trunk(num_of_segment):
    threshold = [np.pi, np.pi/6]
    reference = []
    for n in range(num_of_segment):
        # theta = np.random.uniform(0, 1, 1) * threshold[0]
        # phi = np.random.uniform(-1, 1, 1) * threshold[1]
        theta = 1. * threshold[0]
        phi = 1. * threshold[1]
        reference.append(np.squeeze(np.array([theta * np.cos(phi), theta * np.sin(phi)])))

    reference.append(np.zeros([2*num_of_segment], dtype=np.float32))
    return np.concatenate(reference, axis=0)

if __name__ == '__main__':

    e = TrunkArmSimulator()
    for i in range(2):
        e.reset()
        for t in range(1000):
            a = e.action_space.sample()
            print(e.step(a))
