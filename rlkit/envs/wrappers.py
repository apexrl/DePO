import numpy as np
import gym
from gym import Env
from gym.spaces import Box
from collections import deque

from rlkit.core.serializable import Serializable

EPS = np.finfo(np.float32).eps.item()


class ProxyEnv(Serializable, Env):
    def __init__(self, wrapped_env, changing_dynamics=False):
        self._wrapped_env = wrapped_env
        Serializable.quick_init(self, locals())
        super(ProxyEnv, self).__init__()

        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

        self.changing_dynamics = changing_dynamics
        if changing_dynamics:
            wrapped_env.sim.model.opt.gravity[2] *= 0.8
            if type(self.action_space) == Box:
                try:
                    self.action_space = Box(
                        low=np.concatenate(
                            [self.action_space.low, self.action_space.low]
                        ),
                        high=np.concatenate(
                            [self.action_space.high, self.action_space.high]
                        ),
                        shape=(self.action_space.shape[0] * 2,),
                    )
                except:
                    self.action_space = Box(
                        low=self.action_space.low[0],
                        high=self.action_space.high[0],
                        shape=(self.action_space.shape[0] * 2,),
                    )

        self.action_dim = self.action_space.shape[0]
        print("\n Proxy action space: ", self.action_space)

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        if self.changing_dynamics:
            action = (
                np.exp(action[: self.action_dim // 2] + 1)
                - np.exp(action[self.action_dim // 2 :])
            ) / 1.5
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self._wrapped_env, "terminate"):
            self._wrapped_env.terminate()

    def seed(self, seed):
        return self._wrapped_env.seed(seed)

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)


class ScaledEnv(ProxyEnv, Serializable):
    """
    Scale the obs if desired
    Unscale the acts if desired
    """

    def __init__(
        self,
        env,
        obs_mean=None,
        obs_std=None,
        acts_mean=None,
        acts_std=None,
        meta=False,
    ):
        self._wrapped_env = env
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        if obs_mean is not None:
            assert obs_std is not None
            self._scale_obs = True
        else:
            assert obs_std is None
            self._scale_obs = False

        if acts_mean is not None:
            assert acts_std is not None
            self._unscale_acts = True
        else:
            assert acts_std is None
            self._unscale_acts = False

        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.acts_mean = acts_mean
        self.acts_std = acts_std

    def get_unscaled_obs(self, obs):
        if self._scale_obs:
            return obs * (self.obs_std + EPS) + self.obs_mean
        else:
            return obs

    def get_scaled_obs(self, obs):
        if self._scale_obs:
            return (obs - self.obs_mean) / (self.obs_std + EPS)
        else:
            return obs

    def get_unscaled_acts(self, acts):
        if self._unscale_acts:
            return acts * (self.acts_std + EPS) + self.acts_mean
        else:
            return acts

    def get_scaled_acts(self, acts):
        if self._unscale_acts:
            return (acts - self.acts_mean) / (self.acts_std + EPS)
        else:
            return acts

    def step(self, action):
        if self._unscale_acts:
            action = action * (self.acts_std + EPS) + self.acts_mean
        obs, r, done, info = self._wrapped_env.step(action)
        if self._scale_obs:
            obs = (obs - self.obs_mean) / (self.obs_std + EPS)
        return obs, r, done, info

    def reset(self, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        if self._scale_obs:
            obs = (obs - self.obs_mean) / (self.obs_std + EPS)
        return obs

    def log_statistics(self, *args, **kwargs):
        if hasattr(self._wrapped_env, "log_statistics"):
            return self._wrapped_env.log_statistics(*args, **kwargs)
        else:
            return {}

    def log_new_ant_multi_statistics(self, paths, epoch, log_dir):
        if hasattr(self._wrapped_env, "log_new_ant_multi_statistics"):
            return self._wrapped_env.log_new_ant_multi_statistics(paths, epoch, log_dir)
        else:
            return {}


class MinmaxEnv(ProxyEnv, Serializable):
    """
    Scale the obs if desired
    """

    def __init__(
        self,
        env,
        obs_min=None,
        obs_max=None,
    ):
        self._wrapped_env = env
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        if obs_min is not None:
            assert obs_max is not None
            self._scale_obs = True
        else:
            assert obs_max is None
            self._scale_obs = False

        self.obs_min = obs_min
        self.obs_max = obs_max

    def get_unscaled_obs(self, obs):
        if self._scale_obs:
            return obs * (self.obs_max - self.obs_min + EPS) + self.obs_min
        else:
            return obs

    def get_scaled_obs(self, obs):
        if self._scale_obs:
            return (obs - self.obs_min) / (self.obs_max - self.obs_min + EPS)
        else:
            return obs

    def step(self, action):
        obs, r, done, info = self._wrapped_env.step(action)
        if self._scale_obs:
            obs = (obs - self.obs_min) / (self.obs_max - self.obs_min + EPS)
        return obs, r, done, info

    def reset(self, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        if self._scale_obs:
            obs = (obs - self.obs_min) / (self.obs_max - self.obs_min + EPS)
        return obs

    def log_statistics(self, *args, **kwargs):
        if hasattr(self._wrapped_env, "log_statistics"):
            return self._wrapped_env.log_statistics(*args, **kwargs)
        else:
            return {}

    def log_new_ant_multi_statistics(self, paths, epoch, log_dir):
        if hasattr(self._wrapped_env, "log_new_ant_multi_statistics"):
            return self._wrapped_env.log_new_ant_multi_statistics(paths, epoch, log_dir)
        else:
            return {}


class ScaledMetaEnv(ProxyEnv, Serializable):
    """
    Scale the obs if desired
    Unscale the acts if desired
    """

    def __init__(
        self,
        env,
        obs_mean=None,
        obs_std=None,
        acts_mean=None,
        acts_std=None,
    ):
        self._wrapped_env = env
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        if obs_mean is not None:
            assert obs_std is not None
            self._scale_obs = True
        else:
            assert obs_std is None
            self._scale_obs = False

        if acts_mean is not None:
            assert acts_std is not None
            self._unscale_acts = True
        else:
            assert acts_std is None
            self._unscale_acts = False

        self.obs_mean = obs_mean
        self.obs_std = obs_std
        self.acts_mean = acts_mean
        self.acts_std = acts_std

    def step(self, action):
        if self._unscale_acts:
            action = action * (self.acts_std + EPS) + self.acts_mean
        obs, r, done, info = self._wrapped_env.step(action)
        if self._scale_obs:
            obs["obs"] = (obs["obs"] - self.obs_mean) / (self.obs_std + EPS)
            obs["obs"] = obs["obs"][0]
        return obs, r, done, info

    def reset(self, **kwargs):
        obs = self._wrapped_env.reset(**kwargs)
        if self._scale_obs:
            obs["obs"] = (obs["obs"] - self.obs_mean) / (self.obs_std + EPS)
            obs["obs"] = obs["obs"][0]
        return obs

    @property
    def task_identifier(self):
        return self._wrapped_env.task_identifier

    def task_id_to_obs_task_params(self, task_id):
        return self._wrapped_env.task_id_to_obs_task_params(task_id)

    def log_statistics(self, paths):
        return self._wrapped_env.log_statistics(paths)

    def log_diagnostics(self, paths):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            return self._wrapped_env.log_diagnostics(paths)


class NormalizedBoxActEnv(ProxyEnv, Serializable):
    """
    Normalize action to in [-1, 1].
    """

    def __init__(
        self,
        env,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        self.action_space = None
        if isinstance(self._wrapped_env.action_space, Box):
            ub = np.ones(self._wrapped_env.action_space.shape)
            self.action_space = Box(-1 * ub, ub, dtype=np.float64)
        else:
            self.action_space = self._wrapped_env.action_space

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)

    def step(self, action):
        scaled_action = None
        if isinstance(self.action_space, Box):
            action = np.clip(action, -1.0, 1.0)
            lb = self._wrapped_env.action_space.low
            ub = self._wrapped_env.action_space.high
            scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
            # scaled_action = np.clip(scaled_action, lb, ub)
            scaled_action = scaled_action
        else:
            scaled_action = action

        return self._wrapped_env.step(scaled_action)

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def log_diagnostics(self, paths, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            return self._wrapped_env.log_diagnostics(paths, **kwargs)
        else:
            return None

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)


class FrameStackEnv(ProxyEnv, Serializable):
    def __init__(self, env, k):
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)
