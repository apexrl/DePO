import math
import torch
import numpy as np
import multiprocessing
import random


def collect_samples(
    pid,
    queue,
    memory,
    env,
    policy,
    render,
    batch_size,
    additional_memory=None,
    epsilon=0.0,
):
    if pid > 0:
        torch.manual_seed(torch.randint(0, 5000, (1,)) * pid)
        if hasattr(env, "np_random"):
            env.np_random.seed(env.np_random.randint(5000) * pid)
        if hasattr(env, "seed"):
            env.seed(np.random.randint(5000) * pid)
    log = dict()
    num_steps = 0
    total_reward = 0
    total_eplen = 0
    num_episodes = 0

    while num_steps < batch_size:
        obs = env.reset()
        reward_episode = 0
        eplen_episode = 0

        for t in range(10000):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action, agent_info = policy.get_action(obs)
                action = action.cpu().numpy()

            next_obs, reward, done, _ = env.step(action)
            eplen_episode += 1

            reward_episode += reward

            memory.add(obs, action, reward, next_obs, done)
            if additional_memory is not None:
                additional_memory.add(obs, action, reward, next_obs, done)

            if render:
                env.render()
            if done:
                # print(np.argmax(obs))
                break

            obs = next_obs

        # log stats
        num_steps += t + 1
        num_episodes += 1
        total_reward += reward_episode
        total_eplen += eplen_episode

    log["num_steps"] = num_steps
    log["num_episodes"] = num_episodes
    log["total_reward"] = total_reward
    log["avg_reward"] = total_reward / num_episodes
    log["total_eplen"] = total_eplen
    log["avg_eplen"] = total_eplen / num_episodes

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        if additional_memory is not None:
            return memory, additional_memory, log
        return memory, log


def merge_log(log_list):
    log = dict()
    log["total_reward"] = sum([x["total_reward"] for x in log_list])
    log["total_eplen"] = sum([x["total_eplen"] for x in log_list])
    log["num_episodes"] = sum([x["num_episodes"] for x in log_list])
    log["num_steps"] = sum([x["num_steps"] for x in log_list])
    log["avg_reward"] = log["total_reward"] / log["num_episodes"]
    log["avg_eplen"] = log["total_eplen"] / log["num_episodes"]
    return log


class Sampler:
    def __init__(
        self,
        env,
        policy,
        memory,
        device,
        clear_memory=True,
        writer=None,
        num_threads=1,
        additional_memory=None,
    ):
        self.env = env
        self.policy = policy
        self.memory = memory
        self.writer = writer
        self.clear_memory = clear_memory
        self.num_threads = num_threads
        self.device = device
        self._cnt = 0
        self.additional_memory = additional_memory

    def collect_samples(self, batch_size, render=False, epsilon=0.0):
        self.policy.set_device(torch.device("cpu"))
        thread_batch_size = int(math.floor(batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []
        if self.clear_memory:
            self.memory.clear()

        for i in range(self.num_threads - 1):
            worker_args = (
                i + 1,
                queue,
                self.memory,
                self.env,
                self.policy,
                False,
                thread_batch_size,
            )
            workers.append(
                multiprocessing.Process(target=collect_samples, args=worker_args)
            )
        for worker in workers:
            worker.start()

        if self.additional_memory is not None:
            self.memory, self.additional_memory, log = collect_samples(
                0,
                None,
                self.memory,
                self.env,
                self.policy,
                render,
                thread_batch_size,
                additional_memory=self.additional_memory,
                epsilon=epsilon,
            )
        else:
            self.memory, log = collect_samples(
                0,
                None,
                self.memory,
                self.env,
                self.policy,
                render,
                thread_batch_size,
                epsilon=epsilon,
            )

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            self.memory.append(worker_memory)
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)

        print(
            "=iter {}, eplen {:.1f}, reward {:.2f}".format(
                self._cnt, log["avg_eplen"], log["avg_reward"]
            ),
        )

        if self.writer is not None:
            self.writer.add_scalar(
                "Global/Info/episode_length",
                log["avg_eplen"],
                self._cnt,
            )

        self.policy.set_device(self.device)

        metrics_dict = {}
        metrics_dict["avg_eplen"] = log["avg_eplen"]

        self._cnt += 1
        if self.additional_memory:
            return self.memory, self.additional_memory, metrics_dict
        return self.memory, metrics_dict, log["avg_reward"]
