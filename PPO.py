import os
import numpy as np
import gym
import traci
import torch
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import random
import time
from collections import defaultdict
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class EpisodeMetrics:
    def __init__(self):
        self.waiting_times = []
        self.queue_lengths = []
        self.num_vehicles = []
        self.rewards = []
        self.unique_vehicles = set()
        self.exit_vehicles = set()

    def update(self, waiting_time, queue_length, vehicles, reward):
        self.waiting_times.append(waiting_time)
        self.queue_lengths.append(queue_length)
        self.num_vehicles.append(len(vehicles))
        self.rewards.append(reward)

        # Identify newly exited vehicles
        exited_vehicles = self.unique_vehicles - set(vehicles)
        self.exit_vehicles.update(exited_vehicles)

        # Update the unique vehicle set
        self.unique_vehicles.update(vehicles)

    def get_stats(self):
        if not self.waiting_times:
            return None
        return {
            'avg_waiting_time': np.mean(self.waiting_times),
            'avg_queue_length': np.mean(self.queue_lengths),
            'total_throughput': len(self.exit_vehicles),
            'avg_reward': np.mean(self.rewards) if self.rewards else 0.0,
            'total_vehicles_served': len(self.exit_vehicles),
            'max_queue_length': max(self.queue_lengths),
            'throughput_rate': len(self.exit_vehicles) / max(len(self.waiting_times), 1),
            'avg_completion_time': np.mean([t for t in self.waiting_times if t > 0]),
            'queue_stability': np.std(self.queue_lengths),
            'system_efficiency': len(self.exit_vehicles) / max(sum(self.queue_lengths), 1)
        }

def evaluate_ppo_model(iterations=5):
    print("Starting evaluation...")
    episode1_stats = []
    
    for iteration in range(iterations):
        print(f"\nIteration {iteration + 1}/{iterations}")
        
        env = DummyVecEnv([lambda: TrafficLightEnv(Config.SUMO_CONFIG_FILE)])
        metrics = EpisodeMetrics()
        
        try:
            model = PPO.load(Config.MODEL_SAVE_PATH, env=env)
            if iteration == 0:
                print("Loaded trained model successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
        
        obs = env.reset()
        base_env = env.envs[0]
        done = False
        step_count = 0
        last_stats_before_step50 = None
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            
            for _ in range(Config.SIMULATION_STEPS):
                obs, rewards, dones, info = env.step(action)
                step_count += 1
                
                waiting_time, queue_length, vehicles = base_env._get_traffic_metrics()
                metrics.update(waiting_time, queue_length, vehicles, rewards[0])
                
                print(f"Step {step_count}: Vehicles={len(vehicles)}, Queue={queue_length}, "
                      f"Waiting={waiting_time:.2f}, Reward={rewards[0]:.2f}")
                
                if step_count == 49:  # Save stats before Step 50
                    last_stats_before_step50 = metrics.get_stats()
                
                done = dones[0]
                if done:
                    break
        
        if last_stats_before_step50:
            print(f"Episode 1 Stats: {last_stats_before_step50}")
            episode1_stats.append(last_stats_before_step50)
        
        env.close()
    
    # Calculate final statistics 
    if episode1_stats:
        overall_stats = {
            'avg_waiting_time': np.mean([stat['avg_waiting_time'] for stat in episode1_stats]),
            'avg_queue_length': np.mean([stat['avg_queue_length'] for stat in episode1_stats]),
            'avg_throughput': np.mean([stat['total_throughput'] for stat in episode1_stats]),
            'avg_reward': np.mean([stat['avg_reward'] for stat in episode1_stats])
        }
        
        overall_stats_std = {
            'waiting_time_std': np.std([stat['avg_waiting_time'] for stat in episode1_stats]),
            'queue_length_std': np.std([stat['avg_queue_length'] for stat in episode1_stats]),
            'throughput_std': np.std([stat['total_throughput'] for stat in episode1_stats]),
            'reward_std': np.std([stat['avg_reward'] for stat in episode1_stats])
        }
        
        print("\nOverall Performance:")
        for metric, value in overall_stats.items():
            print(f"  {metric}: {value:.2f} ± {overall_stats_std[metric.replace('avg_', '') + '_std']:.2f}")
        
        return overall_stats, overall_stats_std
    else:
        print("\nNo valid statistics collected during evaluation")
        return None, None

class Config:
    LANE_NAMES = {
        'NORTH': ['N_0', 'N_1'],
        'EAST': ['E_0', 'E_1'],
        'SOUTH': ['S_0', 'S_1'],
        'WEST': ['W_0', 'W_1']
    }
    SUMO_CONFIG_FILE = "cross/cross.sumocfg"
    MODEL_SAVE_PATH = "ppo_traffic_model"
    MAX_STEPS = 500
    TRAIN_TIMESTEPS = 10000
    LEARNING_RATE = 1e-4
    N_STEPS = 512
    BATCH_SIZE = 64
    N_EPOCHS = 10
    SIMULATION_STEPS = 10
    REWARD_WEIGHTS = {
        'waiting_time': 0.4,
        'queue_length': 0.5,
        'emergency': 0.8,
        'throughput': 0.3
    }
    TL_ID = "0"
    NUM_PHASES = 4
    VEHICLE_TYPES = ['motorcycles', 'truck', 'bus', 'car','emergency']

class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.episode_metrics = defaultdict(list)
        self.current_vehicles = set()  # Currently active vehicles
        self.completed_vehicles = set()  # Vehicles that have exited
        self.total_throughput = 0  # Track completed vehicles
        self.episode_metrics['reward_components'] = [] 
        
    def update(self, waiting_time, queue_length, vehicles_in_system, reward):
        if reward is None:
            reward = 0.0
        
        # Track basic metrics
        self.episode_metrics['waiting_times'].append(waiting_time)
        self.episode_metrics['queue_lengths'].append(queue_length)
        
        # Find newly completed vehicles (vehicles that were present but are now gone)
        completed = self.current_vehicles - set(vehicles_in_system)
        self.completed_vehicles.update(completed)
        self.total_throughput += len(completed)  # Increment throughput by completed vehicles
        
        # Update current vehicles
        self.current_vehicles = set(vehicles_in_system)
        
        # Store metrics
        self.episode_metrics['throughput'].append(len(completed))  # Store incremental throughput
        self.episode_metrics['rewards'].append(reward)
        
    def get_episode_stats(self):
        stats = {
            'avg_waiting_time': np.mean(self.episode_metrics['waiting_times']),
            'avg_queue_length': np.mean(self.episode_metrics['queue_lengths']),
            'total_throughput': len(self.completed_vehicles),  # Use total completed vehicles
            'avg_reward': np.mean(self.episode_metrics['rewards']) if self.episode_metrics['rewards'] else 0.0,
            'total_vehicles_served': len(self.completed_vehicles),
            'max_queue_length': max(self.episode_metrics['queue_lengths'], default=0),
            'throughput_rate': len(self.completed_vehicles) / max(len(self.episode_metrics['waiting_times']), 1),
            'avg_completion_time': np.mean([t for t in self.episode_metrics['waiting_times'] if t > 0]),
            'queue_stability': np.std(self.episode_metrics['queue_lengths']),
            'system_efficiency': len(self.completed_vehicles) / max(sum(self.episode_metrics['queue_lengths']), 1)
        }
        return stats




class TrafficLightEnv(gym.Env):
    def __init__(self, sumo_config):
        super().__init__()
        self.sumo_config = sumo_config
        num_features = len(Config.VEHICLE_TYPES) + 4 + 4 + 1
        self.observation_space = spaces.Box(
            low=0, high=float('inf'), shape=(num_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(Config.NUM_PHASES)
        self.current_step = 0
        self.metrics = MetricsTracker()
        self.connection_label = f"conn_{random.randint(0, 10000)}"
        self.episode_metrics = {'reward_components': []}  # Initialize episode_metrics
        # Initialize tracking variables for reward calculation
        self._last_completed_count = 0
        self._last_queue_length = 0

    def _get_lane_direction(self, lane_id):
        for direction, lanes in Config.LANE_NAMES.items():
            if any(lane_id.startswith(lane) for lane in lanes):
                return list(Config.LANE_NAMES.keys()).index(direction)
        return None

    def _get_traffic_metrics(self):
        total_waiting_time, queue_length, vehicles_in_system = 0, 0, []
        for lane_id in traci.lane.getIDList():
            if lane_id.startswith(":"):
                continue
            waiting_time = traci.lane.getWaitingTime(lane_id)
            total_waiting_time += waiting_time
            vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
            queue_vehicles = [v for v in vehicles if traci.vehicle.getSpeed(v) < 0.1]
            queue_length += len(queue_vehicles)
            vehicles_in_system.extend(vehicles)
        return total_waiting_time, queue_length, vehicles_in_system

    def _calculate_reward(self, state):
        """
        Calculate reward based on multiple traffic management objectives:
        - Minimize waiting time
        - Minimize queue length
        - Maximize throughput
        - Prioritize emergency vehicles
        - Maintain traffic flow efficiency
        
        Args:
            state: Current state observation containing traffic metrics
            
        Returns:
            float: Calculated reward value
        """
        try:
            # Get current traffic metrics
            waiting_time, queue_length, vehicles = self._get_traffic_metrics()
            
            # 1. Throughput Component
            # Count newly completed vehicles since last step
            completed_vehicles = len(self.metrics.completed_vehicles) - getattr(self, '_last_completed_count', 0)
            self._last_completed_count = len(self.metrics.completed_vehicles)
            throughput_reward = completed_vehicles * 2.0  # Significant positive reward for completing vehicles
            
            # 2. Waiting Time Component
            # Normalize waiting time penalty
            max_acceptable_wait = 300.0  # 5 minutes
            normalized_wait = min(waiting_time / max_acceptable_wait, 1.0)
            waiting_penalty = -normalized_wait * 1.5
            
            # 3. Queue Length Component
            # Exponential penalty for growing queues
            max_acceptable_queue = 20.0
            normalized_queue = min(queue_length / max_acceptable_queue, 1.0)
            queue_penalty = -(normalized_queue ** 2) * 1.0
            
            # 4. Emergency Vehicle Priority
            # Extract emergency vehicle count from state
            emergency_count = 0
            for veh_id in vehicles:
                try:
                    if traci.vehicle.getTypeID(veh_id) == 'emergency':
                        emergency_count += 1
                except:
                    continue
                    
            # Higher penalty if emergency vehicles are waiting
            emergency_multiplier = 1.0
            if emergency_count > 0:
                emergency_multiplier = 2.0 + (emergency_count * 0.5)
                waiting_penalty *= emergency_multiplier
            
            # 5. Traffic Flow Efficiency
            # Reward for maintaining good flow (ratio of completed to waiting vehicles)
            total_vehicles = max(len(vehicles), 1)
            flow_efficiency = (self.metrics.total_throughput / total_vehicles) if total_vehicles > 0 else 0
            efficiency_reward = flow_efficiency * 1.0
            
            # 6. Queue Growth Rate Penalty
            # Penalize rapidly growing queues
            previous_queue = getattr(self, '_last_queue_length', queue_length)
            queue_growth = max(0, queue_length - previous_queue)
            self._last_queue_length = queue_length
            queue_growth_penalty = -queue_growth * 0.5
            
            # 7. Green Wave Bonus
            # Reward for maintaining continuous flow (less stop-and-go)
            moving_vehicles = sum(1 for v in vehicles if traci.vehicle.getSpeed(v) > 0.1)
            green_wave_bonus = (moving_vehicles / max(total_vehicles, 1)) * 0.5
            
            # Combine all reward components
            total_reward = (
                throughput_reward +      # Weight: 2.0
                waiting_penalty +        # Weight: 1.5
                queue_penalty +          # Weight: 1.0
                efficiency_reward +      # Weight: 1.0
                queue_growth_penalty +   # Weight: 0.5
                green_wave_bonus        # Weight: 0.5
            )
            
            # Normalize final reward
            reward = float(np.clip(total_reward, -10.0, 10.0))
            
            # Store metrics for analysis
            self.metrics.episode_metrics['reward_components'].append({
                'throughput': throughput_reward,
                'waiting': waiting_penalty,
                'queue': queue_penalty,
                'efficiency': efficiency_reward,
                'growth': queue_growth_penalty,
                'green_wave': green_wave_bonus,
                'total': reward
            })
            
            return reward
            
        except Exception as e:
            print(f"Error in reward calculation: {str(e)}")
            return 0.0


    def step(self, action):
        traci.trafficlight.setPhase(Config.TL_ID, action)
        for _ in range(Config.SIMULATION_STEPS):
            traci.simulationStep()
            self.current_step += 1
        state = self._get_state()
        reward = self._calculate_reward(state)
        done = self.current_step >= Config.MAX_STEPS
        waiting_time, queue_length, vehicles = self._get_traffic_metrics()
        self.metrics.update(waiting_time, queue_length, vehicles, reward)

        # Print metrics after each episode
        if done:
            episode_stats = self.metrics.get_episode_stats()
            print(f"Episode {self.current_step // Config.MAX_STEPS} Stats: {episode_stats}")

        return state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            traci.close()
        except:
            pass
        self.current_step = 0
        self.metrics.reset()
        self.episode_metrics = {'reward_components': []}  # Reset episode_metrics
        self._last_completed_count = 0
        self._last_queue_length = 0
        traci.start(["sumo", "-c", self.sumo_config], label=self.connection_label)
        state = self._get_state()
        return state, {}

    def _get_state(self):
        try:
            vehicle_counts = defaultdict(int)
            waiting_times, queue_lengths = np.zeros(4), np.zeros(4)
            for lane_id in traci.lane.getIDList():
                if lane_id.startswith(":"):
                    continue
                direction = self._get_lane_direction(lane_id)
                if direction is not None:
                    waiting_times[direction] += traci.lane.getWaitingTime(lane_id)
                    queue_lengths[direction] += len(
                        [v for v in traci.lane.getLastStepVehicleIDs(lane_id) if traci.vehicle.getSpeed(v) < 0.1]
                    )
                    for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                        vehicle_counts[traci.vehicle.getTypeID(veh_id)] += 1
            current_phase = traci.trafficlight.getPhase(Config.TL_ID)
            state = np.concatenate([
                [vehicle_counts.get(vtype, 0) for vtype in Config.VEHICLE_TYPES],
                waiting_times, queue_lengths, [current_phase]
            ]).astype(np.float32)
            return state
        except Exception as e:
            print(f"Error getting state: {e}")
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

def train_model():
    env = DummyVecEnv([lambda: TrafficLightEnv(Config.SUMO_CONFIG_FILE)])
    model = PPO(
        "MlpPolicy", env,
        learning_rate=Config.LEARNING_RATE,
        n_steps=Config.N_STEPS,
        batch_size=Config.BATCH_SIZE,
        n_epochs=Config.N_EPOCHS,
        verbose=1,
        tensorboard_log="./traffic_control_tensorboard/"
    )
    model.learn(total_timesteps=Config.TRAIN_TIMESTEPS)
    model.save(Config.MODEL_SAVE_PATH)
    print("Training completed!")

if __name__ == "__main__":
    train_model()
    final_metrics, std_metrics = evaluate_ppo_model(iterations=5)
