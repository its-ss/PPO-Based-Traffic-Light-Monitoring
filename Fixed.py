import os
import numpy as np
import gym
import traci
import random
from gym import spaces
from collections import defaultdict

class Config:
    LANE_NAMES = {
        'NORTH': ['N_0', 'N_1'],
        'EAST': ['E_0', 'E_1'],
        'SOUTH': ['S_0', 'S_1'],
        'WEST': ['W_0', 'W_1']
    }
    SUMO_CONFIG_FILE = "cross/cross.sumocfg"
    MAX_STEPS = 500
    SIMULATION_STEPS = 10
    NUM_PHASES = 4
    FIXED_TIMES = [20, 15, 20, 15]  # Each phase lasts for a fixed duration in seconds (NORTH, EAST, SOUTH, WEST)
    TL_ID = "0"
    VEHICLE_TYPES = ['motorcycles', 'truck', 'bus', 'car', 'emergency']

class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.episode_metrics = defaultdict(list)
        self.current_vehicles = set()
        self.completed_vehicles = set()
        
    def update(self, waiting_time, queue_length, vehicles_in_system, reward):
        if reward is None:
            reward = 0.0
        
        self.episode_metrics['waiting_times'].append(waiting_time)
        self.episode_metrics['queue_lengths'].append(queue_length)
        
        new_vehicles = set(vehicles_in_system) - self.current_vehicles
        completed = self.current_vehicles - set(vehicles_in_system)
        self.completed_vehicles.update(completed)
        self.current_vehicles = set(vehicles_in_system)
        
        self.episode_metrics['throughput'].append(len(completed))
        self.episode_metrics['rewards'].append(reward)
        self.episode_metrics['total_vehicles'].append(len(self.current_vehicles))
        self.episode_metrics['completed_vehicles'].append(len(self.completed_vehicles))
        
    def get_episode_stats(self):
        stats = {
            'avg_waiting_time': np.mean(self.episode_metrics['waiting_times']),
            'avg_queue_length': np.mean(self.episode_metrics['queue_lengths']),
            'total_throughput': sum(self.episode_metrics['throughput']),
            'avg_reward': np.mean(self.episode_metrics['rewards']) if self.episode_metrics['rewards'] else 0.0,
            'total_vehicles_served': len(self.completed_vehicles),
            'max_queue_length': max(self.episode_metrics['queue_lengths'], default=0)
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
        self.action_space = spaces.Discrete(Config.NUM_PHASES)  # 4 fixed phases
        self.current_step = 0
        self.metrics = MetricsTracker()
        self.connection_label = f"conn_{random.randint(0, 10000)}"
        self.phase_counter = 0  # To track which phase is currently active
        self.fixed_times = Config.FIXED_TIMES  # Fixed duration for each phase in seconds

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

    def step(self, action=None):
        # Use fixed-time phases
        traci.trafficlight.setPhase(Config.TL_ID, self.phase_counter)
        phase_duration = self.fixed_times[self.phase_counter]
        
        for _ in range(phase_duration * Config.SIMULATION_STEPS):
            traci.simulationStep()
            self.current_step += 1
        
        state = self._get_state()
        waiting_time, queue_length, vehicles = self._get_traffic_metrics()
        self.metrics.update(waiting_time, queue_length, vehicles, reward=None)

        # Update phase counter and loop back to the first phase when the last phase is reached
        self.phase_counter = (self.phase_counter + 1) % Config.NUM_PHASES
        done = self.current_step >= Config.MAX_STEPS
        
        return state, 0.0, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            traci.close()
        except:
            pass
        self.current_step = 0
        self.metrics.reset()
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

def run_fixed_time_simulation(episodes_per_iteration=10, iterations=20):
    env = TrafficLightEnv(Config.SUMO_CONFIG_FILE)
    
    all_iteration_stats = []
    
    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}/{iterations}")
        iteration_stats = []
        
        for episode in range(episodes_per_iteration):
            done = False
            state, _ = env.reset()
            while not done:
                state, reward, done, _, _ = env.step()
            
            # Collect episode stats after each episode
            episode_stats = env.metrics.get_episode_stats()
            print(f"  Fixed-Time Episode {episode + 1} Stats: {episode_stats}")
            iteration_stats.append(episode_stats)
        
        # Compute and store average stats for this iteration
        avg_iteration_stats = {
            'avg_waiting_time': np.mean([stat['avg_waiting_time'] for stat in iteration_stats]),
            'avg_queue_length': np.mean([stat['avg_queue_length'] for stat in iteration_stats]),
            'avg_throughput': np.mean([stat['total_throughput'] for stat in iteration_stats]),
            'avg_reward': np.mean([stat['avg_reward'] for stat in iteration_stats])
        }
        
        print(f"Iteration {iteration + 1} Average Stats: {avg_iteration_stats}")
        all_iteration_stats.append(avg_iteration_stats)
    
    # Compute and print overall average performance across all iterations
    overall_avg_stats = {
        'avg_waiting_time': np.mean([stat['avg_waiting_time'] for stat in all_iteration_stats]),
        'avg_queue_length': np.mean([stat['avg_queue_length'] for stat in all_iteration_stats]),
        'avg_throughput': np.mean([stat['avg_throughput'] for stat in all_iteration_stats]),
        'avg_reward': np.mean([stat['avg_reward'] for stat in all_iteration_stats])
    }
    
    print("\nOverall Average Fixed-Time Control Stats Across All Iterations:")
    print(overall_avg_stats)

if __name__ == "__main__":
    run_fixed_time_simulation(episodes_per_iteration=10, iterations=20)
