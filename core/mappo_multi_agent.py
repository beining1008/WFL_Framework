import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from collections import deque

class MultiAgentReinforcementLearningFramework:
    def compute_policy_gradient(self, states, actions, advantages):
        raise NotImplementedError

    def update_value_function(self, states, returns):
        raise NotImplementedError

class BaseProximalPolicyOptimization(MultiAgentReinforcementLearningFramework):
    def __init__(self, mappo_config):
        self.beta = mappo_config['beta']
        self.gamma = mappo_config['gamma']
        self.lambda_gae = mappo_config['lambda_gae']
        self.epsilon_clip = mappo_config['epsilon_clip']
        self.value_loss_coef = mappo_config['value_loss_coef']
        self.entropy_coef = mappo_config['entropy_coef']
        self.batch_size = mappo_config['batch_size']
        self.episodes_per_update = mappo_config['episodes_per_update']

class MAPPOActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(MAPPOActorNetwork, self).__init__()
        
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.policy_network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state_tensor, action_mask=None):
        policy_logits = self.policy_network(state_tensor)
        
        if action_mask is not None:
            masked_logits = policy_logits + (1 - action_mask) * (-1e9)
            policy_distribution = F.softmax(masked_logits, dim=-1)
        else:
            policy_distribution = F.softmax(policy_logits, dim=-1)
        
        return policy_distribution, policy_logits

class MAPPOCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims=[128, 128]):
        super(MAPPOCriticNetwork, self).__init__()
        
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in self.value_network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state_tensor):
        value_estimate = self.value_network(state_tensor)
        return value_estimate.squeeze(-1)

class MAPPOAgent(BaseProximalPolicyOptimization):
    def __init__(self, agent_id, state_dim, action_dim, mappo_config):
        super().__init__(mappo_config)
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor_network = MAPPOActorNetwork(state_dim, action_dim)
        self.critic_network = MAPPOCriticNetwork(state_dim)
        
        self.actor_optimizer = torch.optim.Adam(
            self.actor_network.parameters(), 
            lr=self.beta
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_network.parameters(), 
            lr=self.beta
        )
        
        self.experience_buffer = deque(maxlen=10000)
        
    def select_action(self, state_tensor, action_mask=None):
        with torch.no_grad():
            policy_probs, policy_logits = self.actor_network(state_tensor, action_mask)
            
            action_distribution = torch.distributions.Categorical(policy_probs)
            selected_action = action_distribution.sample()
            action_log_prob = action_distribution.log_prob(selected_action)
            
            value_estimate = self.critic_network(state_tensor)
        
        return selected_action.item(), action_log_prob.item(), value_estimate.item()
    
    def store_experience(self, state, action, reward, next_state, done, log_prob, value):
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        }
        self.experience_buffer.append(experience)
    
    def compute_gae_advantages(self, rewards, values, dones):
        advantages = []
        gae_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            td_error = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae_advantage = td_error + self.gamma * self.lambda_gae * (1 - dones[t]) * gae_advantage
            advantages.insert(0, gae_advantage)
        
        return torch.tensor(advantages, dtype=torch.float32)
    
    def compute_policy_gradient(self, states, actions, advantages, old_log_probs):
        current_policy_probs, current_logits = self.actor_network(states)
        current_action_dist = torch.distributions.Categorical(current_policy_probs)
        current_log_probs = current_action_dist.log_prob(actions)
        
        probability_ratio = torch.exp(current_log_probs - old_log_probs)
        
        clipped_ratio = torch.clamp(
            probability_ratio, 
            1 - self.epsilon_clip, 
            1 + self.epsilon_clip
        )
        
        policy_loss_1 = probability_ratio * advantages
        policy_loss_2 = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        entropy_bonus = current_action_dist.entropy().mean()
        total_actor_loss = policy_loss - self.entropy_coef * entropy_bonus
        
        return total_actor_loss, policy_loss, entropy_bonus
    
    def update_value_function(self, states, returns):
        current_values = self.critic_network(states)
        value_loss = F.mse_loss(current_values, returns)
        return value_loss
    
    def update_networks(self):
        if len(self.experience_buffer) < self.batch_size:
            return
        
        batch_experiences = list(self.experience_buffer)[-self.batch_size:]
        
        states = torch.stack([exp['state'] for exp in batch_experiences])
        actions = torch.tensor([exp['action'] for exp in batch_experiences])
        rewards = [exp['reward'] for exp in batch_experiences]
        dones = [exp['done'] for exp in batch_experiences]
        old_log_probs = torch.tensor([exp['log_prob'] for exp in batch_experiences])
        values = [exp['value'] for exp in batch_experiences]
        
        advantages = self.compute_gae_advantages(rewards, values, dones)
        returns = advantages + torch.tensor(values)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        actor_loss, policy_loss, entropy = self.compute_policy_gradient(
            states, actions, advantages, old_log_probs
        )
        
        value_loss = self.update_value_function(states, returns)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 0.5)
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 0.5)
        self.critic_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item()
        }

class MAPPOMultiAgentSystem:
    def __init__(self, num_agents, state_dim, action_dim, mappo_config):
        self.num_agents = num_agents
        self.agents = []
        
        for agent_id in range(num_agents):
            agent = MAPPOAgent(agent_id, state_dim, action_dim, mappo_config)
            self.agents.append(agent)
        
        self.global_episode_count = 0
        self.training_metrics = {
            'episode_rewards': [],
            'actor_losses': [],
            'value_losses': []
        }
    
    def compute_joint_actions(self, joint_states, action_masks=None):
        joint_actions = []
        joint_log_probs = []
        joint_values = []
        
        for agent_id, agent in enumerate(self.agents):
            agent_state = joint_states[agent_id]
            agent_mask = action_masks[agent_id] if action_masks is not None else None
            
            action, log_prob, value = agent.select_action(agent_state, agent_mask)
            
            joint_actions.append(action)
            joint_log_probs.append(log_prob)
            joint_values.append(value)
        
        return joint_actions, joint_log_probs, joint_values
    
    def compute_reward_function(self, joint_states, joint_actions, flmd_values, delta_values):
        alpha_weight = 3.76
        beta_weight = 1.0
        M = 5
        K = 4
        
        total_reward = 0.0
        
        for agent_id in range(self.num_agents):
            agent_delta = delta_values.get(agent_id, 0.0)
            agent_flmd = flmd_values.get(agent_id, 0.0)
            
            agent_reward = -(alpha_weight * agent_delta + beta_weight * (agent_flmd ** 2)) / (M * K)
            total_reward += agent_reward
        
        return total_reward / self.num_agents

    def compute_td_residual(self, agent_id, reward, current_state, next_state, done):
        with torch.no_grad():
            current_value = self.agents[agent_id].critic_network(current_state)
            next_value = self.agents[agent_id].critic_network(next_state) if not done else 0.0

            td_residual = reward + self.gamma * next_value - current_value

        return td_residual.item()

    def compute_gae_advantage(self, agent_id, rewards, values, next_values, dones):
        advantages = []
        gae_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t] if not dones[t] else 0.0
            else:
                next_value = values[t + 1]

            td_residual = rewards[t] + self.gamma * next_value - values[t]
            gae_advantage = td_residual + self.gamma * self.lambda_gae * gae_advantage * (1 - dones[t])
            advantages.insert(0, gae_advantage)

        return torch.tensor(advantages, dtype=torch.float32)

    def compute_policy_ratio(self, agent_id, states, actions, old_log_probs):
        current_policy_probs, _ = self.agents[agent_id].actor_network(states)
        current_action_dist = torch.distributions.Categorical(current_policy_probs)
        current_log_probs = current_action_dist.log_prob(actions)

        policy_ratio = torch.exp(current_log_probs - old_log_probs)

        return policy_ratio

    def update_agent_policy(self, agent_id, states, actions, advantages, old_log_probs):
        policy_ratio = self.compute_policy_ratio(agent_id, states, actions, old_log_probs)

        clipped_ratio = torch.clamp(policy_ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)

        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = clipped_ratio * advantages

        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        self.agents[agent_id].actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent_id].actor_network.parameters(), self.max_grad_norm)
        self.agents[agent_id].actor_optimizer.step()

        return policy_loss.item()

    def update_agent_value(self, agent_id, states, target_values):
        predicted_values = self.agents[agent_id].critic_network(states).squeeze()

        value_loss = F.mse_loss(predicted_values, target_values)

        self.agents[agent_id].critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents[agent_id].critic_network.parameters(), self.max_grad_norm)
        self.agents[agent_id].critic_optimizer.step()

        return value_loss.item()

    def construct_m_order_mdp_state(self, communication_round, M=5, N=20, K=4):
        state_sequence = []

        for m in range(M):
            subcycle_states = {}

            for k in range(K):
                agent_subcycle_state = []

                for n in range(N):
                    device_state = {
                        'flmd_value': self.get_device_flmd(n, communication_round),
                        'channel_gain_squared': self.get_channel_gain_squared(n, m),
                        'age_of_information': self.get_device_aoi(n, communication_round, m)
                    }

                    state_vector = torch.tensor([
                        device_state['flmd_value'],
                        device_state['channel_gain_squared'],
                        device_state['age_of_information']
                    ], dtype=torch.float32)

                    agent_subcycle_state.append(state_vector)

                subcycle_states[k] = torch.stack(agent_subcycle_state)

            state_sequence.append(subcycle_states)

        return state_sequence

    def get_device_flmd(self, device_id, communication_round):
        return np.random.uniform(0.0, 0.2)

    def get_channel_gain_squared(self, device_id, subcycle):
        return np.random.exponential(1.0)

    def get_device_aoi(self, device_id, communication_round, subcycle):
        return communication_round * 5 + subcycle + np.random.uniform(0, 1)

    def apply_flmd_based_masking(self, action_logits, flmd_values, threshold=0.1):
        basic_mask = (flmd_values <= threshold).float()

        adaptive_mask = torch.where(
            flmd_values <= threshold,
            torch.ones_like(flmd_values),
            torch.exp(-1.0 * flmd_values * (1 - 0.01) ** (-1))
        )

        masked_logits = action_logits * adaptive_mask

        return masked_logits, adaptive_mask
    
    def store_joint_experience(self, joint_states, joint_actions, joint_rewards, 
                              joint_next_states, joint_dones, joint_log_probs, joint_values):
        for agent_id, agent in enumerate(self.agents):
            agent.store_experience(
                state=joint_states[agent_id],
                action=joint_actions[agent_id],
                reward=joint_rewards[agent_id],
                next_state=joint_next_states[agent_id],
                done=joint_dones[agent_id],
                log_prob=joint_log_probs[agent_id],
                value=joint_values[agent_id]
            )
    
    def update_all_agents(self):
        training_metrics = {
            'actor_losses': [],
            'value_losses': [],
            'policy_losses': [],
            'entropies': []
        }
        
        for agent in self.agents:
            agent_metrics = agent.update_networks()
            if agent_metrics:
                training_metrics['actor_losses'].append(agent_metrics['actor_loss'])
                training_metrics['value_losses'].append(agent_metrics['value_loss'])
                training_metrics['policy_losses'].append(agent_metrics['policy_loss'])
                training_metrics['entropies'].append(agent_metrics['entropy'])
        
        return training_metrics

def create_mappo_system():
    MAPPO_PARAMS = {
        'beta': 1e-4,
        'gamma': 0.98,
        'lambda_gae': 0.95,
        'epsilon_clip': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'batch_size': 32,
        'episodes_per_update': 10,
        'total_episodes': 500
    }
    
    num_agents = 20
    state_dim = 64
    action_dim = 20
    
    mappo_system = MAPPOMultiAgentSystem(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        mappo_config=MAPPO_PARAMS
    )
    
    return mappo_system
