from collections import OrderedDict
from pathlib import Path
from logging import INFO
from typing import List
import pickle
from datetime import datetime

import sys, numpy as np
import torch
from flwr.client import NumPyClient, Client as FlwrClient, start_client
# from flwr.client.mod import secaggplus_mod
from flwr.common import (
    Code, Status, FitIns, FitRes, GetParametersIns,
    GetParametersRes, EvaluateIns, EvaluateRes, log,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from optsfc.envs.mo_fiveg_mdp import initialize_model_for_flwr, SaveOnBestTrainingRewardCallback, MOfiveG_net
from optsfc.envs.morl_train import eval_agent, train_eupg, train_Envelope, eupg_model_save, rewards_coeff
from optsfc.envs.short_simulated_testbed import is_action_possible

sys.path.append('/home/sous/MTDFed/FL-SPDZ/.venv/MP-SPDZ')          # parent of ExternalIO
from ExternalIO.client import *

num_parties = 2   # Total number of parties in the MPC run
scale = 1 << 16   # Scale factor for fixed-point representation
chunk = 10000
round_number = 5


def save_layer_shapes(shares: List[np.ndarray], path: str = "Player-Data/layer_shapes.pkl"):
    shapes = [s.shape for s in shares]
    print(f"[Client {client_id}] Saving layer shapes: {shapes} to {path}")
    with open(path, "wb") as f:
        pickle.dump(shapes, f)

def split_into_shares(array: np.ndarray, num_shares: int) -> List[np.ndarray]:
    shares = [np.random.uniform(-1, 1, size=array.shape).astype(array.dtype) for _ in range(num_shares - 1)]
    final_share = array - sum(shares)
    shares.append(final_share)
    return shares


def unflatten_with_shapes(flat_array: np.ndarray, path: str = "Player-Data/layer_shapes.pkl") -> List[np.ndarray]:
    """
    Convert a flat NumPy vector back to a list of arrays whose shapes were
    stored in `layer_shapes.pkl` (written earlier by save_layer_shapes()).
    Args:
        flat (np.ndarray): The flat array to reshape.
    Returns:
        List[np.ndarray]: A list of reshaped arrays corresponding to the original model layers.
    """
    with open(path, "rb") as f:
        shapes = pickle.load(f)

    layers, offset = [], 0
    for shp in shapes:
        size   = int(np.prod(shp))
        layer  = flat_array[offset : offset + size].reshape(shp)
        layers.append(layer.astype(np.float32))
        offset += size
    return layers


class FlowerACClient(FlwrClient):
    def __init__(self, partition_id, rl_algo, model, train_env, eval_env, use_spdz=True):
        """
        Args:
            model: Stable-Baselines3 PPO model.
            train_env: Environment used for training.
            eval_env: Environment used for evaluation.
        """
        super().__init__()
        self.partition_id = partition_id
        self.rl_algo = rl_algo
        self.model = model
        self.train_env = train_env
        self.eval_env = eval_env
        self.spdz_update = None  # Placeholder for SPDZ update
        self.shares = []
        self.use_spdz = use_spdz

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        if self.rl_algo in ["PPO", "A2C", "MaskablePPO"]:
            parameters_dict = self.model.get_parameters()["policy"]
            parameters_dict = self.model.get_parameters()["policy"] # SB3 get method
            # print("Retrieved parameters (dictionary):", parameters_dict)
            # Convert the dictionary of tensors to a list of NumPy arrays
            parameters_list = [param.detach().cpu().numpy() for param in parameters_dict.values()]
            # print("Converted parameters (list):", parameters_list)
        elif self.rl_algo == "EUPG":
            parameters_dict = self.model.net.state_dict()
            parameters_list = [val.cpu().numpy() for _, val in parameters_dict.items()]
        elif self.rl_algo == "Envelope":
            parameters_dict = self.model.q_net.state_dict()
            parameters_list = [val.cpu().numpy() for _, val in parameters_dict.items()]
        
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(parameters_list),
        )

    def set_parameters(self, parameters):
        if self.rl_algo in ["PPO", "A2C", "MaskablePPO"]:
            # Convert the list of NumPy arrays back to a dictionary of tensors
            parameter_keys = list(self.model.get_parameters()["policy"].keys())  # Get keys from the model

            parameters_dict = {
                key: torch.tensor(param, dtype=torch.float32)
                for key, param in zip(parameter_keys, parameters)
                if isinstance(param, (np.ndarray, float, int))
            }
            # Set the parameters in the model
            full_parameters = self.model.get_parameters()
            full_parameters["policy"] = parameters_dict
            self.model.set_parameters(full_parameters)
            log(INFO, "New parameters set successfully.")
            # reinitialize the optimizer
            self._reinitialize_optimizer()
        elif self.rl_algo == "EUPG":
            parameters = zip(self.model.net.state_dict().keys(), parameters)
            self.model.net.load_state_dict(OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in parameters}))
            log(INFO, "New parameters set successfully.")
            # reinitialize the optimizer
            self._reinitialize_optimizer()
        elif self.rl_algo == "Envelope":
            parameters = zip(self.model.q_net.state_dict().keys(), parameters)
            # ate dictionary from the list
            self.model.q_net.load_state_dict(OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in parameters}))
            self.model.target_q_net.load_state_dict(self.model.q_net.state_dict())
            # reinitialize the optimizer
            self._reinitialize_optimizer()


    def _reinitialize_optimizer(self):
        """
        Reinitialize the optimizer after setting new parameters.
        """
        # Recreate the optimizer instance
        if self.rl_algo in ["PPO", "A2C", "MaskablePPO"]:
            policy = self.model.policy
            policy.optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=policy.optimizer.defaults["lr"],  # Keep the original learning rate
                eps=policy.optimizer.defaults["eps"],  # Keep other defaults
            )
        elif self.rl_algo == "EUPG":
            self.model.optimizer = torch.optim.Adam(self.model.net.parameters(), lr=self.model.learning_rate)
        elif self.rl_algo == "Envelope":
            self.model.q_optim = torch.optim.Adam(self.model.q_net.parameters(), lr=self.model.learning_rate)

        log(INFO, "Optimizer reinitialized successfully.")


    # def load_model(self):
    #     model_dir = "./tested_models/" + self.rl_algo + "_model_" + str(self.partition_id) + "/"
    #     if self.rl_algo == "PPO":
    #         self.model = PPO.load(model_dir)
    #     elif self.rl_algo == "A2C":
    #         self.model = A2C.load(model_dir)
    #     elif self.rl_algo == "MaskablePPO":
    #         self.model = MaskablePPO.load(model_dir)
    #     log(INFO, "Model loaded successfully.")


    def fit(self, ins: FitIns) -> FitRes:
        log(INFO, "Training model")
        # 1) update local model with global weights
        # global_weights = parameters_to_ndarrays(ins.parameters)
        if self.use_spdz is False or self.spdz_update is None:
            global_weights = parameters_to_ndarrays(ins.parameters)
        else:
            print("----USING SPDZ------")
            global_weights = self.spdz_update

        # Set parameters received from the server
        self.set_parameters(global_weights)

        # Train the PPO agent locally
        #       Create the callback: check every 10000 steps
        training_size = 2000  # Number of timesteps used in training (can be modified)
        client_id = self.partition_id
        print("training in ROUND NUMBER ", round_number, "at ",self.rl_algo," with round length equal to", training_size, "timesteps")
        training_start_time = datetime.now()
        if training_size > 0:
            log_dir = "./monitor_logs/client" + str(client_id) + "/" + str(
                round_number) + "/callback_logs/"  # Directory to save logs (can be modified)
            # create middle directories of the log_dir
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            model_name = "local_FL_" + self.rl_algo
            if self.rl_algo in ["PPO", "A2C", "MaskablePPO"]:
                policy = "MlpPolicy"
                callback = SaveOnBestTrainingRewardCallback(check_freq=training_size, log_dir=log_dir, model_name=model_name,
                                                            policy=policy, env=self.train_env)  # (HYPER)

                self.model.set_env(self.train_env)
                self.model.learn(total_timesteps=training_size, callback=callback, reset_num_timesteps = False)
            elif self.rl_algo == "EUPG":
                train_eval_env = MOfiveG_net("MlpPolicy", "daily")
                self.model.env = self.train_env
                self.model.train(total_timesteps=training_size, eval_env=train_eval_env)
                eupg_model_save(self.model, log_dir, model_name)
            elif self.rl_algo == "Envelope":
                self.model.env = self.train_env
                print("Starting Envelope training...")
                self.model.train(total_timesteps=training_size, eval_freq=1000)
                print("Ended Envelope training and starting to save model...")
                self.model.save(save_dir=log_dir, filename=model_name, save_replay_buffer=True)
                print("Ended saving the model.")
        training_end_time = datetime.now()
        training_duration_seconds = (training_end_time - training_start_time).total_seconds()

        print("CLIENT ", client_id, " finished training for ", training_size, " timesteps!!!")
        updated_parameters = parameters_to_ndarrays(self.get_parameters(ins).parameters)

        # 2) save layer shape
        layer_shapes_path = "Player-Data/layer_shapes_" + str(self.rl_algo).lower() + ".pkl"
        if client_id == 0:
            save_layer_shapes(updated_parameters, layer_shapes_path)

        if self.use_spdz is False:
            # If not using SPDZ, return the updated parameters directly
            log(INFO, "One training round done successfully without SPDZ.")

            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                # Client sends the updated model parameters to server for evaluation
                parameters=ndarrays_to_parameters(updated_parameters),
                num_examples=training_size,
                metrics={"training_duration": training_duration_seconds, "training_end_epoch": training_end_time.timestamp()},
            )

        spdz_start_time = datetime.now()
        # 3) flatten and split weights into shares
        flat_weights = np.concatenate([w.flatten() for w in updated_parameters])
        self.shares.clear()
        self.shares = split_into_shares(flat_weights, num_parties)

        total = len(flat_weights)
        batches = total // chunk
        rest = total - batches * chunk 

        # 4) calculate weighted shares
        num_examples = training_size
        weighted_shares = [(share * scale).astype(np.int64) for share in self.shares]

        # 5) Send shares to SPDZ
        party_sum = np.zeros_like(weighted_shares[0], dtype=np.int64)  # Initialize weighted sum
        for pid in range(num_parties):
            spdz_client = Client(['localhost'], 5100 + pid, client_id)
            print(f"[Client {client_id}] Sending shares to SPDZ party {pid}")
            spdz_client.send_public_inputs([num_examples * scale])  # Send scaled count
            for i in range(batches):
                spdz_client.send_public_inputs(weighted_shares[pid][i * chunk:(i + 1) * chunk])
                print(f"[Client {client_id}] Sent chunk {i + 1}/{batches} to SPDZ party {pid}")
            if rest > 0:
                spdz_client.send_public_inputs(weighted_shares[pid][batches * chunk:])
                print(f"[Client {client_id}] Sent remaining chunk to SPDZ party {pid}")

            # 6) Receive output from SPDZ
            received = spdz_client.receive_plain_values()
            print(f"[Client {client_id}] Received share from SPDZ party {pid}")
            party_sum += np.array(received, dtype=np.int64)

        # 7) Average the weighted shares
        avg_weights = np.asarray(party_sum, dtype=np.int64) / scale # Unscale the weighted sum
        # Convert the averaged weights back to the original shapes
        self.spdz_update = unflatten_with_shapes(avg_weights, path=layer_shapes_path)

        log(INFO, "One training round done successfully.")
        spdz_end_time = datetime.now()
        spdz_duration_seconds = (spdz_end_time - spdz_start_time).total_seconds()

        # # save model
        # model_dir = "./tested_models/" + self.rl_algo + "_model_" + str(self.partition_id) + "/"
        # Path(model_dir).mkdir(parents=True, exist_ok=True)
        # self.model.save(model_dir)

        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            # Client sends the received global model as updated model parameters to server for evaluation
            parameters=ndarrays_to_parameters(self.spdz_update),
            num_examples=training_size,
            metrics={"training_duration": training_duration_seconds, "training_end_epoch": training_end_time.timestamp()},
        )


    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        log(INFO, "Evaluating model")
        # policy_weights_list = [(np.frombuffer(param, dtype=np.float32), len(param)) for param in parameters]
        # print("the parameters in the evaluation is ", policy_weights_list)

        # Set parameters received from the server
        self.set_parameters(parameters_to_ndarrays(ins.parameters))

        # Evaluate the model
        episode_rewards = []
        n_steps = 2160
        step = 0
        if n_steps == 0:
            return float(0), 1, {"avg_reward": float(0)}
        else:
            for _ in range(1):
                obs, info = self.eval_env.reset()
                done = False
                total_reward = 0
                vec_return = np.zeros(3)
                remaining_steps = n_steps - step
                for step in range(remaining_steps):
                    same = 0
                    counter = 0
                    previous_action = 0
                    if self.rl_algo in ["PPO", "A2C", "MaskablePPO"]:
                        action = self.model.predict(observation=obs, deterministic=True)
                        print("the action predicted is ", action)
                        action = action[0]
                    else:
                        while counter < 100:
                            if self.rl_algo == "EUPG":
                                action = self.model.eval(obs, vec_return)
                            elif self.rl_algo == "Envelope":
                                action = self.model.eval(obs, rewards_coeff)

                            if previous_action != None and previous_action == action:
                                same += 1
                            else:
                                same = 0
                            if same > 10 and not is_action_possible(self.eval_env.environment, action)[0]:
                                print("same invalid action selected 10 consecutive times")
                                action = 0
                                break
                            if is_action_possible(self.eval_env.environment, action)[0]:
                                break
                            counter += 1
                        if counter == 100:
                            action = 0
                    # print("the action predicted is ", action)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    if self.rl_algo in ["PPO", "A2C", "MaskablePPO"]:
                        total_reward += reward
                    else:
                        total_reward += info["rew"]
                        vec_return[0] = reward[0]
                        vec_return[1] = reward[1]
                        vec_return[2] = reward[2]

                    if done:
                        break
                episode_rewards.append(total_reward/step)

            # Calculate mean reward
            mean_reward = sum(episode_rewards) / len(episode_rewards)
            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss= float(mean_reward) * -1, # float(mean_reward)
                num_examples=n_steps,
                metrics={"avg_reward": float(mean_reward)},
            )

ALGORITHM_MAPPING = {
    "eupg": "EUPG",
    "envelope": "Envelope",
    "ppo": "PPO",
    "a2c": "A2C",
    "maskableppo": "MaskablePPO"
}

def construct_flower_client(partition_id, use_spdz=True, algorithm="EUPG") -> FlwrClient:
    rl_algo = ALGORITHM_MAPPING.get(algorithm.lower(), "EUPG")

    print("construction in ROUND NUMBER ", round_number)
    # Create environments
    log_dir = "./monitor_logs/client" + str(partition_id) + "/" + str(round_number) + "/callback_logs/"  # Directory to save logs (can be modified)
    budget_reset = "daily"
    model, train_env, eval_env = initialize_model_for_flwr(rl_algo, log_dir, budget_reset)
    log(INFO, f"MDP and {rl_algo} models initialized for client {partition_id}")

    # Create and start Flower client
    flower_client = FlowerACClient(partition_id, rl_algo, model, train_env, eval_env ,use_spdz=use_spdz)
    # start_numpy_client(server_address="localhost:8080", client=flower_client)
    # flower_client.set_context(context)
    return flower_client.to_client()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python client_spdz_mtd.py <client_id> <use_spdz> <algorithm>")
        sys.exit(1)
    client_id = int(sys.argv[1])
    use_spdz = bool(int(sys.argv[2]))
    algorithm = sys.argv[3]

    """Create a Flower client representing a single organization."""
    log(INFO, f"Starting client {client_id}")
 
    # Construct the client
    flower_client = construct_flower_client(
        partition_id=client_id, use_spdz=use_spdz, algorithm=algorithm
    )

    start_client(
        server_address="127.0.0.1:5006",
        client=flower_client,
    )

# # Create an instance of the mod with the required params
# local_dp_obj = LocalDpMod(
#     0.8, 0.2, 0.001, 0.001
# )


