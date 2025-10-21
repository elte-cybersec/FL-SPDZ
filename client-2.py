import os
import pickle
from collections import OrderedDict
from typing import List
import pickle

import numpy as np
import torch
from flwr.client import Client as FlwrClient, start_client
from flwr.common import (
    Code, Status, FitIns, FitRes, GetParametersIns,
    GetParametersRes, EvaluateIns, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from task import load_data, load_model, train, test

import sys, numpy as np
sys.path.append('/home/aya/.venv/MP-SPDZ')          # parent of ExternalIO
from ExternalIO.client import *
from Compiler.types import regint, sfix

client_id = 2    # Change for each client (0, 1, ...)
num_parties = 2   # Total number of parties in the MPC run
scale = 1 << 16   # Scale factor for fixed-point representation
chunk = 10000

net = load_model()
trainloader, testloader = load_data(client_id, False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Model Utilities ===

def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters: List[np.ndarray]):
    keys = model.state_dict().keys()
    state_dict = OrderedDict({k: torch.tensor(v, device=device) for k, v in zip(keys, parameters)})
    model.load_state_dict(state_dict, strict=True)

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

def unflatten_with_shapes(flat_array: np.ndarray) -> List[np.ndarray]:
    """
    Convert a flat NumPy vector back to a list of arrays whose shapes were
    stored in `layer_shapes.pkl` (written earlier by save_layer_shapes()).
    Args:
        flat (np.ndarray): The flat array to reshape.
    Returns:
        List[np.ndarray]: A list of reshaped arrays corresponding to the original model layers.
    """
    shapes_path= "Player-Data/layer_shapes.pkl"
    with open(shapes_path, "rb") as f:
        shapes = pickle.load(f)

    layers, offset = [], 0
    for shp in shapes:
        size   = int(np.prod(shp))
        layer  = flat_array[offset : offset + size].reshape(shp)
        layers.append(layer.astype(np.float32))
        offset += size
    return layers




# === Flower Client Implementation ===
class FlowerClient(FlwrClient):
    def __init__(self):
        super().__init__()
        self.spdz_update = None  # Placeholder for SPDZ update
        self.shares = []

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        ndarrays = get_parameters(net)
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(ndarrays),
        )

    def fit(self, ins: FitIns) -> FitRes:
        # 1) update local model with global weights
        # global_weights = parameters_to_ndarrays(ins.parameters)
        if self.spdz_update is None:
            global_weights = parameters_to_ndarrays(ins.parameters)
        else:
            print("----USING SPDZ------")
            global_weights = self.spdz_update

        # 2) train the local model
        set_parameters(net, global_weights)
        train(net, trainloader, epochs=1)
        updated_weights = get_parameters(net)

        # 3) Save layer shapes once
        if client_id == 0:
            save_layer_shapes(updated_weights)

        # 4) flatten and split weights into shares
        flat_weights = np.concatenate([w.flatten() for w in updated_weights])
        self.shares.clear()
        self.shares = split_into_shares(flat_weights, num_parties)

        total = len(flat_weights)
        batches = total // chunk
        rest = total - batches * chunk  

        # 5) calculate weighted shares
        num_examples = len(trainloader.dataset)
        weighted_shares = [(share * scale).astype(np.int64) for share in self.shares]

        # 6) Send shares to SPDZ
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

        # 7) Receive output from SPDZ
            received = spdz_client.receive_plain_values()
            print(f"[Client {client_id}] Received share from SPDZ party {pid}")
            party_sum += np.array(received, dtype=np.int64)

        # 8) Average the weighted shares
        avg_weights = np.asarray(party_sum, dtype=np.int64) / scale # Unscale the weighted sum
        # Convert the averaged weights back to the original shapes
        self.spdz_update = unflatten_with_shapes(avg_weights)
        print(f"[Client {client_id}] Received updated model from SPDZ: {avg_weights[0:10]}")
        #check new shape
        print(f"[Client {client_id}] Updated model shape: {[w.shape for w in self.spdz_update]}")


        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            # Client sends the received global model as updated model parameters to server for evaluation
            parameters=ndarrays_to_parameters(self.spdz_update),
            num_examples=len(trainloader),
            metrics={},
        )


    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        global_weights = parameters_to_ndarrays(ins.parameters)
        set_parameters(net, global_weights)
        loss, accuracy = test(net, testloader)
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(testloader),
            metrics={"accuracy": float(accuracy)},
        )


# === Main Entry ===

if __name__ == "__main__":
    start_client(
        server_address="127.0.0.1:5006",
        client=FlowerClient().to_client(),
    )
