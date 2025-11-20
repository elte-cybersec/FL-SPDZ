from typing import Dict, List, Optional, Tuple, Union
import os
import subprocess
from logging import INFO
from flwr.server.strategy import FedAvg
import numpy as np


from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import (
    Code, Status, FitIns, FitRes, GetParametersIns, Parameters,
    GetParametersRes, EvaluateIns, EvaluateRes, Scalar, Metrics,
    ndarrays_to_parameters, parameters_to_ndarrays, log,
)

def launch_party(total: int, pid: int, prog: str):
    """Launch a single MP-SPDZ party and return the Popen handle."""
    cmd = [
        "./semi2k-party.x",
        "-p", str(pid),
        "-N", str(total),
        # "-pn", str(5100),
        prog,          # ExternalIO on stdin/stdout; remove if not needed
    ]
    print("[launcher] starting:", " ".join(cmd))
    # Use a new process group so we can kill children on Ctrl‑C
    return subprocess.Popen(cmd, preexec_fn=os.setsid)
       



# === Custom Strategy ===
class FedMPCStrategy(FedAvg):
    def __init__(self, num_parties, program, **fedavg_kwargs):
        super().__init__(**fedavg_kwargs)
        # --- SPDZ specific parameters ----------------------------------------
        self.NUM_PARTIES = num_parties
        self.PROGRAM = program

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> Tuple[FitIns, Dict[str, Scalar]]:
        """Configure the fit process for the server.
        Args:
            server_round (int): The current round number.
            parameters (Parameters): The global model parameters.
            client_manager (ClientManager): The client manager.
        Returns:
            Tuple[FitIns, Dict[str, Scalar]]: The fit input and additional metadata."""
        
        # # Launch the SPDZ party 0
        launch_party(self.NUM_PARTIES, 0, self.PROGRAM)

        # Call the usual FedAvg configuration
        return super().configure_fit(server_round, parameters, client_manager)
    

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[str, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, Dict[str, Metrics]]]:
        # --- run the usual FedAvg aggregation --------------------------
        agg = super().aggregate_fit(rnd, results, failures)
        if agg is None:
            # No aggregated parameters due to client failures
            return None

        parameters, metrics = agg
        if parameters is None:
            # Propagate metrics without preview if parameters are missing
            return agg

        # --- flatten and show the first 10 numbers ---------------------
        try:
            ndarrays = parameters_to_ndarrays(parameters)
            flat     = np.concatenate([w.flatten() for w in ndarrays])
            print(f"[Round {rnd}] First 10 values of aggregated model:", flat[:10])
        except Exception as e:
            print(f"[Round {rnd}] Unable to preview aggregated model: {e}")

        return parameters, metrics


class MTDFedSPDZStrategy(FedMPCStrategy):
    """Strategy for federated learning with SPDZ secure aggregation and MTD capabilities."""
    
    def __init__(
        self, mtd_strategies, **fedavg_kwargs):
        super().__init__(**fedavg_kwargs)
        # --- MTDFed specific paramenters ------------------------------------
        self.mtd_strategies = mtd_strategies or ["random_sampling"]
        
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, Dict]]:
        """Configure the next round of training with MTD strategies."""
        # Get base configuration from parent class
        client_configs = super().configure_fit(server_round, parameters, client_manager)
        
        # Apply MTD strategies
        for _, config in client_configs:
            # Select a random MTD strategy for this round
            import random
            mtd_strategy = random.choice(self.mtd_strategies)
            config["mtd_strategy"] = mtd_strategy
            log(INFO, f"Applied MTD strategy: {mtd_strategy} for round {server_round}")
            
        return client_configs