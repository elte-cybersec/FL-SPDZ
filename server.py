from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import (EvaluateRes, FitIns, FitRes, Parameters, Scalar,  parameters_to_ndarrays, Metrics)
from typing import List, Tuple, Optional, Union, Dict
from flwr.common import (FitIns,  Parameters, Scalar)
from typing import Tuple, Optional, Dict
import subprocess
import os
import numpy as np



def launch_party(total: int, pid: int, prog: str):
    """Launch a single MP‑SPDZ party and return the Popen handle."""
    cmd = [
        "./semi2k-party.x",
        "-p", str(pid),
        "-N", str(total),
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
        
        # Launch the SPDZ party 0
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
            return None

        parameters, metrics = agg

        # --- flatten and show the first 10 numbers ---------------------
        ndarrays = parameters_to_ndarrays(parameters)
        flat     = np.concatenate([w.flatten() for w in ndarrays])

        print(f"[Round {rnd}] First 10 values of aggregated model:",
              flat[:10])

        return parameters, metrics
        
    

def weighted_average(metrics):
    """Aggregate metrics using a weighted average.
    Args:
        metrics (List[Tuple[int, Dict[str, float]]]): A list of tuples containing the number of examples and the metrics for each client.
    Returns:
        Dict[str, float]: A dictionary containing the aggregated metrics.
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}



# === Server Main ===
def main():
    strategy = FedMPCStrategy(
        num_parties=2,  # Number of parties in the MPC setup
        program="fedavg",  # The program to run for each party
        min_available_clients = 3,   # server won't start a round until 3 ready
        min_fit_clients       = 3,
        fraction_fit          = 1.0, # sample 100 % each round
        evaluate_metrics_aggregation_fn = weighted_average,
    )
    config = ServerConfig(num_rounds=5)

    start_server(
        server_address="0.0.0.0:5006",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
