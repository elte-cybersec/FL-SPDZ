from logging import INFO, WARNING
from typing import List, Tuple

from flwr.common import Context, Metrics, Parameters, log
from flwr.server.strategy import FedAvg
from flwr.server import ServerAppComponents, ServerConfig, ServerApp, start_server, Driver, LegacyContext, ClientManager
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

import sys
sys.path.append('/home/ubuntu/FL-SPDZ/.venv/MP-SPDZ')      
from Programs.Source.spdz_strategy import FedMPCStrategy

# Configure the server for training
config = ServerConfig(num_rounds=5, round_timeout=None)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print("the metrics are: ", metrics)
    # Multiply accuracy of each client by number of examples used
    total_reward = [num_examples * m["avg_reward"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"avg_reward": sum(total_reward) / sum(examples)}


# def server_fn(context: Context) -> ServerAppComponents:
#     """Construct components that set the ServerApp behaviour.

#     You can use the settings in `context.run_config` to parameterize the
#     construction of all elements (e.g the strategy or the number of rounds)
#     wrapped in the returned ServerAppComponents object.
#     """
#     # Use SPDZ strategy if specified in context, otherwise use FedAvg
#     use_spdz = context.run_config.get("use_spdz", True)
#     rl_algo = context.run_config.get("rl_algo", "PPO")

#     program = program_map.get(rl_algo, "fedavg_ppo")

#     log(INFO, f"Using {'SPDZ+MTD' if use_spdz else 'FedAvg'} strategy (program={program})")

#     return ServerAppComponents(
#         strategy=strategy,
#         config=config
#     )


# === Server Main ===
def main():
    if len(sys.argv) < 2:
        print("Usage: python server_rl_mtd.py <use_spdz>")
        sys.exit(1)
    use_spdz = bool(int(sys.argv[1]))

    if use_spdz is True:
        program_list = [
            "fedavg_ppo",
            "fedavg_a2c",
            "fedavg_envelope",
            "fedavg_eupg"
        ]
        program = "fedavg_envelope"
        
        strategy = FedMPCStrategy(
            num_parties=2,
            program= program,
            fraction_fit=0.6,
            fraction_evaluate=0.5,
            min_fit_clients=3,
            min_evaluate_clients=2,
            min_available_clients=3,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:
        strategy = FedAvg(
            fraction_fit=0.6,
            fraction_evaluate=0.5,
            min_fit_clients=3,
            min_evaluate_clients=2,
            min_available_clients=3,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    config = ServerConfig(num_rounds=10, round_timeout=None)

    start_server(
        server_address="0.0.0.0:5006",
        config=config,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()



## Comment-out the following code block ###
## to disable SecAgg+ Secure Aggregation ###

# @app.main()
# def main(driver: Driver, context: Context) -> None:
#     # Construct the LegacyContext
#     num_rounds = context.run_config["num-server-rounds"]
#     context = LegacyContext(
#         context=context,
#         config=ServerConfig(num_rounds=num_rounds),
#         strategy=strategy1,
#     )
#
#     fit_workflow = SecAggPlusWorkflow(
#         num_shares=context.run_config["num-shares"],
#         reconstruction_threshold=context.run_config["reconstruction-threshold"],
#         max_weight=context.run_config["max-weight"],
#     )
#
#     # Create the workflow
#     workflow = DefaultWorkflow(fit_workflow=fit_workflow)
#
#     # Execute
#     workflow(driver, context)
