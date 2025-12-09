from logging import INFO, WARNING
from typing import List, Tuple
from datetime import datetime

from flwr.common import Context, Metrics, Parameters, log
from flwr.server.strategy import FedAvg
from flwr.server import ServerAppComponents, ServerConfig, ServerApp, start_server, Driver, LegacyContext, ClientManager
from flwr.server.workflow import SecAggPlusWorkflow, DefaultWorkflow

import sys
sys.path.append('/home/sous/MTDFed/FL-SPDZ/.venv/MP-SPDZ')      
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


def weighted_average_with_time_logging(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    now_epoch =  datetime.now().timestamp()
    agg_metrics = {}
    max_training_end_epoch = -1.0

    for num_examples, m in metrics:
        for key, value in m.items():
            # handle training_end_epoch separately to find the max
            if key == "training_end_epoch":
                if value > max_training_end_epoch:
                    max_training_end_epoch = value
                continue

            if key not in agg_metrics:
                agg_metrics[key] = 0.0
            agg_metrics[key] += num_examples * value
    total_examples = sum(num_examples for num_examples, _ in metrics)
    for key in agg_metrics:
        agg_metrics[key] /= total_examples

    now_epoch - max_training_end_epoch
    agg_metrics["aggregation_duration"] = now_epoch - max_training_end_epoch
    
    print("#####Duration metrics#####")
    duration_metrics = [(key, agg_metrics[key]) for key in agg_metrics if "duration" in key]
    for key, value in duration_metrics:
        print(f"{key}: {value} seconds")

    return agg_metrics


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
PROGRAM_MAPPING = {
    "ppo": "fedavg_ppo",
    "a2c": "fedavg_a2c",
    "envelope": "fedavg_envelope",
    "eupg": "fedavg_eupg"
}

def main():
    if len(sys.argv) < 3:
        print("Usage: python server_rl_mtd.py <use_spdz> <algorithm>")
        sys.exit(1)
    use_spdz = bool(int(sys.argv[1]))
    algorithm = sys.argv[2]

    if use_spdz is True:
        program = PROGRAM_MAPPING.get(algorithm.lower(), "fedavg_eupg")
        
        strategy = FedMPCStrategy(
            num_parties=2,
            program=program,
            fraction_fit=0.6,
            fraction_evaluate=0.5,
            min_fit_clients=3,
            min_evaluate_clients=2,
            min_available_clients=3,
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=weighted_average_with_time_logging,
        )
    else:
        strategy = FedAvg(
            fraction_fit=0.6,
            fraction_evaluate=0.5,
            min_fit_clients=3,
            min_evaluate_clients=2,
            min_available_clients=3,
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=weighted_average_with_time_logging,
        )
    config = ServerConfig(num_rounds=25, round_timeout=None)

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
