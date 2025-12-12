# MTDFed: Privacy-Preserving MTD Framework using Federated Learning (FL)

In this folder, you can find the implementation of MTDFed system, integrated into the FL-SPDZ framework which is based on Flower and MP-SPDZ.

The set of instructions outlined here assumes that you have already followed through the README file in the parent directory, and you have successfully set up a working instance of the FL-SPDZ framework. If you have not done so already, please refer to the README file in the parent directory, where it is explained how to run FL-SPDZ.


## 📂 File Descriptions

- `server_rl_mtd.py:` Implements the Flower server for the MTD scheme, which also functions as SPDZ Party 0, performing secure model aggregation using MPC.
- `party-1_mtd.py:` Implements an additional SPDZ party that collaborate with the server to execute the MTD task with MP-SPDZ protocol. This file is only required if you want to run MTDFed in the MP-SPDZ mode. If you want to use more parties, you can create corresponding party files, `party-i_mtd.py`, where _i > 1_ for additional files.
- `client_spdz_mtd.py:` Defines the Flower client logic to handle local MTD training and secure sharing of model parameters via different privacy preservation modes.
- `spdz_strategy.py`: The customized FL aggregation strategy that supports an integration with the MP-SPDZ protocol
- `pyproject.toml`: The configuration file to run MTDFed simulations using the `flwr run` API of Flower.
- `mpc_files/*.mpc:` SPDZ programs for each Reinforcement Learning (RL) algorithm supported in MTDFed, namely: A2C, EUPG, Envelope, PPO.
- `requirements.txt:` Lists all Python dependencies required to run MTDFed (except the OptSFC library)
- `layer_shapes/*.pkl:` Metadata about each supporterd RL model’s layer dimensions and parameter shapes, used by the corresponding `mpc` file to calculate the model size.


## ⚙️ Environment Setup
We recommend using the _same virtual environment_ you set up for **FL-SPDZ**. Therefore, you can activate that environment which should reside in the parent directory.

```bash
# enter virtual environment
source ../.venv/bin/activate

# install Python dependencies related to the MTD scheme
pip3 install -r requirements.txt
```

However, there is an additional library that need installation from source, which is the _OptSFC_ codebase that constitutes the core MTD logic. To install OptSFC, please continue with the following steps:

```bash
cd OptSFC

pip3 install -e .

cd .. # go back to the MTD folder containing this README file
```

With this, the environment should be ready to run MTDFed experiments.

## 🧪 Running MTDFed Experiments

### 1. Prepare file locations

Copy the following files into the corresponding MP-SPDZ directories:
| File                                                                      | Destination                      |
| ------------------------------------------------------------------------- | -------------------------------- |
| `client_spdz_mtd.py`                                                      | `.venv/MP-SPDZ/ExternalIO/`      |
| `server_rl_mtd.py`, `party-*.py`, `./mpc_files/*.mpc`, `spdz_strategy.py` | `.venv/MP-SPDZ/Programs/Source/` |
| `./layer_shapes/*.pkl`                                                    | `.venv/MP-SPDZ/Player-Data/`     |

Even when MP-SPDZ protocol is not used, we need to execute all of the simulation within the MP-SPDZ folder, so that we would not have to use separate configuration for MP-SPDZ and different modes such as Secure Aggregation (specifically, SecAgg+) or Differential Privacy (DP).


### 2. Compile all relevant `mpc` file

You need compile the `mpc` files for which you want to execute the corresponding RL model with MP-SPDZ.

```bash
cd ../.venv/MP-SPDZ

./compile.py Programs/Source/fedavg_a2c.mpc
./compile.py Programs/Source/fedavg_envelope.mpc
./compile.py Programs/Source/fedavg_eupg.mpc
./compile.py Programs/Source/fedavg_ppo.mpc
```

### 3. Generate SSL keys

It is assumed that you have already completed this step during the FL-SPDZ setup in the parent directory. Otherwise, please refer to the README file in the root folder of FL-SPDZ repository.


### 4. Compile relevant MPC protocols

It is assumed that you have already completed this step during the FL-SPDZ setup in the parent directory. Otherwise, please refer to the README file in the root folder of FL-SPDZ repository.

### 5. Run MPC parties

Unlike in the base FL-SPDZ configuration, you do not have to execute each client or the server in a separate terminal session. Only the party files (i.e., party-*_mtd.py) need to be executed in separate terminals. The rest of the FL process will require only one blocking shell session.

Please ensure that you are inside the root `MP-SPDZ` folder in each of these terminal sessions.

If you want to use the MP-SPDZ mode during the experiments, you need to first start the MPC party file(s) using following command:

```bash
python3 ./Programs/Source/party-1_mtd.py <rl_algo>     # SPDZ Party 1
```

Here, the `<rl_algo>` refers to one of the following options:
- a2c
- ppo
- envelope
- eupg

Repeat this process for each additional MPC party you want to use. Note that all parties **must** use the same RL algorithm simultaneously, so it is not possible to use differemt arguments for different MPC parties at the same time. In addition, make sure that you have previously compiled the relevant `mpc` file of the RL option you want to proceed with.

### 6. Execute MTDFed experiments

After the set of MPC parties have started, you can run an MTDFed experiment based on the newer Flower API, using a single command:

```bash
flwr run . local-simulation --stream --run-config 'use-spdz=<use_spdz> rl-algo="<rl_algo>" num-server-rounds=<num_server_rounds> use-dp=<use_dp> use-secagg=<use_secagg>'
```

The following table provides information on the parameters and their possible values that can be used as arguments during an experiment:

| Argument              | Possible Values (separated by comma) | Description                        |
| --------------------- | ------------------------------------ | ---------------------------------- |
| `<rl_algo>`           | `a2c` , `ppo` , `eupg` , `envelope`  | Which RL algorithm to use          |
| `<use_spdz>`          | `true` , `false`                     | Whether to use MP-SPDZ or not      |
| `<use_dp>`            | `true` , `false`                     | Whether to use DP or not           |
| `<use_secagg>`        | `true` , `false`                     | Whethher to use SecAgg+ or not     |
| `<num_server_rounds>` | _any positive integer_               | How many FL rounds to use          |


With this configuration, it is possible to run multiple privacy preservation method simultaneously. Internal parameters with regards to SecAgg+ or DP modes can be found and adjusted in the **pyproject.toml** file in this folder. 

## ⚠️ Note:
If you change the number of clients or parties, make sure to update the following files accordingly:

- `./mpc_files/*.mpc` — Update client counts, then recompile.
- `./party-*_mtd.py` — Update party ID and total number of parties.
- `server_rl_mtd.py` — Update the total number of parties and minimum clients.
