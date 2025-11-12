# MPC-based Federated Learning with SPDZ

This repository provides an implementation of decentralized Federated Learning using secure Multi-Party Computation (MPC) through the MP-SPDZ framework, integrated into a Flower Federated Learning system.

The system enables privacy-preserving model aggregation by using SPDZ parties to securely compute the federated average (FedAvg) of model parameters without exposing any client's private data.


## 📂 File Descriptions

- `server.py:` Implements the Flower server, which also functions as SPDZ Party 0, performing secure model aggregation using MPC.
- `party-i.py:` Implements additional SPDZ parties (i > 0) that collaborate with the server to execute the secure aggregation protocol.
- `client-i.py:` Defines Flower clients, each performing local training and securely sharing model parameters via additive secret sharing to SPDZ parties.
- `task.py:` Contains dataset loading and model definition used by clients during federated training.
- `fedavg.mpc:` SPDZ program that performs secure federated averaging (FedAvg) over the secret-shared model parameters.
- `requirements.txt:` Lists all Python dependencies required to run the project.
- `layer_shapes.pkl:` Stores metadata about the model’s layer dimensions and parameter shapes, used by the `fedavg.mpc` file as well to calculate the model size.


## ⚙️ Environment Setup
We recommend using a Python virtual environment.

```bash
# create a virtual environment
python3 -m venv .venv

# enter virtual environment
source .venv/bin/activate

# install Python dependencies
pip3 install -r requirements.txt
```

## 🔐 MP-SPDZ
This project integrates [MP-SPDZ](https://github.com/data61/MP-SPDZ), an open-source framework for secure multiparty computation.

*(Inside the Python environment)*  
Clone the following repository:

```bash
# Clone repo
git clone git@github.com:data61/MP-SPDZ.git

# Build executables
make -j$(nproc)
```

## 🧪 Running Experiments
### 1. Prepare MP-SPDZ Folders

Copy the following files into the corresponding MP-SPDZ directories:
| File                                    | Destination                      |
| ----------------------------------------| -------------------------------- |
| `client-i.py`, `task.py`                | `.venv/MP-SPDZ/ExternalIO/`      |
| `server.py`, `party-i.py`, `fedavg.mpc` | `.venv/MP-SPDZ/Programs/Source/` |
| `layer_shapes.pkl`                      | `.venv/MP-SPDZ/Player-Data/`     |



### 2. Compile the `fedavg.mpc` file
```bash
cd ./MP-SPDZ/Programs/Source
./compile.py fedavg.mpc
```
     
### 3. Generate SSL keys:
Make sure that you are inside the root `MP-SPDZ` folder before proceeding.

-  *Keys for clients*:
```bash
Scripts/setup-clients.sh 3
```

-   *Keys for parties*:

```bash
Scripts/setup-ssl.sh 2
```

### 4. Compile relevant MPC protocols
Use make to compile specific MPC protocol(s) that you intend to use in your experiments.

For instance, use the following command to compile **semi2k-party**:
```bash
make -j$(nproc) semi2k-party.x
```

In case you receive any error with regards to _boost::filesystem_, you can try using explicit linking:
```bash
make -j$(nproc) semi2k-party.x -lboost_system -lboost_filesystem
```

Now you should be able to see the _semi2k-party.x_ file inside the `MP-SPDZ` folder.

For other MPC protocols you need to use (e.g., **mascot-party**), repeat this process.

### 5. Run the server and MPC parties.
Here, each MPC party and client should be initialized in a separate terminal, since each one will start a blocking process. Please ensure that you are inside the root `MP-SPDZ` folder in each of these terminal sessions.

It is recommended to first start MPC parties using following commands:
```bash
python3 ./Programs/Source/server.py      # SPDZ Party 0 (also acts as the FL server)
python3 ./Programs/Source/party-1.py     # SPDZ Party 1
```

After making sure that the MPC parties have spinned up, you can start the clients:
```bash
python3 ./ExternalIO/client-1.py
python3 ./ExternalIO/client-2.py
python3 ./ExternalIO/client-3.py
```

In case you encounter `No module named "ExternalIO"` during the execution of client scripts, you might resolve it by explicitly setting _PYTHONPATH_ ENV variable to the current directory, which correspond to the root `MP-SPDZ` folder.
```bash
PYTHONPATH=./ python3 ./ExternalIO/client-1.py
PYTHONPATH=./ python3 ./ExternalIO/client-2.py
PYTHONPATH=./ python3 ./ExternalIO/client-3.py
```

## ⚠️ Note:
When changing the number of clients or parties, make sure to update the following files accordingly:

- `fedavg.mpc` — Update client counts, then recompile.
- `party-i.py` — Update party ID and total number of parties.
- `server.py` — Update the total number of parties and minimum clients.
- `client-i.py` — Update client ID.