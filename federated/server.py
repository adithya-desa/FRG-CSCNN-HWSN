import flwr as fl

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=1,
    min_available_clients=1,
)

fl.server.start_server(
    server_address="localhost:8081",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)

