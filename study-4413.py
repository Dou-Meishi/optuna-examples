import optuna

from fit_sin_GS import objective

name = "4413-RedoGS"
storage = f"sqlite:///../outputs/study-{name}.db"

search_space = {
    "batch_size": [16, 32, 128],
    "num_epochs": [10, 100, 1000],
    "hidden_size": [32],
    "lr": [1e-4],
    "momentum": [0.9],
}

study = optuna.create_study(
    storage=storage,
    sampler=optuna.samplers.GridSampler(search_space=search_space),
    study_name=f"Study-{name}",
    load_if_exists=True,
)

study.optimize(objective)
print(study.best_trial)
print("Finished")
