import optuna

from fit_sin_GS import objective

name = "0714-RedoRS"
storage = f"sqlite:///../outputs/study-{name}.db"
study = optuna.create_study(
    storage=storage,
    sampler=optuna.samplers.RandomSampler(),
    study_name=f"Study-{name}",
    load_if_exists=True,
)

study.optimize(objective, timeout=600)
print(study.best_trial)
print("Finished")
