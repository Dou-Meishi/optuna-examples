import optuna

from fit_sin_GS import objective

name = "0411-RedoRS2"
storage = f"sqlite:///../outputs/study-{name}.db"
study = optuna.create_study(
    storage=storage,
    sampler=optuna.samplers.RandomSampler(),
    study_name=f"Study-{name}",
    load_if_exists=True,
)

study.optimize(objective, timeout=300)
print(study.best_trial)
print("Finished")
