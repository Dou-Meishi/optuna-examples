import optuna

from fit_sin_GS import objective

name = "4907-RedoRS3"
storage = f"sqlite:///../outputs/study-{name}.db"
study = optuna.create_study(
    storage=storage,
    sampler=optuna.samplers.RandomSampler(),
    study_name=f"Study-{name}",
    load_if_exists=True,
)

study.optimize(objective)
print(study.best_trial)
print("Finished")
