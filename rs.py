import optuna

from fit_sin_GS import objective

name = "1522-TestRemoteTrain"
storage = f"sqlite:///../outputs/study-{name}.db"
study = optuna.create_study(
    storage=storage,
    sampler=optuna.samplers.RandomSampler(),
    study_name=f"Study-{name}",
)

study.optimize(objective, timeout=300)
print(study.best_trial)
print("Finished")
