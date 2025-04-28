import optuna

from fit_sin_GS import objective

name = "2720-DefaultSampler"
storage = f"sqlite:///../outputs/study-{name}.db"
study = optuna.create_study(
    storage=storage,
    study_name=f"Study-{name}",
)

study.optimize(objective)
