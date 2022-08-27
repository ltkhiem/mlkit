import mlflow


def load_experiment(experiment_name, exp_exists_ok):
    expm = mlflow.get_experiment_by_name(experiment_name)
    if expm is not None:
        if exp_exists_ok:
            return = dict(expm)['experiment_id']
        else:
            confirm = input(f"Experiment {experiment_name} exists. Press Enter to continue, type 'quit' to abort operation.\n")
            if confirm == '': 
                return dict(expm)['experiment_id']
            raise Exception("Experiment exists")

    return mlflow.create_experiment(experiment_name)


def check_active_run(func):
    def wrapper(*args, **kwargs):
        print("Decorator working")
        active_run = mlflow.active_run()
        if active_run is not None:
            confirm = input("Run with run_id {active_run.info.run_id} is currently active. Press Enter to continue or type 'quit' to abort operation.\n")
            if confirm != 'quit':
                raise Exception("There is another run running on mlflow")
            else:
                mlflow.end_run()

        func(*args, **kwargs)
    return wrapper
