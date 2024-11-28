from util import load_pkl
import optuna


def get_info(study):
    number_of_trials = len(study.trials)
    # Access the best trial
    best_trial = study.best_trial

    # Get the best value (objective value)
    best_value = best_trial.value
    # Get the best parameters
    best_params = best_trial.params
    best_trial_index = best_trial.number
    return number_of_trials, best_value, best_params, best_trial_index

for period in [8,16,32,64,128,256]:
    data_dir = f'processed_data_{period}'
    df_train_X_all = load_pkl(f"{data_dir}/df_train_X_all.pkl")
    # load the valid tickers.
    with open(f'{data_dir}/valid_tickers.txt', 'r') as f:
        valid_tickers = f.readlines()
        
    mysql_url = "mysql://root@192.168.2.34:3306/mysql"
    n_columns = len(df_train_X_all[0].columns)
    study_rf_name = f'study_rf_columns_{n_columns}_stocks_{len(valid_tickers)}_period_{period}'
    study_rf = optuna.create_study(study_name=study_rf_name, storage=mysql_url, load_if_exists=True)
    number_of_trials, best_value, best_params, best_trial_index = get_info(study_rf)
    print(f"-----------Information of horizon period: {period}----------")
    print(f"RF: Number of finished trials: {number_of_trials}")
    print(f"RF: Best value: {best_value}")
    print(f"RF: The best trial parameters are: {best_params}")
    print(f"RF: The best trial index is: {best_trial_index}")
    
    study_svm_name = f'study_svm_columns_{n_columns}_stocks_{len(valid_tickers)}_period_{period}'
    study_svm = optuna.create_study(study_name=study_svm_name, storage=mysql_url, load_if_exists=True)
    number_of_trials, best_value, best_params, best_trial_index = get_info(study_svm)
    print(f"SVM: Number of finished trials: {number_of_trials}")
    print(f"SVM: Best value: {best_value}")
    print(f"SVM: The best trial parameters are: {best_params}")
    print(f"SVM: The best trial index is: {best_trial_index}")