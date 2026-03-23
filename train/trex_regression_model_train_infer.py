import os
import yaml
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from itertools import chain
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning)

out_pkl = "./trex_1gb_64k_lang_results.pkl"
config_dir = "./config_mn"

baseline = pd.read_pickle("./k-exaone-236b-a23b_lang_results.pkl") # create it using calculate_length.sh

rows = []
for fp in Path(config_dir).glob("n*.yaml"):
    with fp.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        train = cfg.get("train", {}) or {}
        rows.append({"tok_name": fp.stem, **train})
X_train = pd.DataFrame(rows)
X_train = X_train.sort_values(by="tok_name", key=lambda s: s.str.extract(r'(\d+)', expand=False).astype(int))

Y_train = pd.read_pickle(out_pkl)
Y_train = Y_train.sort_values(by="tok_name", key=lambda s: s.str.extract(r'(\d+)', expand=False).astype(int)).reset_index(drop=True)

domains = X_train.columns[1:]
def prepare_arr_y_from_df(df_y, baseline):
    dist = []
    for idx in range(len(df_y)):  
        dist_per_tokenizer = []
        for domain in domains:
            num = sum(df_y.loc[idx, domain])  
            den = sum(baseline.loc[0, domain]) 

            ratio = num / den if den != 0 else np.nan
            dist_per_tokenizer.append(ratio)
        dist.append(dist_per_tokenizer)
    return np.array(dist)

x_train = X_train[domains].values
y_train = prepare_arr_y_from_df(Y_train, baseline)

y_train = y_train.mean(axis=1, keepdims=True)

num_valid_sample = 64 # FIXME
x_train, x_valid = x_train[:-num_valid_sample, :], x_train[-num_valid_sample:, :]
y_train, y_valid = y_train[:-num_valid_sample, :], y_train[-num_valid_sample:, :]

hyper_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1','l2'],
    "num_iterations": 1000, 
    'seed': 42,
    'learning_rate': 5e-2,
    "verbosity": -1,
}

np.random.seed(42)

print("=============================================================")

target = y_train[:,-1]
test_target = y_valid[:,-1]

gbm = lgb.LGBMRegressor(**hyper_params)
reg = gbm.fit(x_train, target,
    eval_set=[(x_valid, test_target)],
    eval_metric='l2', callbacks=[
])

print("### Feature Importance")
for col, imp in zip(domains, reg.feature_importances_):
    print(f"{col:10s} >>> {imp}")

def calc_mape(y_true, y_pred, eps=1e-8):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0
print("===========================================================")
print("Predictor Performance")
test_target = y_valid[:,-1]

r, p = spearmanr(reg.predict(x_valid), test_target)
mape = calc_mape(test_target, reg.predict(x_valid))
print("Correlation: {:.3f}".format(r))
print("MAPE: {:.8f}".format(mape))
print("===========================================================")

def generate_train_group(groups, weights, precision=5):
    assert len(groups) == len(weights), "Length of groups and weights must be equal"
    
    def format_weight(weight):
        return f"{weight:.{precision}f}".rstrip('0').rstrip('.')
    
    output_group = [f"  {group}: {format_weight(num)}" 
                    for group, num in zip(groups, weights)]
    
    return "\n".join(output_group)

def generate_valid_group(groups):
    weights = [1.0] * len(groups)
    output_group = [f"  {group}: {num}" for group, num in zip(groups, weights)]
    return "\n".join(output_group)

def save_config(output_folder, optimal_data_mixture):
    # if not exist, create the folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = f"{output_folder}/n1.yaml"

    train_groups = list(domains)
    valid_groups = list(domains)
    weights = list(optimal_data_mixture)

    # get the train and valid group
    train_group = generate_train_group(train_groups, weights)
    valid_group = generate_valid_group(valid_groups)

    with open(output_path, "w", encoding="utf8") as f:
        f.write("train:\n")
        f.write(train_group)
        f.write("\n")
        f.write("valid:\n")
        f.write(valid_group)
        f.write("\n")
        
        # these are configurations for the model
        content = ""
        content += "\n" + "model_name: Mistral-Nemo" # FIXME
        content += "\n" + "max_step: 1000"
        # constant learning rate for the small model
        content += "\n" + "learning_rate: 0.05"
        f.write(content)

np.random.seed(42)

# FIXME
prior_dist = [
    0.1666666666666666,
    0.1666666666666666,
    0.1666666666666666,
    0.1666666666666666,
    0.1666666666666666,
    0.1666666666666666,
]

samples = np.random.dirichlet(prior_dist * 1, 50000000)

all_d_preds = []

pred = reg.predict(samples)
all_d_preds.append(pred)

o = np.column_stack(all_d_preds)

k = 1024 # FIXME
col = o[:, 0]
topk_idx = np.argsort(col)[:k]
topk_vals = col[topk_idx]

print("Top-k indices:", topk_idx)
print("Top-k values:", topk_vals)

optimal_data_mixture = samples[topk_idx].mean(0)
print("Optimal Data Mixture : ", optimal_data_mixture)

save_config(
    optimal_data_mixture=optimal_data_mixture,
    output_folder=f"{config_dir}_optimal"
)