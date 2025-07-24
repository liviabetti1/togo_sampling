import os
import sys
import argparse
import numpy as np
import dill
import json
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

import torch

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

# Local imports
from pycls.al.ActiveLearning import ActiveLearning
from pycls.core.config import cfg, dump_cfg
from pycls.datasets.data import Data
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def argparser():
    parser = argparse.ArgumentParser(description='Subset Selection - Ridge Regression')
    parser.add_argument('--cfg', dest='cfg_file', required=True, type=str)
    parser.add_argument('--exp-name', required=True, type=str)
    parser.add_argument('--sampling_fn', required=True, type=str)
    #parser.add_argument('--random_strategy', default=None, type=str)
    parser.add_argument('--budget', required=True, type=int)
    parser.add_argument('--initial_size', default=0, type=int)
    parser.add_argument('--id_path', default=None, type=str)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--initial_set_str', default="empty_initial_set", type=str)

    parser.add_argument('--cost_func', default=None, type=str)
    parser.add_argument('--cost_name', default=None, type=str)
    parser.add_argument('--cost_array_path', default=None, type=str)

    parser.add_argument('--group_type', default=None, type=str)
    parser.add_argument('--group_assignment_path', default=None, type=str)

    parser.add_argument('--unit_type', default=None, type=str)
    parser.add_argument('--points_per_unit', default=None, type=int)
    parser.add_argument('--unit_assignment_path', default=None, type=str)
    parser.add_argument('--unit_cost_path', default=None, type=str)

    parser.add_argument('--region_assignment_path', default=None, type=str)
    parser.add_argument('--in_region_unit_cost', default=None, type=int)
    parser.add_argument('--out_of_region_unit_cost', default=None, type=int)

    parser.add_argument('--util_lambda', default=0.5, type=float)

    parser.add_argument('--similarity_matrix_path', default=None, type=str)
    parser.add_argument('--distance_matrix_path', default=None, type=str)
    return parser

import csv

def write_results_csv(cfg, initial_r2, updated_r2):
    results_file = os.path.join(cfg.OUT_DIR, "experiment_results.csv")
    file_exists = os.path.isfile(results_file)

    header = [
        "exp_name",
        "seed",
        "sampling_fn",
        "budget",
        "cost_fn",
        "group_type",
        "initial_r2",
        "updated_r2"
    ]

    row = [
        cfg.EXP_NAME,
        cfg.RNG_SEED,
        cfg.ACTIVE_LEARNING.SAMPLING_FN,
        cfg.ACTIVE_LEARNING.BUDGET_SIZE,
        cfg.COST.NAME,
        getattr(cfg.GROUPS, "GROUP_TYPE", ""),
        initial_r2 if initial_r2 is not None else "",
        updated_r2 if updated_r2 is not None else ""
    ]

    with open(results_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def main(cfg):
    cfg.OUT_DIR = os.path.abspath(cfg.OUT_DIR)
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    dataset_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME)
    os.makedirs(dataset_dir, exist_ok=True)


    if cfg.EXP_NAME == 'auto':
        now = datetime.now()
        exp_dir = now.strftime('%Y%m%d_%H%M%S')
    else:
        seed_str = f"seed_{cfg.RNG_SEED}"

        if cfg.ACTIVE_LEARNING.SAMPLING_FN in ["match_population_proportion", "poprisk"]:
            sampling_str = f"{cfg.ACTIVE_LEARNING.SAMPLING_FN}/{cfg.GROUPS.GROUP_TYPE}"
        else:
            sampling_str = f"{cfg.ACTIVE_LEARNING.SAMPLING_FN}"

        base_dir = (f"{cfg.INITIAL_SET.STR}/{cfg.COST.NAME}/opt/{sampling_str}/budget_{cfg.ACTIVE_LEARNING.BUDGET_SIZE}"
                    if cfg.ACTIVE_LEARNING.OPT
                    else f"{cfg.INITIAL_SET.STR}/{cfg.COST.NAME}/{sampling_str}/budget_{cfg.ACTIVE_LEARNING.BUDGET_SIZE}")

        # Add random_strategy subfolder if cost_name is cluster_based and sampling_fn is random
        # if cfg.COST.NAME == "cluster_based" and cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":
        #     random_strategy = getattr(cfg.ACTIVE_LEARNING, "RANDOM_STRATEGY", None)
        #     if random_strategy:
        #         base_dir = f"{base_dir}/{random_strategy}"

        # Append util_lambda if sampling_fn is poprisk
        if cfg.ACTIVE_LEARNING.SAMPLING_FN == "poprisk":
            util_lambda = getattr(cfg.ACTIVE_LEARNING, "UTIL_LAMBDA", None)
            if util_lambda is not None:
                base_dir = f"{base_dir}/util_lambda_{util_lambda}"

        # Check if INITIAL_SET.STR already ends with seed_{seed}
        if not cfg.INITIAL_SET.STR.endswith(seed_str):
            exp_dir = f"{base_dir}/{seed_str}"
        else:
            exp_dir = base_dir

    exp_dir = os.path.join(dataset_dir, exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    cfg.EXP_DIR = exp_dir
    cfg.INITIAL_SET_DIR = os.path.join(dataset_dir, cfg.INITIAL_SET.STR)
    cfg.SAMPLING_DIR = os.path.join(dataset_dir, base_dir)
    dump_cfg(cfg)

    lu.setup_logging(cfg)
    logger.info(f"Experiment directory: {exp_dir}")

    cfg.DATASET.ROOT_DIR = os.path.abspath(cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    train_data, _ = data_obj.getDataset(isTrain=True)
    test_data, _ = data_obj.getDataset(isTrain=False)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridgecv', RidgeCV(alphas=np.logspace(-5, 5, 10), scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=42)))
    ])

    if cfg.LSET_IDS:
        lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets_from_ids(cfg.LSET_IDS, data=train_data, save_dir=cfg.EXP_DIR)
        lSet, uSet, _= data_obj.loadPartitions(lSetPath=lSet_path, \
            uSetPath=uSet_path, valSetPath = valSet_path )
    else:
        lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets_from_ids([], data=train_data, save_dir=cfg.EXP_DIR)
        lSet, uSet, _= data_obj.loadPartitions(lSetPath=lSet_path, \
            uSetPath=uSet_path, valSetPath = valSet_path )


    def evaluate_r2(model, X_test, y_test):
        return model.score(X_test, y_test)
    
    X_test, y_test = test_data[:][0], test_data[:][1]

    if len(lSet) > 0:
        initial_set_r2_path = f"{cfg.INITIAL_SET_DIR}/initial_set_r2_seed_{cfg.RNG_SEED}.json"
        if os.path.exists(initial_set_r2_path):

            print("Loading initial set r2 performance...")
            with open(initial_set_r2_path, "r") as f:
                r2 = json.load(f)
            logger.info(f"Initial R² score: {r2:.4f}")
        else:
            n_splits = 5
            print("Training ridge regression on initial set...")
            logger.info("Training ridge regression on initial set...")
            lSet = lSet.astype(int)
            uSet = uSet.astype(int)
            X_train, y_train = train_data[lSet][0], train_data[lSet][1]

            if X_train.shape[0] < 5:
                print("Not enough samples...")
            else:
                model.fit(X_train, y_train)
                r2_initial = evaluate_r2(model, X_test, y_test)
                logger.info(f"Initial R² score: {r2_initial:.4f}")

                os.makedirs(cfg.INITIAL_SET_DIR, exist_ok=True)
                with open(initial_set_r2_path, "w") as f:
                    json.dump(r2_initial, f)
            train_data.plot_subset_on_map(lSet, country_shape_file='shapefiles/tgo_admbnda_adm0_inseed_itos_20210107.shp', save_path=f"{cfg.INITIAL_SET_DIR}/lSet_plot.png")
    else:
        logger.info("Initial labeled set is empty; skipping to subset selection.")

    logger.info("Starting subset selection...")
    al_obj = ActiveLearning(data_obj, cfg)
    activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data)
    print(f"Sampled {len(activeSet)} points!")
    train_data.plot_subset_on_map(activeSet, country_shape_file='shapefiles/tgo_admbnda_adm0_inseed_itos_20210107.shp', save_path=f"{exp_dir}/activeSet_plot.png")

    logger.info(f"Selected {len(activeSet)} new samples.")
    data_obj.saveSets(lSet, uSet, activeSet, os.path.join(cfg.EXP_DIR, 'episode_0'))

    lSet_updated = np.concatenate((lSet, activeSet))
    lSet_updated = lSet_updated.astype(int)
    train_data.plot_subset_on_map(lSet_updated, country_shape_file='shapefiles/tgo_admbnda_adm0_inseed_itos_20210107.shp', save_path=f"{exp_dir}/lSet_updated_plot.png")

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridgecv', RidgeCV(alphas=np.logspace(-5, 5, 10), scoring='r2', cv=KFold(n_splits=5, shuffle=True, random_state=42)))
    ])

    # Train again on the updated labeled set
    if lSet_updated.size > 0:
        n_splits = 5
        logger.info("Training ridge regression on updated labeled set...")
        X_train_updated, y_train_updated = train_data[lSet_updated][0], train_data[lSet_updated][1]

        if n_splits > X_train_updated.shape[0]:
            print("Not enough samples...")
        else:
            model.fit(X_train_updated, y_train_updated)
            r2_updated = evaluate_r2(model, X_test, y_test)
            logger.info(f"Updated R² score: {r2_updated:.4f}")

    data_obj.saveSets(lSet_updated, new_uSet, activeSet, os.path.join(cfg.EXP_DIR, 'episode_1'))
    write_results_csv(cfg, r2_initial, r2_updated)