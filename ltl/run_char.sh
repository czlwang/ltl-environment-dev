#!/bin/bash

mode=$1  # all, train (only), baseline, langemb
nformula=$2  # number of formulas in training
runid=$3  # index for the run

env_flag="--env_name CharStream --num_steps 15"
gen_train_formula="--gen_formula_only --num_train_ltls $nformula"
gen_test_formula="--gen_formula_only --num_test_ltls 100"
baseline_flag=""

train_path="--formula_pickle data/formula_CharStream_abcde_${nformula}_${runid}.pickle"
test_path_0="data/formula_CharStream_test_in_abcde_${nformula}_${runid}.pickle"
test_path_1="data/formula_CharStream_test_out_abcde_${nformula}_${runid}.pickle"
test_path_2="data/formula_CharStream_test_out15_abcde_${nformula}_${runid}.pickle"
test_path_3="data/formula_CharStream_test_out20_abcde_${nformula}_${runid}.pickle"

model_path="models/a2c/CharStream_abcde_${nformula}_${runid}/"
model_name="CharStream_abcde_${nformula}_${runid}/model"


# set a different model path for baselines
if [ $mode = "baseline" ]; then
    model_path="models/a2c/CharStream_base_abcde_${nformula}_${runid}/"
    model_name="CharStream_base_abcde_${nformula}_${runid}/model"
    baseline_flag="--baseline"
fi
if [ $mode = "langemb" ]; then
    model_path="models/a2c/CharStream_lang_abcde_${nformula}_${runid}/"
    model_name="CharStream_lang_abcde_${nformula}_${runid}/model"
    baseline_flag="--baseline --lang_emb"
fi
if [ $mode = "notime" ]; then
    baseline_flag="--no_time"
fi


# generate data for 'all' mode
if [ $mode = "all" ]; then
    # generate training data
    python main.py $env_flag $gen_train_formula $train_path

    # generate testing data
    python main.py $env_flag $gen_test_formula $train_path \
        --test_formula_pickle_1 $test_path_0 \
        --num_test_ltls 100 --test_in_domain

    python main.py $env_flag $gen_test_formula $train_path \
        --test_formula_pickle_1 $test_path_1 \
        --num_test_ltls 100 --test_out_domain

    python main.py $env_flag $gen_test_formula $train_path \
        --test_formula_pickle_1 $test_path_2 \
        --num_test_ltls 100 --test_out_domain \
        --min_symbol_len 10 --max_symbol_len 15

    python main.py $env_flag $gen_test_formula $train_path \
        --test_formula_pickle_1 $test_path_3 \
        --num_test_ltls 100 --test_out_domain \
        --test_out_domain --min_symbol_len 15 --max_symbol_len 20
fi

# make the directory for the models
mkdir $model_path

# train the model
python main.py $env_flag --algo a2c --no_cuda \
    --prefix_reward_decay 0.8 \
    --use_gae --entropy_coef 0.1 --gamma 0.9 --train \
    --load_formula_pickle $train_path --num_train_ltls $nformula \
    --save_model_name $model_name $baseline_flag \
    --num_processes 15 --num_epochs 1000 \
    --test_formula_pickle_1 $test_path_0 \
    --test_formula_pickle_2 $test_path_1 \
    --test_formula_pickle_3 $test_path_2 \
    --test_formula_pickle_4 $test_path_3 \
    --load_eval_train \
    --num_env_steps 225 --log_interval 100
