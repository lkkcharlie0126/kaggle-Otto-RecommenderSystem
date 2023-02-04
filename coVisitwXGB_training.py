import pandas as pd, numpy as np
from tqdm.notebook import tqdm
import os, gc

import xgboost as xgb
from sklearn.model_selection import GroupKFold
import gensim

inputPath = '../input/newSplited/'
outputPath = '../output/newSplited/'

type_labels = {'clicks':0, 'carts':1, 'orders':2}

SET = 2
TOPN_candidate = 100
predictTypes = ['clicks', 'carts', 'orders']
note = 'cand100'
outputNote = 'cand100'
data4xgb_path = outputPath + f'data4xgb_set{SET}_{note}/'

underSampleRate = {'clicks': 0.05, 'carts': 0.05, 'orders': 0.05} 
type2num_boost_round = {'clicks': 100, 'carts': 100, 'orders': 100}
xgb_parms = {'objective':'rank:ndcg', 'tree_method':'gpu_hist', 'lambda': 20, 'alpha': 20}


aid2vec_model = gensim.models.KeyedVectors.load_word2vec_format('../output/newSplited/savedModel/otto_aid2vec_5d.bin', binary=True)
aid2vec = {}
for aid in aid2vec_model.index_to_key:
    aid2vec[int(aid)] = aid2vec_model[aid]
aid2vec_df = pd.DataFrame.from_dict(aid2vec, orient='index').reset_index()
aid2vec_df.columns=['aid', 'd0', 'd1', 'd2', 'd3', 'd4']

for predictType in predictTypes:
    print('========= type', predictType, '==========')
    predCols = {'clicks': 'click', 'carts': 'cart', 'orders': 'order'}
    for s in range(4):
        data4xgb = pd.read_parquet(f'{data4xgb_path}data4xgb_{predictType}_{s}.pqt')
        positives = data4xgb.loc[data4xgb[predCols[predictType]] == 1]
        negatives = data4xgb.loc[data4xgb[predCols[predictType]] == 0]
        negatives = negatives.sample(frac=underSampleRate[predictType])
        if s == 0:
            candidates = pd.concat([positives,negatives],axis=0,ignore_index=True)
        else:
            candidates = pd.concat([candidates, positives, negatives],axis=0,ignore_index=True)
    candidates = candidates.merge(aid2vec_df, on=['aid'], how='left')

    # Normalize orderofRule
    candidates.order_by_rule = candidates.order_by_rule / TOPN_candidate

    modelName = 'xgb_' + predictType + '_' + outputNote
    modelSavedPAth = outputPath + f'savedModel_set{SET}/{modelName}/'
    try: 
        os.mkdir(modelSavedPAth) 
    except OSError as error: 
        print(error)  

    skf = GroupKFold(n_splits=5)
    predsVal = np.zeros(len(candidates))
    for fold,(train_idx, valid_idx) in enumerate(skf.split(candidates, candidates[predCols[predictType]], groups=candidates['session'] )):
        print('========= fold', fold, '==========')

        train = candidates.iloc[train_idx]
        val = candidates.iloc[valid_idx]


        train = train.sort_values('session')
        groupsTrain = train.groupby('session').aid.agg('count').values
        dropCol = ['session', 'aid', predCols[predictType]]
        dtrain = xgb.DMatrix(train.drop(dropCol, axis=1), train[predCols[predictType]], group=groupsTrain)

        val = val.sort_values('session')
        groupsVal = val.groupby('session').aid.agg('count').values
        dvalid = xgb.DMatrix(val.drop(dropCol, axis=1), val[predCols[predictType]], group=groupsVal)

        model = xgb.train(xgb_parms, 
            dtrain=dtrain,
            evals=[(dtrain,'train'),(dvalid,'valid')],
            num_boost_round=type2num_boost_round[predictType],
            verbose_eval=100)
        model.save_model(modelSavedPAth + modelName + f'_fold{fold}.xgb')

        # Validate on 1 fold
        dvalid = xgb.DMatrix(val.drop(dropCol, axis=1))
        pred = model.predict(dvalid)
        predsVal[valid_idx] = pred

    # xgb.plot_importance(model)
    # del data4xgb, candidates, predsVal, pred, train, val, groupsTrain, dtrain, groupsVal, dvalid, model
    # gc.collect()




