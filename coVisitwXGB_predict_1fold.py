import pandas as pd, numpy as np
from tqdm.notebook import tqdm
import gc
import xgboost as xgb
import time

inputPath = '../input/newSplited/'
outputPath = '../output/newSplited/'

type_labels = {'clicks':0, 'carts':1, 'orders':2}
TOPN_candidate = 100
SET = 3
SUBSETNUM = 6
chunkSize = 5000000

predictTypes = ['orders']
input_note = 'covisit_20_20_20_newSuggester2_w2vSim_last_4'
note_model = 'final_3'


for predictType in predictTypes:
    print('================type:', predictType, '=================')
    if predictType == 'orders':
        input_note = 'covisit_20_20_20_newSuggester2_w2vSim_last_4'
    else:
        input_note = 'covisit_20_20_20_newSuggester2_add_freq'
    for s in range(SUBSETNUM):
        startTime = time.time()
        print('==============sub:', s, '==================')
        sub = f'_{s}'

        modelName = f'xgb_{predictType}'
        modelSavedPAth = outputPath + f'savedModel_set2/{modelName}_top_{TOPN_candidate}_{note_model}/'

        data4xgb_path = outputPath + f'data4xgb/set{SET}_top_{TOPN_candidate}/{input_note}'
        data4xgb = pd.read_parquet(f'{data4xgb_path}/{predictType}_{s}.pqt')
        print(data4xgb.shape)

        # Normalize orderofRule
        data4xgb.order_by_rule = data4xgb.order_by_rule / TOPN_candidate

        print('======= predict ===========')


        model = xgb.Booster()
        model.load_model(modelSavedPAth + modelName + f'.xgb')
        model.set_param({'predictor': 'gpu_predictor'})

        n = len(data4xgb)
        i = 0
        preds = np.array([])
        while (i * chunkSize < n):
            # dtest = xgb.DMatrix(data4xgb.iloc[i*chunkSize:(i+1)*chunkSize, 2:])
            dtest = xgb.DMatrix(data4xgb.iloc[i*chunkSize:(i+1)*chunkSize, 1:])
            preds = np.append(preds, model.predict(dtest))
            i += 1                


        data4xgb = data4xgb[['session','aid']]
        data4xgb['pred'] = preds
        data4xgb = data4xgb.sort_values(['session','pred'], ascending=[True,False]).reset_index(drop=True)
        data4xgb['n'] = data4xgb.groupby('session').aid.cumcount().astype('int8')
        data4xgb = data4xgb.loc[data4xgb.n<20]
        if s == 0:
            predConcate = data4xgb.copy()
        else:
            predConcate = pd.concat([predConcate, data4xgb], axis=0, ignore_index=True)

        endTime = time.time()
        print('time: ', endTime - startTime)
        
    predConcate = predConcate.groupby('session').aid.apply(list).to_frame().reset_index()
    predConcate.aid = predConcate.aid.apply(lambda x: " ".join(map(str,x)))
    predConcate.columns = ['session_type','labels']
    predConcate.session_type = predConcate.session_type.astype('str')+ '_' + predictType    

    predConcate.to_parquet(outputPath + 'predictions/' + f'xgb_{predictType}_top_{TOPN_candidate}{note_model}.pqt')


# integrate predictions
predictTypes = ['clicks', 'carts', 'orders']
subs = []
for i, predictType in enumerate(predictTypes):
    sub = pd.read_parquet(outputPath + 'predictions/' + f'xgb_{predictType}_top_{TOPN_candidate}{note_model}.pqt')
    subs.append(sub)
pred_df = pd.concat(subs).reset_index(drop=True)
pred_df.to_csv(f'../submissions/Xgb_top_{TOPN_candidate}_{note_model}.csv', index=False)