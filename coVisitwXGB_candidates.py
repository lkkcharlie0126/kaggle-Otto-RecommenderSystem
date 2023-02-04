import pandas as pd, numpy as np
from tqdm.auto import tqdm
import os, sys, pickle, glob, gc, shutil
import math
from pathlib import Path
from collections import Counter
import itertools
from eventsSuggesterNew import ClicksSuggester, BuysSuggester, CartsSuggester
import gensim

inputPath = '../input/newSplited/'
outputPath = '../output/newSplited/'
type_labels = {'clicks':0, 'carts':1, 'orders':2}

TOPN_clicks = 20
TOPN_b2b = 20
TOPN_cartsOrders = 20
TOPN_c2c = 20
TOPN_click2cart = 20
TOPN_click2order = 20
candidatesNum = 100
SETS = [2, 3]
VER = 6
note_covisit = 'covisit_20_20_20'
note_candidate = 'suggester_addLast'


def load_test(path):    
    dfs = []
    for e, chunk_file in enumerate(glob.glob(path)):
        chunk = pd.read_parquet(chunk_file)
        chunk.ts = (chunk.ts/1000).astype('int32')
        chunk['type'] = chunk['type'].map(type_labels).astype('int8')
        dfs.append(chunk)
    return pd.concat(dfs).reset_index(drop=True) #.astype({"ts": "datetime64[ms]"})

def pqt_to_dict(df):
    return df.groupby('aid_x').aid_y.apply(list).to_dict()

def saveChunk(data, chunkSize, path, predType):
    n = len(data)
    i = 0
    while(i*chunkSize < n):
        sub = data[i*chunkSize:(i+1)*chunkSize]
        sub.to_parquet(path + f'{predType}_{i}.pqt')
        i += 1


for SET in SETS:
    if SET == 1:
        testA = load_test('../input/split_2-1_pqt/test_parquets/*')
    elif SET == 2:
        testA = pd.read_parquet('../input/splited/test.parquet')

    elif SET == 3:
        testA = load_test('../input/parquets/test_parquets/*')

    coVisitSaveFolder = outputPath + f'/coVisit/set{SET}/top20_20_20/'

    # top_clicks_covisit = {}
    # for k in range(0, 8):
    #     top_clicks_covisit.update( pqt_to_dict( pd.read_parquet(coVisitSaveFolder + f'top_{TOPN_clicks}_clicks_v{VER}_{k}.pqt') ) )

    # top_c2c_covisit = {}
    # for k in range(0, 8): 
    #     top_c2c_covisit.update( pqt_to_dict( pd.read_parquet(coVisitSaveFolder + f'top_{TOPN_c2c}_click2click_v{VER}_{k}.pqt') ) )

    top_buy2buy_covisit = {}
    for k in range(0, 2):
        top_buy2buy_covisit.update( pqt_to_dict( pd.read_parquet(coVisitSaveFolder + f'top_{TOPN_b2b}_buy2buy_v{VER}_{k}.pqt') ) )

    top_cartsOrders_covisit = {}
    for k in range(0, 8): 
        top_cartsOrders_covisit.update( pqt_to_dict( pd.read_parquet(coVisitSaveFolder + f'top_{TOPN_cartsOrders}_cartsOrders_v{VER}_{k}.pqt') ) )

    top_click2cart_covisit = {}
    for k in range(0, 8): 
        top_click2cart_covisit.update( pqt_to_dict( pd.read_parquet(coVisitSaveFolder + f'top_{TOPN_click2cart}_click2cart_v{VER}_{k}.pqt') ) )

    top_click2order_covisit = {}
    for k in range(0, 8): 
        top_click2order_covisit.update( pqt_to_dict( pd.read_parquet(coVisitSaveFolder + f'top_{TOPN_click2order}_click2order_v{VER}_{k}.pqt') ) )

    # top_clicks = testA.loc[testA['type']==0,'aid'].value_counts().index.values[:candidatesNum].astype(np.int32)
    # top_carts = testA.loc[testA['type'] == 1,'aid'].value_counts().index.values[:candidatesNum].astype(np.int32)
    top_orders = testA.loc[testA['type'] == 2,'aid'].value_counts().index.values[:candidatesNum].astype(np.int32)
    aid2vec_model = gensim.models.KeyedVectors.load_word2vec_format(f'../output/newSplited/savedModel/set_{SET}/otto_aid2vec_5d.bin', binary=True)
    # Suggest
    # clicksSuggester = ClicksSuggester(top_clicks, top_c2c_covisit, top_clicks_covisit, aid2vec_model)
    # cartsSuggester = CartsSuggester(top_carts, top_buy2buy_covisit, top_cartsOrders_covisit, top_click2cart_covisit, aid2vec_model)
    ordersSuggester = BuysSuggester(top_orders, top_buy2buy_covisit, top_cartsOrders_covisit, top_click2order_covisit, aid2vec_model)

    candidatesSavePath = outputPath + f'candidates/set{SET}_top_{candidatesNum}/{note_candidate}/'
    try: 
        os.makedirs(candidatesSavePath) 
    except OSError as error: 
        print(error)

    # clicks
    print('clicks')
    # tqdm.pandas()
    # pred_df_clicks = testA.sort_values(["session", "ts"]).groupby(["session"]).progress_apply(lambda x: clicksSuggester.suggest(x, candidatesNum)).to_frame().reset_index()
    # # pred_df_clicks = testA.sort_values(["session", "ts"]).groupby(["session"]).parallel_apply(lambda x: clicksSuggester.suggest(x, TOPN_candidate)).to_frame().reset_index()
    # pred_df_clicks.columns = ['session', 'labels']
    # pred_df_clicks['session'] = pred_df_clicks['session'].astype('int32')

    # aids = pred_df_clicks.labels.explode().astype('int32').rename('aid')
    # clicksCandidate = pred_df_clicks[['session']].astype('int32')
    # clicksCandidate = clicksCandidate.merge(aids, left_index=True, right_index=True, how='left').reset_index(drop=True)
    # saveChunk(clicksCandidate, 650000 * 100, candidatesSavePath, 'clicks')
    #  # carts
    # print('carts')
    # tqdm.pandas()
    # pred_df_carts = testA.sort_values(["session", "ts"]).groupby(["session"]).progress_apply(lambda x: cartsSuggester.suggest(x, candidatesNum)).to_frame().reset_index()
    # pred_df_carts.columns = ['session', 'labels']
    # pred_df_carts['session'] = pred_df_carts['session'].astype('int32')
    # pred_df_carts

    # aids = pred_df_carts.labels.explode().astype('int32').rename('aid')
    # cartsCandidate = pred_df_carts[['session']].astype('int32')
    # cartsCandidate = cartsCandidate.merge(aids, left_index=True, right_index=True, how='left').reset_index(drop=True)
    # saveChunk(cartsCandidate, 650000 * 100, candidatesSavePath, 'carts')
    # orders
    print('orders')
    tqdm.pandas()
    pred_df_orders = testA.sort_values(["session", "ts"]).groupby(["session"]).progress_apply(lambda x: ordersSuggester.suggest(x, candidatesNum)).to_frame().reset_index()
    pred_df_orders.columns = ['session', 'labels']
    pred_df_orders['session'] = pred_df_orders['session'].astype('int32')
    pred_df_orders

    aids = pred_df_orders.labels.explode().astype('int32').rename('aid')
    ordersCandidate = pred_df_orders[['session']].astype('int32')
    ordersCandidate = ordersCandidate.merge(aids, left_index=True, right_index=True, how='left').reset_index(drop=True)
    saveChunk(ordersCandidate, 330000 * 100, candidatesSavePath, 'orders')
