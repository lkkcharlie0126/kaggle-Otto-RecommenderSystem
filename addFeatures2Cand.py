import pandas as pd, numpy as np
from tqdm.notebook import tqdm
import os, sys, pickle, glob, gc
from collections import Counter
import itertools
from pathlib import Path
from operator import itemgetter
import gensim
from annoy import AnnoyIndex
tqdm.pandas()


class AddFeatures2Cand():

    inputPath = '../input/newSplited/'
    outputPath = '../output/newSplited/'
    type_labels = {'clicks':0, 'carts':1, 'orders':2}
    # input_note = 'covisit_20_20_20_newSuggester2'
    output_note = ''
    topNinHistory = 30
    timeWeights = list(np.logspace(1, 0.1, topNinHistory, base=2, endpoint=True) - 1)

    def __init__(self, TOPN_candidate, SETS, predTypes, SUBSETNUM, input_note, isTimeWgt=False):
        self.TOPN_candidate = TOPN_candidate
        self.SETS = SETS
        self.predTypes = predTypes
        self.SUBSETNUM = SUBSETNUM
        self.input_note = input_note
        self.isTimeWgt = isTimeWgt

    def load_df(self, path):    
        dfs = []
        for e, chunk_file in enumerate(glob.glob(path)):
            chunk = pd.read_parquet(chunk_file)
            chunk.ts = (chunk.ts/1000).astype('int32')
            chunk['type'] = chunk['type'].map(self.type_labels).astype('int8')
            dfs.append(chunk)
        return pd.concat(dfs).reset_index(drop=True) #.astype({"ts": "datetime64[ms]"})

    def process(self):
        for SET in self.SETS:
            print('set:', SET)
            self.candidates_input_path = self.outputPath + f'candidates/set{SET}_top_{self.TOPN_candidate}/{self.input_note}/'
            self.candidates_output_path = self.outputPath + f'candidates/set{SET}_top_{self.TOPN_candidate}/{self.input_note}_{self.output_note}_t{int(self.isTimeWgt)}/'
            try: 
                os.makedirs(self.candidates_output_path) 
            except OSError as error: 
                print(error)

            self.featureProcess(SET)
            for predType in self.predTypes:
                print('type:', predType)
                for s in range(self.SUBSETNUM):
                    print('sub: ', s)
                    self.merge(predType, s)

    def load_ValA(self, SET):
        if SET == 1:
            return self.load_df('../input/split_2-1_pqt/test_parquets/*')
        elif SET == 2:
            return pd.read_parquet('../input/splited/test.parquet')
        elif SET == 3:
            return self.load_df('../input/parquets/test_parquets/*')

    def featureProcess(self, SET):
        pass

    def merge(self, predType, s):
        pass


class ADD_covWgt(AddFeatures2Cand):

    covisitTypes = ['clicks', 'click2click', 'cartsOrders', 'buy2buy', 'click2cart', 'click2order', 'buy2buy_q']
    topN = {'clicks': 20, 'click2click': 20, 'cartsOrders': 20, 'buy2buy': 20, 'click2cart': 20, 'click2order': 20, 'buy2buy_q': 20}
    diskPart = {'clicks': 8, 'click2click': 8, 'cartsOrders': 8, 'buy2buy': 2, 'click2cart': 8, 'click2order': 8, 'buy2buy_q': 2}
    VER = 6
    covisitMat = {}
    covisitNote = 'top20_20_20'
    output_note = 'covWgt'

    def featureProcess(self, SET):
        self.val_A = self.load_ValA(SET)
        self.val_A = self.val_A.sort_values(['session', 'ts'], ascending=[True, False]).reset_index(drop=True)
        self.val_A = self.val_A.drop_duplicates((['session', 'aid', 'type'])).reset_index(drop=True)
        if self.isTimeWgt:
            self.val_A['ts'] = 1 + 3*(self.val_A.ts - self.val_A.ts.min())/(self.val_A.ts.max()-self.val_A.ts.min())

        self.coVisitSaveFolder = self.outputPath + f'coVisit/set{SET}/{self.covisitNote}/'
        self.readCovisitMat()

    def readCovisitMat(self):
        for covisitType in self.covisitTypes:
            for k in range(0, self.diskPart[covisitType]):
                if k == 0:
                    self.covisitMat[covisitType] = pd.read_parquet(self.coVisitSaveFolder + f'top_{self.topN[covisitType]}_{covisitType}_v{self.VER}_{k}.pqt')
                else:
                    self.covisitMat[covisitType] = pd.concat([self.covisitMat[covisitType], pd.read_parquet(self.coVisitSaveFolder + f'top_{self.topN[covisitType]}_{covisitType}_v{self.VER}_{k}.pqt')], axis=0)

    def merge(self, predType, s):
        candidate = pd.read_parquet(self.candidates_input_path + f'{predType}_{s}.pqt')
        candidate['session'] = candidate.session.astype('int32')
        
        df = self.val_A.merge(candidate[['session', 'aid']],on='session')
        if predType == 'clicks':
            toMergeTypes = ['clicks', 'click2click']
            clicksCol = ['wgt_click2click']
            cartsOrdersCol = []
        elif predType == 'carts':
            toMergeTypes = ['cartsOrders', 'buy2buy', 'click2cart']
            cartsOrdersCol = ['wgt_buy2buy']
            clicksCol = ['wgt_click2cart']
        elif predType == 'orders':
            toMergeTypes = ['cartsOrders', 'buy2buy', 'click2order']
            cartsOrdersCol = ['wgt_buy2buy']
            clicksCol = ['wgt_click2order']

        for toMergeType in toMergeTypes:
            df = df.merge(self.covisitMat[toMergeType], on=['aid_x', 'aid_y'], how='left').fillna(0)

        newCols = list(map(lambda x: 'wgt_'+x, toMergeTypes))
        df.columns = ['session', 'aid_x', 'ts', 'type', 'aid_y'] + newCols
        for col in clicksCol:
            df.loc[df.type.isin([1, 2]), col] = 0
        for col in cartsOrdersCol:
            df.loc[df.type == 0, col] = 0
        if self.isTimeWgt:
            for col in newCols:
                df[col] = df[col] * df['ts']

        df = df.groupby(['session', 'aid_y']).agg({col: 'mean' for col in newCols}).reset_index()
        df.rename(columns = {'aid_y': 'aid'}, inplace = True)
        candidate = candidate.merge(df, on=['session', 'aid'], how='left')
        candidate.to_parquet(self.candidates_output_path + f'{predType}_{s}.pqt')


class ADD_covWgt_cnt(AddFeatures2Cand):

    covisitTypes = ['clicks', 'click2click', 'cartsOrders', 'buy2buy', 'click2cart', 'click2order', 'buy2buy_q']
    topN = {'clicks': 20, 'click2click': 20, 'cartsOrders': 20, 'buy2buy': 20, 'click2cart': 20, 'click2order': 20, 'buy2buy_q': 20}
    diskPart = {'clicks': 8, 'click2click': 8, 'cartsOrders': 8, 'buy2buy': 2, 'click2cart': 8, 'click2order': 8, 'buy2buy_q': 2}
    VER = 6
    covisitMat = {}
    covisitNote = 'top20_20_20'
    output_note = 'covWgtCnt'

    def featureProcess(self, SET):
        self.val_A = self.load_ValA(SET)
        self.val_A = self.val_A.sort_values(['session', 'ts'], ascending=[True, False]).reset_index(drop=True)
        self.val_A = self.val_A.drop_duplicates((['session', 'aid'])).reset_index(drop=True)

        self.val_A['n'] = self.val_A.groupby('session').aid.cumcount()
        self.val_A = self.val_A.loc[self.val_A.n < 10].drop('n',axis=1)

        if self.isTimeWgt:
            self.val_A['ts'] = 1 + 3*(self.val_A.ts - self.val_A.ts.min())/(self.val_A.ts.max()-self.val_A.ts.min())

        self.coVisitSaveFolder = self.outputPath + f'coVisit/set{SET}/{self.covisitNote}/'
        self.readCovisitMat()

    def readCovisitMat(self):
        for covisitType in self.covisitTypes:
            for k in range(0, self.diskPart[covisitType]):
                if k == 0:
                    self.covisitMat[covisitType] = pd.read_parquet(self.coVisitSaveFolder + f'top_{self.topN[covisitType]}_{covisitType}_v{self.VER}_{k}.pqt')
                else:
                    self.covisitMat[covisitType] = pd.concat([self.covisitMat[covisitType], pd.read_parquet(self.coVisitSaveFolder + f'top_{self.topN[covisitType]}_{covisitType}_v{self.VER}_{k}.pqt')], axis=0)

            self.covisitMat[covisitType].wgt = 1

    def merge(self, predType, s):
        candidate = pd.read_parquet(self.candidates_input_path + f'{predType}_{s}.pqt')
        candidate['session'] = candidate.session.astype('int32')
        df = self.val_A.merge(candidate[['session', 'aid']],on='session')

        if predType == 'clicks':
            toMergeTypes = ['clicks', 'click2click']
            clicksCol = ['wgt_cnt_click2click']
            cartsOrdersCol = []
        elif predType == 'carts':
            toMergeTypes = ['cartsOrders', 'buy2buy', 'click2cart']
            cartsOrdersCol = ['wgt_cnt_buy2buy']
            clicksCol = ['wgt_cnt_click2cart']
        elif predType == 'orders':
            toMergeTypes = ['cartsOrders', 'buy2buy', 'click2order']
            cartsOrdersCol = ['wgt_cnt_buy2buy']
            clicksCol = ['wgt_cnt_click2order']

        for toMergeType in toMergeTypes:
            print('merge', toMergeType)
            df = df.merge(self.covisitMat[toMergeType], on=['aid_x', 'aid_y'], how='left').fillna(0)

        newCols = list(map(lambda x: 'wgt_cnt_' + x, toMergeTypes))


        df.columns = ['session', 'aid_x', 'ts', 'type', 'aid_y'] + newCols
        for col in clicksCol:
            df.loc[df.type.isin([1, 2]), col] = 0
        for col in cartsOrdersCol:
            df.loc[df.type == 0, col] = 0
        if self.isTimeWgt:
            for col in newCols:
                df[col] = df[col] * df['ts']

        df = df.groupby(['session', 'aid_y']).agg({col: 'sum' for col in newCols}).reset_index()
        df.rename(columns = {'aid_y': 'aid'}, inplace = True)
        candidate = candidate.merge(df, on=['session', 'aid'], how='left')
        candidate.to_parquet(self.candidates_output_path + f'{predType}_{s}.pqt')


class ADD_covScore_lastN(AddFeatures2Cand):
    lastN = 1
    output_note = 'covScore_last'

    covisitTypes = ['clicks', 'click2click', 'cartsOrders', 'buy2buy', 'click2cart', 'click2order']
    topN = {'clicks': 20, 'click2click': 20, 'cartsOrders': 20, 'buy2buy': 20, 'click2cart': 20, 'click2order': 20, 'buy2buy_q': 20}
    diskPart = {'clicks': 8, 'click2click': 8, 'cartsOrders': 8, 'buy2buy': 2, 'click2cart': 8, 'click2order': 8, 'buy2buy_q': 2}
    VER = 6
    covisitMats = {}
    covisitNote = 'top20_20_20'

    def featureProcess(self, SET):
        self.coVisitSaveFolder = self.outputPath + f'coVisit/set{SET}/{self.covisitNote}/'
        self.readCovisitMat()

        self.val_A = self.load_ValA(SET)
        self.val_A = self.val_A.sort_values(['session', 'ts'], ascending=[True, False]).reset_index(drop=True)
        self.val_A = self.val_A.drop_duplicates((['session', 'aid'])).reset_index(drop=True)

        self.val_A['n'] = self.val_A.groupby('session').aid.cumcount()
        self.val_A = self.val_A.loc[self.val_A.n == self.lastN - 1].drop('n',axis=1)

    def readCovisitMat(self):
        for covisitType in self.covisitTypes:
            for k in range(0, self.diskPart[covisitType]):
                if k == 0:
                    self.covisitMats[covisitType] = pd.read_parquet(self.coVisitSaveFolder + f'top_{self.topN[covisitType]}_{covisitType}_v{self.VER}_{k}.pqt')
                else:
                    self.covisitMats[covisitType] = pd.concat([self.covisitMats[covisitType], pd.read_parquet(self.coVisitSaveFolder + f'top_{self.topN[covisitType]}_{covisitType}_v{self.VER}_{k}.pqt')], axis=0)
            self.covisitMats[covisitType].rename(columns = {'wgt': f'covScore_{covisitType}_{self.lastN}'}, inplace = True)


    def merge(self, predType, s):
        candidate = pd.read_parquet(self.candidates_input_path  + f'{predType}_{s}.pqt')
        candidate['session'] = candidate.session.astype('int32')
        df = self.val_A.merge(candidate[['session', 'aid']],on='session')[['session', 'aid_x', 'aid_y']]
        if predType == 'clicks':
            covisitTypes = ['clicks', 'click2click']
        elif predType == 'carts':
            covisitTypes = ['cartsOrders', 'buy2buy', 'click2cart']
        elif predType == 'orders':
            covisitTypes = ['cartsOrders', 'buy2buy', 'click2order']
            # covisitTypes = ['buy2buy_q']

        for covisitType in covisitTypes:
            print('merge', covisitType, '...')
            df = df.merge(self.covisitMats[covisitType], on=['aid_x', 'aid_y'], how='left').fillna(0)
        df = df.drop(columns=['aid_x'])
        df.rename(columns = {'aid_y': 'aid'}, inplace = True)

        candidate = candidate.merge(df, on=['session', 'aid'], how='left')
        candidate.to_parquet(self.candidates_output_path + f'{predType}_{s}.pqt')




class ADD_cfSim(AddFeatures2Cand):
    output_note = 'cfSim'
    def featureProcess(self, SET):
        cf_save_path = Path(f'../output/newSplited/cf_matrix/set_{SET}')
        itemSimMatrix = np.load(cf_save_path / 'itemSimMatrix.npy', allow_pickle='TRUE').item()
        self.aid2aidCfScore = {
            'aid_x': [],
            'aid_y': [],
            'sim': [],
            }
        for aid_x in tqdm(itemSimMatrix):
            for aid_y, score in sorted(itemSimMatrix[aid_x].items(), key=itemgetter(1), reverse=True)[1:21]:
                self.aid2aidCfScore['aid_x'].append(aid_x)
                self.aid2aidCfScore['aid_y'].append(aid_y)
                self.aid2aidCfScore['sim'].append(score)
        self.aid2aidCfScore = pd.DataFrame(self.aid2aidCfScore)

        self.val_A = self.load_ValA(SET)
        self.val_A = self.val_A.sort_values(['session', 'ts'], ascending=[True, False]).reset_index(drop=True)
        self.val_A = self.val_A.drop_duplicates((['session', 'aid', 'type'])).reset_index(drop=True)
        if self.isTimeWgt:
            self.val_A['ts'] = 1 + 3*(self.val_A.ts - self.val_A.ts.min())/(self.val_A.ts.max()-self.val_A.ts.min())

    def merge(self, predType, s):
        candidate = pd.read_parquet(self.candidates_input_path  + f'{predType}_{s}.pqt')
        candidate['session'] = candidate.session.astype('int32')
        df = self.val_A.merge(candidate[['session', 'aid']],on='session')

        df = df.merge(self.aid2aidCfScore, on=['aid_x', 'aid_y'], how='left').fillna(0)
        if self.isTimeWgt:
            df['sim'] = df['sim'] * df['ts']
        df = df.groupby(['session', 'aid_y']).sim.sum().reset_index()
        df.columns = ['session', 'aid', 'cfSim']

        candidate = candidate.merge(df, on=['session', 'aid'], how='left')
        candidate.to_parquet(self.candidates_output_path + f'{predType}_{s}.pqt')




class ADD_cfSim_lastN(AddFeatures2Cand):
    lastN = 1
    output_note = 'cfSim_last'
    def featureProcess(self, SET):
        cf_save_path = Path(f'../output/newSplited/cf_matrix/set_{SET}')
        itemSimMatrix = np.load(cf_save_path / 'itemSimMatrix.npy', allow_pickle='TRUE').item()
        self.aid2aidCfScore = {
            'aid_x': [],
            'aid_y': [],
            'sim': [],
            }
        for aid_x in tqdm(itemSimMatrix):
            for aid_y, score in sorted(itemSimMatrix[aid_x].items(), key=itemgetter(1), reverse=True):#[1:21]:
                self.aid2aidCfScore['aid_x'].append(aid_x)
                self.aid2aidCfScore['aid_y'].append(aid_y)
                self.aid2aidCfScore['sim'].append(score)
        self.aid2aidCfScore = pd.DataFrame(self.aid2aidCfScore)

        self.val_A = self.load_ValA(SET)
        self.val_A = self.val_A.sort_values(['session', 'ts'], ascending=[True, False]).reset_index(drop=True)
        self.val_A = self.val_A.drop_duplicates((['session', 'aid'])).reset_index(drop=True)

        self.val_A['n'] = self.val_A.groupby('session').aid.cumcount()
        self.val_A = self.val_A.loc[self.val_A.n == self.lastN - 1].drop('n',axis=1)


    def merge(self, predType, s):
        candidate = pd.read_parquet(self.candidates_input_path  + f'{predType}_{s}.pqt')
        candidate['session'] = candidate.session.astype('int32')
        df = self.val_A.merge(candidate[['session', 'aid']],on='session')

        df = df.merge(self.aid2aidCfScore, on=['aid_x', 'aid_y'], how='left').fillna(0)
        df = df[['session', 'aid_y', 'sim']]
        df.rename(columns = {'aid_y': 'aid', 'sim': f'cfSim_last{self.lastN}'}, inplace = True)

        candidate = candidate.merge(df, on=['session', 'aid'], how='left')
        candidate.to_parquet(self.candidates_output_path + f'{predType}_{s}.pqt')


class ADD_w2vSim(AddFeatures2Cand):

    output_note = 'w2vSim'
    def featureProcess(self, SET):
        self.aidxaid = pd.read_parquet(f'../output/newSplited/features/{self.input_note}/set_{SET}/aid2aid_annoySim.pqt')

        self.val_A = self.load_ValA(SET)
        self.val_A = self.val_A.sort_values(['session', 'ts'], ascending=[True, False]).reset_index(drop=True)
        self.val_A = self.val_A.drop_duplicates((['session', 'aid', 'type'])).reset_index(drop=True)
        if self.isTimeWgt:
            self.val_A['ts'] = 1 + 3*(self.val_A.ts - self.val_A.ts.min())/(self.val_A.ts.max()-self.val_A.ts.min())

    def merge(self, predType, s):
        candidate = pd.read_parquet(self.candidates_input_path  + f'{predType}_{s}.pqt')
        candidate['session'] = candidate.session.astype('int32')
        df = self.val_A.merge(candidate[['session', 'aid']], on='session')

        df = df.merge(self.aidxaid, on=['aid_x', 'aid_y'], how='left').fillna(0)
        if self.isTimeWgt:
            df['annoySim'] = df['annoySim'] * df['ts']
        df = df.groupby(['session', 'aid_y']).annoySim.mean().reset_index()
        df.columns = ['session', 'aid', 'w2vSim']

        candidate = candidate.merge(df, on=['session', 'aid'], how='left')
        candidate.to_parquet(self.candidates_output_path + f'{predType}_{s}.pqt')



class ADD_w2vSim_lastN(AddFeatures2Cand):

    lastN = 1
    output_note = 'w2vSim_last'

    def featureProcess(self, SET):
        self.aidxaid = pd.read_parquet(f'../output/newSplited/features/{self.input_note}/aid2aid_annoySim.pqt')

        self.val_A = self.load_ValA(SET)
        self.val_A = self.val_A.sort_values(['session', 'ts'], ascending=[True, False]).reset_index(drop=True)
        self.val_A = self.val_A.drop_duplicates((['session', 'aid'])).reset_index(drop=True)

        self.val_A['n'] = self.val_A.groupby('session').aid.cumcount()
        self.val_A = self.val_A.loc[self.val_A.n == self.lastN-1].drop('n',axis=1)


    def merge(self, predType, s):
        candidate = pd.read_parquet(self.candidates_input_path  + f'{predType}_{s}.pqt')
        candidate['session'] = candidate.session.astype('int32')
        df = self.val_A.merge(candidate[['session', 'aid']],on='session')

        df = df.merge(self.aidxaid, on=['aid_x', 'aid_y'], how='left').fillna(0)
        df = df[['session', 'aid_y', 'annoySim']]
        df.rename(columns = {'aid_y': 'aid', 'annoySim': f'w2vSim_last{self.lastN}'}, inplace = True)

        candidate = candidate.merge(df, on=['session', 'aid'], how='left')
        candidate.to_parquet(self.candidates_output_path + f'{predType}_{s}.pqt')



class Add_features(AddFeatures2Cand):
    featureNote = 'norm_addLast'
    output_note = f'features_{featureNote}'
    addLabel = False

    def featureProcess(self, SET):
        featuresPath = self.outputPath + f'features/set{SET}/{self.featureNote}/'
        self.item_features = pd.read_parquet(featuresPath + 'item_features.pqt')
        self.user_features = pd.read_parquet(featuresPath + 'user_features.pqt')
        self.userItem_features = pd.read_parquet(featuresPath + 'userItem_features.pqt')

        if SET == 1 or SET ==2:
            self.addLabel = True
            if SET == 1:
                self.val_labels_pd = pd.read_parquet('../input/split_2-1_pqt/test_labels.parquet')
            else:
                self.val_labels_pd = pd.read_parquet('../input/splited/test_labels.parquet')
        else: 
            self.addLabel = False

    def merge(self, predType, s):
        candidate = pd.read_parquet(self.candidates_input_path + f'{predType}_{s}.pqt')
        candidate['session'] = candidate.session.astype('int32')
        candidate['order_by_rule'] = candidate.groupby('session').cumcount(ascending=False)
        candidate = candidate.merge(self.item_features, left_on='aid', right_index=True, how='left').fillna(-1)
        candidate = candidate.merge(self.user_features, left_on='session', right_index=True, how='left').fillna(-1)
        candidate = candidate.merge(self.userItem_features, left_on=['session', 'aid'], right_index=True, how='left').fillna(0)
        if self.addLabel:
            val_labels_type = self.val_labels_pd.loc[self.val_labels_pd['type'] == predType]
            aids = val_labels_type.ground_truth.explode().astype('int32').rename('aid')
            val_labels_type = val_labels_type[['session']].astype('int32')
            val_labels_type = val_labels_type.merge(aids, left_index=True, right_index=True, how='left')
            val_labels_type[predType[:-1]] = 1
            candidate = candidate.merge(val_labels_type, on=['session','aid'],how='left')
        candidate.to_parquet(self.candidates_output_path + f'{predType}_{s}.pqt')


class Add_freq_features(AddFeatures2Cand):
    featureNote = 'norm_freq'
    output_note = f'features_{featureNote}'

    def featureProcess(self, SET):
        featuresPath = self.outputPath + f'features/set{SET}/{self.featureNote}/'
        self.item_features = pd.read_parquet(featuresPath + 'item_freq_features.pqt')
        self.user_features = pd.read_parquet(featuresPath + 'user_freq_features.pqt')

    def merge(self, predType, s):
        candidate = pd.read_parquet(self.candidates_input_path + f'{predType}_{s}.pqt')
        candidate['session'] = candidate.session.astype('int32')
        candidate = candidate.merge(self.item_features, left_on='aid', right_index=True, how='left').fillna(-1)
        candidate = candidate.merge(self.user_features, left_on='session', right_index=True, how='left').fillna(-1)
       
        candidate.to_parquet(self.candidates_output_path + f'{predType}_{s}.pqt')