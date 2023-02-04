import numpy as np
import cudf
import gc

class CoVisitation:
    VER = 6
    cvType = ''
    # type_weight = {0:1, 1:6, 2:3}
    type_weight = {0:0.5, 1:9, 2:0.5}
    # USE SMALLEST DISK_PIECES POSSIBLE WITHOUT MEMORY ERROR
    # CHUNK PARAMETERS
    DISK_PIECES = 8
    READ_CT = 3
    CHUNK = 25
    # saveFolder = '../output/coVisitationPqt'

    def __init__(self, files, data_cache, saveFolder):
        self.files = files
        self.saveFolder = saveFolder
        self.data_cache = data_cache
        self.SIZE = 1.86e6/self.DISK_PIECES
        self.CHUNK_SEG = int( np.ceil(len(files)/self.CHUNK) )
        print(f'We will process {len(files)} files, in groups of {self.READ_CT} and chunks of {self.CHUNK}.')

    def read_file(self, f):
        return cudf.DataFrame( self.data_cache[f] )

    def processDisks(self, topN):
        for PART in range(self.DISK_PIECES):
            print()
            print('### DISK PART',PART+1)
            # MERGE IS FASTEST PROCESSING CHUNKS WITHIN CHUNKS
            # => OUTER CHUNKS
            for j in range(self.CHUNK_SEG):
                a = j*self.CHUNK
                b = min( (j+1)*self.CHUNK, len(self.files) )
                print(f'Processing files {a} thru {b-1} in groups of {self.READ_CT}...')
                # => INNER CHUNKS
                for k in range(a, b, self.READ_CT):
                    # READ FILE
                    df = [self.read_file(self.files[k])]
                    for i in range(1, self.READ_CT): 
                        if k+i<b: df.append( self.read_file(self.files[k+i]) )
                    df = self.fileProcessing(df, PART)
                    # COMBINE INNER CHUNKS
                    if k==a: tmp2 = df
                    else: tmp2 = tmp2.add(df, fill_value=0)
                    del df
                    gc.collect()
                    print(k,', ',end='')
                print()
                # COMBINE OUTER CHUNKS
                if a==0: tmp = tmp2
                else: tmp = tmp.add(tmp2, fill_value=0)
                del tmp2
                gc.collect()
            # CONVERT MATRIX TO DICTIONARY
            tmp = tmp.reset_index()
            tmp = tmp.sort_values(['aid_x','wgt'],ascending=[True,False])
            # SAVE TOP 40
            tmp = tmp.reset_index(drop=True)
            tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
            tmp = tmp.loc[tmp.n<topN].drop('n',axis=1)
            # SAVE PART TO DISK (convert to pandas first uses less memory)
            tmp.to_pandas().to_parquet(f'{self.saveFolder}/top_{topN}_{self.cvType}_v{self.VER}_{PART}.pqt')
            del tmp
            gc.collect()

    def fileProcessing(self, PART):
        pass



class CV_B2B(CoVisitation):
    cvType = 'buy2buy'
    DISK_PIECES = 2

    def fileProcessing(self, df, PART):
        df = cudf.concat(df,ignore_index=True,axis=0)
        df = df.loc[df['type'].isin([1,2])] # ONLY WANT CARTS AND ORDERS
        df = df.sort_values(['session','ts'],ascending=[True,False])
        # USE TAIL OF SESSION
        df = df.reset_index(drop=True)
        df['n'] = df.groupby('session').cumcount()
        df = df.loc[df.n<30].drop('n',axis=1)
        # CREATE PAIRS
        df = df.merge(df,on='session')
        df = df.loc[ ((df.ts_x - df.ts_y).abs()< 14 * 24 * 60 * 60) & (df.aid_x != df.aid_y) ] # 14 DAYS
        # MEMORY MANAGEMENT COMPUTE IN PARTS
        df = df.loc[(df.aid_x >= PART*self.SIZE)&(df.aid_x < (PART+1)*self.SIZE)]
        # ASSIGN WEIGHTS
        df = df[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y', 'type_y'])
        df['wgt'] = 1
        df = df[['aid_x','aid_y','wgt']]
        df.wgt = df.wgt.astype('float32')
        df = df.groupby(['aid_x','aid_y']).wgt.sum()
        return df




class ClicktoClick(CoVisitation):
    cvType = 'click2click'
    DISK_PIECES = 8

    def fileProcessing(self, df, PART):
        df = cudf.concat(df,ignore_index=True,axis=0)
        df = df.loc[df['type'] == 0] # ONLY WANT Click
        df = df.sort_values(['session','ts'],ascending=[True,False])
        # USE TAIL OF SESSION
        df = df.reset_index(drop=True)
        df['n'] = df.groupby('session').cumcount()
        df = df.loc[df.n<30].drop('n',axis=1)
        # CREATE PAIRS
        df = df.merge(df,on='session')
        df = df.loc[ (df.ts_y > df.ts_x) & ((df.ts_y - df.ts_x) < 5 * 60) & (df.aid_x != df.aid_y)]
        # MEMORY MANAGEMENT COMPUTE IN PARTS
        df = df.loc[(df.aid_x >= PART*self.SIZE)&(df.aid_x < (PART+1)*self.SIZE)]
        # ASSIGN WEIGHTS
        df = df[['session', 'aid_x', 'aid_y', 'ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
        df['wgt'] = (1 + 3*(df.ts_x - 1659304800)/(1662328791-1659304800))
        df = df[['aid_x','aid_y','wgt']]
        df.wgt = df.wgt.astype('float32')
        df = df.groupby(['aid_x','aid_y']).wgt.sum()
        return df



class ClicktoCart(CoVisitation):
    cvType = 'click2cart'
    DISK_PIECES = 8

    def fileProcessing(self, df, PART):
        df = cudf.concat(df,ignore_index=True,axis=0)
        df = df.loc[df['type'].isin([0, 1])] # ONLY WANT Click, Cart
        df = df.sort_values(['session','ts'],ascending=[True,False])
        # USE TAIL OF SESSION
        df = df.reset_index(drop=True)
        df['n'] = df.groupby('session').cumcount()
        df = df.loc[df.n<30].drop('n',axis=1)
        # CREATE PAIRS
        df = df.merge(df,on='session')
        df = df.loc[ (df.type_x == 0) & (df.type_y == 1) & (df.ts_y > df.ts_x) & ((df.ts_y - df.ts_x) < 30 * 60)]
        # MEMORY MANAGEMENT COMPUTE IN PARTS
        df = df.loc[(df.aid_x >= PART*self.SIZE)&(df.aid_x < (PART+1)*self.SIZE)]
        # ASSIGN WEIGHTS
        df = df[['session', 'aid_x', 'aid_y', 'ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
        df['wgt'] = (1 + 3*(df.ts_x - 1659304800)/(1662328791-1659304800))
        df = df[['aid_x','aid_y','wgt']]
        df.wgt = df.wgt.astype('float32')
        df = df.groupby(['aid_x','aid_y']).wgt.sum()
        return df


class ClicktoOrder(CoVisitation):
    cvType = 'click2order'
    DISK_PIECES = 8

    def fileProcessing(self, df, PART):
        df = cudf.concat(df,ignore_index=True,axis=0)
        df = df.loc[df['type'].isin([0, 1])] # ONLY WANT Click, Cart
        df = df.sort_values(['session','ts'],ascending=[True,False])
        # USE TAIL OF SESSION
        df = df.reset_index(drop=True)
        df['n'] = df.groupby('session').cumcount()
        df = df.loc[df.n<30].drop('n',axis=1)
        # CREATE PAIRS
        df = df.merge(df,on='session')
        df = df.loc[ (df.type_x == 0) & (df.type_y == 2) & (df.ts_y > df.ts_x) & ((df.ts_y - df.ts_x) < 60 * 60)]
        # MEMORY MANAGEMENT COMPUTE IN PARTS
        df = df.loc[(df.aid_x >= PART*self.SIZE)&(df.aid_x < (PART+1)*self.SIZE)]
        # ASSIGN WEIGHTS
        df = df[['session', 'aid_x', 'aid_y', 'ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
        df['wgt'] = (1 + 3*(df.ts_x - 1659304800)/(1662328791-1659304800))
        df = df[['aid_x','aid_y','wgt']]
        df.wgt = df.wgt.astype('float32')
        df = df.groupby(['aid_x','aid_y']).wgt.sum()
        return df




class CV_toClicks(CoVisitation):
    cvType = 'clicks'
    DISK_PIECES = 8

    def fileProcessing(self, df, PART):
        df = cudf.concat(df,ignore_index=True,axis=0)
        df = df.sort_values(['session','ts'],ascending=[True,False])
        # USE TAIL OF SESSION
        df = df.reset_index(drop=True)
        df['n'] = df.groupby('session').cumcount()
        df = df.loc[df.n<30].drop('n',axis=1)
        # CREATE PAIRS
        df = df.merge(df,on='session')
        df = df.loc[ ((df.ts_x - df.ts_y).abs()< 24 * 60 * 60) & (df.aid_x != df.aid_y)]
        # MEMORY MANAGEMENT COMPUTE IN PARTS
        df = df.loc[(df.aid_x >= PART*self.SIZE)&(df.aid_x < (PART+1)*self.SIZE)]
        # ASSIGN WEIGHTS
        df = df[['session', 'aid_x', 'aid_y', 'ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
        df['wgt'] = (1 + 3*(df.ts_x - 1659304800)/(1662328791-1659304800))
        df = df[['aid_x','aid_y','wgt']]
        df.wgt = df.wgt.astype('float32')
        df = df.groupby(['aid_x','aid_y']).wgt.sum()
        return df



class CV_carts_orders(CoVisitation):
    cvType = 'cartsOrders'
    DISK_PIECES = 8

    def fileProcessing(self, df, PART):
        df = cudf.concat(df,ignore_index=True,axis=0)
        df = df.sort_values(['session','ts'],ascending=[True,False])
        # USE TAIL OF SESSION
        df = df.reset_index(drop=True)
        df['n'] = df.groupby('session').cumcount()
        df = df.loc[df.n<30].drop('n',axis=1)
        # CREATE PAIRS
        df = df.merge(df,on='session')
        # df = df.loc[ ((df.ts_x - df.ts_y).abs()< 24 * 60 * 60) & (df.aid_x != df.aid_y) ]
        df = df.loc[ ((df.ts_x - df.ts_y).abs()< 24 * 60 * 60) & (df.aid_x != df.aid_y)]
        # MEMORY MANAGEMENT COMPUTE IN PARTS
        df = df.loc[(df.aid_x >= PART*self.SIZE)&(df.aid_x < (PART+1)*self.SIZE)]
        # ASSIGN WEIGHTS
        df = df[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y', 'type_y'])
        df['wgt'] = df.type_y.map(self.type_weight)
        df = df[['aid_x','aid_y','wgt']]
        df.wgt = df.wgt.astype('float32')
        df = df.groupby(['aid_x','aid_y']).wgt.sum()
        return df



class CV_B2B_Q(CoVisitation):
    cvType = 'buy2buy_q'
    DISK_PIECES = 2

    def fileProcessing(self, df, PART):
        df = cudf.concat(df,ignore_index=True,axis=0)
        df = df.loc[df['type'].isin([1,2])] # ONLY WANT CARTS AND ORDERS
        df = df.sort_values(['session','ts'],ascending=[True,False])
        df = df.drop_duplicates(['session', 'aid', 'type'])
        # USE TAIL OF SESSION
        df = df.reset_index(drop=True)
        df['n'] = df.groupby('session').cumcount()
        df = df.loc[df.n<50].drop('n',axis=1)
        # CREATE PAIRS
        df = df.merge(df,on='session')
        df = df.loc[ ((df.ts_x - df.ts_y).abs()< 24 * 60 * 60) & (df.type_x == 1) & (df.type_y == 2)] # 14 DAYS
        # MEMORY MANAGEMENT COMPUTE IN PARTS
        df = df.loc[(df.aid_x >= PART*self.SIZE)&(df.aid_x < (PART+1)*self.SIZE)]
        # ASSIGN WEIGHTS
        df = df[['session', 'aid_x', 'aid_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
        df['wgt'] = 1
        df = df[['aid_x','aid_y','wgt']]
        df.wgt = df.wgt.astype('float32')
        df = df.groupby(['aid_x','aid_y']).wgt.sum()
        return df