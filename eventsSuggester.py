import itertools
from collections import Counter
import numpy as np

class EventsSuggester:
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}

    def __init__(self, topEvents):
        self.topEvents = list(topEvents)

    def dfProcess(self):
        # USER HISTORY AIDS AND TYPES
        self.aids = self.df.aid.tolist()
        self.types = self.df.type.tolist()
        self.unique_aids = list(dict.fromkeys(self.aids[::-1] ))

    def suggest(self, df, topN):
        self.df = df
        self.topN = topN
        self.dfProcess()
        if len(self.unique_aids) >= topN:
            self.candidatesGenerateIfGreaterTopN()
            return self.reRankIfGreaterTopN()
        else:
            self.candidatesGenerate()
            return self.reRank()

    def candidatesGenerateIfGreaterTopN(self):
        self.candidates = Counter()
        # RERANK CANDIDATES USING WEIGHTS
        weights = np.logspace(0.1,1,len(self.aids),base=2, endpoint=True)-1
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid,w,t in zip(self.aids, weights, self.types):
            self.candidates[aid] += w * self.type_weight_multipliers[t]

    def candidatesGenerate(self):
        pass

    def reRankIfGreaterTopN(self):
        # RERANK CANDIDATES
        sorted_aids = [aid for aid, cnt in self.candidates.most_common(self.topN)]
        return sorted_aids

    def reRank(self):
        # RERANK CANDIDATES
        top_aids = [aid for aid, cnt in self.candidates.most_common(self.topN)]  
        result = self.unique_aids + top_aids[:self.topN - len(self.unique_aids)]
        # USE TOP20 TEST CLICKS
        idx = 0
        resultSet = set(result)
        while(len(result) < self.topN):
            if self.topEvents[idx] not in resultSet:
                result.append(self.topEvents[idx])
            idx += 1
        return result



class ClicksSuggester(EventsSuggester):

    def __init__(self, topEvents, top_n_clicks):
        self.topEvents = list(topEvents)
        self.top_n_clicks = top_n_clicks

    def candidatesGenerate(self):
        # USE "CLICKS" CO-VISITATION MATRIX
        aids2 = list(itertools.chain(*[self.top_n_clicks[aid] for aid in self.unique_aids if aid in self.top_n_clicks]))
        self.candidates = Counter(aids2)
        for aid in self.unique_aids:
            if aid in self.candidates:
                del self.candidates[aid]
    
        

class BuysSuggester(EventsSuggester):  

    def __init__(self, topEvents, top_15_buy2buy, top_15_buys):
        self.topEvents = list(topEvents)
        self.top_15_buy2buy = top_15_buy2buy
        self.top_15_buys = top_15_buys
    
    def dfProcess(self):
        super().dfProcess()
        self.df = self.df.loc[self.df['type'].isin([1, 2])]
        self.unique_buys = list(dict.fromkeys( self.df.aid.tolist()[::-1] ))

    def candidatesGenerateIfGreaterTopN(self):
        super().candidatesGenerateIfGreaterTopN()
        # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[self.top_15_buy2buy[aid] for aid in self.unique_buys if aid in self.top_15_buy2buy]))
        for aid in aids3: 
            self.candidates[aid] += 0.1

    def candidatesGenerate(self):
        # USE "CART ORDER" CO-VISITATION MATRIX
        aids2 = list(itertools.chain(*[self.top_15_buys[aid] for aid in self.unique_aids if aid in self.top_15_buys]))
        # USE "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[self.top_15_buy2buy[aid] for aid in self.unique_buys if aid in self.top_15_buy2buy]))
        self.candidates = Counter(aids2 + aids3)
        for aid in self.unique_aids:
            if aid in self.candidates:
                del self.candidates[aid]


class CartsSuggesterNew(EventsSuggester):  

    def __init__(self, topEvents, top_15_buy2buy, top_15_buys):
        self.topEvents = list(topEvents)
        self.top_15_buy2buy = top_15_buy2buy
        self.top_15_buys = top_15_buys

    def suggest(self, df, topN):
        self.df = df
        self.topN = topN
        self.dfProcess()
        if len(self.unique_clicks) >= topN:
            self.candidatesGenerateIfGreaterTopN()
            return self.reRankIfGreaterTopN()
        else:
            self.candidatesGenerate()
            return self.reRank()
    
    def dfProcess(self):
        super().dfProcess()
        self.df_buys = self.df.loc[self.df['type'].isin([1, 2])]
        self.unique_buys = list(dict.fromkeys( self.df_buys.aid.tolist()[::-1] ))
        self.df_clicks = self.df.loc[self.df['type'] == 0]
        self.unique_clicks = list(dict.fromkeys( self.df_clicks.aid.tolist()[::-1] ))

    def candidatesGenerateIfGreaterTopN(self):
        super().candidatesGenerateIfGreaterTopN()
        # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[self.top_15_buy2buy[aid] for aid in self.unique_buys if aid in self.top_15_buy2buy]))
        for aid in aids3: 
            self.candidates[aid] += 0.1

    def candidatesGenerate(self):
        # USE "CART ORDER" CO-VISITATION MATRIX
        aids2 = list(itertools.chain(*[self.top_15_buys[aid] for aid in self.unique_aids if aid in self.top_15_buys]))
        # USE "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[self.top_15_buy2buy[aid] for aid in self.unique_buys if aid in self.top_15_buy2buy]))
        self.candidates = Counter(aids2 + aids3)
        for aid in self.unique_clicks:
            if aid in self.candidates:
                del self.candidates[aid]

    def reRank(self):
        # RERANK CANDIDATES
        top_aids = [aid for aid, cnt in self.candidates.most_common(self.topN)]  
        result = self.unique_clicks + top_aids[:self.topN - len(self.unique_clicks)]
        # USE TOP20 TEST CLICKS
        idx = 0
        resultSet = set(result)
        while(len(result) < self.topN):
            if self.topEvents[idx] not in resultSet:
                result.append(self.topEvents[idx])
            idx += 1
        return result



class CartsSuggester(EventsSuggester):  

    def __init__(self, topEvents, top_15_buy2buy, top_15_carts):
        self.topEvents = list(topEvents)
        self.top_15_buy2buy = top_15_buy2buy
        self.top_15_carts = top_15_carts
    
    def dfProcess(self):
        super().dfProcess()
        self.df = self.df.loc[self.df['type'].isin([1, 2])]
        self.unique_buys = list(dict.fromkeys( self.df.aid.tolist()[::-1] ))

    def candidatesGenerateIfGreaterTopN(self):
        super().candidatesGenerateIfGreaterTopN()
        # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[self.top_15_buy2buy[aid] for aid in self.unique_buys if aid in self.top_15_buy2buy]))
        for aid in aids3: 
            self.candidates[aid] += 0.1

    def candidatesGenerate(self):
        # USE "CART ORDER" CO-VISITATION MATRIX
        aids2 = list(itertools.chain(*[self.top_15_carts[aid] for aid in self.unique_aids if aid in self.top_15_carts]))
        # USE "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[self.top_15_buy2buy[aid] for aid in self.unique_buys if aid in self.top_15_buy2buy]))
        self.candidates = Counter(aids2 + aids3)
        for aid in self.unique_aids:
            if aid in self.candidates:
                del self.candidates[aid]



class OrdersSuggester(EventsSuggester):  

    def __init__(self, topEvents, top_15_buy2buy, top_15_orders):
        self.topEvents = list(topEvents)
        self.top_15_buy2buy = top_15_buy2buy
        self.top_15_orders = top_15_orders
    
    def dfProcess(self):
        super().dfProcess()
        self.df = self.df.loc[self.df['type'].isin([1, 2])]
        self.unique_buys = list(dict.fromkeys( self.df.aid.tolist()[::-1] ))

    def candidatesGenerateIfGreaterTopN(self):
        super().candidatesGenerateIfGreaterTopN()
        # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[self.top_15_buy2buy[aid] for aid in self.unique_buys if aid in self.top_15_buy2buy]))
        for aid in aids3: 
            self.candidates[aid] += 0.1

    def candidatesGenerate(self):
        # USE "CART ORDER" CO-VISITATION MATRIX
        aids2 = list(itertools.chain(*[self.top_15_orders[aid] for aid in self.unique_aids if aid in self.top_15_orders]))
        # USE "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[self.top_15_buy2buy[aid] for aid in self.unique_buys if aid in self.top_15_buy2buy]))
        self.candidates = Counter(aids2 + aids3)
        for aid in self.unique_aids:
            if aid in self.candidates:
                del self.candidates[aid]
