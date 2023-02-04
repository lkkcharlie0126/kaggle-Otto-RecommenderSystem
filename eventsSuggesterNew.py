import itertools
from collections import Counter
import numpy as np
from operator import itemgetter
import gensim
import math
import heapq


class EventsSuggester:
    # type_weight_multipliers = {0: 1, 1: 6, 2: 3}
    type_weight_multipliers = {0:0.5, 1:9, 2:0.5}

    def __init__(self, topEvents):
        self.topEvents = list(topEvents)

    def dfProcess(self):
        # USER HISTORY AIDS AND TYPES
        self.aids = self.df.aid.tolist()
        self.types = self.df.type.tolist()
        self.unique_aids = list(dict.fromkeys(self.aids[::-1] ))
        self.lastItem = str(self.unique_aids[0])

    def suggest(self, df, topN):
        self.df = df
        self.topN = topN
        self.dfProcess()
        if len(self.unique_aids) >= 20:
            self.candidatesGenerate20up()
            return self.reRank20up()
        else:
            self.candidatesGenerate()
            return self.reRank()

    def candidatesGenerate20up(self):
        self.candidates = Counter()
        # RERANK CANDIDATES USING WEIGHTS
        weights = np.logspace(0.1,1,len(self.aids),base=2, endpoint=True)-1
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid,w,t in zip(self.aids, weights, self.types):
            self.candidates[aid] += w * self.type_weight_multipliers[t]

    def candidatesGenerate(self):
        pass

    def reRank20up(self):
        # RERANK CANDIDATES
        result = [aid for aid, cnt in self.candidates.most_common(self.topN)]
        # if (len(result) < self.topN):
        #     resultSet = set(result)
        #     w2vScore = []
        #     for aid in self.topEvents:
        #         if aid not in resultSet:
        #             if len(w2vScore) < (self.topN - len(result)):
        #                 heapq.heappush(w2vScore, (-math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem)), aid))
        #             else:
        #                 heapq.heappushpop(w2vScore, (-math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem)), aid))
        #                 # w2vScore.append((-math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem)), aid))
        #     w2vScore.sort(reverse=True)
        #     idx = 0
        #     while(len(result) < self.topN):
        #         result.append(w2vScore[idx][1])
        #         idx += 1

        resultSet = set(result)
        idx = 0
        while(len(result) < self.topN):
            if self.topEvents[idx] not in resultSet:
                result.append(self.topEvents[idx])
            idx += 1
        return result

    def reRank(self):
        # RERANK CANDIDATES
        result = self.unique_aids + [aid for aid, _ in self.candidates.most_common(self.topN - len(self.unique_aids))]
        # USE TOP20 TEST CLICKS
        idx = 0
        resultSet = set(result)
        while(len(result) < self.topN):
            if self.topEvents[idx] not in resultSet:
                result.append(self.topEvents[idx])
            idx += 1
        # if (len(result) < self.topN):
        #     resultSet = set(result)
        #     w2vScore = []
        #     for aid in self.topEvents:
        #         if aid not in resultSet:
        #             if len(w2vScore) < (self.topN - len(result)):
        #                 heapq.heappush(w2vScore, (-math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem)), aid))
        #             else:
        #                 heapq.heappushpop(w2vScore, (-math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem)), aid))
        #     w2vScore.sort(reverse=True)
        #     idx = 0
        #     while(len(result) < self.topN):
        #         result.append(w2vScore[idx][1])
        #         idx += 1

        return result[:self.topN]



# class ClicksSuggester(EventsSuggester):

#     def __init__(self, topEvents, top_click2click, top_n_clicks):
#         self.topEvents = list(topEvents)
#         self.top_click2click = top_click2click
#         self.top_n_clicks = top_n_clicks

#     def dfProcess(self):
#         super().dfProcess()
#         self.df = self.df.loc[self.df['type']==0]
#         self.unique_clicks = list(dict.fromkeys( self.df.aid.tolist()[::-1] ))

#     def candidatesGenerate20up(self):
#         super().candidatesGenerate20up()
#         # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
#         aids2 = list(itertools.chain(*[self.top_n_clicks[aid] for aid in self.unique_aids if aid in self.top_n_clicks][:2]))
#         aids3 = list(itertools.chain(*[self.top_click2click[aid] for aid in self.unique_aids if aid in self.top_click2click][:5]))
#         for aid in aids3 + aids2: 
#             self.candidates[aid] += 0.1

#     def candidatesGenerate(self):
#         # USE "CLICKS" CO-VISITATION MATRIX
#         aids2 = list(itertools.chain(*[self.top_n_clicks[aid] for aid in self.unique_aids if aid in self.top_n_clicks][:2]))
#         aids3 = list(itertools.chain(*[self.top_click2click[aid] for aid in self.unique_clicks if aid in self.top_click2click][:4]))
#         self.candidates = Counter(aids2+aids3)
#         for aid in self.unique_aids:
#             if aid in self.candidates:
#                 del self.candidates[aid]

class ClicksSuggester(EventsSuggester):

    def __init__(self, topEvents, top_click2click, top_n_clicks, aid2vec_model):
        self.topEvents = list(topEvents)
        self.top_click2click = top_click2click
        self.top_n_clicks = top_n_clicks
        self.aid2vec_model = aid2vec_model


    def dfProcess(self):
        super().dfProcess()
        self.df = self.df.loc[self.df['type']==0]
        self.unique_clicks = list(dict.fromkeys( self.df.aid.tolist()[::-1] ))

    def candidatesGenerate20up(self):
        super().candidatesGenerate20up()
        # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
        aids2 = list(itertools.chain(*[self.top_n_clicks[aid] for aid in self.unique_aids if aid in self.top_n_clicks][:5]))
        aids3 = list(itertools.chain(*[self.top_click2click[aid] for aid in self.unique_aids if aid in self.top_click2click][:10]))
        for aid in aids3 + aids2: 
            self.candidates[aid] += 0.1
        for aid in self.candidates:
            self.candidates[aid] *= -math.log(max(1 - self.aid2vec_model.similarity(str(aid), self.lastItem), 0.000001))

    def candidatesGenerate(self):
        # USE "CLICKS" CO-VISITATION MATRIX
        aids2 = list(itertools.chain(*[self.top_n_clicks[aid] for aid in self.unique_aids if aid in self.top_n_clicks][:5]))
        aids3 = list(itertools.chain(*[self.top_click2click[aid] for aid in self.unique_clicks if aid in self.top_click2click][:10]))

        self.candidates = Counter(aids2+aids3)

        for aid in self.unique_aids:
            if aid in self.candidates:
                del self.candidates[aid]

        for aid in self.candidates:
            self.candidates[aid] *= -math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem))

# class ClicksSuggester(EventsSuggester):

#     def __init__(self, topEvents, top_click2click, top_n_clicks, aid2vec_model):
#         self.topEvents = list(topEvents)
#         self.top_click2click = top_click2click
#         self.top_n_clicks = top_n_clicks
#         self.aid2vec_model = aid2vec_model


#     def dfProcess(self):
#         super().dfProcess()
#         self.df = self.df.loc[self.df['type']==0]
#         self.unique_clicks = list(dict.fromkeys( self.df.aid.tolist()[::-1] ))

#     def candidatesGenerate20up(self):
#         super().candidatesGenerate20up()
#         # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
#         aids2 = list(itertools.chain(*[self.top_n_clicks[aid] for aid in self.unique_aids if aid in self.top_n_clicks][:2]))
#         aids3 = list(itertools.chain(*[self.top_click2click[aid] for aid in self.unique_aids if aid in self.top_click2click][:5]))
#         for aid in aids3 + aids2: 
#             self.candidates[aid] += 0.1
#         for aid in self.candidates:
#             self.candidates[aid] *= -math.log(max(1 - self.aid2vec_model.similarity(str(aid), self.lastItem), 0.000001))

#     def candidatesGenerate(self):
#         # USE "CLICKS" CO-VISITATION MATRIX
#         aids2 = list(itertools.chain(*[self.top_n_clicks[aid] for aid in self.unique_aids if aid in self.top_n_clicks][:2]))
#         aids3 = list(itertools.chain(*[self.top_click2click[aid] for aid in self.unique_clicks if aid in self.top_click2click][:4]))

#         self.candidates = Counter(aids2+aids3)

#         for aid in self.unique_aids:
#             if aid in self.candidates:
#                 del self.candidates[aid]

#         for aid in self.candidates:
#             self.candidates[aid] *= -math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem))



class CartsSuggester(EventsSuggester): 
    a = 1

    def __init__(self, topEvents, top_15_buy2buy, top_cartsOrders, top_click2cart, aid2vec_model):
        self.topEvents = list(topEvents)
        self.top_15_buy2buy = top_15_buy2buy
        self.top_cartsOrders = top_cartsOrders
        self.top_click2cart = top_click2cart
        self.aid2vec_model = aid2vec_model
    
    def dfProcess(self):
        super().dfProcess()
        self.unique_buys = list(dict.fromkeys( self.df.loc[self.df['type'].isin([1, 2])].aid.tolist()[::-1] ))
        self.unique_clicks = list(dict.fromkeys( self.df.loc[self.df['type'] == 0].aid.tolist()[::-1] ))
        

    def candidatesGenerate20up(self):
        super().candidatesGenerate20up()
        # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
        aids2 = list(itertools.chain(*[self.top_cartsOrders[aid] for aid in self.unique_aids if aid in self.top_cartsOrders][:1]))
        aids4 = list(itertools.chain(*[self.top_click2cart[aid] for aid in self.unique_clicks if aid in self.top_click2cart][:3]))
        aids3 = list(itertools.chain(*[self.top_15_buy2buy[aid] for aid in self.unique_buys if aid in self.top_15_buy2buy][:3]))
        for aid in aids4 + aids3 + aids2: 
            self.candidates[aid] += 0.1
        for aid in self.candidates:
            self.candidates[aid] *= -math.log(max(1 - self.aid2vec_model.similarity(str(aid), self.lastItem), 0.000001))

    def candidatesGenerate(self):
        # USE "CART ORDER" CO-VISITATION MATRIX
        aids2 = list(itertools.chain(*[self.top_cartsOrders[aid] for aid in self.unique_aids if aid in self.top_cartsOrders][:2]))
        # USE "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[self.top_15_buy2buy[aid] for aid in self.unique_buys if aid in self.top_15_buy2buy][:6]))
        aids4 = list(itertools.chain(*[self.top_click2cart[aid] for aid in self.unique_clicks if aid in self.top_click2cart][:6]))
        self.candidates = Counter(aids2 + aids3 + aids4)
        for aid in self.unique_aids:
            if aid in self.candidates:
                del self.candidates[aid]
        for aid in self.candidates:
            self.candidates[aid] *= -math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem))


class BuysSuggester(EventsSuggester):  
    a = 3
    def __init__(self, topEvents, top_15_buy2buy, top_15_buys, top_click2order, aid2vec_model):
        self.topEvents = list(topEvents)
        self.top_15_buy2buy = top_15_buy2buy
        self.top_15_buys = top_15_buys
        self.top_click2order = top_click2order
        self.aid2vec_model = aid2vec_model
        
    
    def dfProcess(self):
        super().dfProcess()
        self.unique_carts = list(dict.fromkeys( self.df.loc[self.df['type'] == 1].aid.tolist()[::-1] ))
        self.unique_buys = list(dict.fromkeys( self.df.loc[self.df['type'].isin([1, 2])].aid.tolist()[::-1] ))
        self.unique_clicks = list(dict.fromkeys( self.df.loc[self.df['type'] == 0].aid.tolist()[::-1] ))

    def candidatesGenerate20up(self):
        self.candidates = Counter()
        # RERANK CANDIDATES USING WEIGHTS
        weights = np.logspace(0.1,1,len(self.aids),base=2, endpoint=True)-1
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid,w,t in zip(self.aids, weights, self.types):
            self.candidates[aid] += w * self.type_weight_multipliers[t]

        # RERANK CANDIDATES USING "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[self.top_15_buy2buy[aid] for aid in self.unique_buys if aid in self.top_15_buy2buy][:3]))
        aids2 = list(itertools.chain(*[self.top_15_buys[aid] for aid in self.unique_aids if aid in self.top_15_buys][:5]))
        aids4 = list(itertools.chain(*[self.top_click2order[aid] for aid in self.unique_clicks if aid in self.top_click2order][:1]))
        
        for aid in aids3 + aids2 + aids4: 
            self.candidates[aid] += 0.1

        for aid in self.unique_carts:
            if aid in self.candidates:
                del self.candidates[aid]

    def reRank20up(self):
        # RERANK CANDIDATES
        result = self.unique_carts + [aid for aid, _ in self.candidates.most_common(self.topN - len(self.unique_carts))]

        # result = [aid for aid, _ in self.candidates.most_common(self.topN)]
        # if (len(result) < self.topN):
        #     resultSet = set(result)
        #     w2vScore = []
        #     for aid in self.topEvents:
        #         if aid not in resultSet:
        #             if len(w2vScore) < (self.topN - len(result)):
        #                 heapq.heappush(w2vScore, (-math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem)), aid))
        #             else:
        #                 heapq.heappushpop(w2vScore, (-math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem)), aid))
        #                 # w2vScore.append((-math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem)), aid))
        #     w2vScore.sort(reverse=True)
        #     idx = 0
        #     while(len(result) < self.topN):
        #         result.append(w2vScore[idx][1])
        #         idx += 1

        resultSet = set(result)
        idx = 0
        while(len(result) < self.topN):
            if self.topEvents[idx] not in resultSet:
                result.append(self.topEvents[idx])
            idx += 1
        return result

    def candidatesGenerate(self):
        # USE "CART ORDER" CO-VISITATION MATRIX
        aids2 = list(itertools.chain(*[self.top_15_buys[aid] for aid in self.unique_aids if aid in self.top_15_buys][:13]))
        # USE "BUY2BUY" CO-VISITATION MATRIX
        aids3 = list(itertools.chain(*[self.top_15_buy2buy[aid] for aid in self.unique_buys if aid in self.top_15_buy2buy][:2]))
        aids4 = list(itertools.chain(*[self.top_click2order[aid] for aid in self.unique_clicks if aid in self.top_click2order][:1]))
        self.candidates = Counter(aids2 + aids3 + aids4)
        for aid in self.unique_aids:
            if aid in self.candidates:
                del self.candidates[aid]
        for aid in self.candidates:
            if self.aid2vec_model.similarity(str(aid), self.lastItem) < 0.95:
                self.candidates[aid] *= 0.15
            # self.candidates[aid] *= -math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem))

    def reRank(self):
        # RERANK CANDIDATES
        result = self.unique_aids + [aid for aid, _ in self.candidates.most_common(self.topN - len(self.unique_aids))]
        # USE TOP20 TEST CLICKS
        # idx = 0
        # resultSet = set(result)
        # while(len(result) < self.topN):
        #     if self.topEvents[idx] not in resultSet:
        #         result.append(self.topEvents[idx])
        #     idx += 1
        if (len(result) < self.topN):
            resultSet = set(result)
            w2vScore = []
            for aid in self.topEvents:
                if aid not in resultSet:
                    if len(w2vScore) < (self.topN - len(result)):
                        heapq.heappush(w2vScore, (-math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem)), aid))
                    else:
                        heapq.heappushpop(w2vScore, (-math.log(1 - self.aid2vec_model.similarity(str(aid), self.lastItem)), aid))
            w2vScore.sort(reverse=True)
            idx = 0
            while(len(result) < self.topN):
                result.append(w2vScore[idx][1])
                idx += 1

        return result[:self.topN]
