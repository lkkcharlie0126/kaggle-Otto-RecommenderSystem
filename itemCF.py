from tqdm import tqdm
from collections import defaultdict
import math
from operator import itemgetter

def itemCFTrain(df):
    
    # create list for dict
    user_item_list = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        user = int(row['session'])
        item = int(row['aid'])
        user_item_list.append([user, item])
    
    # create dict
    user_item_dict = dict()
    for user, item in tqdm(user_item_list):
         # change set() to list()
        user_item_dict.setdefault(user, list())
        user_item_dict[user].append(item)
    
    return user_item_dict


def ItemMatrix_fn(user_item_dict):
    
    # 采用python字典存储稀疏矩阵
    # N[i]表示物品i被操作（包括clicks，carts和orders）的次数
    N = defaultdict(int)
    itemMatrix = defaultdict(int)
    for user, items in tqdm(user_item_dict.items()):
        for i in items:
            itemMatrix.setdefault(i, dict())
            N[i] += 1
            for j in items:
                itemMatrix[i].setdefault(j, 0)
                # 若aid i与j同时出现，则将物品相似度矩阵i行j列递增1，否则不存储于稀疏矩阵中
                itemMatrix[i][j] += 1
    
    return itemMatrix, N


def ItemSimilarityMatrix_fn(ItemMatrix, N):
    
    itemSimMatrix = defaultdict(int)
    # cosine similarity
    for i, related_items in tqdm(ItemMatrix.items()):
        itemSimMatrix.setdefault(i, dict())
        for j, cij in related_items.items():
            itemSimMatrix[i].setdefault(j, 0)
            itemSimMatrix[i][j] = cij / math.sqrt(N[i] * N[j])
    
    # normalization
    for i, relations in tqdm(itemSimMatrix.items()):
        max_num = relations[max(relations, key=relations.get)]
        if max_num == 0:
            continue
        itemSimMatrix[i] = {k : v / max_num for k, v in relations.items()}
    
    return itemSimMatrix


def recommend(trainData, itemSimMatrix, user, popularity):

    recommends = dict()
    items = trainData[user]
    for item in items:
        # for every item in session, get top 100 similarity scores
        for i, sim in sorted(itemSimMatrix[item].items(), key=itemgetter(1), reverse=True)[:100]:
            recommends.setdefault(i, 0.)
            recommends[i] += sim
    # sort and return top 20
    result = list(dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:20]).keys())
    if len(result) < 20:
        result = result + popularity # if num of items < 20, use most popular items overall
        result = result[:20]
    
    return result