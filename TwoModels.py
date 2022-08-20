import json
import pandas as pd
import numpy as np
import codecs
import time
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
import sys
import joblib

def color(rus_file, eng_file):    
    f = codecs.open(rus_file, 'r', "utf-8")
    rus_words = [line.strip() for line in f]
    f1 = open(eng_file, 'r')
    eng_words = [line.strip() for line in f1]

    colors = dict(zip(eng_words, rus_words))
    return colors

def make_tokens(input_str, reg, colors):
    tokens = input_str.lower().replace("/", " ").replace("-", " ").replace("\t", " ").replace("pro", " pro")
    tokens = reg.sub('', tokens).replace("ё", "е").replace("ghz", "ггц").replace("gb", "гб")
    tokens = tokens.split(" ")
    new_tokens = []
    for i in tokens:
        if i in colors.keys():
            i = colors[i]
        new_tokens += re.findall(r'[a-zа-я]+', i)
        new_tokens += re.findall(r'\d+', i)
    return new_tokens

def model_T(agora_data, data_goods):
    agora_data_goods = data_goods
    agora_data_prime = agora_data[agora_data['is_reference'] == True]

    colors = color('russian_colors.txt', 'english_colors.txt')

    Xname = {}
    Xprops = []
    reg = re.compile('[^a-zа-я0-9 ]')
    for index, row in agora_data_goods.iterrows():
        tmp = make_tokens(row['name'], reg, colors)
        Xname[agora_data_goods.product_id[index]] = set(tmp)
        tmp = make_tokens(' '.join(row['props']), reg, colors)
        Xprops.append(set(tmp))

    etalonsname = {}
    etalonsprops = []
    for index, row in agora_data_prime.iterrows():
        tmp = make_tokens(row['name'], reg, colors)
        etalonsname[agora_data_prime.product_id[index]] = set(tmp)
        tmp = make_tokens(' '.join(row['props']), reg, colors)
        etalonsprops.append(set(tmp))

    y = np.array(agora_data_goods.reference_id)

    ans = []
    for i, j in zip(Xname.values(), Xprops):
        comp_name = 0 # число совпадений в имени
        comp_props = 0 # число совпадений в свойствах
        tmp_ans = 0
        for k, l, m in zip(etalonsname.keys(), etalonsname.values(), etalonsprops):
            # если общее число совпадений у двух эталонов равняется
            if len(i & l) + len(m & j) == comp_name + comp_props:
                # выбираем эталон, с которым совпало больше свойств
                if len(m & j) > comp_props:
                    comp_name = len(i & l)
                    comp_props = len(m & j)
                    tmp_ans = k
            if (len(i & l) + len(m & j) > comp_name + comp_props):
                comp_name = len(i & l)
                comp_props = len(m & j)
                tmp_ans = k
        ans.append(tmp_ans)
    
    print(accuracy_score(y, ans))
    return agora_data_goods.id, ans
    
def prepare_data(agora_data, data_goods):
    agora_data_goods = data_goods
    
    X = []
    y = []
    for index, row in agora_data_goods.iterrows():
        tmp = row['name']
        X.append(tmp)
        y.append(row.reference_id)
    X = np.array(X)
    y = np.array(y)
    
    scaler = TfidfVectorizer()
    scaler.fit(X)
    X_vec = scaler.transform(X)
    
    return y, X_vec

def train_ET(agora_data, data_goods):
    y, X_vec = prepare_data(agora_data, data_goods)
    forest = ExtraTreesClassifier()
    forest.fit(X_vec, y)
    filename = 'model_ET.bin'
    joblib.dump(forest, filename)
    
#     start = time.time()
    
#     t = time.time()-start
#     print('Время:', t, 'Количество:', len(X), 'Скорость:', 100 * t / len(X))
#     print(accuracy_score(y, ans))

def test_ET(forest, agora_data, data_goods):
    y, X_vec = prepare_data(agora_data, data_goods)
    ans = forest.predict(X_vec)     
    matr = forest.predict_proba(X_vec)
    
    matr[matr < 0.2] = -1
    ids = np.sum(matr<0, axis=1) == matr.shape[1]
    ans[ids] = None
    
    print(accuracy_score(y, ans))
    return data_goods.id, ans

if __name__=='__main__':
    data_goods = pd.read_json(json_file) # получаем файл с товарами 
    agora_data = pd.read_json('agora_hack_products.json')
    if(sys.argv[1] == 'token'):
        id_product_1, ref_id_1 = model_T(agora_data, data_goods)
        res_T = pd.DataFrame({"id":id_product_1, "reference_id":ref_id_1})
    if sys.argv[1] == 'train':    
        train_ET(agora_data, data_goods)
    if sys.argv[1] == 'test':
        forest = joblib.load('model_ET.bin')
        id_product_2, ref_id_2 = test_ET(forest, agora_data, data_goods)
        res_ET = pd.DataFrame({"id":id_product_2, "reference_id":ref_id_2})
    
    # result = res.to_json(orient="records")
    # parsed = json.loads(result)
    # with open("result.json", "w") as file:
    #     json.dump(parsed, file, indent=4)
    # file.close()
