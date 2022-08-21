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
import pickle

ET_PATH = 'model_ET.bin'
ET_TFIDF_PATH = 'model_ET_tfidf.bin'

def make_colors_dict():    
    f = codecs.open('russian_colors.txt', 'r', "utf-8")
    rus_words = [line.strip() for line in f]
    f1 = open('english_colors.txt', 'r')
    eng_words = [line.strip() for line in f1]

    colors = dict(zip(eng_words, rus_words))
    return colors

def make_tokens(input_str):
    # отбрасываем небуквенные и нецифровые символы и разбиваем строку на отдельные слова
    reg = re.compile('[^a-zа-я0-9 ]')
    tokens = input_str.lower().replace("/", " ").replace("-", " ").replace("\t", " ").replace("pro", " pro")
    tokens = reg.sub(' ', tokens).replace("ё", "е").replace("ghz", "ггц").replace("gb", "гб")
    tokens = tokens.split(" ")
    colors = make_colors_dict()
    new_tokens = []
    for i in tokens:
        # перевод всех цветов на русский
        if i in colors.keys():
            i = colors[i]
        if len(i) > 1 or i.isdigit():
            # разбиваем буквенно-цифровые токены на цифровые и буквенные
            new_tokens += re.findall(r'[a-zа-я]+', i)
            new_tokens += re.findall(r'\d+', i)
    return new_tokens

def train_T(agora_data_prime):
    etalonsname = {}
    etalonsprops = {}
    for index, row in agora_data_prime.iterrows():
        tmp = make_tokens(row['name'])
        etalonsname[agora_data_prime.product_id[index]] = set(tmp)
        tmp = make_tokens(' '.join(row['props']))
        etalonsprops[agora_data_prime.product_id[index]] = set(tmp)

    with open('etalons.bin', "wb") as f:
        pickle.dump(etalonsname, f)
    with open('etalonsprops.bin', "wb") as f:
        pickle.dump(etalonsprops, f)

# tokenization model
def model_T(data_goods):
    Xname = {}
    Xprops = []
    for index, row in data_goods.iterrows():
        tmp = make_tokens(row['name'])
        Xname[data_goods.product_id[index]] = set(tmp)
        tmp = make_tokens(' '.join(row['props']))
        Xprops.append(set(tmp))

    etalonsname = {}
    etalonsprops = {}
    with open('etalons.bin', "rb") as f:
        etalonsname = pickle.load(f)
    with open('etalonsprops.bin', "rb") as f:
        etalonsprops = pickle.load(f)
 
    ans = []
    for i, j in zip(Xname.values(), Xprops):
        name_matches = 0 # число совпадений в имени
        props_matches = 0 # число совпадений в свойствах
        tmp_ans = 0
        for k, l, m in zip(etalonsname.keys(), etalonsname.values(), etalonsprops.values()):
            tmp_name_matches = len(i & set(l))
            tmp_props_matches = len(j & set(m))
            if tmp_name_matches + tmp_props_matches > name_matches + props_matches or \
               tmp_name_matches + tmp_props_matches == name_matches + props_matches and tmp_props_matches > props_matches:
                    name_matches = tmp_name_matches
                    props_matches = tmp_props_matches
                    tmp_ans = k
        # если имя товара совпадает менее чем на четверть с наиболее подходящим эталоном,
        # то не выбираем эталон вовсе
        if name_matches / len(etalonsname[tmp_ans]) < 0.25:
            ans.append('0')
        # получаем id эталона с наибольши числом соответствий
        else: ans.append(tmp_ans)
    return data_goods.product_id, ans
    
def prepare_data(agora_data, data_goods):
    X = []
    y = []
    for index, row in data_goods.iterrows():
        tmp = row['name']
        X.append(tmp)
        y.append(row.reference_id)
    X = np.array(X)
    y = np.array(y)
    
    scaler = pickle.load(open(ET_TFIDF_PATH, 'rb'))
    X_vec = scaler.transform(X)
    
    return y, X_vec

def train_ET(agora_data, data_goods):
    y, X_vec = prepare_data(agora_data, data_goods)
    forest = ExtraTreesClassifier()
    forest.fit(X_vec, y)
    filename = 'model_ET1.bin'
    joblib.dump(forest, filename)
    
#     start = time.time()
    
#     t = time.time()-start
#     print('Время:', t, 'Количество:', len(X), 'Скорость:', 100 * t / len(X))
#     print(accuracy_score(y, ans))

def test_ET(forest, data_goods):
    X = data_goods['name'].to_numpy()
    scaler = pickle.load(open(ET_TFIDF_PATH, 'rb'))
    X_vec = scaler.transform(X)
    ans = forest.predict(X_vec)     
    matr = forest.predict_proba(X_vec)
    
    matr[matr < 0.2] = -1
    ids = np.sum(matr<0, axis=1) == matr.shape[1]
    ans[ids] = None
    
    return data_goods.product_id, ans

if __name__=='__main__':
    agora_data = pd.read_json('agora_hack_products.json')
    agora_data = pd.read_csv('Teee.csv')
    # request_data = pd.read_json('request.json')
    data_goods = agora_data#agora_data[agora_data['is_reference'] == False]
    if(sys.argv[1] == 'token'):
        id_product_1, ref_id_1 = model_T(data_goods)
       #id_product_1, ref_id_1 = model_T(request_data)
        print(f'Accuracy token: {100 * accuracy_score(data_goods["reference_id"], ref_id_1):.3f}%')
        res_T = pd.DataFrame({"id":id_product_1, "reference_id":ref_id_1})
    if sys.argv[1] == 'train':    
        train_ET(agora_data, data_goods)
        train_T(agora_data[agora_data['is_reference'] == True])
    if sys.argv[1] == 'test':
        forest = joblib.load('model_ET1.bin')
        id_product_2, ref_id_2 = test_ET(forest, data_goods)
       #id_product_2, ref_id_2 = test_ET(forest, request_data)
        res_ET = pd.DataFrame({"id":id_product_2, "reference_id":ref_id_2})
        print(accuracy_score(data_goods['reference_id'], ref_id_2))
        
