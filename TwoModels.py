import json
import pandas as pd
import numpy as np
import codecs
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier

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

# tokenization model
def model_T(agora_data, data_goods):
    agora_data_goods = data_goods
    agora_data_prime = agora_data[agora_data['is_reference'] == True]

    Xname = {}
    Xprops = []
    for index, row in agora_data_goods.iterrows():
        tmp = make_tokens(row['name'])
        Xname[agora_data_goods.product_id[index]] = set(tmp)
        tmp = make_tokens(' '.join(row['props']))
        Xprops.append(set(tmp))

    etalonsname = {}
    etalonsprops = []
    for index, row in agora_data_prime.iterrows():
        tmp = make_tokens(row['name'])
        etalonsname[agora_data_prime.product_id[index]] = set(tmp)
        tmp = make_tokens(' '.join(row['props']))
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
        # если имя товара или свойства товара совпадают менее чем на порог,
        # то не выбираем эталона
        if name_matches / len(i) < 0.2 or props_matches / len(j) < 0.2:
            ans.append('0')
        # иначе получаем id эталона с наибольши числом соответствий
        else: ans.append(tmp_ans)
    return agora_data_goods.product_id, ans
    
# model with ExtraTree
def model_ET(agora_data, data_goods):
    agora_data_goods = data_goods
    agora_data_prime = agora_data[agora_data['is_reference'] == True]
    
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
    
    forest = ExtraTreesClassifier()
    forest.fit(X_vec, y)
    ans = forest.predict(X_vec)
    return agora_data_goods.product_id, ans

def main(json_file):
    data_goods = pd.read_json(json_file)
    agora_data = pd.read_json('agora_hack_products.json')
    
    id_product_1, ref_id_1 = model_T(agora_data, data_goods)
    id_product_2, ref_id_2 = model_ET(agora_data, data_goods)
    
    res_T = pd.DataFrame({"id":id_product_1, "reference_id":ref_id_1})
    res_ET = pd.DataFrame({"id":id_product_2, "reference_id":ref_id_2})
    
    result = res.to_json(orient="records")
    parsed = json.loads(result)
    with open("result.json", "w") as file:
        json.dump(parsed, file, indent=4)
    file.close()

goods = input()
main(goods)




