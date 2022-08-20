import json
import pandas as pd
import numpy as np
import codecs
import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier

def model(agora_data, data_goods):
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
    print(accuracy_score(y, ans))
    return agora_data_goods.product_id, ans

def main(json_file):
    data_goods = pd.read_json(json_file)
    agora_data = pd.read_json('agora_hack_products.json')
    id_product, ref_id = model(agora_data, data_goods)
    
    res_json = pd.DataFrame({"id":id_product, "reference_id":ref_id})
    
    result = res_json.to_json(orient="records")
    parsed = json.loads(result)
    with open("result1.json", "w") as file:
        json.dump(parsed, file, indent=4)
    file.close()
    
goods = input()
main(goods)





