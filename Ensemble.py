from re import T
import arcNet
import TwoModels
import joblib
from time import time
import numpy as np

def extract_predictions(data):
    data_goods = data#data[data['is_reference']==False]

    start_ET = time()
    ET_classifier = joblib.load('model_ET.bin')
    _, ET_predictions = TwoModels.test_ET(ET_classifier, data_goods)

    print(f'ET {time()-start_ET} {100 * (time()-start_ET)/data_goods.shape[0]:.3f}')

    start_T = time()
    _, Token_predictions = TwoModels.model_T(data_goods)

    print(f'T {time()-start_T} {100 * (time()-start_T)/data_goods.shape[0]:.3f}')

    start_N = time()
    net = arcNet.load_net()
    net.eval()
    net.emb_net.to('cpu')

    scaler = arcNet.load_sklearn_model(arcNet.TFIDF_PATH)
    head = arcNet.load_sklearn_model(arcNet.KNN_PATH)

    Net_predictions = arcNet.arc_predict(data_goods['name'], data_goods['props'], scaler, net, head)
    print(f'N {time()-start_N}')

    return ET_predictions, Token_predictions, Net_predictions

if __name__ == '__main__':
    import pandas as pd
    data = pd.read_json('agora_hack_products.json')
    data_goods = data[data['is_reference']==False]

    #train_t = pd.read_csv('TTT.csv')

    #TwoModels.train_ET(train_t, train_t)

    from time import time

    data = pd.read_csv('Teee.csv')
    data_goods = data
    start = time()
    ET_preds, Token_preds, NN_preds = extract_predictions(data)

    voting = pd.DataFrame(index=data_goods['product_id'])
    voting['ET'] = ET_preds
    voting['Token'] = Token_preds
    voting['NN'] = NN_preds

    check = voting.mode(axis='columns')
    second_p = check[1].fillna(0)
    modes = check[0]
    modes[second_p!=0] = 'null'

    print(TwoModels.accuracy_score(modes, data_goods['reference_id']))
    print(f'Time {time() - start} 100: {100 * (time()-start)/data.shape[0]}')
