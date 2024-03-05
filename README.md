# DjangoAPI SOLUTION

# Solution

1. Download DjangoAPI.zip, unpack;
2. Go to the `\DjangoAPI` directory;
3. Start the server using the command `python Manage.py runserver`;
4. Now instructions for the specified result of the json file to be sent to the server.
    - Use the HTTP request `POST http://127.0.0.1:8000/match_products`;
    - Receive a json response with predictions in the body;

# Model training

1. Download **TwoModels.py**, **arcNet.py**, **Ensemble.py**, in the same directory `agora_hack_products.json`
2. ExtraTreesClassifier[Classic ML solution] and Tokenizator[Data usage and mapping]:
    1. Run the command `python TwoModels.py train` - as a result, models and additional files will be saved;
    2. Running the command `python TwoModels.py token` will show the result of Tokenizator's work on all data;
    3. Running the command `python TwoModels.py test` will show the result of ExtraTreeClassifier running on all data;
3. ArcNet [Neural network solution]:
    1. Running the command `python arcNet.py` will output the expected parameters as input to the py file;
    2. Running the command `python arcNet.py train [true/false] [true/false] [true/false] [true/false] [true/false]` will train the models and save them to files upon request, also saving additional files;
    3. Running the command `python arcNet.py test [true/false]` will check the operation of the model on test data, and also save the head of the model (KNN-1) if necessary.
    4. Running the command `python arcNet.py check` will show the accuracy and running time on all data;
