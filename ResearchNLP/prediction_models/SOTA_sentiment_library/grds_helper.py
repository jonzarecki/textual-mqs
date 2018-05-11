import os

from sklearn.externals import joblib

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
model = None
encoder = None
sentiment_dict = dict()


def prepare_model():
    global model, encoder
    if encoder is None:
        from encoder import Model
        encoder = Model()
    if model is None:
        from ResearchNLP import Constants as cn
        import config
        model_path = config.CODE_DIR + 'prediction_models/SOTA_sentiment_library/model/' + cn.data_name + '.pkl'

        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:  # load model and save it
            from ResearchNLP.prediction_models.SOTA_sentiment_library.utils import train_model, test_model
            import pandas as pd
            train_df = pd.concat([cn.base_training_df, cn.pool_df])
            trX = train_df[cn.col_names.text].values
            trY = train_df[cn.col_names.tag].values
            tstX = cn.validation_data_df[cn.col_names.text].values
            tstY = cn.validation_data_df[cn.col_names.tag].values
            trXt = encoder.transform(trX)
            tstXt = encoder.transform(tstX)
            model = train_model(trXt, trY, tstXt, tstY)  # train on all data
            print test_model(model, tstXt, tstY)
            joblib.dump(model, model_path)


def predict_sentiment(sent):
    return predict_sentiment_bulk([sent])[0]


def predict_sentiment_bulk(sents):
    unknown_sents = filter(lambda sent: sent not in sentiment_dict, sents)
    if len(unknown_sents) != 0:
        predicted_sentiments = actual_predict(unknown_sents)
        # manager = multiprocessing.Manager()
        # return_dict = manager.dict()
        # fail_count = 0
        # while True:
        #     try:
        #         p = multiprocessing.Process(
        #             target=lambda sents, return_dict: return_dict.__setitem__(1, actual_predict(unknown_sents)),
        #             args=(sents, return_dict))
        #         p.start()
        #         p.join()
        #         predicted_sentiments = return_dict[1]
        #         manager.shutdown()
        #         break  # success
        #     except:
        #         fail_count += 1
        #         print "waiting for GPU"
        #         time.sleep(1)
        #         if fail_count == 5:
        #             raise  # failure, re-raise
        sentiment_dict.update(zip(unknown_sents, predicted_sentiments))

    return map(lambda sent: sentiment_dict[sent], sents)


def actual_predict(sents):
    prepare_model()
    feats = encoder.transform(sents + ['i love you', 'i hate you'])[:-2]
    probas = model.predict_proba(feats)
    return map(lambda (neg_prob, pos_prob): 1.0 if pos_prob >= 0.5 else 0.0, probas)