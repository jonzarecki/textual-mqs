import multiprocessing
import unittest
from functools import partial

import numpy as np
import pandas as pd

from ResearchNLP.feature_extraction import AvgGloveExtractor
from ResearchNLP.prediction_models import SvmModel
from ResearchNLP.text_synthesis.heuristic_functions import SynStateTestDataGain
from ResearchNLP.util_files import pandas_util
from ResearchNLP.util_files.libact_utils.ideal_text_labeler import IdealTextLabeler
from libact.models import LogisticRegression
from libact.query_strategies import UncertaintySampling

from ResearchNLP import Constants as cn
from ResearchNLP.knowledge_bases import kb_helper
from ResearchNLP.util_files.libact_utils.text_dataset import TextDataset
from ResearchNLP.z_experiments.experiment_util import label_df_with_expert, run_classifier, run_active_learning, \
    prepare_balanced_dataset, prepare_pool_based_dataset, assert_ends_and_beginnings_are_the_same, prepare_trn_ds
from ResearchNLP.text_synthesis import sentence_generation as sg
from ResearchNLP.z_experiments.ex_insertion_order import insert_in_AL_fashion


class TestGeneralSystem(unittest.TestCase):
    def test_libact_first_try_results_are_the_same(self):
        """
        test that the first libact example work the same way as the original example taken from github\

        very long test !
        """

        # self.skipTest(reason="too long")

        cn.Inner_PredictionModel = SvmModel
        cn.Feature_Extractor = AvgGloveExtractor
        cn.load_codementor_sentiment_analysis_parameters()
        kb_helper.load_WordNet_model()

        quota = 5  # ask labeler to label 5 samples (tops)
        base_training_df, validation_data_df = prepare_balanced_dataset()
        pos_sents = pandas_util.get_all_positive_sentences(base_training_df, cn.col_names.text,
                                                           cn.col_names.tag, cn.pos_tags)

        # prepare all data
        generated_pool_df = sg.generate_sents_using_random_synthesis(pos_sents, base_training_df, cn.col_names)
        labeled_pool_df = label_df_with_expert(generated_pool_df, cn.col_names)

        enriched_train_df = pd.concat([base_training_df, generated_pool_df], ignore_index=True)
        ideal_df = pd.concat([base_training_df, labeled_pool_df], ignore_index=True)

        extractor = cn.Feature_Extractor(enriched_train_df, cn.col_names)  # build the feature extractor

        lbr = IdealTextLabeler(TextDataset(ideal_df, cn.col_names, extractor))
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        jobs = []
        # first job
        p = multiprocessing.Process(target=self.libact_first_try_first_run, args=(enriched_train_df, extractor,
                                                                                  lbr, quota,
                                                                                  validation_data_df, return_dict))
        jobs.append(p)
        p.start()
        # second job
        p = multiprocessing.Process(target=self.libact_first_try_second_run, args=(enriched_train_df, extractor,
                                                                                   ideal_df, lbr, quota,
                                                                                   validation_data_df, return_dict))
        jobs.append(p)
        p.start()

        for proc in jobs:
            proc.join()

        self.assertTrue(np.array_equal(return_dict[1], return_dict[2]))

    def libact_first_try_second_run(self, enriched_train_df, extractor, ideal_df, lbr, quota, validation_data_df,
                                    return_dict):

        trn_ds = TextDataset(enriched_train_df, cn.col_names, extractor)
        qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
        E_out1 = []
        E_out1 = np.append(E_out1,
                           run_classifier(trn_ds.extract_labeled_dataframe(), validation_data_df).f1)
        for i in range(quota):
            if len(trn_ds.get_unlabeled_entries()) == 0:
                break  # finished labeling all examples
            ask_id = qs.make_query()
            lb = lbr.label(trn_ds.extract_sentence(ask_id))
            self.assertEqual(lb, ideal_df[cn.tag_col][ask_id])
            trn_ds.update(ask_id, lb)
            # model.train(trn_ds)
            E_out1 = np.append(E_out1,
                               run_classifier(trn_ds.extract_labeled_dataframe(), validation_data_df).f1)
        return_dict[2] = E_out1

    def libact_first_try_first_run(self, enriched_train_df, extractor, lbr, quota, validation_data_df, return_dict):

        trn_ds = TextDataset(enriched_train_df, cn.col_names, extractor)
        qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
        scoring_fun = lambda ds: run_classifier(ds.extract_labeled_dataframe(), validation_data_df).f1
        query_num, E_out1 = run_active_learning(trn_ds, scoring_fun, lbr, qs, quota)
        return_dict[1] = E_out1

    def test_heuristic_test_data_gain_works(self):
        cn.load_codementor_sentiment_analysis_parameters()
        kb_helper.load_WordNet_model()

        base_train_df, pool_df, validation_data_df = prepare_pool_based_dataset()
        all_sents = list(base_train_df[cn.col_names.text])
        tns = 2

        pool_name, prep_pools = ("orig pool", lambda *_: (pool_df.iloc[:tns+5], cn.labeled_pool_df.iloc[:tns+5]))
        train_with_pool_df = pd.concat([base_train_df, pool_df], ignore_index=True)
        generated_pool_df, labeled_pool_df = prep_pools(all_sents, train_with_pool_df, cn.col_names, tns)
        cn.experiment_purpose += pool_name + " "

        trn_ds, lbr, extractor = prepare_trn_ds(base_train_df, generated_pool_df, labeled_pool_df)
        final_scoring_fun = partial(lambda en_df: run_classifier(en_df, validation_data_df).acc)

        table_headers = ['#added examples']
        data = [range(0, tns + 1)]
        compared_heuristics = [("test-data-gain", lambda: SynStateTestDataGain), ("random", lambda: "random")]

        for (heuristic_name, prepare_usage) in compared_heuristics:
            ss_type = prepare_usage()
            table_headers.append(heuristic_name)
            print heuristic_name
            _, heur_scores = insert_in_AL_fashion(trn_ds, final_scoring_fun, lbr, ss_type, labeled_pool_df, quota=tns)
            data.append(heur_scores)

        print data[1]
        print data[2]
        self.assertEqual(data[1][0], data[2][0], "starts are same")
        self.assertGreater(data[1][-1], data[2][-1], "test-data-gain should be better than random")


if __name__ == '__main__':
    unittest.main()
