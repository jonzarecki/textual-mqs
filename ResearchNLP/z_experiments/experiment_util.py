import warnings

import numpy as np
import pandas as pd
from libact.base.interfaces import Labeler, QueryStrategy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold

import ResearchNLP.Constants as cn
import ResearchNLP.classification_experts.classification_expert as ce
from ResearchNLP.prediction_models import PredictionModel
from ResearchNLP.util_files import pandas_util, ColumnNames, ExprScores, combinatorics_util
from ResearchNLP.util_files.libact_utils import IdealTextLabeler, TextDataset


def prepare_dataset(data_df=None, tag_col=None, train_validation_split=None, pos_ex_split=None, random_ex_split=None):
    # type: (pd.DataFrame, str, float, float, float) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """

    :param data_df: all the tagged examples
    :param tag_col: the name of the tag column
    :param train_validation_split: the % of the data going into training (all other goes to validation)
    :param pos_ex_split: the % of positive examples in our data distribution
    :param random_ex_split: the % of the data extracted from training going as 'random examples from distribution'
    :return: (base_training_df, random_examples_df, validation_data_df) as a Tuple
    """
    # default parameter init
    data_df = data_df if data_df is not None else cn.data_df
    tag_col = tag_col if tag_col is not None else cn.tag_col
    train_validation_split = train_validation_split if train_validation_split is not None else cn.train_validation_split
    pos_ex_split = pos_ex_split if pos_ex_split is not None else cn.pos_ex_split
    random_ex_split = random_ex_split if random_ex_split is not None else cn.random_distribution_split

    ############ function start ###############
    train_data_df, validation_data_df = \
        pandas_util.split_dataset(data_df, train_validation_split)  # split the data into training and validation

    # change the training data to t% positive examples, 100-t% negative examples
    train_data_df, pos_rest = pandas_util.imbalance_dataset(train_data_df, tag_col, pos_ex_split,
                                                            cn.pos_tags, cn.neg_tags, shuffle=True, return_rest=True)

    base_training_df, random_examples_df = \
        pandas_util.split_dataset(train_data_df, random_ex_split)  # some part is used to enrich the training

    print "Training : "
    print base_training_df.groupby(tag_col).size()
    print "Validation : "
    print validation_data_df.groupby(tag_col).size()
    print "Random Distribution : "
    print random_examples_df.groupby(tag_col).size()
    print "Pos Rest: "
    print pos_rest.groupby(tag_col).size()

    # make it accessible for the test_data gain heuristic
    cn.base_training_df, cn.random_examples_df, cn.validation_data_df, cn.pos_rest = \
        base_training_df, random_examples_df, validation_data_df, pos_rest

    return base_training_df, random_examples_df, validation_data_df


def prepare_balanced_dataset(data_df=None, tag_col=None, print_expert_acc=False):
    # type: (pd.DataFrame, str, bool) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """

    :param data_df: all the tagged examples
    :param tag_col: the name of the tag column
    :return: (base_training_df, random_examples_df, validation_data_df) as a Tuple
    """
    # default parameter init
    data_df = data_df if data_df is not None else cn.data_df
    tag_col = tag_col if tag_col is not None else cn.tag_col

    ############ function start ###############
    if cn.balance_dataset:
        data_df = pandas_util.imbalance_dataset(data_df, tag_col, cn.init_balance, cn.pos_tags, cn.neg_tags)
    train_data_df, validation_data_df = \
        pandas_util.split_dataset(data_df, 0.6)  # split the data into training and validation

    balanced_train_df = train_data_df[:cn.init_pool_size].copy(deep=True)
    pool_df = train_data_df.iloc[cn.init_pool_size:].copy(deep=True)

    print "Training : "
    print balanced_train_df.groupby(tag_col).size()
    print "Validation : "
    print validation_data_df.groupby(tag_col).size()

    # make it accessible for the test_data gain heuristic
    cn.base_training_df, cn.pool_df, cn.validation_data_df = \
        balanced_train_df, pool_df, validation_data_df

    if print_expert_acc:
        calculate_machine_experts_accuracy()

    return balanced_train_df, validation_data_df


def calculate_machine_experts_accuracy():
    cv = KFold(n_splits=8, shuffle=True, random_state=42)
    total_acc = 0.0
    for train_index, test_index in cv.split(cn.data_df):
        train_df = cn.data_df.iloc[train_index].copy(deep=True)
        test_df = cn.data_df.iloc[test_index].copy(deep=True)
        unlabeled_test_df = test_df.copy(deep=True)
        pandas_util.switch_df_tag(unlabeled_test_df, cn.col_names.tag, 0.0, None)
        pandas_util.switch_df_tag(unlabeled_test_df, cn.col_names.tag, 1.0, None)
        from ResearchNLP.classification_experts import MachineExpert
        clsExpert = MachineExpert(cn.col_names, cn.relevant_tags, train_df)
        labeled_test_df = clsExpert.classify_df(unlabeled_test_df)
        total_acc += accuracy_score(test_df[cn.col_names.tag].values, labeled_test_df[cn.col_names.tag].values)
    print "expert's CV test accuracy", total_acc / cv.n_splits


def prepare_pool_based_dataset(data_df=None, tag_col=None, print_expert_acc=False):
    # type: (pd.DataFrame, str, bool) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
    """
    :param data_df: all the tagged examples
    :param tag_col: the name of the tag column
    :return: (base_training_df, pool_df, validation_data_df) as a Tuple
    """
    # default parameter init
    data_df = data_df if data_df is not None else cn.data_df
    tag_col = tag_col if tag_col is not None else cn.tag_col

    ############ function start ###############
    if cn.balance_dataset:
        data_df = pandas_util.imbalance_dataset(data_df, tag_col, cn.init_balance, cn.pos_tags, cn.neg_tags)
    train_data_df, validation_data_df = \
        pandas_util.split_dataset(data_df, 0.6)  # split the data into training and validation
    # validation_data_df = validation_data_df.iloc[:1000]

    base_training_df = train_data_df[:cn.init_pool_size].copy(deep=True)
    pool_df = train_data_df.iloc[cn.init_pool_size:].copy(deep=True)

    labeled_pool_df = pool_df.copy(deep=True)
    pandas_util.switch_df_tag(pool_df, tag_col, 1.0, None)
    pandas_util.switch_df_tag(pool_df, tag_col, 0.0, None)

    print "Training : "
    print base_training_df.groupby(tag_col).size()
    print "Pool orig labels:"
    print labeled_pool_df.groupby(tag_col).size()
    print "Validation : "
    print validation_data_df.groupby(tag_col).size()
    # make it accessible for the test_data gain heuristic
    cn.base_training_df, cn.pool_df, cn.labeled_pool_df, cn.validation_data_df = \
        base_training_df, pool_df, labeled_pool_df, validation_data_df

    if print_expert_acc:
        calculate_machine_experts_accuracy()

    return base_training_df, pool_df, validation_data_df


def label_df_with_expert(unlabeled_df, col_names, print_status=False):
    # type: (pd.DataFrame, ColumnNames) -> pd.DataFrame
    """
    add to the training new generated examples, classified by expert
    :param unlabeled_df: the df we want to label
    :param col_names: namedtuple containing all column names
    :return: a DataFrame with the label examples
    """
    if len(unlabeled_df) == 0:
        return unlabeled_df.copy(deep=True)  # return another empty df

    if cn.use_human_expert:  # classify new sentences with human expert
        ce.load_human_expert(col_names, cn.relevant_tags, cn.pos_statement, cn.neg_statement)
    else:  # classify new sentences with machine expert
        ce.load_machine_expert(col_names, cn.relevant_tags, cn.data_df)

    labeled_df = ce.clsExpert.classify_df(unlabeled_df)
    assert (labeled_df[col_names.text] == unlabeled_df[col_names.text]).all()


    return pandas_util.all_positive_rows_df(labeled_df, col_names.tag, cn.relevant_tags)  # extract only relevant rows


def prepare_classifier(train_data_df, validation_data_df, col_names):
    # type: (pd.DataFrame, pd.DataFrame, ColumnNames) -> (PredictionModel, np.ndarray, np.ndarray)

    # build the feature extractor
    extractor = cn.Feature_Extractor(train_data_df, col_names)  # this is actually faster than DataFrame.append()

    # extract all the features
    combined_df = pd.concat([train_data_df, validation_data_df])
    X_all = extractor.transform(combined_df, col_names)  # save time by using fit_transform

    X_train = X_all[:len(train_data_df[col_names.text])]
    y_train = train_data_df[col_names.tag].tolist()
    X_test = X_all[len(train_data_df[col_names.text]):]
    y_test = validation_data_df[col_names.tag]

    model = cn.Inner_PredictionModel(X_train, y_train)  # expert model trains on all data (makes it an expert)

    return model, X_test, np.array(y_test, dtype=int)


def run_classifier(training_df, validation_data_df):
    # type: (pd.DataFrame, pd.DataFrame) -> ExprScores
    """
    build a classfier from the data, test it on the validation data
    :param training_df: the basic dataset we will train on
    :param validation_data_df: validation set to check accuracy
    :return: returns Constants.py's ScoresList
    """

    model, X_test, y_test = prepare_classifier(training_df, validation_data_df, cn.col_names)
    y_pred = model.train_model_and_predict(X_test)

    y_pred = y_pred.astype(float)  # bugfix
    y_test = y_test.astype(float)
    try:
        roc_auc = roc_auc_score(y_test, y_pred)
    except ValueError:
        warnings.warn("ROC_AUC not defined", RuntimeWarning)
        roc_auc = 0.0
    return ExprScores(f1_score(y_test, y_pred), roc_auc, accuracy_score(y_test, y_pred))


def run_active_learning(trn_ds, score_function, lbr, qs, quota):
    # type: (TextDataset, callable, Labeler, QueryStrategy, int) -> (list, list)
    E_out = []
    query_num = np.arange(0, min(quota, len(trn_ds.get_unlabeled_entries())) + 1)

    E_out = np.append(E_out, score_function(trn_ds))
    for _ in range(quota):
        if len(trn_ds.get_unlabeled_entries()) == 0:
            break  # finished labeling all examples

        if callable(qs):
            ask_id = qs(trn_ds)
        else:
            ask_id = qs.make_query()
        lb = lbr.label(trn_ds.extract_sentence(ask_id))
        trn_ds.update(ask_id, lb)

        E_out = np.append(E_out, score_function(trn_ds))

    return query_num.tolist(), E_out.tolist()


def assert_ends_and_beginnings_are_the_same(data):
    for i, j in combinatorics_util.get_all_possible_matchings(range(0, len(data)), range(0, len(data))):
        assert data[i][-1] == data[j][-1], \
            "scores should have same end as they insert the same data" + str(i + 1) + " " + str(j + 1)
    assert data[i][0] == data[j][0], \
        "scores should have same beggining as they insert the same data" + str(i + 1) + " " + str(j + 1)


def prepare_trn_ds(balanced_train_df, generated_pool_df, labeled_pool_df):
    enriched_train_df = pd.concat([balanced_train_df, generated_pool_df], ignore_index=True)

    extractor = cn.Feature_Extractor(enriched_train_df, cn.col_names)  # build the feature extractor
    trn_ds = TextDataset(enriched_train_df, cn.col_names, extractor)

    ideal_df = pd.concat([balanced_train_df, labeled_pool_df], ignore_index=True)
    lbr = IdealTextLabeler(TextDataset(ideal_df, cn.col_names, extractor))
    return trn_ds, lbr, extractor


def check_all_train_data_accuracy():
    balanced_train_df, validation_data_df = prepare_balanced_dataset()
    all_train_df = pd.concat([balanced_train_df, cn.pool_df], ignore_index=True)
    return run_classifier(all_train_df, validation_data_df).acc


def check_initial_train_data_accuracy():
    balanced_train_df, validation_data_df = prepare_balanced_dataset()
    return run_classifier(balanced_train_df, validation_data_df).acc
