import csv
import os

import pandas as pd

from ResearchNLP.feature_extraction import FeatureExtractor, AvgGloveExtractor, BOWExtractor
from ResearchNLP.prediction_models import PredictionModel, SOTASentimentModel, LinearModel
from ResearchNLP.util_files import pandas_util
from ResearchNLP.util_files.column_names import ColumnNames, prepare_df_columns

data_folder_abspath = os.path.join(os.path.join(os.path.dirname(__file__), "experiment_data"))


def add_experiment_param(param):
    # type: (str) -> None
    global curr_experiment_params, curr_experiment_params_list
    if param not in curr_experiment_params_list:
        curr_experiment_params_list.append(param)
        curr_experiment_params = os.path.join(curr_experiment_params, param)


def _load_search_algorithm_params():
    global search_alg_code, search_iter_lim, bm_size, ss_type
    from ResearchNLP.text_synthesis.heuristic_functions import SynStateUncertainty
    _load_generic_params()
    search_alg_code = "SWHC"
    search_iter_lim = 4
    bm_size = 8
    ss_type = SynStateUncertainty


def _load_generic_params():
    global curr_experiment_params, curr_experiment_params_list, total_mod_ops, inst_count, balance_dataset, sem_dist, \
        Expert_PredictionModel, Expert_FeatureExtractor, Inner_PredictionModel, Feature_Extractor, experiment_purpose, \
        init_pool_size, init_balance

    experiment_purpose = ""
    curr_experiment_params = ""
    curr_experiment_params_list = list()
    total_mod_ops = 0
    inst_count = 0
    sem_dist = 1
    balance_dataset = True
    init_pool_size = 25
    init_balance = 0.5

    Inner_PredictionModel = LinearModel
    Feature_Extractor = AvgGloveExtractor
    Expert_PredictionModel = SOTASentimentModel
    Expert_FeatureExtractor = Feature_Extractor


def _load_csv_from_file(_data_csv_delimiter, _data_csv_quoting, _data_csv_encoding, _data_name, file_name='all_data.csv'):
    """
    load the data_df csv file, use parameters given for the function
    """
    global data_csv_delimiter, data_csv_quoting, data_csv_encoding, data_name, filename, data_df, \
        tok_word_pool, experiment_purpose

    data_csv_delimiter = _data_csv_delimiter
    data_csv_quoting = _data_csv_quoting
    data_csv_encoding = _data_csv_encoding
    data_name = _data_name
    filename = os.path.join(data_folder_abspath, data_name, file_name)
    # load csv from file
    data_df = pd.read_csv(filename, header=None, delimiter=data_csv_delimiter, quoting=data_csv_quoting,
                          encoding=data_csv_encoding, dtype=object)
    data_df.columns = [tag_col, text_col]
    data_df = prepare_df_columns(data_df, col_names)
    data_df = pandas_util.shuffle_df_rows(data_df)  # SHUFFLES results, put after load



def _load_default_sentiment_analysis_params(d_measure=None):
    """
    load all default sentiment analysis parameters
    """
    global train_validation_split, pos_ex_split, pos_ex_split, random_distribution_split, \
        col_names, tag_col, text_col, tok_text_col, feature_repr_col, prev_states_col, sent_sim_repr_col, \
        pos_tags, neg_tags, relevant_tags, pos_statement, neg_statement, use_human_expert, choose_sword_with_expert, \
        distance_measure, switch_word_column
    train_validation_split = 0.6
    pos_ex_split = 0.05
    random_distribution_split = 0.80
    tag_col = "Sentiment"
    text_col = "Text"
    tok_text_col = "Tokenized Text"
    feature_repr_col = "Feature Representation"
    prev_states_col = "Previous States"
    sent_sim_repr_col = "Similarity Representation"
    col_names = ColumnNames(text_col, tag_col, tok_text_col, feature_repr_col, prev_states_col, sent_sim_repr_col)
    pos_tags = [1.0]
    neg_tags = [0.0]
    relevant_tags = [1.0, 0.0]
    pos_statement = 'Positive Sentiment'
    neg_statement = 'Negative Sentiment'
    use_human_expert = False
    choose_sword_with_expert = False
    distance_measure = d_measure if d_measure is not None else 5  # number of words in each distance unit
    add_experiment_param("d_measure_" + str(distance_measure))
    switch_word_column = 'switch_word'


def _load_default_subjectivity_params(d_measure=None):
    """
    load all default subjectivity parameters
    """
    global train_validation_split, pos_ex_split, pos_ex_split, random_distribution_split, \
        col_names, tag_col, text_col, tok_text_col, feature_repr_col, prev_states_col, sent_sim_repr_col, \
        pos_tags, neg_tags, relevant_tags, pos_statement, neg_statement, use_human_expert, choose_sword_with_expert, \
        distance_measure, switch_word_column
    train_validation_split = 0.6
    pos_ex_split = 0.05
    random_distribution_split = 0.80
    tag_col = "Subjectivity"
    text_col = "Text"
    tok_text_col = "Tokenized Text"
    feature_repr_col = "Feature Representation"
    prev_states_col = "Previous States"
    sent_sim_repr_col = "Similarity Representation"
    col_names = ColumnNames(text_col, tag_col, tok_text_col, feature_repr_col, prev_states_col, sent_sim_repr_col)
    pos_tags = [1.0]
    neg_tags = [0.0]
    relevant_tags = [1.0, 0.0]
    pos_statement = 'Subjective'
    neg_statement = 'Objective'
    use_human_expert = False
    choose_sword_with_expert = False
    distance_measure = d_measure if d_measure is not None else 5  # number of words in each distance unit
    add_experiment_param("d_measure_" + str(distance_measure))
    switch_word_column = 'switch_word'


def _load_default_hate_speech_params(d_measure=None):
    """
    load all default subjectivity parameters
    """
    global train_validation_split, pos_ex_split, pos_ex_split, random_distribution_split, \
        col_names, tag_col, text_col, tok_text_col, feature_repr_col, prev_states_col, sent_sim_repr_col, \
        pos_tags, neg_tags, relevant_tags, pos_statement, neg_statement, use_human_expert, choose_sword_with_expert, \
        distance_measure, switch_word_column
    train_validation_split = 0.6
    pos_ex_split = 0.05
    random_distribution_split = 0.80
    tag_col = "Hate speech"
    text_col = "Text"
    tok_text_col = "Tokenized Text"
    feature_repr_col = "Feature Representation"
    prev_states_col = "Previous States"
    sent_sim_repr_col = "Similarity Representation"
    col_names = ColumnNames(text_col, tag_col, tok_text_col, feature_repr_col, prev_states_col, sent_sim_repr_col)
    pos_tags = [1.0]
    neg_tags = [0.0]
    relevant_tags = [1.0, 0.0]
    pos_statement = 'Hate speech / offensive language'
    neg_statement = 'Neutral'
    use_human_expert = False
    choose_sword_with_expert = False
    distance_measure = d_measure if d_measure is not None else 5  # number of words in each distance unit
    add_experiment_param("d_measure_" + str(distance_measure))
    switch_word_column = 'switch_word'


def _load_default_spam_params(d_measure=None):
    """
    load all default sentiment analysis parameters
    """
    global train_validation_split, pos_ex_split, pos_ex_split, random_distribution_split, \
        col_names, tag_col, text_col, tok_text_col, feature_repr_col, prev_states_col, sent_sim_repr_col, \
        pos_tags, neg_tags, relevant_tags, pos_statement, neg_statement, use_human_expert, choose_sword_with_expert, \
        distance_measure, switch_word_column
    train_validation_split = 0.6
    pos_ex_split = 0.03
    random_distribution_split = 0.80
    tag_col = "label"
    text_col = "Text"
    tok_text_col = "Tokenized Text"
    feature_repr_col = "Feature Representation"
    prev_states_col = "Previous States"
    sent_sim_repr_col = "Similarity Representation"
    col_names = ColumnNames(text_col, tag_col, tok_text_col, feature_repr_col, prev_states_col, sent_sim_repr_col)
    pos_tags = [1.0]
    neg_tags = [0.0]
    relevant_tags = [1.0, 0.0]
    pos_statement = 'Spam'
    neg_statement = 'Non Spam'
    use_human_expert = False
    choose_sword_with_expert = False
    distance_measure = d_measure if d_measure is not None else 5  # number of words in each distance unit
    switch_word_column = 'switch_word'


def load_codementor_sentiment_analysis_parameters(d_measure=None):
    """
    loads all saved parameters for sentiment analysis, and saves them in the global space
    """
    _load_search_algorithm_params()
    _load_default_sentiment_analysis_params(d_measure)
    global pos_ex_split, init_pool_size
    pos_ex_split = 0.06
    init_pool_size = 10
    data_name = "codementor_SA"
    add_experiment_param(data_name)
    data_csv_delimiter = "\t"
    data_csv_quoting = 3
    data_csv_encoding = 'utf-8'
    _load_csv_from_file(data_csv_delimiter, data_csv_quoting, data_csv_encoding, data_name)


def load_stanford_sentiment_treebank_parameters(d_measure=None):
    """
    loads all saved parameters for sentiment analysis, and saves them in the global space
    """
    _load_search_algorithm_params()
    _load_default_sentiment_analysis_params(d_measure)
    global pos_ex_split, balance_dataset, init_pool_size
    pos_ex_split = 0.06
    # balance_dataset = False
    init_pool_size = 40


    data_name = "stanford_SA"
    add_experiment_param(data_name)
    data_csv_delimiter = ","
    data_csv_quoting = csv.QUOTE_MINIMAL
    data_csv_encoding = 'utf-8'
    _load_csv_from_file(data_csv_delimiter, data_csv_quoting, data_csv_encoding, data_name)


def load_uci_spam_parameters(d_measure=None):
    """
    loads all saved parameters for sentiment analysis, and saves them in the global space
    """
    _load_search_algorithm_params()
    _load_default_spam_params(d_measure)

    data_name = "UCI_spam"
    add_experiment_param(data_name)
    data_csv_delimiter = ","
    data_csv_quoting = csv.QUOTE_MINIMAL
    data_csv_encoding = 'latin-1'
    _load_csv_from_file(data_csv_delimiter, data_csv_quoting, data_csv_encoding, data_name)

    global Expert_PredictionModel
    Expert_PredictionModel = LinearModel


mtl_options = ["apparel", "dvd", "magazines", "sports_outdoors",
               "baby", "electronics", "MR", "toys_games",
               "books", "health_personal_care", "music", "video",
               "camera_photo", "imdb", "kitchen_housewares", "software"]


def load_mtl_16_sentiment_dataset_parameters(d_measure=None, genre="books"):
    """
    loads all saved parameters for sentiment analysis, and saves them in the global space
    """
    _load_search_algorithm_params()
    _load_default_sentiment_analysis_params(d_measure)

    data_name = os.path.join("mtl-dataset", genre)
    add_experiment_param(data_name)
    global pos_ex_split
    pos_ex_split = 0.03

    data_csv_delimiter = "\t"
    data_csv_quoting = 3
    data_csv_encoding = 'utf-8'
    _load_csv_from_file(data_csv_delimiter, data_csv_quoting, data_csv_encoding, data_name)


def load_cornell_sentiment_parameters(d_measure=None):
    """
    loads all saved parameters for sentiment analysis, and saves them in the global space
    """
    global pos_ex_split, init_pool_size
    _load_search_algorithm_params()
    _load_default_sentiment_analysis_params(d_measure)
    pos_ex_split = 0.01
    init_pool_size = 20

    data_name = "cornell-sent-polarity"
    add_experiment_param(data_name)
    data_csv_delimiter = "\t"
    data_csv_quoting = 3
    data_csv_encoding = 'latin-1'
    _load_csv_from_file(data_csv_delimiter, data_csv_quoting, data_csv_encoding, data_name)


def load_customer_reviews_parameters(d_measure=None):
    """
    loads all saved parameters for sentiment analysis, and saves them in the global space
    """
    global pos_ex_split, init_pool_size
    _load_search_algorithm_params()
    _load_default_sentiment_analysis_params(d_measure)
    pos_ex_split = 0.01
    init_pool_size = 20

    data_name = "customer-reviews"
    add_experiment_param(data_name)
    data_csv_delimiter = "\t"
    data_csv_quoting = 3
    data_csv_encoding = 'latin-1'
    _load_csv_from_file(data_csv_delimiter, data_csv_quoting, data_csv_encoding, data_name)


def load_cornell_subjectivity_parameters(d_measure=None):
    """
    loads all saved parameters for subjectivity analysis, and saves them in the global space
    """
    global pos_ex_split, init_pool_size, Expert_PredictionModel, Expert_FeatureExtractor
    _load_search_algorithm_params()
    _load_default_subjectivity_params(d_measure)
    pos_ex_split = 0.01
    init_pool_size = 20
    # Expert_PredictionModel = LinearModel
    # Expert_FeatureExtractor = BOWExtractor

    data_name = "cornell-sent-subjectivity"
    add_experiment_param(data_name)
    data_csv_delimiter = "\t"
    data_csv_quoting = 3
    data_csv_encoding = 'latin-1'
    _load_csv_from_file(data_csv_delimiter, data_csv_quoting, data_csv_encoding, data_name)


def load_hate_speech_parameters(d_measure=None):
    """
    loads all saved parameters for subjectivity analysis, and saves them in the global space
    """
    global pos_ex_split, Expert_PredictionModel, init_pool_size, init_balance, Expert_FeatureExtractor
    _load_search_algorithm_params()
    _load_default_hate_speech_params(d_measure)
    pos_ex_split = 0.01
    Expert_PredictionModel = LinearModel
    Expert_FeatureExtractor = BOWExtractor
    init_pool_size = 10
    init_balance = 0.6

    data_name = "hate-speech-and-offensive-language"
    add_experiment_param(data_name)
    data_csv_delimiter = ","
    data_csv_quoting = csv.QUOTE_MINIMAL
    data_csv_encoding = 'latin-1'
    _load_csv_from_file(data_csv_delimiter, data_csv_quoting, data_csv_encoding, data_name)


def load_handmade_experiments_parameters(d_measure=None):
    """
    loads all saved parameters for subjectivity analysis, and saves them in the global space
    """
    global pos_ex_split, init_pool_size
    _load_search_algorithm_params()
    _load_default_subjectivity_params(d_measure)
    pos_ex_split = 0.01
    init_pool_size = 20

    data_name = "handmade-experiments"
    add_experiment_param(data_name)
    data_csv_delimiter = "\t"
    data_csv_quoting = 3
    data_csv_encoding = 'latin-1'
    _load_csv_from_file(data_csv_delimiter, data_csv_quoting, data_csv_encoding, data_name)
    global Expert_PredictionModel
    Expert_PredictionModel = LinearModel


#########################################
# list of parameters, so the IDE will recognize them
data_csv_delimiter = "\t"
data_csv_quoting = 3
data_csv_encoding = 'utf-8'
train_validation_split = 0.6
pos_ex_split = 0.07
random_distribution_split = 0.80

filename = os.path.join(os.path.dirname(__file__), 'all_data.csv')
tag_col = "Sentiment"  # tag_column: name of tag column
text_col = "Text"  # text_column: name of text column
tok_text_col = "Tokenized Text"  # tok_text_column: name of tokenized text column
feature_repr_col = "Feature Representation"
prev_states_col = "Previous States"
sent_sim_repr_col = "Similarity Representation"
col_names = ColumnNames(text_col, tag_col, tok_text_col, feature_repr_col, prev_states_col, sent_sim_repr_col)
pos_tags = [1.0]  # pos_tags: list of tags we consider positive
neg_tags = [0.0]  # neg_tags: list of tags we consider negative
relevant_tags = [1.0, 0.0]  # relevant_tags : all the tags we consider as legal
pos_statement = 'Positive Sentiment'
neg_statement = 'Negative Sentiment'
use_human_expert = False  # use_human_expert: stating if we want to use a human for the tagging of new sentences
choose_sword_with_expert = False  # boolean stating if we want to use an expert for switch word
data_df = pd.DataFrame(columns=list(col_names), dtype=object)
distance_measure = 5  # number of words in each distance unit
switch_word_column = 'switch_word'
Inner_PredictionModel = PredictionModel # the prediction model the model that trains on the generated examples uses
Feature_Extractor = FeatureExtractor  # the feature extractor the inner model and machine experts will use
Expert_PredictionModel = PredictionModel  # the prediction model the machine expert will use
Expert_FeatureExtractor = Feature_Extractor
tok_word_pool = []
data_name = "None"  # also folder's name where data is stored
verbose = False  # if to make the code be verbose and add lots of prints
serial_parmap = False
# max_procs = -1
search_alg_code = "None"
search_iter_lim = 3
bm_size = 8
curr_experiment_params = ''
curr_experiment_params_list = list()
total_mod_ops = 0
inst_count = 0
ss_type = type
tmp_expr_foldpath = ""  # placeholder for using the tmp experiment folder to store files (such as CSVs)
experiment_purpose = ""  # saves space for writing the purpose of the new experiment, on the graph and in the output
expr_id = ""  # unique identifier for experiments
furthest_away_reverse = False
sem_dist = -1
qs = None
_load_generic_params()
balance_dataset = True
init_pool_size = 25
init_balance = 0.5
