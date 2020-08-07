import cPickle

from ResearchNLP import Constants as cn
from ResearchNLP.text_synthesis.heuristic_functions import find_heuristic
from ResearchNLP.text_synthesis.heuristic_functions.choose_best_for_addition import choose_best_by_heuristic_template
from ResearchNLP.text_synthesis.sentence_generation import generate_sents_using_enhanced_search, \
    generate_sents_using_search_alg, generate_sents_using_random_synthesis
from ResearchNLP.text_synthesis.sentence_generation_special import curr_pool_synthesis_from_sents, \
    generate_lecun_augmentation_from_sents
from ResearchNLP.util_files import pandas_util, prepare_df_columns, combinatorics_util
from ResearchNLP.util_files.column_names import fix_sentences_encoding
from ResearchNLP.util_files.pandas_util import switch_df_tag
from ResearchNLP.z_experiments.experiment_util import label_df_with_expert
from config import CODE_DIR


def prepare_pools_template(pool_name, pool_gen_fun):
    def prepare_pools(base_sents, base_df, col_names, tns):
        gen_pool_df = pool_gen_fun(base_sents, base_df, col_names, total_new_sents=tns)
        labeled_gen_pool_df = label_df_with_expert(gen_pool_df, col_names)
        return gen_pool_df, labeled_gen_pool_df

    return pool_name, prepare_pools


def curr_pool_gen_template(heuristic_name):
    def pool_gen_fun(base_sents, base_df, col_names, total_new_sents):
        ss_type = find_heuristic(heuristic_name)()
        ret_df = curr_pool_synthesis_from_sents(base_sents, base_df, col_names, total_new_sents=total_new_sents,
                                                choose_bulk_method=choose_best_by_heuristic_template(ss_type))
        with open("/home/yonatanz/Dropbox/Research/Interesting results/" + cn.data_name + "_sentences_generated.txt", "a+") as f:
            f.writelines([sent + "\n" for sent in ret_df[cn.text_col]])
        return curr_pool_synthesis_from_sents(base_sents, base_df, col_names, total_new_sents=total_new_sents,
                                              choose_bulk_method=choose_best_by_heuristic_template(ss_type))
    return prepare_pools_template("beam search " + heuristic_name, pool_gen_fun)


def local_search_gen_template(heuristic_name, iter_lim, use_enhanced=False):
    def pool_gen_fun(base_sents, base_df, col_names, total_new_sents):
        ss_type = find_heuristic(heuristic_name)()
        old_ss_type = cn.ss_type
        cn.ss_type = ss_type
        if use_enhanced:
            retval = generate_sents_using_enhanced_search(base_sents, base_df, col_names, cn.search_alg_code,
                                                          iter_lim=iter_lim, total_new_sents=total_new_sents)
        else:
            retval = generate_sents_using_search_alg(base_sents, base_df, col_names, cn.search_alg_code,
                                                     iter_lim=iter_lim, total_new_sents=total_new_sents)
        cn.ss_type = old_ss_type
        return retval
    pool_name = heuristic_name + (" en" if use_enhanced else " ")+"search lim" +str(iter_lim)
    return prepare_pools_template(pool_name, pool_gen_fun)


def generate_pool_using_random_synthesis():
    return prepare_pools_template("random synthesis", generate_sents_using_random_synthesis)


##### Competitors


def generate_pool_lecun_augmentation():
    def lecun_pools_gen_fun(base_sents, base_df, col_names, tns):
        orig_sents_list = cn.data_df[col_names.text].tolist()
        base_sents = filter(lambda s: s in orig_sents_list, base_sents)
        gen_pool_df = generate_lecun_augmentation_from_sents(base_sents, base_df, col_names, total_new_sents=tns)
        gen_origins = map(lambda (i, row): row[col_names.prev_states][-1], gen_pool_df.iterrows())
        orig_sent_tags = map(lambda orig: float(cn.data_df[col_names.tag][orig_sents_list.index(orig)]),
                             gen_origins)  # find original tags
        labeled_gen_pool_df = gen_pool_df.copy(deep=True)
        labeled_gen_pool_df[col_names.tag] = orig_sent_tags
        return gen_pool_df, labeled_gen_pool_df
    # return prepare_pools_template("lecun augmentation", generate_lecun_augmentation_from_sents)
    return "lecun augmentation", lecun_pools_gen_fun


def generate_sents_using_lstm_generator():
    file_lstm_pool = open(CODE_DIR + 'z_experiments/init_pools/lstm-generator/' + cn.data_name + '.txt', 'r').readlines()
    # file_lstm_pool = filter(lambda s: s.upper() != s.lower(), fix_sentences_encoding(file_lstm_pool))  # has characters
    file_lstm_pool = fix_sentences_encoding(file_lstm_pool)

    # import pdb; pdb.set_trace()

    def pool_gen_fun(base_sents, base_df, col_names, total_new_sents):
        lstm_pool = set(file_lstm_pool).difference(set(base_df[col_names.text]))
        lstm_pool = list(lstm_pool)
        assert len(lstm_pool) >= total_new_sents, "not enough sentences in tcg pool" + str(len(lstm_pool)) + "  " + str(total_new_sents)
        chosen_idxs = combinatorics_util.random_choice_bulk([1.0] * len(lstm_pool), total_new_sents)

        generated_sents = map(lambda i: lstm_pool[i], chosen_idxs)
        generated_pool_df = pandas_util.append_rows_to_dataframe(pandas_util.copy_dataframe_structure(base_df),
                                                                 col_names.text, generated_sents)
        generated_pool_df = prepare_df_columns(generated_pool_df, col_names)
        labeled_gen_pool_df = label_df_with_expert(generated_pool_df, col_names, print_status=False)
        return generated_pool_df, labeled_gen_pool_df
    return "lstm-generator", pool_gen_fun


def prepare_orig_examples_pools():
    def pools_gen_fun(base_sents, base_df, col_names, total_new_sents):
        tns = total_new_sents
        pool_idxs = combinatorics_util.random_choice_bulk(range(len(cn.pool_df)), tns, allow_dups=True)
        orig_labeled_pool_df = cn.pool_df.iloc[pool_idxs].copy(deep=True)
        orig_pool_df = orig_labeled_pool_df.copy(deep=True)
        switch_df_tag(orig_pool_df, cn.col_names.tag, 0.0, None)
        switch_df_tag(orig_pool_df, cn.col_names.tag, 1.0, None)
        # print orig_labeled_pool_df.groupby(cn.col_names.tag).size()
        return orig_pool_df, orig_labeled_pool_df
    return "original examples", pools_gen_fun