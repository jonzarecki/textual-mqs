import numpy as np
import pandas as pd  # data analysis lib for python
from pandas import DataFrame


def shuffle_df_rows(df):
    # type: (pd.DataFrame) -> pd.DataFrame
    """
    returns a new df with $df rows shuffled
    :param df: the original df
    :return: shuffled $df
    """
    return df.iloc[np.random.permutation(len(df))].reset_index(drop=True)


def get_all_positive_sentences(df, text_col, tag_col, pos_tags):
    # type: (pd.DataFrame, str, str, list) -> list
    """
    puts all positive sentences in df in a list
    :param df: input DataFrame
    :param text_col: the name of the column that contains the text
    :param tag_col: the name of the column that contains the tag
    :param pos_tags: a list of tags we consider as positive
    :return: a list, that contains all 'positive' sentences in $df
    """
    all_positive_df = all_positive_rows_df(df, tag_col, pos_tags)
    all_positive_sentences = all_positive_df[text_col].tolist()
    return all_positive_sentences


def append_rows_to_dataframe(df, text_col, new_sents):
    # type: (pd.DataFrame, str, list) -> pd.DataFrame
    """
    appends $new_sents as new rows in $df
    :param df: input DataFrame
    :param text_col: the name of the column that contains the text
    :param new_sents: list of new sentences we want to add to the dataframe
    :return: $df appended with new_sents as the bottom rows (doesn't change df)
    """
    new_sent_df = copy_dataframe_structure(df)
    new_sent_df[text_col] = new_sents  # fill the text column with the new sentences
    return df.append(new_sent_df, ignore_index=True)


def copy_dataframe_structure(df):
    # type: (pd.DataFrame) -> pd.DataFrame
    """
    copies the df column to a new df with no data in it
    :param df: the original df (pandas'es DataFrame) from which we want to extract the structure
    :return: an empty df, with the same structure as $df
    """
    df_copy = pd.DataFrame(data=None, columns=df.columns, index=None)
    return df_copy


def get_tag_count(df, tag_col, tag_vals):
    # type: (pd.DataFrame, str, list) -> int
    """
    returns the number of accuracies of the values in $tag_vals (which resides in $tag_column) in $df
    :param df: contains the data
    :param tag_col: the name of the column in which the tag is
    :param tag_vals: str list, contains the tag values we want to count
    :rtype: int
    """
    return len(all_positive_rows_df(df, tag_col, tag_vals))


def get_cell_val(df, row_index, column_name):
    # type: (pd.DataFrame, int, str) -> object
    """
    :param df: contains the table data
    :param row_index: the row index of the cell
    :param column_name: the column name of the cell
    :return: the cell value in df[row_index][column_name], type depends on content
    """
    assert 0 <= row_index <= len(df.index), "get_cell_val: row index out of range"
    return df.loc[row_index].get(column_name)


def set_cell_val(df, row_index, column_name, new_val):
    """
    sets a given cell in the DataFrame with new_val, adds the row if it doesn't exist
    :param new_val: the new value to put in the cell
    :type df: DataFrame
    :param row_index: the row index of the cell
    :type row_index: int
    :param column_name: the column name of the cell
    :type column_name: str
    :return: None
    """
    assert 0 <= row_index <= len(df.index), "set_cell_val: row index out of range"
    df.at[row_index, column_name] = new_val


def all_positive_rows_df(df, tag_col, pos_tags):
    # type: (pd.DataFrame, str, list) -> pd.DataFrame
    """
    :param df: pandas read_csv file (DataFrame), lets the user choose it's parameters
    :param tag_col: the column in $in_csv that contains the tag  (string)
    :param pos_tags: list of Strings, each a valid positive tag
    :return: a new df with all the positively tagged rows
    """
    assert len(pos_tags) != 0, "should have possible tags in pos_tags"
    return df[df[tag_col].astype(type(pos_tags[0])).isin(pos_tags)].sort_index().reset_index(drop=True)


def imbalance_dataset(df, tag_col, pos_percent, pos_tags, neg_tags, shuffle=False, return_rest=False):
    # type: (pd.DataFrame, str, float, list, list, bool, bool) -> pd.DataFrame
    """
    builds a new data set from $in_csv,
        containing $pos_percent examples with 1 tag, and 1-$positive_percent with 0 tag
    :param df: pandas read_csv file (DataFrame), lets the user choose it's parameters
    :param tag_col: the column in $in_csv that contains the tag  (string)
    :param pos_percent: 0<=x<=1, percent of positive examples in the output dataset (real)
    :param pos_tags: list of Strings, each a valid positive tag
    :param neg_tags: list of Strings, each a valid negative tag
    :return: new_df with the wanted percent of positive examples
    """
    assert 0.0 <= pos_percent <= 1.0, "imbalanced_dataset: invalid $pos_percent parameter"

    df_row_count = len(df)
    # split into positive df and negative df
    positive_df = all_positive_rows_df(df, tag_col, pos_tags)
    negative_df = all_positive_rows_df(df, tag_col, neg_tags)

    # determines how many positive and negatives examples to keep to maintain the wanted distribution
    # (partition_count / partition_%) = total number of examples,
    # (examples_num) - partition_count = number of other class examples
    if df_row_count * pos_percent < len(positive_df):
        neg_count = len(negative_df)
        pos_count = (pos_percent / (1-pos_percent)) * neg_count
    else:
        pos_count = len(positive_df)
        neg_count = ((1-pos_percent) / pos_percent) * pos_count

    # merge the wanted amount of positive and negative examples while keeping example order
    imbalanced_df = pd.concat([negative_df[:int(neg_count)], positive_df[:int(pos_count)]]).sort_index().reset_index(drop=True)
    rest_of_data_df = pd.concat([negative_df[int(neg_count):], positive_df[int(pos_count):]]).sort_index().reset_index(drop=True)
    if shuffle:
        imbalanced_df = shuffle_df_rows(imbalanced_df)
        rest_of_data_df = shuffle_df_rows(rest_of_data_df)
    if return_rest:
        return imbalanced_df, rest_of_data_df
    else:
        return imbalanced_df


def shrink_dataset(df, percent, shuffle=False):
    # type: (pd.DataFrame, float, bool) -> pd.DataFrame
    """
    Shrinks the given dataset to a fraction of its original size
    :param df: contains the data
    :param percent: the % of rows we want to leave ( real)
    :param shuffle: decides if to shuffle the rows beforehand
    :return: a new DataFrame obj, with only $percent of the rows of $df
    """
    if shuffle:
        df = shuffle_df_rows(df)

    shrinked_df = DataFrame(data=df[:int(len(df.index) * percent)], columns=df.columns, copy=True)

    return shrinked_df


def extract_row_from_df(df, idx):
    # type: (pd.DataFrame, int) -> pd.DataFrame
    """
    Extracts the i'th row and puts it in a new DataFrame
    :param df: contains the data
    :param idx: the index of the row we want to extract
    :return: a new DataFrame obj, with only the idx's row from $df
    """
    new_df = copy_dataframe_structure(df)  # will contain the chosen rows for the new
    new_df.loc[0] = df.loc[idx]  # copy row
    return new_df


def split_dataset(df, percent):
    # type: (pd.DataFrme, float) -> (DataFrame, DataFrame)
    """
    splits $df into 2 DataFrames, each contain a split of $df's data
    :param df: the original df we want to split
    :param percent: the % of $df we want in the first split
    :return: 2 DataFrames, each contains a split of $df
    """
    first_df_len = int(len(df) * percent)
    first_df = DataFrame(data=df[:first_df_len], columns=df.columns, copy=True).reset_index(drop=True)
    second_df = DataFrame(data=df[first_df_len:], columns=df.columns, copy=True).reset_index(drop=True)

    return first_df, second_df


def save_lists_as_csv(header_list, data_lists, file_path):
    # type: (list, list, basestring) -> None
    """
    save lists as csv and to the dropbox folder
    :param file_path: the output file path
    :param header_list: list of strings, each contains the header for the corresponding column
    :param data_lists: list of lists, each contains the data for the column
    :return: None
    """

    assert len(header_list) == len(data_lists), "save_lists_as_csv: lists lens not equal"

    # put headers in the start of each column
    column_lists = []
    for i in range(len(data_lists)):
        column_lists.append([header_list[i]] + data_lists[i])
    column_lists_lens = map(len, column_lists)

    columns_lens_list = zip(column_lists, column_lists_lens)

    with open(file_path, 'w') as f:
        for i in range(max(column_lists_lens)):
            row = ""
            for col, col_len in columns_lens_list:
                if i < col_len:
                    row += str(col[i])
                row += "\t"
            row = row[:-1]
            f.write(row + '\n')


def get_pos_neg_balance(df, tag_col, pos_tags):
    # type: (pd.DataFrame, str, list) -> float
    """
    calculates the percent of $pos_tags in $df
    :param df: contains the data
    :param tag_col: the name of the column in which the tag is
    :param pos_tags: str list, contains the tag values we consider as positive
    :return: the portion of positive examples in the dataset (0<=x<=1)
    """
    pos_count = get_tag_count(df, tag_col, pos_tags)
    return float(pos_count) / len(df.index)


def switch_df_tag(df, tag, old_tag, new_tag):
    df.loc[df[tag] == old_tag, tag] = [new_tag] * len(df.loc[df[tag] == old_tag, tag])