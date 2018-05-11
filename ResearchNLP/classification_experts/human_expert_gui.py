import tkMessageBox as MessageBox
from Tkinter import *

from pandas import DataFrame

from ResearchNLP.util_files import pandas_util


class ExpertClassificationApp:
    def __init__(self, master, df, text_column, tag_column, positive_text, negative_text):
        """
        UI for expert classification
        adds tags in the provided DataFrame, 1-positive, 0-negative, (-1)-not relevant
        :param master: the tk's root widget
        :param tag_column: str, the name of the column which contains the tag
        :param text_column: str, the name of the column which contains the text
        :param positive_text: str, text to classify as positive
        :param negative_text: str, text to classify as negative
        :param df: an input DataFrame which contains texts without classification
        """
        # initialize object fields
        self.df = df
        self.df_len = len(df.index)
        self.df_row_index = -1
        self.tag_column = tag_column
        self.text_column = text_column

        # initialize frame (widget container)
        self.frame = Frame(master)
        self.frame.pack()

        # initialize text field
        self.text_message = StringVar()
        self.text_label = Label(self.frame, textvariable=self.text_message, anchor=W, justify=CENTER,
                                font=("Helvetica", 10))
        self.text_label.pack()

        # initialize radio buttons
        self.check_button_val = IntVar()  # will contain the radio button's value
        self.positive_radio = Radiobutton(self.frame, text=positive_text,
                                          variable=self.check_button_val, value=1)
        self.negative_radio = Radiobutton(self.frame, text=negative_text,
                                          variable=self.check_button_val, value=0)
        # use to determine how good the generated sentences are
        self.not_relevant_radio = Radiobutton(self.frame, text='Not Relevant',
                                              variable=self.check_button_val, value=-1)
        self.positive_radio.pack(anchor=W)
        self.negative_radio.pack(anchor=W)
        self.not_relevant_radio.pack(anchor=W)

        # initialize next task button
        self.next_button = Button(self.frame, text="Next", command=self.next_task)
        self.next_button.pack(side=LEFT)

        self.next_task()

    def quit(self):
        self.frame.quit()

    def next_task(self):
        if self.df_row_index != -1:  # not init
            pandas_util.set_cell_val(self.df, self.df_row_index, self.tag_column, self.check_button_val.get())

        self.df_row_index += 1
        if self.df_row_index >= self.df_len or self.df_len == 0:
            MessageBox.showinfo("Finished Classifying !", "Thanks for the work !")  # yay
            self.quit()  # terminate window
            return
        else:
            self.text_message.set(
                pandas_util.get_cell_val(self.df, self.df_row_index, self.text_column))  # set text to next value


class ExpertBestWordApp:
    def __init__(self, master, df, text_column, best_word_column):
        """
        :param df: an input DataFrame which contains texts without the most important word in each sentence
        :type df: DataFrame
        :param master: the tk's root widget
        :param best_word_column: str, the name of the column which will contains the most important word in the sentence
        :param text_column: str, the name of the column which contains the text
        """
        # initialize object fields
        self.df = df
        self.df_len = len(self.df.index)
        self.df_row_index = -1  # init val
        self.best_word_column = best_word_column
        self.text_column = text_column

        # initialize frame (widget container)
        self.frame = Frame(master)
        self.frame.pack()

        # initialize text field
        self.text_message = StringVar()
        self.text_label = Label(self.frame, textvariable=self.text_message, anchor=W, justify=CENTER,
                                font=("Helvetica", 10))
        self.text_label.pack()

        self.text_label = Text(self.frame, font=("Helvetica", 7))
        # initialize text field
        # Create an Entry Widget in textFrame
        self.text_field = Entry(self.frame)
        self.text_field["width"] = 50
        self.text_field.pack(side=LEFT)

        # initialize next task button
        self.next_button = Button(self.frame, text="Next", command=self.next_task)
        self.next_button.pack(side=LEFT)

        self.next_task()

    def quit(self):
        self.frame.quit()

    def next_task(self):
        if self.df_row_index != -1:  # not init
            print self.df_row_index.__str__() + self.text_field.get()
            pandas_util.set_cell_val(self.df, self.df_row_index, self.best_word_column, self.text_field.get())
            self.text_field.delete(0, END)

        self.df_row_index += 1
        if self.df_row_index >= self.df_len or self.df_len == 0:
            MessageBox.showinfo("Finished Classifying !", "Thanks for the work !")  # yay
            self.quit()  # terminate window
            return
        else:
            self.text_message.set(
                pandas_util.get_cell_val(self.df, self.df_row_index, self.text_column))  # set text to next value


def classify_by_expert(df, text_column, tag_column, positive_text, negative_text):
    # type: (DataFrame, str, str, str, str) -> DataFrame
    """

    :param df: untagged DataFrame obj, contains text with no tags
    :param text_column: str, the name of the column which contains the text
    :param tag_column: str, the name of the column which contains the tag
    :param positive_text: str, text to classify as positive
    :param negative_text: str, text to classify as negative
    :return: a tagged DataFrame, in each line a human annotated the text
    """
    root = Tk()

    app = ExpertClassificationApp(root, df, text_column, tag_column, positive_text, negative_text)
    root.protocol("WM_DELETE_WINDOW", app.quit())
    root.mainloop()
    root.destroy()

    return app.df  # return the new tagged df


def choose_most_important_word_by_expert(df, text_column, best_word_column):
    """

    :param df: untagged DataFrame obj, contains text lines
    :param text_column: str, the name of the column which contains the text
    :param best_word_column: str, the name of the column which will contain the most important word in the sentence
    :return: a tagged DataFrame, in each line a human annotated the text
    """
    root = Tk()

    app = ExpertBestWordApp(root, df, text_column, best_word_column)
    root.protocol("WM_DELETE_WINDOW", app.quit())
    root.mainloop()
    root.destroy()

    return app.df  # return the new tagged df
