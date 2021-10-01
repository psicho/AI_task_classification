import spacy
import pandas as pd
import json
import csv
import operator
import re
import requests
from io import BytesIO
import pickle
import numpy as np
from definitions import ROOT_DIR


def load(text):
    nlp = spacy.load("en_core_web_sm")
    document = nlp(text)
    return document


def split_to_sentence(document) -> list:
    """
    Split input text to sentence
    """
    sentences = []
    for sent in document.sents:
        sentences.append([sent[i] for i in range(len(sent))])

    return sentences


def one_digit_in_task(tokenized):

    one_digit = 0
    count_num = 0

    for token_index in range(len(tokenized)):
        if tokenized[token_index]['pos_'] == 'NUM':
            count_num += 1

    if count_num == 1:
        one_digit = 1

    return one_digit


def get_data_from_lemma_statistics_json(*, validation=False, grouping=False):
    """ Load data for lemma statistics json files """

    # Load statistics from task validation json
    if validation:
        try:
            with open(ROOT_DIR + "/Statistics/group_lemma_dict_validation_filtered.json", "r") as group_lemma_dict_validation:
                group_lemma_dict_validation = json.load(group_lemma_dict_validation)
        except FileNotFoundError:
            raise 'file "group_lemma_dict_validation.json" not found'

        return group_lemma_dict_validation

    # Load statistics from task grouping json
    if grouping:
        try:
            with open(ROOT_DIR + "/Statistics/groups_new_lemma_dict.json", "r") as groups_lemma_dict:
                groups_lemma_dict = json.load(groups_lemma_dict)
        except FileNotFoundError:
            raise 'file "group_lemma_dict.json" not found'

        return groups_lemma_dict


def exist_question_in_task(sentences, count_index=False) -> [bool, bool]:
    # exist '?' in task
    question = False
    counter = 0
    for sent in sentences:
        # print(type(sent[-1]))
        if '?' == str(sent[-1]):
            question = True
            break
        counter += 1

    if count_index is False:
        return question
    else:
        return question, counter


def exist_question_in_end_of_sentence(sentences) -> [bool, bool]:
    question_end = 0
    question_start = 0
    counter = 0
    for sent in sentences:
        # print(sent)
        if '?' in str(sent[-1]):
            question_end = 1
            counter += 1
        elif '?' in str(sent[:-1]):
            question_start = 1

    return question_end, question_start


def exist_questions_noun_in_other_task_text(sentences):

    question_in_task, count_index = exist_question_in_task(sentences, count_index=True)
    exist_word = 0

    if question_in_task is True:
        for noun in sentences[count_index]:
            if noun.pos_ == 'NOUN':
                for i in range(len(sentences)):
                    if i != count_index:
                        for other_noun in sentences[i]:
                            if noun.lemma_.lower() == other_noun.lemma_.lower():
                                exist_word = 1
                                break

                    if exist_word is True:
                        break
            if exist_word is True:
                break
        return exist_word
    else:
        return exist_word


def calculate_lemma_statistic_validation_parameter_percent(tokenized, lemma_dict, percent=100):
    statistics_dict = {
        'none': 0,
        'bad': 0,
        'good': 0,
        'agood': 0,
    }
    groups = {'bad': 0, 'good': 1, 'agood': 2}

    token_task_statistics_dict = {}

    def add_lemma_statistics(token_task_statistics_dict1, lemma, group1, lemma_dict1):
        if token_task_statistics_dict1.get(lemma) is not None:
            token_task_statistics_dict1[lemma][group1] \
                = lemma_dict1[group1].get(lemma)
        else:
            token_task_statistics_dict1[lemma] = statistics_dict.copy()
            token_task_statistics_dict1[lemma][group1] \
                = lemma_dict1[group1].get(lemma)

    # Сохраняем в массив выбранный вариант отношения к группе по каждому токену
    score_dict = {'bad': 0, 'good': 1, 'agood': 2, 'none': 3}

    for token_index in range(len(tokenized)):
        group_rate = [0, 0, 0, 0]
        for group in groups:
            if lemma_dict[group].get(tokenized[token_index]['lemma_'].lower()) is not None:
                group_rate[groups[group]] = lemma_dict[group].get(tokenized[token_index]['lemma_'].lower())

                add_lemma_statistics(token_task_statistics_dict,
                                     tokenized[token_index]['lemma_'].lower(),
                                     group,
                                     lemma_dict)
            else:
                group_rate[3] += 1

    round_num = int(percent/100 * len(tokenized))
    count_lemmas_percent = round_num if round_num > 0 else 1

    result_list = []
    for lemma_stat in token_task_statistics_dict:
        lemma_list = [max(token_task_statistics_dict[lemma_stat].items(), key=operator.itemgetter(1))[1],
                      max(token_task_statistics_dict[lemma_stat].items(), key=operator.itemgetter(1))[0],
                      lemma_stat]
        result_list.append(lemma_list)

    sorted_result_list = sorted(result_list, reverse=True)
    if len(sorted_result_list) == 0:
        return score_dict['none']

    if count_lemmas_percent == 1:
        return score_dict[sorted_result_list[0][1]]
    else:
        statistics_dict_result = statistics_dict.copy()
        for i in range(len(sorted_result_list[:count_lemmas_percent])):
            statistics_dict_result[sorted_result_list[i][1]] += sorted_result_list[i][0]
        # print(statistics_dict_result)
        return score_dict[max(statistics_dict_result.items(), key=operator.itemgetter(1))[0]]


def calculate_lemma_statistic_grouping_parameter(tokenized, lemma_dict):
    statistics_dict = {
        'number_properties': 0,
        'geometry': 0,
        'measurement': 0,
        'algebra': 0,
        'data_and_probability': 0,
        'none': 0
    }

    groups = {
        'number_properties': 0,
        'geometry': 1,
        'measurement': 2,
        'algebra': 3,
        'data_and_probability': 4
        }

    # Сохраняем в массив выбранный вариант отношения к группе по каждому токену
    score_dict = {
        'number_properties': 1,
        'geometry': 2,
        'measurement': 3,
        'algebra': 4,
        'data_and_probability': 5,
        'none': 0
    }

    for token_index in range(len(tokenized)):
        group_rate = [0, 0, 0, 0, 0, 0]

        for group in groups:
            if lemma_dict[group].get(tokenized[token_index]['lemma_'].lower()) is not None:
                group_rate[groups[group]] = lemma_dict[group].get(tokenized[token_index]['lemma_'].lower())
            else:
                group_rate[5] += 1

        index, value = max(enumerate(group_rate[:-1]), key=operator.itemgetter(1))
        if index == 0 and group_rate[5] < 5:
            statistics_dict['number_properties'] += 1
        elif index == 1 and group_rate[5] < 5:
            statistics_dict['geometry'] += 1
        elif index == 2 and group_rate[5] < 5:
            statistics_dict['measurement'] += 1
        elif index == 2 and group_rate[5] < 5:
            statistics_dict['algebra'] += 1
        elif index == 2 and group_rate[5] < 5:
            statistics_dict['data_and_probability'] += 1
        else:
            statistics_dict['none'] += 1

    return score_dict[max(statistics_dict.items(), key=operator.itemgetter(1))[0]]


def calculate_lemma_statistic_grouping_parameter_percent(tokenized, lemma_dict, percent=100):
    statistics_dict = {
        'number_properties': 0,
        'geometry': 0,
        'measurement': 0,
        'algebra': 0,
        'data_and_probability': 0,
        'none': 0
    }

    groups = {
        'number_properties': 0,
        'geometry': 1,
        'measurement': 2,
        'algebra': 3,
        'data_and_probability': 4
        }

    # Сохраняем в массив выбранный вариант отношения к группе по каждому токену
    score_dict = {
        'number_properties': 1,
        'geometry': 2,
        'measurement': 3,
        'algebra': 4,
        'data_and_probability': 5,
        'none': 0
    }

    token_task_statistics_dict = {}

    def add_lemma_statistics(token_task_statistics_dict1, lemma, group1, lemma_dict1):
        # lemma = str(lemma)
        if token_task_statistics_dict1.get(lemma) is not None:
            token_task_statistics_dict1[lemma][group1] \
                = lemma_dict1[group1].get(lemma)
        else:
            token_task_statistics_dict1[lemma] = statistics_dict.copy()
            token_task_statistics_dict1[lemma][group1] \
                = lemma_dict1[group1].get(lemma)

    for token_index in range(len(tokenized)):
        group_rate = [0, 0, 0, 0, 0, 0]

        for group in groups:
            if lemma_dict[group].get(tokenized[token_index]['lemma_'].lower()) is not None:
                group_rate[groups[group]] = lemma_dict[group].get(tokenized[token_index]['lemma_'].lower())
                add_lemma_statistics(token_task_statistics_dict,
                                     tokenized[token_index]['lemma_'].lower(),
                                     group,
                                     lemma_dict)
            else:
                group_rate[5] += 1

    round_num = int(percent / 100 * len(tokenized))
    count_lemmas_percent = round_num if round_num > 0 else 1

    result_list = []
    for lemma_stat in token_task_statistics_dict:
        # print(max(token_task_statistics_dict[lemma_stat].items(), key=operator.itemgetter(1))[1])
        lemma_list = [max(token_task_statistics_dict[lemma_stat].items(), key=operator.itemgetter(1))[1],
                      max(token_task_statistics_dict[lemma_stat].items(), key=operator.itemgetter(1))[0],
                      lemma_stat]
        result_list.append(lemma_list)

    sorted_result_list = sorted(result_list, reverse=True)
    if len(sorted_result_list) == 0:
        return score_dict['none']
    # print(sorted(result_list, reverse=True))
    # print('count_lemma_percent >> ', count_lemmas_percent)
    if count_lemmas_percent == 1:
        return score_dict[sorted_result_list[0][1]]
    else:
        statistics_dict_result = statistics_dict.copy()
        for i in range(len(sorted_result_list[:count_lemmas_percent])):
            statistics_dict_result[sorted_result_list[i][1]] += sorted_result_list[i][0]
        # print(statistics_dict_result)
        return score_dict[max(statistics_dict_result.items(), key=operator.itemgetter(1))[0]]


def calculate_lemma_statistic_new_grouping_parameter_percent(tokenized, lemma_dict, percent=100):
    statistics_dict = {
        "Multiplication and division": 0,
        "Addition and subtraction": 0,
        "Fractions": 0,
        "Mixed operations": 0,
        "Measurements": 0,
        "Figures": 0,
        "Number": 0,
        "Modelling": 0,
        "Geometry": 0,
        "Time": 0,
        "Comparison": 0,
        "Estimation": 0,
        "Logic": 0,
        "Series and pattern": 0,
        "Graph": 0,
        "Probability": 0,
        "Money": 0,
        "Other": 0,
    }

    groups = {
        "Multiplication and division": 0,
        "Addition and subtraction": 1,
        "Fractions": 2,
        "Mixed operations": 3,
        "Measurements": 4,
        "Figures": 5,
        "Number": 6,
        "Modelling": 7,
        "Geometry": 8,
        "Time": 9,
        "Comparison": 10,
        "Estimation": 11,
        "Logic": 12,
        "Series and pattern": 13,
        "Graph": 14,
        "Probability": 15,
        "Money": 16,
        "Other": 17,
        }

    # Сохраняем в массив выбранный вариант отношения к группе по каждому токену
    score_dict = {
        "Multiplication and division": 1,
        "Addition and subtraction": 2,
        "Fractions": 3,
        "Mixed operations": 4,
        "Measurements": 5,
        "Figures": 6,
        "Number": 7,
        "Modelling": 8,
        "Geometry": 9,
        "Time": 10,
        "Comparison": 11,
        "Estimation": 12,
        "Logic": 13,
        "Series and pattern": 14,
        "Graph": 15,
        "Probability": 16,
        "Money": 17,
        "Other": 18,
        'none': 0
    }

    token_task_statistics_dict = {}

    def add_lemma_statistics(token_task_statistics_dict1, lemma, group1, lemma_dict1):
        # lemma = str(lemma)
        if token_task_statistics_dict1.get(lemma) is not None:
            token_task_statistics_dict1[lemma][group1] \
                = lemma_dict1[group1].get(lemma)
        else:
            token_task_statistics_dict1[lemma] = statistics_dict.copy()
            token_task_statistics_dict1[lemma][group1] = lemma_dict1[group1].get(lemma)

    for token_index in range(len(tokenized)):
        group_rate = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for group in list(groups.keys())[:-1]:
            if lemma_dict[group].get(tokenized[token_index]['lemma_'].lower()) is not None:
                group_rate[groups[group]] = lemma_dict[group].get(tokenized[token_index]['lemma_'].lower())
                add_lemma_statistics(token_task_statistics_dict,
                                     tokenized[token_index]['lemma_'].lower(),
                                     group,
                                     lemma_dict)
            else:
                group_rate[17] += 1

    round_num = int(percent / 100 * len(tokenized))
    count_lemmas_percent = round_num if round_num > 0 else 1

    result_list = []
    for lemma_stat in token_task_statistics_dict:
        # print(max(token_task_statistics_dict[lemma_stat].items(), key=operator.itemgetter(1))[1])
        lemma_list = [max(token_task_statistics_dict[lemma_stat].items(), key=operator.itemgetter(1))[1],
                      max(token_task_statistics_dict[lemma_stat].items(), key=operator.itemgetter(1))[0],
                      lemma_stat]
        result_list.append(lemma_list)

    sorted_result_list = sorted(result_list, reverse=True)
    if len(sorted_result_list) == 0:
        return score_dict['none']
    if count_lemmas_percent == 1:
        return score_dict[sorted_result_list[0][1]]
    else:
        statistics_dict_result = statistics_dict.copy()
        for i in range(len(sorted_result_list[:count_lemmas_percent])):
            statistics_dict_result[sorted_result_list[i][1]] += sorted_result_list[i][0]
        return score_dict[max(statistics_dict_result.items(), key=operator.itemgetter(1))[0]]


def text_in_bad_list(document):

    bad_list = ['[.\s]*([-=#;,.\\/|]+\$)[\.\s]*', '[.\s]*(^)[\.\s]*', '[.\s]*([\^▢■△▞░◡□�▣◈△􏰀|]+)[\.\s]*',
                '"{2}?', '[.\s]*(\$[-=#;,.\\/|]+)[\.\s]*', '[.\s]*([-=*_\^\s<>]{3})[\.\s]*',
                '#ERROR', '√ ', '(\\;)', '(\\})', '(\d[\s]){2,}', '_n_', '$$', '[.\s]*(\?\s){2}[\.\s]*', '  ?  ',
                'ANS: 5', '“IAM,""', '0.12,000', '.""', '_ _', '(√∑)', 'diagram',
                'following', 'fill in the blank', 'of the following', 'table below']

    task_text = document.text

    for bad_symbol in bad_list:
        m = re.search(bad_symbol, task_text)
        if m is not None:
            return 1
    return 0


def tokenize(document, text=True, lemma=True, pos=True, tag=True, dep=True, ent_type=True):
    """
    :param document: Загруженный с помощью функции load текст
    :param text: Выводить текст токена (token.text)
    :param lemma: Выводить лемму токена (token.lemma_)
    :param pos: Тег общей (укрупненной) части речи (глагол, прилагательное и др.) (token.pos_)
    :param tag: Тег уточнённой части речи (например у общей категории 'SYM' есть три уточненные подкатегории:
           $ - для обозначения валют) (token.tag_)
    :param dep: Метки зависимостей (Вспомогательный глагол, Детальный падеж и др.) (token.dep_)
    :param ent_type: Тип именованной сущности (Например геополитическая метра GPE - обозначает страны, города, штаты и
           другие геополитические объекты) (token.ent_type_)
    :return:
    """
    token_dict_temp = {'text': '', 'lemma_': '', 'pos_': '', 'tag_': '', 'dep_': '', 'ent_type_': ''}

    tokens_list = []

    for token in document:
        token_dict = token_dict_temp.copy()
        token_dict['text'] = token.text
        token_dict['lemma_'] = token.lemma_
        token_dict['pos_'] = token.pos_
        token_dict['dep_'] = token.dep_
        token_dict['ent_type_'] = token.ent_type_
        tokens_list.append(token_dict)

    return tokens_list


def create_data(tokens_list, document):
    dict_map = {
        'ADJ': 0,
        'ADP': 0,
        'ADV': 0,
        'AUX': 0,
        'CCONJ': 0,
        'DET': 0,
        'INTJ': 0,
        'NOUN': 0,
        'NUM': 0,
        'PART': 0,
        'PRON': 0,
        'PROPN': 0,
        'PUNCT': 0,
        'SCONJ': 0,
        'SYM': 0,
        'VERB': 0,
        'X': 0,
        # additional parameters:
        'all_to_num': 0,  # Calculate (all type token) / (number type token)
        'exist_question': 0,  # Calculate exist question
        'negative_numbers': 0,  # Calculate exist of negative_numbers
        'more_hundred': 0,  # Calculate exist of more hundred numbers
        'exist_noun_in': 0,  # Calculate exist noun from question in other task text
        'question_in_end': 0,  # Calculate exist question in end of sentences
        'question_not_end': 0,  # Calculate exist question not in end
    }

    error_dict = set()

    # 1 Calculate (all type token) / (number type token)
    num_type_count = 0
    all_type_count = 0
    all_to_num = 0

    # 2 Calculate exist question
    exist_question = 0

    # 3 Calculate exist of negative_numbers
    negative_numbers = 0

    # 4 Calculate exist of more hundred numbers
    more_hundred = 0

    # 5 Calculate exist noun from question in other task text
    sentences = split_to_sentence(document)
    # print(sentences)
    exist_noun_in = exist_questions_noun_in_other_task_text(sentences)
    # print(exist_noun_in)

    # 6 Calculate exist of question in end of sentences
    question_in_end = exist_question_in_end_of_sentence(sentences)[0]

    # 7 Calculate exist of question not in end
    question_not_end = exist_question_in_end_of_sentence(sentences)[1]

    for token in tokens_list:
        try:
            dict_map[token['pos_']] += 1

            # Count num and all tokens type in text
            if token['pos_'] == 'NUM':
                num_type_count += 1
                if '-' in token['text']:
                    negative_numbers = 1
                try:
                    if int(token['text']) > 100:
                        more_hundred += 1
                except ValueError:
                    pass
            else:
                all_type_count += 1

            if '?' in token['text']:
                exist_question = 1

        except KeyError:
            # print('Error pos_')
            error_dict.add(token['pos_'])
            # print('<<< error token pos_ >>> ', token['pos_'], error_dict)
            print(error_dict)

    # 1. Calculate (all type token) / (number type token)
    all_to_num = (all_type_count // num_type_count) if num_type_count > 0 else 0
    dict_map['all_to_num'] = all_to_num

    # 2. Presence of a question
    dict_map['exist_question'] = exist_question

    # 3. Negative numbers
    dict_map['negative_numbers'] = negative_numbers

    # 4. Number > 100
    dict_map['more_hundred'] = more_hundred

    # 5. Exist noun from question in other task text
    dict_map['exist_noun_in'] = exist_noun_in

    # 6. Exist of question in end of sentences
    dict_map['question_in_end'] = question_in_end

    # 7. Calculate exist of question not in end
    dict_map['question_not_end'] = question_not_end

    return dict_map


def handing_input_dataset_from_google_docs(link):
    link = re.search("^(https://docs.google.com/spreadsheets/d/.+/)", link)
    r = requests.get(link.group(0) + 'export?format=csv')
    data_content = r.content
    df = pd.read_csv(BytesIO(data_content))
    data_new = df

    return data_new


def classification_data_set(link):

    # grouping_map = {1: 'number_properties', 2: 'geometry', 3: 'measurement', 4: 'algebra', 5: 'data_and_probability'}
    grouping_map = {
        1: "Multiplication and division",
        2: "Addition and subtraction",
        3: "Fractions",
        4: "Mixed operations",
        5: "Measurements",
        6: "Figures",
        7: "Number",
        8: "Modelling",
        9: "Geometry",
        10: "Time",
        11: "Comparison",
        12: "Estimation",
        13: "Logic",
        14: "Series and pattern",
        15: "Graph",
        16: "Probability",
        17: "Money",
        18: "Other",
    }
    validation_map = {0: 'bad', 1: 'good', 2: 'agood'}

    data = handing_input_dataset_from_google_docs(link)

    """ Load training models """
    validate_model_file = ROOT_DIR + '/training_models/gbrt_validation_model.sav'
    validate_model = pickle.load(open(validate_model_file, 'rb'))

    grouping_model_file = ROOT_DIR + '/training_models/gbrt_grouping_model.sav'
    grouping_model = pickle.load(open(grouping_model_file, 'rb'))

    """ Lemma statistics calculation """
    lemma_dict_validation = get_data_from_lemma_statistics_json(validation=True, grouping=False)
    lemma_dict_grouping = get_data_from_lemma_statistics_json(validation=False, grouping=True)

    with open(ROOT_DIR + '/Data/classification.csv', 'w', newline='') as csvfile:
        fieldnames = ['', 'gen_text', 'Good/Agood/Bad', 'Group']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for string in range(data['gen_text'].count()):
            try:
                task_id = data.iloc[string, 0]
                load_question = load(data.iloc[string, 1])
                # print('id', id, 'load question > ', load_question)
                in_bad_list = text_in_bad_list(load_question)
                tokenized = tokenize(load_question)
                load_data_dict = create_data(tokenized, load_question)

                # calculate parameters
                one_digit = one_digit_in_task(tokenized)

                ################################################
                # calculate selected group from lemma statistics
                ################################################
                group_calculate_validation = calculate_lemma_statistic_validation_parameter_percent(
                    tokenized,
                    lemma_dict_validation,
                    percent=30)
                # group_calculate = calculate_lemma_statistic_validation_parameter(tokenized, lemma_dict)

                group_calculate_grouping = calculate_lemma_statistic_new_grouping_parameter_percent(
                    tokenized,
                    lemma_dict_grouping,
                    percent=30)
                # group_calculate = calculate_lemma_statistic_grouping_parameter(tokenized, lemma_dict)

                map_token_list_validation = [
                    load_data_dict['ADJ'],
                    load_data_dict['ADP'],
                    load_data_dict['ADV'],
                    load_data_dict['AUX'],
                    load_data_dict['CCONJ'],
                    load_data_dict['DET'],
                    load_data_dict['INTJ'],
                    load_data_dict['NOUN'],
                    load_data_dict['NUM'],
                    load_data_dict['PART'],
                    load_data_dict['PRON'],
                    load_data_dict['PROPN'],
                    load_data_dict['PUNCT'],
                    load_data_dict['SCONJ'],
                    load_data_dict['SYM'],
                    load_data_dict['VERB'],
                    load_data_dict['X'],
                    load_data_dict['all_to_num'],  # Calculate (all type token) / (number type token)
                    load_data_dict['exist_question'],  # Calculate exist question
                    load_data_dict['negative_numbers'],  # Calculate exist of negative_numbers
                    load_data_dict['more_hundred'],  # Calculate exist of more hundred numbers
                    load_data_dict['exist_noun_in'],  # Calculate exist noun from question in other task text
                    load_data_dict['question_in_end'],  # Calculate exist question in end of sentences
                    load_data_dict['question_not_end'],  # Calculate exist question not in end
                    group_calculate_validation,  # Calculate selected group from lemma statistics
                    in_bad_list,  # bad list entry calculation
                    one_digit,  # True if one digit exist in task
                ]
                map_token_list_grouping = [
                    load_data_dict['ADJ'],
                    load_data_dict['ADP'],
                    load_data_dict['ADV'],
                    load_data_dict['AUX'],
                    load_data_dict['CCONJ'],
                    load_data_dict['DET'],
                    load_data_dict['INTJ'],
                    load_data_dict['NOUN'],
                    load_data_dict['NUM'],
                    load_data_dict['PART'],
                    load_data_dict['PRON'],
                    load_data_dict['PROPN'],
                    load_data_dict['PUNCT'],
                    load_data_dict['SCONJ'],
                    load_data_dict['SYM'],
                    load_data_dict['VERB'],
                    load_data_dict['X'],
                    load_data_dict['all_to_num'],  # Calculate (all type token) / (number type token)
                    load_data_dict['exist_question'],  # Calculate exist question
                    load_data_dict['negative_numbers'],  # Calculate exist of negative_numbers
                    load_data_dict['more_hundred'],  # Calculate exist of more hundred numbers
                    load_data_dict['exist_noun_in'],  # Calculate exist noun from question in other task text
                    load_data_dict['question_in_end'],  # Calculate exist question in end of sentences
                    load_data_dict['question_not_end'],  # Calculate exist question not in end
                    group_calculate_grouping,  # Calculate selected group from lemma statistics
                ]

                target_validate = np.asarray(map_token_list_validation)
                target_validate = validation_map[validate_model.predict(target_validate.reshape(1, -1))[0]]

                target_grouping = np.asarray(map_token_list_grouping)
                target_grouping = grouping_map[grouping_model.predict(target_grouping.reshape(1, -1))[0]]

            except InterruptedError:
                print('Errore >>> SIGSEGV')
                continue

            writer.writerow({'': task_id, 'gen_text': load_question,
                             'Good/Agood/Bad': target_validate,
                             'Group': target_grouping})
            print(f"Обработано {string+1} из { data['gen_text'].count()} задач")
