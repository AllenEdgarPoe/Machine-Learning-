#Input Layer
import numpy as np
import csv
from collections.abc import Iterable
import random

class InputLayer:
    y_label_starting = None
    y_label_last = None

    y_label_shape = None

    training_input_matrix = None
    training_y_label = None

    validation_input_matrix = None
    validation_y_label = None

    test_input_matrix = None
    test_y_label = None

    one_hot_encoding_mappings = dict()
    one_hot_encoding_mappings_for_prediction = dict()
    y_label_mappings = dict()
    y_label_mappings_for_prediction = dict()

    returning_shape = None


class InputLayer_CSV(InputLayer):
    def __init__(self, y_label_index_starting, y_label_index_last):
        self.y_label_starting = y_label_index_starting
        self.y_label_last = y_label_index_last

    def build_layer(self, csv_path, train_validation_test_ratio=(6,2,2)):
        y_label_index_starting = self.y_label_starting
        y_label_index_last = self.y_label_last

        def csv_to_header_and_data(csv_path):
            file_instance = open(csv_path)
            rdr = csv.reader(file_instance)
            raw_data = [i for i in rdr]
            return raw_data[0], raw_data[1:], len(raw_data[0])


        def typecast_data_accordingly(header, data):
            output = []
            for each_row in data:
                for index in range(len(header)):
                    if header[index] == 'string':
                        each_row[index] = str(each_row[index])
                        continue
                    if header[index] == 'number':
                        each_row[index] = float(each_row[index])
                output.append(each_row)
            return output


        def encoding_rules(header, data):
            all_mappings = dict()
            all_mappings_reverse = dict()

            for header_index in range(len(header)):
                if header[header_index] == 'number':
                    continue

                string_to_one_hot_encoding = dict()
                one_hot_encoding_to_string = dict()

                temp = set()
                for each_list in data:
                    temp.add(each_list[header_index])

                counter = 0
                for i in temp:
                    cal = list(np.zeros(len(temp)))
                    cal[counter] = 1
                    string_to_one_hot_encoding[i] = cal
                    one_hot_encoding_to_string[tuple(cal)] = i
                    counter += 1

                #행번호 밑에 각 string 별 변환해야 할 one-hot-encoding이 딕셔너리로 적혀있다.
                all_mappings[header_index] = string_to_one_hot_encoding
                all_mappings_reverse[header_index] = one_hot_encoding_to_string
            return all_mappings, all_mappings_reverse

        def flatten(items):
            """My homage to pylang of Stack Overflow"""
            for x in items:
                if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                    for sub_x in flatten(x):
                        yield sub_x
                else:
                    yield x

        def apply_flattening(data, encoding_rules):
            for column_index in encoding_rules:
                for row_index in range(len(data)):
                    data[row_index][column_index] = encoding_rules[column_index][data[row_index][column_index]]

            for row_index in range(len(data)):
                data[row_index] = list(flatten(data[row_index]))
            return data

        def assign_train_evaluation_test(data, ratio):
            random.shuffle(data)
            train_last_index = round(len(data)/sum(ratio)*ratio[0])
            evaluation_last_index = train_last_index + round(len(data) / sum(ratio) * ratio[1])
            test_last_index = evaluation_last_index + round(len(data) / sum(ratio) * ratio[2])
            return data[:train_last_index], data[train_last_index:evaluation_last_index], data[evaluation_last_index:test_last_index]

        #JSP에서 코딩해서 보낼 때 index는 0번부터 시작하게 하기. 꼭 확인해야함.
        def seperate_y_label_and_arrayfy(data, y_label_starting, y_label_last, original_array_length, encoding_rule):
            value_to_add = 0
            if y_label_starting == 0:
                for index in encoding_rule:
                    if index <= y_label_last:
                        value_to_add += (len(encoding_rule[index])-1)
                else:
                    y_label_last += value_to_add
                    return np.array([i[y_label_last+1:] for i in  data]), np.array([i[:y_label_last+1] for i in data])

            elif y_label_last+1 == original_array_length:
                for index in encoding_rule:
                    if index < y_label_starting:
                        value_to_add += (len(encoding_rule[index])-1)
                else:
                    y_label_starting += value_to_add
                    return np.array([i[:y_label_starting] for i in data]), np.array([i[y_label_starting:] for i in data])


        header, data, original_array_length = csv_to_header_and_data(csv_path)
        data = typecast_data_accordingly(header, data)
        self.one_hot_encoding_mappings, self.one_hot_encoding_mappings_for_prediction = encoding_rules(header, data)
        data = apply_flattening(data, self.one_hot_encoding_mappings)
        train, evaluation, test = assign_train_evaluation_test(data, train_validation_test_ratio)

        self.training_input_matrix, self.training_y_label = \
            seperate_y_label_and_arrayfy(train, y_label_index_starting, y_label_index_last, original_array_length, self.one_hot_encoding_mappings)
        self.validation_input_matrix, self.validation_y_label = \
            seperate_y_label_and_arrayfy(evaluation, y_label_index_starting, y_label_index_last, original_array_length, self.one_hot_encoding_mappings)
        self.test_input_matrix, self.test_y_label = \
            seperate_y_label_and_arrayfy(test, y_label_index_starting, y_label_index_last, original_array_length, self.one_hot_encoding_mappings)

        #print(self.training_input_matrix)
        #print(self.training_y_label)

        #print(self.training_y_label)

        #hard-coded for 1D-data only
        self.returning_shape = self.training_input_matrix[0].reshape(1,-1).shape
        self.y_label_shape = self.training_y_label[0].reshape(1,-1).shape
        if self.one_hot_encoding_mappings != {}:
            self.y_label_mappings = self.one_hot_encoding_mappings[self.y_label_starting]
            self.y_label_mappings_for_prediction = self.one_hot_encoding_mappings_for_prediction[self.y_label_starting]
            
        return self.training_input_matrix, self.training_y_label, self.test_input_matrix, self.test_y_label# -*- coding: utf-8 -*-

