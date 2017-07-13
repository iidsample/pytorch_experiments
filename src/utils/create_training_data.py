import os
import shutil
import random

import numpy as np


class SplitData(object):
    """
    Interface for creating a splitting the dataset into train, validation and
    test.
    The class should be instantiated using a mode, source path to the parent
    directory containg the folders, and the destination parent directory

    Args:
        mode(int) : 1 for only creating training set
                    2 for creating training, val set
                    3 for creating training, val, test set
        src_directory (str) : source parent directory for images
        dest_directory (str) : destination parent directory, 
                               structure-
                               dest_directory/src_directory(name)/train
                               dest-directory/src_directory(name)/test

    """
    def __init__(self, mode, src_directory, dest_directory):
        self.mode = mode
        self.src_directory = src_directory
        self.dest_directory = dest_directory
    def generate_dataset(self, sample_mode, val_pc = 0.0, test_pc = 0.0):
        """
        Creates dataset
        sample_mode (int) : the way sampling to be done, for now its data dependent
        val_pc (float) : percentage of data for validation
        test_pc (float) : percentage of data for test
        """
        out_path = os.path.join(self.dest_directory,
                                os.path.basename(self.src_directory))

        if self.mode == 1:
            print (('Warning : Full data to be used for training \n No')
                   ('validation test will be created'))
            for directory, _,files in os.walk(self.src_directory):
                for f in files:
                    src_filepath = os.path.join(directory, f)
                    subd_name = os.path.dirname(src_filepath).split('/')[-1]
                    subd_path = os.path.join(out_path, 'train', subd_name)
                    if not os.path.exists(subd_path):
                        os.makedirs(subd_path)
                    shutil.copy(src_filepath, subd_path)
        if self.mode == 2:
            print (('Only train and validation data will be created,\n hope you have separate testing set'))
            for directory, _, files in os.walk(self.src_directory):
                # shuffle the list
                random.shuffle(files)
                length_files = len(files)
                num_validation_files = int(length_files * val_pc)
                validation_file_list = files[:num_validation_files]
                training_file_list = files[num_validation_files:]
                print ('total number of files {}'.format(length_files))
                print ('number of validation files {}'.format(len(validation_file_list)))
                print ('number of training files {}'.format(len(training_file_list)))
                for f in validation_file_list:
                    src_filepath = os.path.join(directory, f)
                    subd_name = os.path.dirname(src_filepath).split('/')[-1]
                    subd_path = os.path.join(out_path, 'valid', subd_name)
                    if not os.path.exists(subd_path):
                        os.makedirs(subd_path)
                    shutil.copy(src_filepath, subd_path)
                for f in training_file_list:
                    src_filepath = os.path.join(directory,f)
                    subd_name = os.path.dirname(src_filepath).split('/')[-1]
                    subd_path = os.path.join(out_path, 'train', subd_name)
                    if not os.path.exists(subd_path):
                        os.makedirs(subd_path)
                    shutil.copy(src_filepath, subd_path)
        if self.mode == 3:
            print ('Creating train, validation and test set')
            for directory, _, files in os.walk(self.src_directory):
                random.shuffle(files)
                length_files = len(files)
                num_validation_files = int(length_files * val_pc)
                num_test_files = int(length_files * test_pc)
                validation_file_list = files[:num_validation_files]
                test_file_indices = num_validation_files + num_test_files
                test_file_list = files[num_validation_files+1:test_file_indices]
                training_file_list = files[test_file_indices:]
                print ('total number of files {}'.format(length_files))
                print ('number of validation files {}'.format(len(validation_file_list)))
                print ('number of training files {}'.format(len(training_file_list)))
                print ('number of test files {}'.format(len(test_file_list))) 
                for f in validation_file_list:
                    src_filepath = os.path.join(directory, f)
                    subd_name = os.path.dirname(src_filepath).split('/')[-1]
                    subd_path = os.path.join(out_path, 'valid', subd_name)
                    if not os.path.exists(subd_path):
                        os.makedirs(subd_path)
                    shutil.copy(src_filepath, subd_path)
                for f in test_file_list:
                    src_filepath = os.path.join(directory, f)
                    subd_name = os.path.dirname(src_filepath).split('/')[-1]
                    subd_path = os.path.join(out_path, 'test', subd_name)
                    if not os.path.exists(subd_path):
                        os.makedirs(subd_path)
                    shutil.copy(src_filepath, subd_path)
                for f in training_file_list:
                    src_filepath = os.path.join(directory, f)
                    subd_name = os.path.dirname(src_filepath).split('/')[-1]
                    subd_path = os.path.join(out_path, 'train', subd_name)
                    if not os.path.exists(subd_path):
                        os.makedirs(subd_path)
                    shutil.copy(src_filepath, subd_path)
