import argparse
import pickle
import os


def save_object(object, directory, file_name):
    path = f'{directory}/{file_name}.pkl'
    with open(path, 'wb') as file:
        pickle.dump(object, file)


def load_object(directory, file_name):
    path = f'{directory}/{file_name}.pkl'
    if os.path.isfile(path):
        with open(path, 'rb') as open_file:
            return pickle.load(open_file)
    else:
        return None


def create_parser():
    parser = argparse.ArgumentParser(description='Parser for run')
    parser.add_argument('--seed_start',
                        '-r',
                        type=int,
                        default=0,
                        help='Seed start')
    parser.add_argument('--seed_end',
                        '-q',
                        type=int,
                        default=10,
                        help='Seed end (inclusive)')
    parser.add_argument('--affinity',
                        '-a',
                        type=int,
                        default=-1,
                        help='Affinity parameter')
    return parser



