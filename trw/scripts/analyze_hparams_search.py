import trw
import argparse
import os


description = """The purpose of this script is to analyze the hyper-parameter search and try
to make sense of the hyper-parameters by plotting the importance hyper-parameter,
loss variation relative to hyper-parameter as well as hyper-parameter covariance"""


parser = argparse.ArgumentParser(description=description)
parser.add_argument('--store_path', help='the root directory where the hyper-parameter results are stored')
parser.add_argument('--output_path', help='path where to export the analysis')
args = parser.parse_args()

store = trw.hparams.RunStoreFile(args.store_path)
runs = store.load_all_runs()
trw.hparams.analyse_hyperparameters(runs, args.output_path)
