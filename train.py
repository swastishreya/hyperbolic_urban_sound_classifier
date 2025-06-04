
"""Command-line entry point for training."""
# import argparse
import pandas as pd
from tqdm import tqdm

from sound_classifier.training import init_model, process_fold
# from sound_classifier.utils.data_utils import show_results

import os
import sys
import pickle
import copy
# from sound_classifier.utils.data_utils import show_results



def main():
    # parser = argparse.ArgumentParser(description="Train the sound classifier.")
    # parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    # args = parser.parse_args()
  # TODO: CHANGE THIS TO YOUR DATA PATH
  dataset_path = "/ivi/zfs/s0/original_homes/smishra/dataset/us8k_df.pkl"

  us8k_df = pd.read_pickle(dataset_path)

  FOLD_K = 1
  REPEAT = 1
  model_name = 'cnn_mm'
  # Example: 10-fold cross validation
  tot_history = []
  for i in range(REPEAT):
      history = process_fold(FOLD_K, us8k_df, epochs=100, num_of_workers=4,model_name=model_name)
      tot_history.append(history)

  print("Training completed.")

if __name__ == "__main__":
    main()
