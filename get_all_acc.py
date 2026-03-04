import os
import numpy as np

run_dir = "./runs"

def read_file(model):
  fn = os.path.join(run_dir, model, "log.txt")
  with open(fn, 'r') as file:


    vals = np.zeros(shape=(150))
    lines = file.readlines()
    header, lines = lines[:10], lines[10:]

    ## Header ##

    ## Body ##
    count = 0
    for i, line in enumerate(lines):
      if i % 2 == 0:
        val_acc = float(line.split(" ")[-2][:-1])
        vals[count] = val_acc
        count += 1
    print(f"{model}\t{np.max(vals)}%\t{round(np.sum(vals) / 150,2)}%")


print(f"Model\tBest \tAvg")
models = [
  "Cnn_0_0",
  # "Cnn_1_0",
  # "test_snn_0",
  # "test_snn_1",
  "test_snn_2",
  "Cnn_2_0_test_0",
  # "Cnn_2_0_test_1",
  # "Cnn_2_0_test_2",
  # "Cnn_2_0_test_3",
  # "Cnn_3_0_test_0",
  "Cnn_3_0_test_1",
  # "Cnn_3_0_test_2",
  # "Cnn_3_0_test_3",
  # "Cnn_4_0_test_0",
  # "Cnn_4_0_test_1",
  "Cnn_4_0_test_2",
  # "Cnn_4_0_test_3",
  # "Cnn_5_0_test_0",
  # "Cnn_5_0_test_1",
  "Cnn_5_0_test_2",
  # "Cnn_5_0_test_3",
  # "Cnn_5_1",
  "Cnn_5_1_extra",
  # "Cnn_5_2",
  "Cnn_5_2_extra",

  # "test_snn_0",
  # "test_snn_1",
  # "test_snn_2",
  # "test_cnn_0",
  # "test_cnn_1",
  # "test_cnn_2",
  # "test_cnn_3",
  # "Cnn_0_0",
  # "Cnn_0_1_4",
  # "Cnn_1_1_3",
  # "Cnn_2_1_2",
  # "Cnn_3_1_1",
  # "Cnn_4_1_0",
  # "Cnn_1_0",
  # "Cnn_2_0",
  # "Cnn_3_0",
  # "DropReLUSpikeCnn_1_0",
  # "SpatialCnn",
  # "SpikeCnn_0_0",
  # "SpikeCnn_1_0",
  # "SpikeCnn_2_0",
  # "SpikeCnn_3_0",
  # "SpikeCnn_4_0",
  # "SpikeCnn_5_0",
  # "SpikeCnn_5_1",
  # "SpikeCnn_5_2",
  "SpikeCnn_5_3",
  # "cnn_dropout"
  ]

for model in models:
  read_file(model)