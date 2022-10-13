import os
from datetime import datetime
import time
import threading
import json
import logging
import subprocess

def run_experiment(num_columns):
    print(f'{datetime.now()} Reading messages from Kafka')
    BATCH_SIZE=64
    SHUFFLE_BUFFER_SIZE=64
    # train_ds = tfio.IODataset.from_kafka('benchmark-train', partition=0, offset=0, servers='kafka:29092',)
    # train_ds = train_ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    # train_ds = train_ds.map(lambda x: decode_kafka_item(x, num_columns))
    # train_ds = train_ds.batch(BATCH_SIZE)

    print(f'{datetime.now()} Training')
    # train(train_ds)

    # TODO: Online training shouldn't happen immediately but only after a certain amount of new data is available.
    # This policy should be editable by the experimental scientist
    # online_training(num_columns)

def main():

    # print("tensorflow-io version: {}".format(tfio.__version__))
    # print("tensorflow version: {}".format(tf.__version__))

    # TODO: The data loading and provisioning should be moved to a different node as this should happen asynchronously to the training
    # Additionally, data storage can or should be implemented on a new node for existing data
    # num_columns = load_data('./data/train/train_mini.csv')

    # run_experiment(num_columns)

    # TODO: Implement a benchmarking node that tests the system
    pass

if __name__ == "__main__":
    main()
