import os
from datetime import datetime
import time
import threading
import json
import logging
import subprocess
timestamp = datetime.now()
logging.basicConfig(filename=f'logs/warnings/{timestamp}.log',level=logging.DEBUG)
logging.captureWarnings(True)

from kafka import KafkaProducer
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio

COLUMNS = ['id','click','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain','app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15','C16','C17','C18','C19','C20','C21']

def setup():
    print(f'{datetime.now()} Starting up Zookeeper')
    zookeeper_log = open(f'logs/zookeeper/{timestamp}.log', 'a')
    c = subprocess.Popen(['zookeeper-server-start /opt/homebrew/etc/kafka/zookeeper.properties'], stdout=zookeeper_log, stderr=zookeeper_log, shell=True)

    print(f'{datetime.now()} Starting up Kafka')
    log = open(f'logs/kafka/{timestamp}.log', 'a')
    c = subprocess.Popen(['kafka-server-start /opt/homebrew/etc/kafka/server.properties'], stdout=log, stderr=log, shell=True)

    print(f'{datetime.now()} Waiting 10s for services to startup')
    time.sleep(10)

    os.system('kafka-topics --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 1 --topic benchmark-train')
    os.system('kafka-topics --create --bootstrap-server 127.0.0.1:9092 --replication-factor 1 --partitions 2 --topic benchmark-test')

    os.system('kafka-topics --describe --bootstrap-server 127.0.0.1:9092 --topic benchmark-train')
    os.system('kafka-topics --describe --bootstrap-server 127.0.0.1:9092 --topic benchmark-test')

def error_callback(exc):
    raise Exception('Error while sendig data to kafka: {0}'.format(str(exc)))

def load_data(train_file):
    file_iterator = pd.read_csv(train_file, header=None, chunksize=100000, names=COLUMNS)

    data_df = next(file_iterator)

    train_df, test_df = train_test_split(data_df, test_size=0.4, shuffle=True)

    x_train_df = train_df.drop(['click'], axis=1)
    y_train_df = train_df['click']

    x_test_df = test_df.drop(['click'], axis=1)
    y_test_df = test_df['click']

    x_train = list(filter(None, x_train_df.to_csv(index=False).split("\n")[1:]))
    y_train = list(filter(None, y_train_df.to_csv(index=False).split("\n")[1:]))

    x_test = list(filter(None, x_test_df.to_csv(index=False).split("\n")[1:]))
    y_test = list(filter(None, y_test_df.to_csv(index=False).split("\n")[1:]))

    write_to_kafka("benchmark-train", zip(x_train, y_train))
    write_to_kafka("benchmark-test", zip(x_test, y_test))

    return len(x_train_df.columns)

def write_to_kafka(topic_name, items):
    count=0
    producer = KafkaProducer(bootstrap_servers=['127.0.0.1:9092'])
    for message, key in items:
        producer.send(topic_name, key=key.encode('utf-8'), value=message.encode('utf-8')).add_errback(error_callback)
        count+=1
    producer.flush()
    print('{0} Wrote {1} messages into topic: {2}'.format(datetime.now(), count, topic_name))

def decode_kafka_item(item, num_columns):
    message = tf.io.decode_csv(item.message, [[0.0] for i in range(num_columns)])
    key = tf.strings.to_number(item.key)
    return (message, key)

def train(train_ds):
    # TODO: Implement
    pass

def fit():
    # TODO: Implement
    pass

def online_training(num_columns):
    online_train_ds = tfio.experimental.streaming.KafkaBatchIODataset(
        topics=['benchmark-train'],
        group_id='cgonline',
        servers='127.0.0.1:9092',
        stream_timeout=10000, # in milliseconds, to block indefinitely, set it to -1.
        configuration=[
            'session.timeout.ms=7000',
            'max.poll.interval.ms=8000',
            'auto.offset.reset=earliest'
        ],
    )

    for mini_ds in online_train_ds:
        mini_ds = mini_ds.shuffle(buffer_size=32)
        mini_ds = mini_ds.map(lambda x: decode_kafka_item(x, num_columns))
        mini_ds = mini_ds.batch(32)
        if len(mini_ds) > 0:
            fit()

def run_experiment(num_columns):
    print(f'{datetime.now()} Reading messages from Kafka')
    BATCH_SIZE=64
    SHUFFLE_BUFFER_SIZE=64
    train_ds = tfio.IODataset.from_kafka('benchmark-train', partition=0, offset=0, servers='127.0.0.1:9092',)
    train_ds = train_ds.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
    train_ds = train_ds.map(lambda x: decode_kafka_item(x, num_columns))
    train_ds = train_ds.batch(BATCH_SIZE)

    print(f'{datetime.now()} Training')
    train(train_ds)

    online_training(num_columns)

def teardown():
    print(f'{datetime.now()} Tearing down')
    os.system('zookeeper-server-stop')
    os.system('kafka-server-stop')

def main():

    print("tensorflow-io version: {}".format(tfio.__version__))
    print("tensorflow version: {}".format(tf.__version__))

    setup()

    num_columns = load_data('../data/train/train_mini.csv')

    run_experiment(num_columns)

if __name__ == "__main__":
    try:
        main()

    finally:
        teardown()
