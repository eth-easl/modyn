from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from kafka import KafkaProducer

def error_callback(exc):
    raise Exception('Error while sendig data to kafka: {0}'.format(str(exc)))

def write_to_kafka(topic_name, items):
    count=0
    producer = KafkaProducer(bootstrap_servers=['kafka:29092'])
    for message, key in items:
        producer.send(topic_name, key=key.encode('utf-8'), value=message.encode('utf-8')).add_errback(error_callback)
        count+=1
    producer.flush()
    print('{0} Wrote {1} messages into topic: {2}'.format(datetime.now(), count, topic_name))


def load_data(train_file):
    # TODO: In the initial training, we should define the number of samples we want to use. 
    # Subsequently, the producer can produce new samples that we then train according to the retrain/fit policy
    file_iterator = pd.read_csv(train_file, header=0, chunksize=100000)

    data_df = next(file_iterator)

    train_df, test_df = train_test_split(data_df, test_size=0.4, shuffle=True)

    # TODO [asridhar]: Please select the training and testing x and y here if you want to train for something different
    # Should be made configurable in future iterations
    x_train_df = train_df.drop(['click'], axis=1)
    y_train_df = train_df['click']

    x_test_df = test_df.drop(['click'], axis=1)
    y_test_df = test_df['click']

    x_train = list(filter(None, x_train_df.to_csv(index=False).split("\n")[1:]))
    y_train = list(filter(None, y_train_df.to_csv(index=False).split("\n")[1:]))

    x_test = list(filter(None, x_test_df.to_csv(index=False).split("\n")[1:]))
    y_test = list(filter(None, y_test_df.to_csv(index=False).split("\n")[1:]))

    # TODO: Best case would be to do this split at the preprocessing engine and not at the producer
    # This would better simulate an actual datastream coming in
    write_to_kafka("benchmark-train", zip(x_train, y_train))
    write_to_kafka("benchmark-test", zip(x_test, y_test))