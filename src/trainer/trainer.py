def train(train_ds):
    # TODO [asridhar]: Implement
    pass

def fit():
    # TODO [asridhar]: Implement
    pass

def online_training(num_columns):
    # Request a new batch(es) from data_loader
    # TODO: We will have to figure out a customizable policy on when and how to retrain/fit to the model
    # After how much time and so forth
    # online_train_ds = tfio.experimental.streaming.KafkaBatchIODataset(
    #    topics=['benchmark-train'],
    #    group_id='cgonline',
    #    servers='kafka:29092',
    #    stream_timeout=10000, # in milliseconds, to block indefinitely, set it to -1.
    #    configuration=[
    #        'session.timeout.ms=7000',
    #        'max.poll.interval.ms=8000',
    #        'auto.offset.reset=earliest'
    #    ],
    #)

    for mini_ds in online_train_ds:
        mini_ds = mini_ds.shuffle(buffer_size=32)
        mini_ds = mini_ds.map(lambda x: decode_kafka_item(x, num_columns))
        mini_ds = mini_ds.batch(32)
        if len(mini_ds) > 0:
            fit()