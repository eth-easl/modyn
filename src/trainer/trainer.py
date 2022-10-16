class Trainer:

    def train(self, train_ds):
        # TODO: Implement
        pass

    def fit(self):
        # TODO: Implement
        pass

    def online_training(self, num_columns):
        pass
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


    def get_next(self):
        # TODO: Implement grpc to access data from the online data feeders as propagated by the data orchestrator
        pass
