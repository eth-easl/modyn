# TODO: Make this system independent

import tensorflow as tf

def decode_kafka_item(item, num_columns):
    message = tf.io.decode_csv(item.message, [[0.0] for i in range(num_columns)])
    key = tf.strings.to_number(item.key)
    return (message, key)