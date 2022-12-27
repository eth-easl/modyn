def process_complex_messages(grpc_dict):

    new_dict = {}
    for key, value in grpc_dict.items():
        if value.HasField("float_value"):
            new_value = value.float_value
        elif value.HasField("int_value"):
            new_value = value.int_value
        elif value.HasField("bool_value"):
            new_value = value.bool_value
        else:
            new_value = value.string_value
        new_dict[key] = new_value

    return new_dict
