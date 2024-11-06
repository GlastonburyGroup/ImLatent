def remove_attributes(config, startswith="lstm_"):
    for attr_name in list(vars(config).keys()):
        if attr_name.startswith(startswith):
            delattr(config, attr_name)