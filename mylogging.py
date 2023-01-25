import logging


def setup_absl_logging():
    # https://stackoverflow.com/questions/59654893/python-absl-logging-without-timestamp-module-name
    from absl import logging as absl_logging

    absl_logging.use_absl_handler()
    absl_logging.set_verbosity(logging.INFO)

    formatter = logging.Formatter('%(asctime)-15s %(name)s %(levelname)s %(message)s')
    handler = absl_logging.get_absl_handler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)


setup_absl_logging()
