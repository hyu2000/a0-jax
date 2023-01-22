import datetime
import logging


def mylog(*argv):
    now = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(now, argv[0] % argv[1:])


logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')
logging.info = mylog
