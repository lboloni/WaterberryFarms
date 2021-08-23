# set up the logging system for the informative path planning project
# silence the annoying warning from GP

import logging
logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger().setLevel(logging.WARNING)
# logging.getLogger().setLevel(logging.ERROR)

def filterConvergenceWarning(logrecord):
    print("filtering")
    if logrecord.msg.find("ConvergenceWarning:") != -1:
        return 0
    return 1

logging.getLogger().addFilter(filterConvergenceWarning)
logging.captureWarnings(True)
