# The basic idea is to NOT have any regular package imports here.
# This just confuses students.
import pandas as pd
from IPython import get_ipython

ipython = get_ipython()

ipython.magic("load_ext autoreload")
ipython.magic("load_ext lab_black")

ipython.magic("matplotlib inline")
ipython.magic("autoreload 2")

pd.options.display.float_format = "{:,.2f}".format
