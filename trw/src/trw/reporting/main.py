import trw
import os

#
# steps:
# 1) main.py must be copied to the model logging location
# 2) start the bokeh server with the model logging location using:
#       bokeh serve mnist_cnn_r0 --show
#
trw.reporting.report(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'reporting_sqlite.db'),
    options=trw.reporting.create_default_reporting_options())
