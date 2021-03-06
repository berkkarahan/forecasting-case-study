# forecasting-case-study
A forecasting case study. The final output for the challenge is reached through running make_prediction.py.

There are several jupyter notebooks as well as a general overview showing / explaining the reasoning behind what is used in the final submission stage. To view the notebooks, please see initial-sub and other branches.

 - What is different from the initial submission is;
  - TimeSeriesSplit is no longer used, replaced with RepeatedKFold with cv=5 and repeats=2.
  - Only inter-quantile range is used for anomaly detection.

## to run the make_prediction.py
Requirements can easily be installed using requirements.txt after creating a virtual environment of your choice.

Once the requirements are met, you can run the file like;

* python make_prediction.py --fname <the_excel_filename> --seed <random_seed>

defaults for filename and seed are, data_analytics.xlsx and [123123](https://www.youtube.com/watch?v=2vjPBrBU-TM).

***final-note;***
In order to run the script and notebooks succesfully, make sure they are on the same folder with the /helpers folder. There are some custom functions / classes which might be used.
