# forecasting-case-study
A forecasting case study. The final output for the challenge is reached through running make_prediction.py . There are several jupyter notebooks showing / explaining the reasoning behind what is used in the final submission stage. 

In order to see how I reached from notebook0 to final notebook please see the flowchart.jpg file.

A general overview of the challenge can be seen in the Forecasting_Case_Study.pdf

## to run the make_prediction.py
Requirements can easily be installed using requirements.txt after creating a virtual environment of your choice.

Once the requirements are met, you can run the file like;

* pyton make_prediction.py --fname <the_excel_filename> --seed <random_seed>

defaults for filename and seed are, data_analytics.xlsx and 123123.

***final-note;***
In order to run the script and notebooks succesfully, make sure they are on the same folder with the /helpers folder. There are some custom functions / classes which might be used.
