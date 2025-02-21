import os

from feature_cleaner import FeatureCleaner
from classifier import Classifier
from csv import DictReader


def run_stages(stages, initial_input_data=[], check_clean=True, **initial_kwargs):
    input_data = initial_input_data
    for i, stage in enumerate(stages):
        print("### STAGE", stage.__name__, "###")
        if check_clean:
            stage_path = stage._stagepath
            if os.path.exists(stage_path) and not os.listdir(stage_path) == []:
                raise RuntimeError(
                    "Make sure the folder %s is empty." % stage_path
                )
        if isinstance(input_data, str):
            with open(input_data) as input_dicts_file:
                input_dicts = list(DictReader(input_dicts_file))
        else:
            input_dicts = input_data
        if i == 0:
            stage(input_dicts, **initial_kwargs).execute()
        else:
            stage(input_dicts).execute()
        input_data = stage.resultspath


run_stages(
    [Classifier],
    FeatureCleaner.resultspath,
    check_clean=False
)
