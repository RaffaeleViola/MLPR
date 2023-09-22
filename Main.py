import os

import DatasetAnalysis
from evaluation import LR, Calibration, FusionModel, GMM, SVM
from validation import LR, SVM, GMM, Gaussians, Calibration, FusionModel


def main():
    # creating main folders
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(f'{absolute_path}/Images'):
        os.mkdir(f'{absolute_path}/Images')
    if not os.path.exists(f'{absolute_path}/scores'):
        os.mkdir(f'{absolute_path}/scores')
    # -------------------------------- DATASET ANALYSIS
    DatasetAnalysis.datasetAnalysis()

    # -------------------------------- VALIDATION
    # Gaussians Validation
    Gaussians.GaussiansValidation()
    # Logistic Regression Evaluation
    LR.LRValidation()
    # SVM Evaluation
    SVM.SVMValidation()
    # GMM Evaluation
    GMM.GMMValidation()
    # Calibration Evaluation
    Calibration.CalibrationValidation()
    # FusionModel Evaluation
    FusionModel.FusionValidation()

    # -------------------------------- EVALUATION
    # Logistic Regression Evaluation
    LR.LogisticRegressionEvaluation()
    # SVM Evaluation
    SVM.SVmEvaluation()
    # GMM Evaluation
    GMM.GMMEvaluation()
    # Calibration Evaluation
    Calibration.CalibrationEvaluation()
    # FusionModel Evaluation
    FusionModel.FusionEvaluation()


main()
