import os
from evaluation import LR, Calibration, FusionModel, GMM, SVM


def main():
    # creating main folders
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    os.mkdir(f'{absolute_path}/Images')
    os.mkdir(f'{absolute_path}/scores')

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
