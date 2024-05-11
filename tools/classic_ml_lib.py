from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import torch


class LinearRegressionModel(object):
    def __init__(self, n_jobs:int = 10):
        self.n_jobs = n_jobs
        self.model = LinearRegression(n_jobs=n_jobs)
        self.degree = 2

    def fit(self, train_df_, inputs_, outputs_, degree: int = 2):
        train_data_X = train_df_[inputs_].to_numpy()
        train_data_Y = train_df_[outputs_].to_numpy()

        polynomial_transform = PolynomialFeatures(degree=degree)
        train_data_X_transformed = polynomial_transform.fit_transform(train_data_X)

        self.model.fit(train_data_X_transformed, train_data_Y)
        self.degree = degree

    def predict(self, X_data, do_poly_transform: bool = True):
        input_X = X_data

        if do_poly_transform:
            polynomial_transform = PolynomialFeatures(degree=self.degree)
            input_X = polynomial_transform.fit_transform(input_X)

        return self.model.predict(input_X)

    def test(self, X_data, Y_actual, do_poly_transform: bool = True):
        predicted = self.predict(X_data, do_poly_transform)
        loss_ = torch.nn.L1Loss()

        return loss_(torch.tensor(predicted), torch.tensor(Y_actual))

