import math
import xgboost as xgb
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score

class ModelRunner:
    """
    This class handles all ML model fit and prediction processes
    """
    def __init__(self,model_type,X,y,problem):
        self.model_type = model_type
        self.X = X
        self.y = y
        self.problem = problem


    def runner(self):
        """
        Runner method
        returns score of model prediction
        """
        #decide model
        model = self._decide_model()
        #get X,y
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=123)
        #run model
        y_pred, model = self._run_model(model,X_train,y_train,X_test)
        #evaluation metrics
        score = self._evaluate(y_test,y_pred)
        return score

    def _decide_model(self):
        if self.model_type == "Linear Regression":
            model = LinearRegression()
        elif self.model_type == "XGBoost":
            model = xgb.XGBRegressor()
        elif self.model_type == "Logistic Regression":
            model = LogisticRegression()
        elif self.model_type == "Decision Tree":
            model = DecisionTreeClassifier()
        return model

    def _run_model(self,model,X_train,y_train,X_test):
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        return y_pred, model


    def _evaluate(self,y_test,y_pred):
        """
        Root mean square for Regression
        Accuracy for Classfication
        """
        if self.problem == "regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = round(math.sqrt(mse),2)
            return rmse
        else:
            score = round(accuracy_score(y_test,y_pred),2)
            return score