import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest,f_classif,RFE,SelectFromModel
from sklearn.linear_model import LinearRegression, LogisticRegression

class FeatureSelector:
    """
    This class handles all feature selection process
    """
    def __init__(self,df,target_feature):
        self.df = df #uploaded dataframe to process
        self.target_feature = target_feature #target feature column name
        

    def get_result_dictionaries(self):
        """
        Returns feature selection params
        """
        self.univariate, self.ref, self.sfm, self.problem = self._runner()
        return self.univariate, self.ref, self.sfm, self.problem


    def _runner(self):
        """
        Runner function.
        """
        #x-y
        X,y,temp,column_types,problem = self.extract_x_y()
        #select k best
        univariate = self.univariate_feature_selection(X,y,temp)
        #rfe
        ref = self.ref_feature_selection(X,y,temp,column_types)
        #select from model
        sfm = self.sfm_feature_selection(X,y,temp,column_types)
        return univariate,ref,sfm,problem


    def extract_x_y(self):
        """
        Processing of dataframe
        """
        #detect feature types
        column_types = self._detect_feature_types()
        _,problem = self.__choose_estimator(column_types)
        #fill missing values with mode
        data = self._fill_missing_with_mode()
        #encoding
        data = self._encode_features(data, column_types)
        #create X and Y feature set
        X, y, temp = self._create_x_y(data)
        return X,y,temp,column_types,problem

    def _detect_feature_types(self):
        """
        detects feature types, returns a dictionary which classify all types
        """
        numerical = self.df.select_dtypes(include=[np.number]).columns.values.tolist()
        categorical = self.df.select_dtypes(include=['object']).columns.values.tolist()
        datetimes = self.df.select_dtypes(include=['datetime',np.datetime64]).columns.values.tolist()
        column_types = {"numerical":numerical,"categorical":categorical,"datetimes":datetimes}
        return column_types



    def _fill_missing_with_mode(self):
        """
        Fills missing values in dataframe,
        as default it fills with the mode of each column.
        """
        data = self.df.copy()
        for c in data.columns:
            data[c] = data[c].fillna(data[c].mode()[0])
        return data


    def _encode_features(self,data,col_types):
        """
        Encoding all features according to types of features
        """
        #standard scaler -> numerical
        data = self.__feature_scaling(col_types["numerical"],data)
        #label encoder -> categorical
        data = self.__label_encoding(col_types["categorical"],data)
        return data

    def __feature_scaling(self,numerical_cols,data):
        """
        Standard scaling function.
        """
        scaler = StandardScaler()
        if len(numerical_cols) == 1:
            data[numerical_cols[0]] = scaler.fit_transform(data[numerical_cols[0]])
        elif len(numerical_cols) > 1:
            data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
        return data
        
    def __label_encoding(self,categorical_cols,data):
        """
        Label encoding function.
        """
        encoder = LabelEncoder()
        if len(categorical_cols) == 1:
            data[categorical_cols[0]] = encoder.fit_transform(data[categorical_cols[0]])
        elif len(categorical_cols) > 1:
            data[categorical_cols] = encoder.fit_transform(data[categorical_cols])
        return data

    def _create_x_y(self,data):
        """
        return X and y arrays to use in models
        """
        y = data[self.target_feature].values.reshape(-1,1)
        temp = data.drop([self.target_feature],axis=1)
        X = temp.values
        return X, y, temp

    def univariate_feature_selection(self,X,y,temp_df,k="all"):
        """
        Univariate feature selection method.
        It uses SelectKBest, f_classif test as default
        """
        selection_params = {"score_func":f_classif,
                   "k":k}
        selector = SelectKBest(**selection_params).fit(X,y)
        scores = selector.scores_
        pvalues = selector.pvalues_
        mask = selector.get_support()
        feature_names = temp_df.iloc[:,mask].columns
        univariate = {"scores":scores,"pvalues":pvalues,"feature_names":feature_names,"X":self.df[feature_names].values}
        return  univariate

    def ref_feature_selection(self,X,y,temp_df,col_types,k=1):
        """
        Recursive Feature selection method.
        It uses RFE from sklearn.
        """
        estimator,_ = self.__choose_estimator(col_types)
        selection_params = {"estimator":estimator,
                   "n_features_to_select": k,
                   "step":1,
                   "verbose":0}
        selector = RFE(**selection_params).fit(X,y)
        ranking = selector.ranking_
        feature_names = temp_df.columns.values.tolist()
        ref = {"ranking":ranking,"feature_names":feature_names,"X":self.df[feature_names].values}
        return ref

    def __choose_estimator(self,col_types):
        """
        chooses default estimators according to target feature type and classify problem as regression or classification
        for regression, Linear Regression is default estimator.
        for classification Logistic Regression is default estimator.
        """
        if self.target_feature in col_types["numerical"]:
            estimator = LinearRegression()
            problem = "regression"
        elif self.target_feature in col_types["categorical"]:
            estimator = LogisticRegression()
            problem = "classification"
        return estimator,problem

    def sfm_feature_selection(self,X,y,temp_df,col_types,k=2):
        """
        Select From Model feature selection method.
        """
        estimator,_ = self.__choose_estimator(col_types)
        selection_params = {"estimator":estimator,
                   "threshold":0.5,
                   "max_features":k,
                   }
        selector = SelectFromModel(**selection_params).fit(X,y)
        coefs = selector.estimator_.coef_
        coefs = np.abs(coefs)
        c_means= coefs.mean(axis=0)
        feature_names = temp_df.columns.values.tolist()
        sfm = {"scores":c_means,"feature_names":feature_names,"X":self.df[feature_names].values}
        return sfm



