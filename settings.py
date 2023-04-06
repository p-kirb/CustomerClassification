labelledDatapath = "../data/existing-customers.xlsx"
unlabelledDatapath = "../data/potential-customers.xlsx"

redundantFeatures = ['RowID', 'education']
categoricalFeatures = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
numericalFeatures = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']



#found from cross-validated grid search
optimalHyperparameters = {
    "n_estimators": 90,
    "min_samples_split": 7,
    "max_features": None,
    "max_depth": 10,
    "class_weight": "balanced_subsample"
}



#values for computing expected profit
sendingCost = -10
highIncomeProfit = 980
lowIncomeProfit = -310

highIncomeAcceptance = 0.1
lowIncomeAcceptance = 0.05


rowIDsFile = "rowIDs.txt"