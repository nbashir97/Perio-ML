'''
Using grid search cross-validation to tune model parameters
Messy code because each model has specific parameters which need to be individually tuned
'''

#%% SPECFIC FUNCTION FOR TUNING MODELS TO THESE DATASETS

features = [taiwanSelectedFeatures,
            taiwanPcaFeatures,
            nhanesSelectedFeatures,
            nhanesPcaFeatures]

labels = [taiwanLabels,
          nhanesLabels]

def modelTuner():
    
    for i in range(0, 4):
        
        if i < 2:
            print(gridSearch(grid = grid, features = features[i], labels = labels[0]))
        
        else:
            print(gridSearch(grid = grid, features = features[i], labels = labels[1]))

#%% OPTIMISING PARAMETERS

#   AdaBoost

param_grid = {'n_estimators' : range(20, 100, 10)}

grid = GridSearchCV(AdaBoostClassifier(random_state = seed,
                                       learning_rate = 0.1),
                    param_grid, cv = 10)

modelTuner()

AdaBoostRfecvTaiwan = AdaBoostClassifier(random_state = seed,
                                         n_estimators = {Insert optimal param value},
                                         learning_rate = 0.1)

AdaBoostPcaTaiwan = AdaBoostClassifier(random_state = seed,
                                       n_estimators = {Insert optimal param value},
                                       learning_rate = 0.1)

AdaBoostRfecvNhanes = AdaBoostClassifier(random_state = seed,
                                         n_estimators = {Insert optimal param value},
                                         learning_rate = 0.1)

AdaBoostPcaNhanes = AdaBoostClassifier(random_state = seed,
                                       n_estimators = {Insert optimal param value},
                                       learning_rate = 0.1)

#   ANN

param_grid = {'hidden_layer_sizes'  : [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,)]}

grid = GridSearchCV(MLPClassifier(random_state = seed),
                    param_grid, cv = 10)

modelTuner()

AnnRfecvTaiwan = MLPClassifier(random_state = seed, hidden_layer_sizes = {Insert optimal param value})

AnnPcaTaiwan = MLPClassifier(random_state = seed, hidden_layer_sizes = {Insert optimal param value})

AnnRfecvNhanes = MLPClassifier(random_state = seed, hidden_layer_sizes = {Insert optimal param value})

AnnPcaNhanes = MLPClassifier(random_state = seed, hidden_layer_sizes = {Insert optimal param value})

#   Decision Trees

param_grid = {'criterion'           : ['gini', 'entropy'],
              'min_samples_split'   : range(2, 10, 2),
              'min_samples_leaf'    : range(1, 5, 1)}

grid = GridSearchCV(DecisionTreeClassifier(random_state = seed),
                    param_grid, cv = 10)

modelTuner()

TreesRfecvTaiwan = DecisionTreeClassifier(random_state = seed,
                                          criterion = {Insert optimal param value},
                                          min_samples_leaf = {Insert optimal param value},
                                          min_samples_split = {Insert optimal param value})

TreesPcaTaiwan = DecisionTreeClassifier(random_state = seed,
                                        criterion = {Insert optimal param value},
                                        min_samples_leaf = {Insert optimal param value},
                                        min_samples_split = {Insert optimal param value})

TreesRfecvNhanes = DecisionTreeClassifier(random_state = seed,
                                          criterion = {Insert optimal param value},
                                          min_samples_leaf = {Insert optimal param value},
                                          min_samples_split = {Insert optimal param value})

TreesPcaNhanes = DecisionTreeClassifier(random_state = seed,
                                        criterion = {Insert optimal param value},
                                        min_samples_leaf = {Insert optimal param value},
                                        min_samples_split = {Insert optimal param value})
#   Gaussian Process

param_grid = {'n_restarts_optimizer'    : range(0, 5),
              'max_iter_predict'        : range(100, 500, 100)}

grid = GridSearchCV(GaussianProcessClassifier(random_state = seed),
                    param_grid, cv = 10)

modelTuner()

ProcessRfecvTaiwan = GaussianProcessClassifier(random_state = seed,
                                               n_restarts_optimizer = {Insert optimal param value},
                                               max_iter_predict = {Insert optimal param value})

ProcessPcaTaiwan = GaussianProcessClassifier(random_state = seed,
                                             n_restarts_optimizer = {Insert optimal param value},
                                             max_iter_predict = {Insert optimal param value})

ProcessRfecvNhanes = GaussianProcessClassifier(random_state = seed,
                                               n_restarts_optimizer = {Insert optimal param value},
                                               max_iter_predict = {Insert optimal param value})

ProcessPcaNhanes = GaussianProcessClassifier(random_state = seed,
                                             n_restarts_optimizer = {Insert optimal param value},
                                             max_iter_predict = {Insert optimal param value})

#   KNN

param_grid = {'n_neighbors'    : range(5, 20, 5),
              'weights'        : ['uniform', 'distance'],
              'p'              : [1, 2]}

grid = GridSearchCV(KNeighborsClassifier(),
                    param_grid, cv = 10)

modelTuner()

KnnRfecvTaiwan = KNeighborsClassifier(n_neighbors = {Insert optimal param value},
                                      weights = {Insert optimal param value},
                                      p = {Insert optimal param value})

KnnPcaTaiwan = KNeighborsClassifier(n_neighbors = {Insert optimal param value},
                                    weights = {Insert optimal param value},
                                    p = {Insert optimal param value})

KnnRfecvNhanes = KNeighborsClassifier(n_neighbors = {Insert optimal param value},
                                      weights = {Insert optimal param value},
                                      p = {Insert optimal param value})

KnnPcaNhanes = KNeighborsClassifier(n_neighbors = {Insert optimal param value},
                                    weights = {Insert optimal param value},
                                    p = {Insert optimal param value})
#   SVC

param_grid = {'penalty'         : ['l1', 'l2'],
              'loss'            : ['loss', 'squared_hinge'],
              'dual'            : [True, False],
              'C'               : [x * 0.1 for x in range(1, 20)],
              'fit_intercept'   : [True, False]}

grid = GridSearchCV(LinearSVC(random_state = seed),
                    param_grid, cv = 10)

modelTuner()

SvcRfecvTaiwan = LinearSVC(random_state = seed,
                           C = {Insert optimal param value},
                           dual = {Insert optimal param value},
                           fit_intercept = {Insert optimal param value},
                           loss = {Insert optimal param value},
                           penalty = {Insert optimal param value})

SvcPcaTaiwan = LinearSVC(random_state = seed,
                           C = {Insert optimal param value},
                           dual = {Insert optimal param value},
                           fit_intercept = {Insert optimal param value},
                           loss = {Insert optimal param value},
                           penalty = {Insert optimal param value})

SvcRfecvNhanes = LinearSVC(random_state = seed,
                           C = {Insert optimal param value},
                           dual = {Insert optimal param value},
                           fit_intercept = {Insert optimal param value},
                           loss = {Insert optimal param value},
                           penalty = {Insert optimal param value})

SvcPcaNhanes = LinearSVC(random_state = seed,
                           C = {Insert optimal param value},
                           dual = {Insert optimal param value},
                           fit_intercept = {Insert optimal param value},
                           loss = {Insert optimal param value},
                           penalty = {Insert optimal param value})
#   LDA

param_grid = {'solver'  :   ['svd', 'lsq', 'eigen']}

grid = GridSearchCV(LinearDiscriminantAnalysis(),
                    param_grid, cv = 10)

modelTuner()

LdaRfecvTaiwan = LinearDiscriminantAnalysis(solver = {Insert optimal param value})

LdaPcaTaiwan = LinearDiscriminantAnalysis(solver = {Insert optimal param value})

LdaRfecvNhanes = LinearDiscriminantAnalysis(solver = {Insert optimal param value})

LdaPcaNhanes = LinearDiscriminantAnalysis(solver = {Insert optimal param value})

#   Logistic Regression

param_grid = {'penalty'         : ['none', 'l1', 'l2', 'elasticnet'],
              'C'               : [x * 0.1 for x in range(1, 20)],
              'fit_intercept'   : [True, False]}

grid = GridSearchCV(LogisticRegression(random_state = seed, solver = 'saga'),
                    param_grid, cv = 10)

modelTuner()

LogisticRfecvTaiwan = LogisticRegression(random_state = seed,
                                         solver = 'saga', penalty = {Insert optimal param value}, 
                                         C = {Insert optimal param value}, fit_intercept = {Insert optimal param value})

LogisticPcaTaiwan = LogisticRegression(random_state = seed,
                                       solver = 'saga', penalty = {Insert optimal param value}, 
                                       C = {Insert optimal param value}, fit_intercept = {Insert optimal param value})

LogisticRfecvNhanes = LogisticRegression(random_state = seed,
                                         solver = 'saga', penalty = {Insert optimal param value}, 
                                         C = {Insert optimal param value}, fit_intercept = {Insert optimal param value})

LogisticPcaNhanes = LogisticRegression(random_state = seed,
                                       solver = 'saga', penalty = {Insert optimal param value}, 
                                       C = {Insert optimal param value}, fit_intercept = {Insert optimal param value})

#   RF

param_grid = {'n_estimators'        : range(20, 200, 20),
              'criterion'           : ['gini', 'entropy'],
              'min_samples_split'   : range(2, 10, 2),
              'min_samples_leaf'    : range(1, 5, 1)}

grid = GridSearchCV(RandomForestClassifier(random_state = seed, max_samples = 0.8),
                    param_grid, cv = 10)

modelTuner()

ForestRfecvTaiwan = RandomForestClassifier(random_state = seed, max_samples = 0.8,
                                           n_estimators = {Insert optimal param value}, criterion = {Insert optimal param value}, 
                                           min_samples_split = {Insert optimal param value}, min_samples_leaf = {Insert optimal param value})

ForestPcaTaiwan = RandomForestClassifier(random_state = seed, max_samples = 0.8,
                                         n_estimators = {Insert optimal param value}, criterion = {Insert optimal param value}, 
                                           min_samples_split = {Insert optimal param value}, min_samples_leaf = {Insert optimal param value})

ForestRfecvNhanes = RandomForestClassifier(random_state = seed, max_samples = 0.8,
                                           n_estimators = {Insert optimal param value}, criterion = {Insert optimal param value}, 
                                           min_samples_split = {Insert optimal param value}, min_samples_leaf = {Insert optimal param value})

ForestPcaNhanes = RandomForestClassifier(random_state = seed, max_samples = 0.8,
                                         n_estimators = {Insert optimal param value}, criterion = {Insert optimal param value}, 
                                           min_samples_split = {Insert optimal param value}, min_samples_leaf = {Insert optimal param value})

#   Naive Bayes

BayesRfecvTaiwan = GaussianNB()

BayesPcaTaiwan = GaussianNB()

BayesRfecvNhanes = GaussianNB()

BayesPcaNhanes = GaussianNB()
