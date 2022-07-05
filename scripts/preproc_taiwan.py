'''
We have cleaned versions of Taiwan and NHANES data prepared
Workflow:
    1) Imputing missing data with multivariate imputation
    2) Computing summary statistics
    3) Making heatmaps of the Spearman rank correlations
    4) Preprocessing features ready for modelling
'''

#%% SUMMARY STATISTICS

#   Reading in data

taiwanData = pd.read_csv('{Insert path to Taiawan data}.csv')

varlist = {'continuous'  : ['Age', 'BMI'],
           'categorical' : ['Sex', 'Education', 'Smoking', 'Alcohol',
                            'Abdominal', 'Hypertension', 'Glucose', 'Triglycerides', 'HDL',
                            'Dentist', 'Mobile', 'Floss', 'Periodontitis']}


contVars = list(varlist.values())[0]
catVars = list(varlist.values())[1]
combinedVars = contVars + catVars

taiwanSummary = taiwanData[combinedVars]

#   Multivariate imputation

taiwanImputed = multiImputation(data = taiwanSummary, state = seed)

print(taiwanImputed['Periodontitis'].value_counts(sort = False),
      taiwanImputed['Periodontitis'].value_counts(sort = False, normalize = True) * 100)
    
#   Summarising data

summaryStats(data = taiwanImputed,
             continuousVars = contVars, 
             categoricalVars = catVars, 
             group = 'Periodontitis')


#%% HEATMAP

##  Heatmap showing correlation between the variables

plotVarsTaiwan = taiwanImputed.copy()

plotVarsTaiwan.rename(columns = {'Sex' : 'Female',
                                 'Hypertension' : 'Hypertens',
                                 'Triglycerides' : 'Triglyc',
                                 'Periodontitis' : 'Perio'},
                      inplace = True)

mapTaiwan = corrMap(data = plotVarsTaiwan,
                    scale = 0.5,
                    corrMethod = 'spearman',
                    size = 5,
                    dp = '.2f')

mapTaiwan.figure.savefig('{Insert path to save Taiawan heatmap}', dpi = 600)

#%% PREPROCESSING

#   Normalising data

taiwanNorm = normaliseFeatures(data = taiwanImputed)

taiwanScaledFeatures = taiwanNorm.loc[: , 'Age' : 'Floss'].copy()
taiwanUnscaledFeatures = taiwanImputed.loc[: , 'Age' : 'Floss'].copy()

taiwanLabels = taiwanImputed.loc[: , 'Periodontitis'].copy()

#   RFECV feature selection

param_grid = {'n_estimators'            : range(20, 100, 20),
              'max_depth'               : range(3, 8, 1)}

grid = GridSearchCV(GradientBoostingClassifier(random_state = seed,
                                               learning_rate = 0.1,
                                               min_samples_split = 40,
                                               min_samples_leaf = 1,
                                               max_features = 'sqrt',
                                               subsample = 0.8),
                    param_grid, cv = 10)

grid.fit(taiwanScaledFeatures, taiwanLabels)
grid.best_params_

taiwanEstimator = GradientBoostingClassifier(random_state = seed,
                                             learning_rate = 0.1,
                                             n_estimators = {Insert optimal param value},
                                             max_depth = {Insert optimal param value},
                                             min_samples_split = 40,
                                             min_samples_leaf = 1,
                                             max_features = 'sqrt',
                                             subsample = 0.8)


taiwanSelectedFeatures = rfecvSelector(features = taiwanScaledFeatures,
                                       labels = taiwanLabels,
                                       estimator = taiwanEstimator,
                                       step = 1, cv = 10)

#   Principal components analysis

taiwanPcaFeatures = pcaFeatures(taiwanUnscaledFeatures)
