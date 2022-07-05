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

nhanesData = pd.read_csv('{Insert path to NHANES data}.csv')

varlist = {'continuous'  : ['Age', 'BMI'],
           'categorical' : ['Sex', 'Education', 'Smoking', 'Alcohol',
                            'Abdominal', 'Hypertension', 'Glucose', 'Triglycerides', 'HDL',
                            'Dentist', 'Mobile', 'Floss', 'Periodontitis']}


contVars = list(varlist.values())[0]
catVars = list(varlist.values())[1]
combinedVars = contVars + catVars

nhanesSummary = nhanesData[combinedVars]

#   Multivariate imputation

nhanesImputed = multiImputation(data = nhanesSummary, state = seed)

print(nhanesImputed['Periodontitis'].value_counts(sort = False),
      nhanesImputed['Periodontitis'].value_counts(sort = False, normalize = True) * 100)
    
#   Summarising data

summaryStats(data = nhanesImputed,
             continuousVars = contVars, 
             categoricalVars = catVars, 
             group = 'Periodontitis')

   
#%% HEATMAP

##  Heatmap showing correlation between the variables

plotVarsNhanes = nhanesImputed.copy()

plotVarsNhanes.rename(columns = {'Sex' : 'Female',
                                 'Hypertension' : 'Hypertens',
                                 'Triglycerides' : 'Triglyc',
                                 'Periodontitis' : 'Perio'},
                      inplace = True)

mapNhanes = corrMap(data = plotVarsNhanes,
                    scale = 0.5,
                    corrMethod = 'spearman',
                    size = 5,
                    dp = '.2f')

mapNhanes.figure.savefig('{Insert path to save NHANES heatmap}', dpi = 600)

#%% PREPROCESSING

#   Normalising data

nhanesNorm = normaliseFeatures(data = nhanesImputed)

nhanesScaledFeatures = nhanesNorm.loc[: , 'Age' : 'Floss'].copy()
nhanesUnscaledFeatures = nhanesImputed.loc[: , 'Age' : 'Floss'].copy()

nhanesLabels = nhanesImputed.loc[: , 'Periodontitis'].copy()

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

grid.fit(nhanesScaledFeatures, nhanesLabels)
grid.best_params_

nhanesEstimator = GradientBoostingClassifier(random_state = seed,
                                             learning_rate = 0.1,
                                             n_estimators = {Insert optimal param value},
                                             max_depth = {Insert optimal param value},
                                             min_samples_split = 40,
                                             min_samples_leaf = 1,
                                             max_features = 'sqrt',
                                             subsample = 0.8)


nhanesSelectedFeatures = rfecvSelector(features = nhanesScaledFeatures,
                                       labels = nhanesLabels,
                                       estimator = nhanesEstimator,
                                       step = 1, cv = 10)

#   Principal components analysis

nhanesPcaFeatures = pcaFeatures(nhanesUnscaledFeatures)
