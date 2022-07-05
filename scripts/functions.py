'''
Defining functions to be used throughout
'''

seed = {Insert your seed here}

#%% MULTIVARIATE IMPUTATION

#   Takes a Pandas dataframe and a random state for IterativeImputer
#   Will clip all values to integer values so they can be used for later purposes in the modelling

def multiImputation(data, state):
    
    imputer = IterativeImputer(random_state = state)
    imputedTransform = imputer.fit_transform(data)
    
    imputedData = pd.DataFrame(imputedTransform,
                               columns = data.columns)
    
    imputedData = imputedData.round(decimals = 0)
    
    imputedData = imputedData.astype(int)
    
    imputedData = imputedData.clip(lower = 0)
    
    return imputedData

#%% SUMMARY STATISTICS

#   Takes a Pandas dataframe, a variable you would like to summarise, 
#   and the variables you would like to stratify (continuous and categrical accepted)
#   For continuous variables the function returns mean and SD
#   For categorical variables the function returns counts and %

def summaryStats(data, continuousVars, categoricalVars, group):
    
    allVars = continuousVars + categoricalVars
    
    results0 = []
    results1 = []
    
    for variable in allVars:
        
        if group in allVars:
            
            allVars.remove(group)
    
        if variable in continuousVars:
        
            average0 = data.groupby(group)[variable].mean().round(decimals = 1)[0]
            deviation0 = data.groupby(group)[variable].std().round(decimals = 1)[0]
        
            average1 = data.groupby(group)[variable].mean().round(decimals = 1)[1]
            deviation1 = data.groupby(group)[variable].std().round(decimals = 1)[1]
        
            summary0 = str(str(average0) + ' (' + str(deviation0) + ')')
            summary1 = str(str(average1) + ' (' + str(deviation1) + ')')
        
            results0.append(summary0)
            results1.append(summary1)
        
    
        else:
        
            count0 = data.groupby(group)[variable].value_counts(sort = False)[0][0]
            proportion0 = data.groupby(group)[variable].value_counts(sort = False, normalize = True)[0][0] * 100
            proportion0 = proportion0.round(decimals = 1)
                
            count1 = data.groupby(group)[variable].value_counts(sort = False)[0][1]
            proportion1 = data.groupby(group)[variable].value_counts(sort = False, normalize = True)[0][1] * 100
            proportion1 = proportion1.round(decimals = 1)
        
            count2 = data.groupby(group)[variable].value_counts(sort = False)[1][0]
            proportion2 = data.groupby(group)[variable].value_counts(sort = False, normalize = True)[1][0] * 100
            proportion2 = proportion2.round(decimals = 1)
                
            count3 = data.groupby(group)[variable].value_counts(sort = False)[1][1]
            proportion3 = data.groupby(group)[variable].value_counts(sort = False, normalize = True)[1][1] * 100
            proportion3 = proportion3.round(decimals = 1)
        
            summary0 = str(str(count0) + ' (' + str(proportion0) + ')')
            summary1 = str(str(count1) + ' (' + str(proportion1) + ')')
            summary2 = str(str(count2) + ' (' + str(proportion2) + ')')
            summary3 = str(str(count3) + ' (' + str(proportion3) + ')')
        
            results0.append([summary0, summary1])
            results1.append([summary2, summary3])
            
    output = pd.DataFrame(list(zip(results0, results1)),
                          columns = ['Group 0', 'Group 1'])
    
    output.index = allVars
    
    return output    
        

#%% HEATMAP

#   Creates heatmap for correlation between variables
#   Takes data, scale (Seaborn), method of correlation, size of plot (Seaborn), and decimal place formatting

def corrMap(data, scale, corrMethod, size, dp):
    
    sb.set(font_scale = scale)
    
    corr = data.corr(method = corrMethod)
    
    heatmap = sb.heatmap(corr,
                         annot = True,
                         annot_kws = {"size": size},
                         fmt = dp)
    
#%% PREPROCESSING

#   Takes data and normalises all columns

def normaliseFeatures(data):
    
    scaler = preprocessing.MinMaxScaler()
    features = data.columns
    features_fit = scaler.fit_transform(data)
    
    normalised = pd.DataFrame(features_fit,
                              columns = features)
    
    return normalised

#   RFECV feature selection
#   Takes features to select from, corresponding labels, the estimator you would like to use, aswell as step and cv params
    
def rfecvSelector(features, labels, estimator, step, cv):
    
    selector = RFECV(estimator, step = step, cv = cv)
    
    selector = selector.fit(features, labels)
    
    status = selector.support_
    
    selectedFeatures = features.loc[: , status]
    
    return selectedFeatures

#   PCA feature transformation
#   Takes features and returns PCs from Minka-MLE PCA

def pcaFeatures(features):
    
    normFeatures = StandardScaler().fit_transform(features)
    pca = PCA(n_components = 'mle', svd_solver = 'full')
    
    principalComps = pca.fit_transform(normFeatures)
    
    return principalComps

#%% MODEL TUNING

#   Tunes algorithms by grid search CV and returns optimal hyperparameter values

def gridSearch(grid, features, labels):
    
    grid.fit(features, labels)
    
    params = grid.best_params_
    
    return params

#%% MODEL VALIDATION

#   Bootstrap sample

def bootstrapSample(data, n, replace, state):
    
    sample = data.sample(n = n,
                         replace = replace,
                         random_state = state)
    
    return sample   

#   Confidence intervals

def confInt(x, n):
    
    upperCI = x + 1.96 * sqrt( (x * (1 - x)) / n)
    lowerCI = x - 1.96 * sqrt( (x * (1 - x)) / n)
    
    return upperCI, lowerCI

#   Fomatting outcomes

def formatter(x, dp):
    
    x = np.round(x * 100, decimals = dp)
    
    return x
    
#   Compute performance metrics

def modelValidation(model, trainPred, trainLabs, testPred, testLabs, dp):
        
    model.fit(trainPred, trainLabs)
    
    predictedVals = model.predict(testPred)
    cm = confusion_matrix(predictedVals, testLabs)
    
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    
    total = tp + tn + fp + fn
        
    accuracy = (tp + tn) / total
    acc_upper = confInt(x = accuracy, n = total)[0]
    acc_lower = confInt(x = accuracy, n = total)[1]
    
    sensitivity = tp / (tp + fn)
    sens_upper = confInt(x = sensitivity, n = total)[0]
    sens_lower = confInt(x = sensitivity, n = total)[1]
    
    specificity = tn / (fp + tn)
    spec_upper = confInt(x = specificity, n = total)[0]
    spec_lower = confInt(x = specificity, n = total)[1]
    
    ppv = tp / (tp + fp)
    ppv_upper = confInt(x = ppv, n = total)[0]
    ppv_lower = confInt(x = ppv, n = total)[1]
    
    npv = tn / (tn + fn)
    npv_upper = confInt(x = npv, n = total)[0]
    npv_lower = confInt(x = npv, n = total)[1] 
    
    roc = round(roc_auc_score(testLabs, model.predict(testPred)), 2)
    roc_upper = round(confInt(x = roc, n = total)[0], 2)
    roc_lower = round(confInt(x = roc, n = total)[1], 2)
    
    results = [str(str(roc) + '\n(' + str(roc_lower) + ', ' + str(roc_upper) + ')'),
               str(str(formatter(accuracy, dp)) + '\n('  + str(formatter(acc_lower, dp)) + ', ' + str(formatter(acc_upper, dp))),
               str(str(formatter(sensitivity, dp)) + '\n('  + str(formatter(sens_lower, dp)) + ', ' + str(formatter(sens_upper, dp))),
               str(str(formatter(specificity, dp)) + '\n('  + str(formatter(spec_lower, dp)) + ', ' + str(formatter(spec_upper, dp))),
               str(str(formatter(ppv, dp)) + '\n('  + str(formatter(ppv_lower, dp)) + ', ' + str(formatter(ppv_upper, dp))),
               str(str(formatter(npv, dp)) + '\n('  + str(formatter(npv_lower, dp)) + ', ' + str(formatter(npv_upper, dp)))]
    
    output = pd.DataFrame(data = results).T
    
    output.set_axis(['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV'],
                    axis = 1, inplace = True)
    
    return output
        
#   Compute performance metrics for list of models
    
def performanceMetrics(modelList, trainPred, trainLabs, testPred, testLabs, rownames, dp):
    
    results = []
    
    for i in modelList:
        
        modelPerformance = modelValidation(model = i,
                                           trainPred = trainPred,
                                           trainLabs = trainLabs,
                                           testPred = testPred,
                                           testLabs = testLabs,
                                           dp = dp)
        
        results.append(modelPerformance)
    
    results = pd.concat(results)
    results.set_axis(rownames, axis = 0, inplace = True)
    
    return results
