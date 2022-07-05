'''
Computing performance metrics of the tuned models
'''

#%% SPECFIC FUNCTIONS FOR CREATING FEATURES AND LABS FROM BOOTSTRAP SAMPLES AND VALIDATING MODELS

def internalSample(data):
    
    sample = bootstrapSample(data = data, n = len(data), replace = True, state = seed)
    
    features = sample.drop('Periodontitis', axis = 1)
    labels = sample.iloc[: , -1]
    
    return features, labels

#%% TAIWAN

##  RFECV models

TaiwanModelsRfecv = list()
TaiwanModelsRfecv.append(AdaBoostRfecvTaiwan)
TaiwanModelsRfecv.append(AnnRfecvTaiwan)
TaiwanModelsRfecv.append(TreesRfecvTaiwan)
TaiwanModelsRfecv.append(ProcessRfecvTaiwan)
TaiwanModelsRfecv.append(KnnRfecvTaiwan)
TaiwanModelsRfecv.append(SvcRfecvTaiwan)
TaiwanModelsRfecv.append(LdaRfecvTaiwan)
TaiwanModelsRfecv.append(LogisticRfecvTaiwan)
TaiwanModelsRfecv.append(ForestRfecvTaiwan)
TaiwanModelsRfecv.append(BayesRfecvTaiwan)

#   Internal validation

taiwanRfecvBootstrapFeatures = internalSample(taiwanSelectedFeatures)[0]
taiwanRfecvBootstrapLabels = internalSample(taiwanSelectedFeatures)[1]

performanceMetrics(modelList = TaiwanModelsRfecv,
                   trainPred = taiwanSelectedFeatures,
                   trainLabs = taiwanLabels,
                   testPred = taiwanRfecvBootstrapFeatures,
                   testLabs = taiwanRfecvBootstrapLabels)

#   External validation

performanceMetrics(modelList = TaiwanModelsRfecv,
                   trainPred = taiwanSelectedFeatures,
                   trainLabs = taiwanLabels,
                   testPred = nhanesSelectedFeatures,
                   testLabs = nhanesLabels)

TaiwanModelsPca = list()
TaiwanModelsPca.append(AdaBoostPcaTaiwan)
TaiwanModelsPca.append(AnnPcaTaiwan)
TaiwanModelsPca.append(TreesPcaTaiwan)
TaiwanModelsPca.append(ProcessPcaTaiwan)
TaiwanModelsPca.append(KnnPcaTaiwan)
TaiwanModelsPca.append(SvcPcaTaiwan)
TaiwanModelsPca.append(LdaPcaTaiwan)
TaiwanModelsPca.append(LogisticPcaTaiwan)
TaiwanModelsPca.append(ForestPcaTaiwan)
TaiwanModelsPca.append(BayesPcaTaiwan)

#   Internal validation

taiwanPcaFeaturesDf = pd.DataFrame(data = taiwanPcaFeatures)
taiwanPcaFeaturesDf['Periodontitis'] = taiwanLabels

taiwanPcaBootstrapFeatures = internalSample(taiwanPcaFeatureDf)[0]
taiwanPcaBootstrapLabels = internalSample(taiwanPcaFeaturesDf)[1]

performanceMetrics(modelList = TaiwanModelsPca,
                   trainPred = taiwanPcaFeatures,
                   trainLabs = taiwanLabels,
                   testPred = taiwanPcaBootstrapFeatures,
                   testLabs = taiwanPcaBootstrapLabels)

#   External validation

performanceMetrics(modelList = TaiwanModelsPca,
                   trainPred = taiwanPcaFeatures,
                   trainLabs = taiwanLabels,
                   testPred = nhanesPcaFeatures,
                   testLabs = nhanesLabels)

#%% NHANES

##  RFECV models

NhanesModelsRfecv = list()
NhanesModelsRfecv.append(AdaBoostRfecvNhanes)
NhanesModelsRfecv.append(AnnRfecvNhanes)
NhanesModelsRfecv.append(TreesRfecvNhanes)
NhanesModelsRfecv.append(ProcessRfecvNhanes)
NhanesModelsRfecv.append(KnnRfecvNhanes)
NhanesModelsRfecv.append(SvcRfecvNhanes)
NhanesModelsRfecv.append(LdaRfecvNhanes)
NhanesModelsRfecv.append(LogisticRfecvNhanes)
NhanesModelsRfecv.append(ForestRfecvNhanes)
NhanesModelsRfecv.append(BayesRfecvNhanes)

#   Internal validation

nhanesRfecvBootstrapFeatures = internalSample(nhanesSelectedFeatures)[0]
nhanesRfecvBootstrapLabels = internalSample(nhanesSelectedFeatures)[1]

performanceMetrics(modelList = NhanesModelsRfecv,
                   trainPred = nhanesSelectedFeatures,
                   trainLabs = nhanesLabels,
                   testPred = nhanesRfecvBootstrapFeatures,
                   testLabs = nhanesRfecvBootstrapLabels)

# External validation

performanceMetrics(modelList = NhanesModelsRfecv,
                   trainPred = nhanesSelectedFeatures,
                   trainLabs = nhanesLabels,
                   testPred = taiwanSelectedFeatures,
                   testLabs = taiwanLabels)

##  PCA models

NhanesModelsPca = list()
NhanesModelsPca.append(AdaBoostPcaNhanes)
NhanesModelsPca.append(AnnPcaNhanes)
NhanesModelsPca.append(TreesPcaNhanes)
NhanesModelsPca.append(ProcessPcaNhanes)
NhanesModelsPca.append(KnnPcaNhanes)
NhanesModelsPca.append(SvcPcaNhanes)
NhanesModelsPca.append(LdaPcaNhanes)
NhanesModelsPca.append(LogisticPcaNhanes)
NhanesModelsPca.append(ForestPcaNhanes)
NhanesModelsPca.append(BayesPcaNhanes)

#   Internal validation

nhanesPcaFeaturesDf = pd.DataFrame(data = nhanesPcaFeatures)
nhanesPcaFeaturesDf['Periodontitis'] = nhanesLabels

nhanesPcaBootstrapFeatures = internalSample(nhanesPcaFeatureDf)[0]
nhanesPcaBootstrapLabels = internalSample(nhanesPcaFeaturesDf)[1]

performanceMetrics(modelList = NhanesModelsPca,
                   trainPred = nhanesPcaFeatures,
                   trainLabs = nhanesLabels,
                   testPred = nhanesPcaBootstrapFeatures,
                   testLabs = nhanesPcaBootstrapLabels)

#   External validation

performanceMetrics(modelList = NhanesModelsPca,
                   trainPred = nhanesPcaFeatures,
                   trainLabs = nhanesLabels,
                   testPred = taiwanPcaFeatures,
                   testLabs = taiwanLabels)
