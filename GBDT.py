
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_20newsgroups
#categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med','misc.forsale'，'comp.os.ms-windows.misc']
categories = ['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x','misc.forsale','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space','soc.religion.christian','talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)


# In[2]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)


# In[15]:


import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


from sklearn.ensemble import GradientBoostingClassifier
GBDT=GradientBoostingClassifier(
learning_rate=0.05
, n_estimators=200
, subsample=0.8
, min_samples_split=250
, min_samples_leaf=5
, max_depth=20
, init=None
, random_state=None
, max_features='sqrt'
, verbose=0
, max_leaf_nodes=None
, warm_start=False)

GBDT.fit(X_train_tfidf,twenty_train.target)

    
'''
使用测试集来评估模型好坏。
'''
from sklearn import metrics
import numpy as np
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
X_test_counts = count_vect.transform(docs_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
predicted = GBDT.predict(X_test_tfidf)
#print(predicted)
#print(twenty_test.target)
print(metrics.classification_report(twenty_test.target, predicted,target_names=twenty_test.target_names))
print("accurary\t"+str(np.mean(predicted == twenty_test.target)))

# param_test1 = {'n_estimators': list(range(20, 81, 10))}
# gsearch1 = GridSearchCV(
#     estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300, min_samples_leaf=20, max_depth=8,
#                                          max_features='sqrt', subsample=0.8, random_state=10),
#     param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
# gsearch1.fit(X, y)
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

# param_test1 = {'n_estimators': list(range(20, 81, 10))}
# gsearch1 = GridSearchCV(
#     estimator=GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300, min_samples_leaf=20, max_depth=8,
#                                          max_features='sqrt', subsample=0.8, random_state=10),
#     param_grid=param_test1, scoring='roc_auc', iid=False, cv=5)
# gsearch1.fit(X_train_tfidf,twenty_train.target)
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




