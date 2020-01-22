from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
import pickle
import numpy as np
import math
from datacleaning import clean_questions,preprocessing,df
# tfidf = TfidfVectorizer(sublinear_tf=True, use_idf=True ,smooth_idf=False, min_df=3,max_df = 52 ,norm='l2', encoding='utf-8', ngram_range=(1, 3), stop_words=preprocessing,max_features=1000)
X_train = clean_questions
y_train = np.array(range(1,289))
# features = tfidf.fit_transform(X_train).toarray()
count_vect = CountVectorizer(ngram_range=(1, 3),stop_words=preprocessing)
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts).toarray()
try:
    filename = 'dataset/SGD_model_log_update.sav'
    # load the model from disk
    model = pickle.load(open(filename, 'rb'))
    # print("loaded model")
except:    
    model = SGDClassifier(loss='log').fit(X_train_tfidf, y_train)
# # save the model to disk
# filename = 'dataset/SGD_model_log_update.sav'
# pickle.dump(model, open(filename, 'wb'))

# probability = model.predict_proba(count_vect.transform(['smiu']))[:,2]
# print(probability)

# accuracy = model.score(X_train_counts,y_train)
# print("OutPut Is = ", accuracy*100)
# y_predictions=model.predict(X_train_counts)
# print(y_predictions.shape)
# print(X_train_counts.toarray().shape)
# mse = mean_squared_error(y_train,y_predictions)
# # print(y_predictions)
# mses = math.sqrt(mse)
# print("Mean Square Error: "+ str(mses))
# q =df['questions'][226]
# print(q)
# results = lr.predict(count_vect.transform([q])) #"smiu for admission"
# result = results
# # indx = np.argmax(result)
# # print(indx)
# print(df['answers'][result])
# probability = model.predict_proba(count_vect.transform([q]))
# print(np.amax(probability))