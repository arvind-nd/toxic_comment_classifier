from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier



models = {
    0:LogisticRegression(max_iter=487, solver='newton-cg', C=15.320595194507824),
    1:LogisticRegression(max_iter=991, solver='liblinear', C=2.058406142145735),
    2:LogisticRegression(max_iter=987, solver='lbfgs', C=10.115976877363867),
    3:LogisticRegression(max_iter=107, solver='newton-cg', C=13.703629872885593),
    4:LogisticRegression(max_iter=401, solver='lbfgs', C=7.64578317335075),
    5:LogisticRegression(max_iter=403, solver='liblinear', C=17.46139181395921)
}
