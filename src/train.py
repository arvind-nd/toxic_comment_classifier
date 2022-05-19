import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from config import model
import warnings
#warnings.filterwarning("ignore")



def eval_metric(y_true,y_preds):
    score = metrics.roc_auc_score(y_true, y_preds)
    return score

def model_fit(model, xtrain, xvalid, ytrain, yvalid):
    model.fit(xtrain, ytrain)
    train_preds = model.predict_proba(xtrain)[:,1]
    valid_preds = model.predict_proba(xvalid)[:,1]
    tr_score = eval_metric(ytrain, train_preds)
    vl_score = eval_metric(yvalid, valid_preds)
    return tr_score, vl_score

def transform(data,tfv):
    return tfv.transform(data)

def train(df, fold, tfv):
    tr_scores = []
    vl_scores = []
    folds = []
    clf_names = []
    train_data = df[df.kfold != fold].reset_index(drop=True)
    valid_data = df[df.kfold == fold].reset_index(drop=True)

    targets = [col for col in train_data.columns if col not in ["id","kfold","comment_text"]]

    xtrain = list(train_data.comment_text.values)
    xvalid = list(valid_data.comment_text.values)

    xtrain = transform(xtrain, tfv)
    xvalid = transform(xvalid, tfv)

    
    for i in range(6):
        clf = model.i
        ytrain = np.array(train_data[targets[i]].values,dtype="int")
        yvalid = np.array(valid_data[targets[i]].values,dtype="int")

        tr_score, vl_score = model_fit(clf, xtrain, xvalid, ytrain, yvalid)

        tr_scores.append(tr_score)
        vl_scores.append(vl_score)
        folds.append(fold)
        clf_names.append(targets[i])

        print(f"fold/clf = {fold}/{targets[i]}, train_auc={tr_score}, valid_auc={vl_score}")

    return tr_scores, vl_scores, folds, clf_names


if __name__ == "__main__":
    tr_scores = []
    vl_scores = []
    folds = []
    clf_names = []

    df = pd.read_csv("../input/train_10folds.csv")

    x_full = df.comment_text.values
    tfv = TfidfVectorizer(ngram_range=(1,2), stopwords="english")
    tfv.fit(x_full)

    for i in range(1):
        tr_score, vl_score, fold, clf_name = train(df=df, fold=i, tfv)
        tr_scores.extend(tr_score)
        vl_scores.extend(vl_score)
        folds.extend(fold)
        clf_names.extend(targets[i])

    details = pd.DataFrame(
            {
                "tr_scores":tr_scores,
                "vl_scores":vl_scores,
                "fold":folds,
                "clf_name":clf_names
            }
        )
    details.to_csv("../input/training_details.csv",index=False)
