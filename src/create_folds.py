import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

df = pd.read_csv("../input/train.csv.zip")
df["kfold"] = -1
targets = [col for col in df.columns if col not in ("id","comment_text","kfold")]
print(targets)
mskf = MultilabelStratifiedKFold(n_splits=10,shuffle=True,random_state=42)

for fold_, (t_,v_) in enumerate(mskf.split(X=df,y=df[targets].values)):
    df.loc[v_,"kfold"] = fold_
df.to_csv("../input/train_10folds.csv",index=False)