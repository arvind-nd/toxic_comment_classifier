import joblib

class inference:
    def __init__(self):
        self.toxic = joblib.load('../models/lr_toxic_0.sav')
        self.severe_toxic = joblib.load('../models/lr_severe_toxic_0.sav')
        self.obscene = joblib.load('../models/lr_obscene_0.sav')
        self.insult = joblib.load('../models/lr_insult_0.sav')
        self.threat = joblib.load('../models/lr_threat_0.sav')
        self.identity_hate = joblib.load('../models/lr_identity_hate_0.sav')
        self.tfv = joblib.load('../models/TfidfVectorizer.sav')

    def predict(self, text):
        score = {}
        text = self.tfv.transform([text])
        score['toxic'] = self.toxic.predict_proba(text)[:,1]
        score['severe_toxic'] = self.severe_toxic.predict_proba(text)[:,1]
        score['obscene'] = self.obscene.predict_proba(text)[:,1]
        score['insult'] = self.insult.predict_proba(text)[:,1]
        score['threat'] = self.threat.predict_proba(text)[:,1]
        score['identity_hate'] = self.identity_hate.predict_proba(text)[:,1]
        print(score)
        return score
