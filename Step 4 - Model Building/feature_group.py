import sys
import random
import warnings
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

if not sys.warnoptions:
    warnings.simplefilter("ignore")

Xtrain = np.load('X_Gpick_train.npy')
Xtest = np.load('X_Gpick_test.npy')
Ytrain = np.load('Y_Gpick_train.npy')
Ytest = np.load('Y_Gpick_test.npy')

features = {1: ['ANF', 0, 40],
            2: ['Binary', 41, 204],
            3: ['CKSNAP 1', 205, 236],
            4: ['CKSNAP 3', 237, 300],
            5: ['CKSNAP 5', 301, 396],
            6: ['CKSNAP 7', 397, 524],
            7: ['DAC 7', 525, 566],
            8: ['EIIP', 567, 607],
            9: ['ENAC 10', 608, 735],
            10: ['ENAC 5', 736, 883],
            11: ['Kmer 1', 884, 887],
            12: ['Kmer 2', 888, 903],
            13: ['Kmer 3', 904, 967],
            14: ['Kmer 4', 968, 1223],
            15: ['PseEIIP', 1224, 1287],
            16: ['TAC 7', 1288, 1301]}

random.seed(9)
index = list(features.keys())
random.shuffle(index)
print(index)

group_1 = [features[index[i]][0] for i in range(0, 4)]
group_2 = [features[index[i]][0] for i in range(4, 8)]
group_3 = [features[index[i]][0] for i in range(8, 12)]
group_4 = [features[index[i]][0] for i in range(12, 16)]

fg1 = np.concatenate([Xtrain[:, features[index[i]][1]:features[index[i]][2]]
                      for i in range(0, 4)], axis=1)
fg2 = np.concatenate([Xtrain[:, features[index[i]][1]:features[index[i]][2]]
                      for i in range(4, 8)], axis=1)
fg3 = np.concatenate([Xtrain[:, features[index[i]][1]:features[index[i]][2]]
                      for i in range(8, 12)], axis=1)
fg4 = np.concatenate([Xtrain[:, features[index[i]][1]:features[index[i]][2]]
                      for i in range(12, 16)], axis=1)

groups = [fg1, fg2, fg3, fg4]

print(group_1, fg1.shape)
print(group_2, fg2.shape)
print(group_3, fg3.shape)
print(group_4, fg4.shape)
print("\n")


index = 1
svc = SVC(kernel='linear')

for group in groups:
    train_acc = []
    train_mcc = []
    train_sen = []
    train_spe = []
    test_acc = []
    test_mcc = []
    test_sen = []
    test_spe = []

    cv = KFold(n_splits=5)
    for train_index, test_index in cv.split(group):
        x_train, x_test = group[train_index], group[test_index]
        y_train, y_test = Ytrain[train_index], Ytrain[test_index]

        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_train)

        cf = confusion_matrix(y_train, y_pred)
        sen = cf[0, 0]/(cf[0, 0] + cf[0, 1])
        spe = cf[1, 1]/(cf[1, 0] + cf[1, 1])
        if sen > 0:
            train_sen.append(sen)
        if spe > 0:
            train_spe.append(spe)
        train_acc.append(accuracy_score(y_train, y_pred))
        train_mcc.append(matthews_corrcoef(y_train, y_pred))

        svc.fit(x_train, y_train)
        y_pred = svc.predict(x_test)

        cf = confusion_matrix(y_test, y_pred)
        sen = cf[0, 0]/(cf[0, 0] + cf[0, 1])
        spe = cf[1, 1]/(cf[1, 0] + cf[1, 1])
        if sen > 0:
            test_sen.append(sen)
        if spe > 0:
            test_spe.append(spe)
        test_acc.append(accuracy_score(y_test, y_pred))
        test_mcc.append(matthews_corrcoef(y_test, y_pred))

    print(f"Feature Group {index}: SVC")
    print(
        f"Accuracy:- train: {round(np.mean(train_acc)*100,2)} %, test: {round(np.mean(test_acc)*100,2)} %")
    print(
        f"MCC:- train: {round(np.mean(train_mcc),2)}, test: {round(np.mean(test_mcc),2)}")
    print(
        f"Sensitivity:- train: {round(np.mean(train_sen)*100,2)} %, test: {round(np.mean(test_sen)*100,2)} %")
    print(
        f"Specificity:- train: {round(np.mean(train_spe)*100,2)} %, test: {round(np.mean(test_spe)*100,2)} %")
    print("\n")
    index += 1


svc = SVC(kernel='linear')
svc.fit(Xtrain, Ytrain)
y_pred = svc.predict(Xtest)
acc = accuracy_score(Ytest, y_pred)
mcc = matthews_corrcoef(Ytest, y_pred)
cf = confusion_matrix(Ytest, y_pred)
sen = (cf[0, 0]/(cf[0, 0] + cf[0, 1]))
spe = (cf[1, 1]/(cf[1, 0] + cf[1, 1]))

print("SVC")
print(f"Accuracy: {round((acc*100), 2)} %")
print(f"MCC: {round(mcc, 2)}")
print(f"Sensitivity: {round((sen*100), 2)} %")
print(f"Specificity: {round((spe*100), 2)} %")
print("\n")

index = 1
lre = LogisticRegression()

for group in groups:
    train_acc = []
    train_mcc = []
    train_sen = []
    train_spe = []
    test_acc = []
    test_mcc = []
    test_sen = []
    test_spe = []

    cv = KFold(n_splits=5)
    for train_index, test_index in cv.split(group):
        x_train, x_test = group[train_index], group[test_index]
        y_train, y_test = Ytrain[train_index], Ytrain[test_index]

        lre.fit(x_train, y_train)
        y_pred = lre.predict(x_train)

        cf = confusion_matrix(y_train, y_pred)
        sen = cf[0, 0]/(cf[0, 0] + cf[0, 1])
        spe = cf[1, 1]/(cf[1, 0] + cf[1, 1])
        if sen > 0:
            train_sen.append(sen)
        if spe > 0:
            train_spe.append(spe)
        train_acc.append(accuracy_score(y_train, y_pred))
        train_mcc.append(matthews_corrcoef(y_train, y_pred))

        lre.fit(x_train, y_train)
        y_pred = lre.predict(x_test)

        cf = confusion_matrix(y_test, y_pred)
        sen = cf[0, 0]/(cf[0, 0] + cf[0, 1])
        spe = cf[1, 1]/(cf[1, 0] + cf[1, 1])
        if sen > 0:
            test_sen.append(sen)
        if spe > 0:
            test_spe.append(spe)
        test_acc.append(accuracy_score(y_test, y_pred))
        test_mcc.append(matthews_corrcoef(y_test, y_pred))

    print(f"Feature Group {index}: Logistic Regression")
    print(
        f"Accuracy:- train: {round(np.mean(train_acc)*100,2)} %, test: {round(np.mean(test_acc)*100,2)} %")
    print(
        f"MCC:- train: {round(np.mean(train_mcc),2)}, test: {round(np.mean(test_mcc),2)}")
    print(
        f"Sensitivity:- train: {round(np.mean(train_sen)*100,2)} %, test: {round(np.mean(test_sen)*100,2)} %")
    print(
        f"Specificity:- train: {round(np.mean(train_spe)*100,2)} %, test: {round(np.mean(test_spe)*100,2)} %")
    print("\n")
    index += 1


lre = LogisticRegression()
lre.fit(Xtrain, Ytrain)
y_pred = lre.predict(Xtest)
acc = accuracy_score(Ytest, y_pred)
mcc = matthews_corrcoef(Ytest, y_pred)
cf = confusion_matrix(Ytest, y_pred)
sen = (cf[0, 0]/(cf[0, 0] + cf[0, 1]))
spe = (cf[1, 1]/(cf[1, 0] + cf[1, 1]))

print("Logistic Regression")
print(f"Accuracy: {round((acc*100), 2)} %")
print(f"MCC: {round(mcc, 2)}")
print(f"Sensitivity: {round((sen*100), 2)} %")
print(f"Specificity: {round((spe*100), 2)} %")
print("\n")

index = 1
rfc = RandomForestClassifier()

for group in groups:
    train_acc = []
    train_mcc = []
    train_sen = []
    train_spe = []
    test_acc = []
    test_mcc = []
    test_sen = []
    test_spe = []

    cv = KFold(n_splits=5)
    for train_index, test_index in cv.split(group):
        x_train, x_test = group[train_index], group[test_index]
        y_train, y_test = Ytrain[train_index], Ytrain[test_index]

        rfc.fit(x_train, y_train)
        y_pred = rfc.predict(x_train)

        cf = confusion_matrix(y_train, y_pred)
        sen = cf[0, 0]/(cf[0, 0] + cf[0, 1])
        spe = cf[1, 1]/(cf[1, 0] + cf[1, 1])
        if sen > 0:
            train_sen.append(sen)
        if spe > 0:
            train_spe.append(spe)
        train_acc.append(accuracy_score(y_train, y_pred))
        train_mcc.append(matthews_corrcoef(y_train, y_pred))

        rfc.fit(x_train, y_train)
        y_pred = rfc.predict(x_test)

        cf = confusion_matrix(y_test, y_pred)
        sen = cf[0, 0]/(cf[0, 0] + cf[0, 1])
        spe = cf[1, 1]/(cf[1, 0] + cf[1, 1])
        if sen > 0:
            test_sen.append(sen)
        if spe > 0:
            test_spe.append(spe)
        test_acc.append(accuracy_score(y_test, y_pred))
        test_mcc.append(matthews_corrcoef(y_test, y_pred))

    print(f"Feature Group {index}: Random Forest Classifier")
    print(
        f"Accuracy:- train: {round(np.mean(train_acc)*100,2)} %, test: {round(np.mean(test_acc)*100,2)} %")
    print(
        f"MCC:- train: {round(np.mean(train_mcc),2)}, test: {round(np.mean(test_mcc),2)}")
    print(
        f"Sensitivity:- train: {round(np.mean(train_sen)*100,2)} %, test: {round(np.mean(test_sen)*100,2)} %")
    print(
        f"Specificity:- train: {round(np.mean(train_spe)*100,2)} %, test: {round(np.mean(test_spe)*100,2)} %")
    print("\n")
    index += 1


rfc = RandomForestClassifier()
rfc.fit(Xtrain, Ytrain)
y_pred = rfc.predict(Xtest)
acc = accuracy_score(Ytest, y_pred)
mcc = matthews_corrcoef(Ytest, y_pred)
cf = confusion_matrix(Ytest, y_pred)
sen = (cf[0, 0]/(cf[0, 0] + cf[0, 1]))
spe = (cf[1, 1]/(cf[1, 0] + cf[1, 1]))

print("Random Forest Classifier")
print(f"Accuracy: {round((acc*100), 2)} %")
print(f"MCC: {round(mcc, 2)}")
print(f"Sensitivity: {round((sen*100), 2)} %")
print(f"Specificity: {round((spe*100), 2)} %")
print("\n")

index = 1
abc = AdaBoostClassifier()

for group in groups:
    train_acc = []
    train_mcc = []
    train_sen = []
    train_spe = []
    test_acc = []
    test_mcc = []
    test_sen = []
    test_spe = []

    cv = KFold(n_splits=5)
    for train_index, test_index in cv.split(group):
        x_train, x_test = group[train_index], group[test_index]
        y_train, y_test = Ytrain[train_index], Ytrain[test_index]

        abc.fit(x_train, y_train)
        y_pred = abc.predict(x_train)

        cf = confusion_matrix(y_train, y_pred)
        sen = cf[0, 0]/(cf[0, 0] + cf[0, 1])
        spe = cf[1, 1]/(cf[1, 0] + cf[1, 1])
        if sen > 0:
            train_sen.append(sen)
        if spe > 0:
            train_spe.append(spe)
        train_acc.append(accuracy_score(y_train, y_pred))
        train_mcc.append(matthews_corrcoef(y_train, y_pred))

        abc.fit(x_train, y_train)
        y_pred = abc.predict(x_test)

        cf = confusion_matrix(y_test, y_pred)
        sen = cf[0, 0]/(cf[0, 0] + cf[0, 1])
        spe = cf[1, 1]/(cf[1, 0] + cf[1, 1])
        if sen > 0:
            test_sen.append(sen)
        if spe > 0:
            test_spe.append(spe)
        test_acc.append(accuracy_score(y_test, y_pred))
        test_mcc.append(matthews_corrcoef(y_test, y_pred))

    print(f"Feature Group {index}: Ada Boost Classifier")
    print(
        f"Accuracy:- train: {round(np.mean(train_acc)*100,2)} %, test: {round(np.mean(test_acc)*100,2)} %")
    print(
        f"MCC:- train: {round(np.mean(train_mcc),2)}, test: {round(np.mean(test_mcc),2)}")
    print(
        f"Sensitivity:- train: {round(np.mean(train_sen)*100,2)} %, test: {round(np.mean(test_sen)*100,2)} %")
    print(
        f"Specificity:- train: {round(np.mean(train_spe)*100,2)} %, test: {round(np.mean(test_spe)*100,2)} %")
    print("\n")
    index += 1


abc = AdaBoostClassifier()
abc.fit(Xtrain, Ytrain)
y_pred = abc.predict(Xtest)
acc = accuracy_score(Ytest, y_pred)
mcc = matthews_corrcoef(Ytest, y_pred)
cf = confusion_matrix(Ytest, y_pred)
sen = (cf[0, 0]/(cf[0, 0] + cf[0, 1]))
spe = (cf[1, 1]/(cf[1, 0] + cf[1, 1]))

print("Ada Boost Classifier")
print(f"Accuracy: {round((acc*100), 2)} %")
print(f"MCC: {round(mcc, 2)}")
print(f"Sensitivity: {round((sen*100), 2)} %")
print(f"Specificity: {round((spe*100), 2)} %")
print("\n")

index = 1
xgb = XGBClassifier(eval_metric='error')

for group in groups:
    train_acc = []
    train_mcc = []
    train_sen = []
    train_spe = []
    test_acc = []
    test_mcc = []
    test_sen = []
    test_spe = []

    cv = KFold(n_splits=5)
    for train_index, test_index in cv.split(group):
        x_train, x_test = group[train_index], group[test_index]
        y_train, y_test = Ytrain[train_index], Ytrain[test_index]

        xgb.fit(x_train, y_train)
        y_pred = xgb.predict(x_train)

        cf = confusion_matrix(y_train, y_pred)
        sen = cf[0, 0]/(cf[0, 0] + cf[0, 1])
        spe = cf[1, 1]/(cf[1, 0] + cf[1, 1])
        if sen > 0:
            train_sen.append(sen)
        if spe > 0:
            train_spe.append(spe)
        train_acc.append(accuracy_score(y_train, y_pred))
        train_mcc.append(matthews_corrcoef(y_train, y_pred))

        xgb.fit(x_train, y_train)
        y_pred = xgb.predict(x_test)

        cf = confusion_matrix(y_test, y_pred)
        sen = cf[0, 0]/(cf[0, 0] + cf[0, 1])
        spe = cf[1, 1]/(cf[1, 0] + cf[1, 1])
        if sen > 0:
            test_sen.append(sen)
        if spe > 0:
            test_spe.append(spe)
        test_acc.append(accuracy_score(y_test, y_pred))
        test_mcc.append(matthews_corrcoef(y_test, y_pred))

    print(f"Feature Group {index}: XGBoost Classifier")
    print(
        f"Accuracy:- train: {round(np.mean(train_acc)*100,2)} %, test: {round(np.mean(test_acc)*100,2)} %")
    print(
        f"MCC:- train: {round(np.mean(train_mcc),2)}, test: {round(np.mean(test_mcc),2)}")
    print(
        f"Sensitivity:- train: {round(np.mean(train_sen)*100,2)} %, test: {round(np.mean(test_sen)*100,2)} %")
    print(
        f"Specificity:- train: {round(np.mean(train_spe)*100,2)} %, test: {round(np.mean(test_spe)*100,2)} %")
    print("\n")
    index += 1


xgb = XGBClassifier(eval_metric='error')
xgb.fit(Xtrain, Ytrain)
y_pred = xgb.predict(Xtest)
acc = accuracy_score(Ytest, y_pred)
mcc = matthews_corrcoef(Ytest, y_pred)
cf = confusion_matrix(Ytest, y_pred)
sen = (cf[0, 0]/(cf[0, 0] + cf[0, 1]))
spe = (cf[1, 1]/(cf[1, 0] + cf[1, 1]))

print("XGBoost Classifier")
print(f"Accuracy: {round((acc*100), 2)} %")
print(f"MCC: {round(mcc, 2)}")
print(f"Sensitivity: {round((sen*100), 2)} %")
print(f"Specificity: {round((spe*100), 2)} %")
print("\n")
