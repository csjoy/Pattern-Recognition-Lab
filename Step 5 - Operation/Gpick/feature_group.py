import numpy as np 
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


data_train_x = np.load('X_Gpick_train.npy')
data_train_y = np.load('Y_Gpick_train.npy')

feature_group_1 = data_train_x[:, range(0, 301)] # ANF, Binary, CKSNAP 1, CKSNAP 3
feature_group_2 = data_train_x[:, range(301, 608)] # CKSNAP 5, CKSNAP 7, DAC 7, EIIP
feature_group_3 = data_train_x[:, range(608, 904)] # ENAC 5, ENAC 10, Kmer 1, Kmer 2
feature_group_4 = data_train_x[:, range(904, 1302)] # Kmer 3, Kmer 4, PseEIIP, TAC 7

svc = SVC(kernel='linear')
lr = LogisticRegression()
rf = RandomForestClassifier()
ab = AdaBoostClassifier()
xb = XGBClassifier()

# ==================================================================================

svc_accuracy = []
svc_mcc = []
svc_sensivity = []
svc_specificity = []

lr_accuracy = []
lr_mcc = []
lr_sensivity = []
lr_specificity = []

rf_accuracy = []
rf_mcc = []
rf_sensivity = []
rf_specificity = []

ab_accuracy = []
ab_mcc = []
ab_sensivity = []
ab_specificity = []

xb_accuracy = []
xb_mcc = []
xb_sensivity = []
xb_specificity = []

cv = KFold(n_splits=5)
for train_index, test_index in cv.split(feature_group_1):
    x_train, x_test, y_train, y_test = feature_group_1[train_index], feature_group_1[test_index], data_train_y[train_index], data_train_y[test_index]
    
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    svc_accuracy.append(svc.score(x_test, y_test))
    svc_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        svc_sensivity.append(sen)
    if spe > 0:
        svc_specificity.append(spe)
    


    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    lr_accuracy.append(lr.score(x_test, y_test))
    lr_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        lr_sensivity.append(sen)
    if spe > 0:
        lr_specificity.append(spe)
    

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    rf_accuracy.append(rf.score(x_test, y_test))
    rf_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        rf_sensivity.append(sen)
    if spe > 0:
        rf_specificity.append(spe)
    

    ab.fit(x_train, y_train)
    y_pred = ab.predict(x_test)
    ab_accuracy.append(ab.score(x_test, y_test))
    ab_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        ab_sensivity.append(sen)
    if spe > 0:
        ab_specificity.append(spe)
    

    xb.fit(x_train, y_train)
    y_pred = xb.predict(x_test)
    xb_accuracy.append(xb.score(x_test, y_test))
    xb_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        xb_sensivity.append(sen)
    if spe > 0:
        xb_specificity.append(spe)

print('Feature Group 1:')
print('svc_accuracy:', np.mean(svc_accuracy))
print('svc_mcc:', np.mean(svc_mcc))
print('svc_sensivity:', np.mean(svc_sensivity))
print('svc_specificity:', np.mean(svc_specificity))
print('lr_accuracy:', np.mean(lr_accuracy))
print('lr_mcc:', np.mean(lr_mcc))
print('lr_sensivity:', np.mean(lr_sensivity))
print('lr_specificity:', np.mean(lr_specificity))
print('rf_accuracy:', np.mean(rf_accuracy))
print('rf_mcc:', np.mean(rf_mcc))
print('rf_sensivity:', np.mean(rf_sensivity))
print('rf_specificity:', np.mean(rf_specificity))
print('ab_accuracy:', np.mean(ab_accuracy))
print('ab_mcc:', np.mean(ab_mcc))
print('ab_sensivity:', np.mean(ab_sensivity))
print('ab_specificity:', np.mean(ab_specificity))
print('xb_accuracy:', np.mean(xb_accuracy))
print('xb_mcc:', np.mean(xb_mcc))
print('xb_sensivity:', np.mean(xb_sensivity))
print('xb_specificity:', np.mean(xb_specificity))

# ==================================================================================

svc_accuracy = []
svc_mcc = []
svc_sensivity = []
svc_specificity = []

lr_accuracy = []
lr_mcc = []
lr_sensivity = []
lr_specificity = []

rf_accuracy = []
rf_mcc = []
rf_sensivity = []
rf_specificity = []

ab_accuracy = []
ab_mcc = []
ab_sensivity = []
ab_specificity = []

xb_accuracy = []
xb_mcc = []
xb_sensivity = []
xb_specificity = []

svc = SVC(kernel='linear')
lr = LogisticRegression()
rf = RandomForestClassifier()
ab = AdaBoostClassifier()
xb = XGBClassifier()
cv = KFold(n_splits=5)
for train_index, test_index in cv.split(feature_group_2):
    x_train, x_test, y_train, y_test = feature_group_2[train_index], feature_group_2[test_index], data_train_y[train_index], data_train_y[test_index]
    
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    svc_accuracy.append(svc.score(x_test, y_test))
    svc_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        svc_sensivity.append(sen)
    if spe > 0:
        svc_specificity.append(spe)


    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    lr_accuracy.append(lr.score(x_test, y_test))
    lr_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        lr_sensivity.append(sen)
    if spe > 0:
        lr_specificity.append(spe)
    

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    rf_accuracy.append(rf.score(x_test, y_test))
    rf_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        rf_sensivity.append(sen)
    if spe > 0:
        rf_specificity.append(spe)
    

    ab.fit(x_train, y_train)
    y_pred = ab.predict(x_test)
    ab_accuracy.append(ab.score(x_test, y_test))
    ab_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        ab_sensivity.append(sen)
    if spe > 0:
        ab_specificity.append(spe)
    

    xb.fit(x_train, y_train)
    y_pred = xb.predict(x_test)
    xb_accuracy.append(xb.score(x_test, y_test))
    xb_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        xb_sensivity.append(sen)
    if spe > 0:
        xb_specificity.append(spe)

print('Feature Group 2:')
print('svc_accuracy:', np.mean(svc_accuracy))
print('svc_mcc:', np.mean(svc_mcc))
print('svc_sensivity:', np.mean(svc_sensivity))
print('svc_specificity:', np.mean(svc_specificity))
print('lr_accuracy:', np.mean(lr_accuracy))
print('lr_mcc:', np.mean(lr_mcc))
print('lr_sensivity:', np.mean(lr_sensivity))
print('lr_specificity:', np.mean(lr_specificity))
print('rf_accuracy:', np.mean(rf_accuracy))
print('rf_mcc:', np.mean(rf_mcc))
print('rf_sensivity:', np.mean(rf_sensivity))
print('rf_specificity:', np.mean(rf_specificity))
print('ab_accuracy:', np.mean(ab_accuracy))
print('ab_mcc:', np.mean(ab_mcc))
print('ab_sensivity:', np.mean(ab_sensivity))
print('ab_specificity:', np.mean(ab_specificity))
print('xb_accuracy:', np.mean(xb_accuracy))
print('xb_mcc:', np.mean(xb_mcc))
print('xb_sensivity:', np.mean(xb_sensivity))
print('xb_specificity:', np.mean(xb_specificity))

# ==================================================================================

svc_accuracy = []
svc_mcc = []
svc_sensivity = []
svc_specificity = []

lr_accuracy = []
lr_mcc = []
lr_sensivity = []
lr_specificity = []

rf_accuracy = []
rf_mcc = []
rf_sensivity = []
rf_specificity = []

ab_accuracy = []
ab_mcc = []
ab_sensivity = []
ab_specificity = []

xb_accuracy = []
xb_mcc = []
xb_sensivity = []
xb_specificity = []

svc = SVC(kernel='linear')
lr = LogisticRegression()
rf = RandomForestClassifier()
ab = AdaBoostClassifier()
xb = XGBClassifier()
cv = KFold(n_splits=5)
for train_index, test_index in cv.split(feature_group_3):
    x_train, x_test, y_train, y_test = feature_group_3[train_index], feature_group_3[test_index], data_train_y[train_index], data_train_y[test_index]
    
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    svc_accuracy.append(svc.score(x_test, y_test))
    svc_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        svc_sensivity.append(sen)
    if spe > 0:
        svc_specificity.append(spe)


    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    lr_accuracy.append(lr.score(x_test, y_test))
    lr_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        lr_sensivity.append(sen)
    if spe > 0:
        lr_specificity.append(spe)
    

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    rf_accuracy.append(rf.score(x_test, y_test))
    rf_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        rf_sensivity.append(sen)
    if spe > 0:
        rf_specificity.append(spe)
    

    ab.fit(x_train, y_train)
    y_pred = ab.predict(x_test)
    ab_accuracy.append(ab.score(x_test, y_test))
    ab_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        ab_sensivity.append(sen)
    if spe > 0:
        ab_specificity.append(spe)
    

    xb.fit(x_train, y_train)
    y_pred = xb.predict(x_test)
    xb_accuracy.append(xb.score(x_test, y_test))
    xb_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        xb_sensivity.append(sen)
    if spe > 0:
        xb_specificity.append(spe)

print('Feature Group 3:')
print('svc_accuracy:', np.mean(svc_accuracy))
print('svc_mcc:', np.mean(svc_mcc))
print('svc_sensivity:', np.mean(svc_sensivity))
print('svc_specificity:', np.mean(svc_specificity))
print('lr_accuracy:', np.mean(lr_accuracy))
print('lr_mcc:', np.mean(lr_mcc))
print('lr_sensivity:', np.mean(lr_sensivity))
print('lr_specificity:', np.mean(lr_specificity))
print('rf_accuracy:', np.mean(rf_accuracy))
print('rf_mcc:', np.mean(rf_mcc))
print('rf_sensivity:', np.mean(rf_sensivity))
print('rf_specificity:', np.mean(rf_specificity))
print('ab_accuracy:', np.mean(ab_accuracy))
print('ab_mcc:', np.mean(ab_mcc))
print('ab_sensivity:', np.mean(ab_sensivity))
print('ab_specificity:', np.mean(ab_specificity))
print('xb_accuracy:', np.mean(xb_accuracy))
print('xb_mcc:', np.mean(xb_mcc))
print('xb_sensivity:', np.mean(xb_sensivity))
print('xb_specificity:', np.mean(xb_specificity))

# ==================================================================================

svc_accuracy = []
svc_mcc = []
svc_sensivity = []
svc_specificity = []

lr_accuracy = []
lr_mcc = []
lr_sensivity = []
lr_specificity = []

rf_accuracy = []
rf_mcc = []
rf_sensivity = []
rf_specificity = []

ab_accuracy = []
ab_mcc = []
ab_sensivity = []
ab_specificity = []

xb_accuracy = []
xb_mcc = []
xb_sensivity = []
xb_specificity = []

svc = SVC(kernel='linear')
lr = LogisticRegression()
rf = RandomForestClassifier()
ab = AdaBoostClassifier()
xb = XGBClassifier()
cv = KFold(n_splits=5)
for train_index, test_index in cv.split(feature_group_4):
    x_train, x_test, y_train, y_test = feature_group_4[train_index], feature_group_4[test_index], data_train_y[train_index], data_train_y[test_index]
    
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    svc_accuracy.append(svc.score(x_test, y_test))
    svc_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        svc_sensivity.append(sen)
    if spe > 0:
        svc_specificity.append(spe)


    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    lr_accuracy.append(lr.score(x_test, y_test))
    lr_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        lr_sensivity.append(sen)
    if spe > 0:
        lr_specificity.append(spe)
    

    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    rf_accuracy.append(rf.score(x_test, y_test))
    rf_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        rf_sensivity.append(sen)
    if spe > 0:
        rf_specificity.append(spe)
    

    ab.fit(x_train, y_train)
    y_pred = ab.predict(x_test)
    ab_accuracy.append(ab.score(x_test, y_test))
    ab_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        ab_sensivity.append(sen)
    if spe > 0:
        ab_specificity.append(spe)
    

    xb.fit(x_train, y_train)
    y_pred = xb.predict(x_test)
    xb_accuracy.append(xb.score(x_test, y_test))
    xb_mcc.append(matthews_corrcoef(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    if sen > 0:
        xb_sensivity.append(sen)
    if spe > 0:
        xb_specificity.append(spe)

print('Feature Group 4:')
print('svc_accuracy:', np.mean(svc_accuracy))
print('svc_mcc:', np.mean(svc_mcc))
print('svc_sensivity:', np.mean(svc_sensivity))
print('svc_specificity:', np.mean(svc_specificity))
print('lr_accuracy:', np.mean(lr_accuracy))
print('lr_mcc:', np.mean(lr_mcc))
print('lr_sensivity:', np.mean(lr_sensivity))
print('lr_specificity:', np.mean(lr_specificity))
print('rf_accuracy:', np.mean(rf_accuracy))
print('rf_mcc:', np.mean(rf_mcc))
print('rf_sensivity:', np.mean(rf_sensivity))
print('rf_specificity:', np.mean(rf_specificity))
print('ab_accuracy:', np.mean(ab_accuracy))
print('ab_mcc:', np.mean(ab_mcc))
print('ab_sensivity:', np.mean(ab_sensivity))
print('ab_specificity:', np.mean(ab_specificity))
print('xb_accuracy:', np.mean(xb_accuracy))
print('xb_mcc:', np.mean(xb_mcc))
print('xb_sensivity:', np.mean(xb_sensivity))
print('xb_specificity:', np.mean(xb_specificity))

# ==================================================================================


# data_test_x = np.load('X_Gpick_test.npy')
# data_test_y = np.load('Y_Gpick_test.npy')

# fg1_accuracy = []
# fg1_mcc = []
# fg1_sensivity = []
# fg1_specificity = []

# fg2_accuracy = []
# fg2_mcc = []
# fg2_sensivity = []
# fg2_specificity = []

# fg3_accuracy = []
# fg3_mcc = []
# fg3_sensivity = []
# fg3_specificity = []

# fg4_accuracy = []
# fg4_mcc = []
# fg4_sensivity = []
# fg4_specificity = []

# fg1 = data_test_x[:, range(0, 301)] # ANF, Binary, CKSNAP 1, CKSNAP 3
# fg2 = data_test_x[:, range(301, 608)] # CKSNAP 5, CKSNAP 7, DAC 7, EIIP
# fg3 = data_test_x[:, range(608, 904)] # ENAC 5, ENAC 10, Kmer 1, Kmer 2
# fg4 = data_test_x[:, range(904, 1302)] # Kmer 3, Kmer 4, PseEIIP, TAC 7

# y_pred = svc.predict(fg1)
# fg1_accuracy.append(svc.score(fg1, data_test_y))
# fg1_mcc.append(matthews_corrcoef(data_test_y, y_pred))
# tn, fp, fn, tp = confusion_matrix(data_test_y, y_pred).ravel()
# sen = tp / (tp + fn)
# spe = tn / (tn + fp)
# if sen > 0:
#     fg1_sensivity.append(sen)
# if spe > 0:
#     fg1_specificity.append(spe)


# y_pred = svc.predict(fg2)
# fg2_accuracy.append(svc.score(fg2, data_test_y))
# fg2_mcc.append(matthews_corrcoef(data_test_y, y_pred))
# tn, fp, fn, tp = confusion_matrix(data_test_y, y_pred).ravel()
# sen = tp / (tp + fn)
# spe = tn / (tn + fp)
# if sen > 0:
#     fg2_sensivity.append(sen)
# if spe > 0:
#     fg2_specificity.append(spe)


# y_pred = svc.predict(fg3)
# fg3_accuracy.append(svc.score(fg3, data_test_y))
# fg3_mcc.append(matthews_corrcoef(data_test_y, y_pred))
# tn, fp, fn, tp = confusion_matrix(data_test_y, y_pred).ravel()
# sen = tp / (tp + fn)
# spe = tn / (tn + fp)
# if sen > 0:
#     fg3_sensivity.append(sen)
# if spe > 0:
#     fg3_specificity.append(spe)


# y_pred = svc.predict(fg4)
# fg4_accuracy.append(svc.score(fg4, data_test_y))
# fg4_mcc.append(matthews_corrcoef(data_test_y, y_pred))
# tn, fp, fn, tp = confusion_matrix(data_test_y, y_pred).ravel()
# sen = tp / (tp + fn)
# spe = tn / (tn + fp)
# if sen > 0:
#     fg4_sensivity.append(sen)
# if spe > 0:
#     fg4_specificity.append(spe)


# print('********************************')
# print('fg1_accuracy:', np.mean(svc_accuracy))
# print('fg1_mcc:', np.mean(svc_mcc))
# print('fg1_sensivity:', np.mean(svc_sensivity))
# print('fg1_specificity:', np.mean(svc_specificity))
# print('fg2_accuracy:', np.mean(lr_accuracy))
# print('fg2_mcc:', np.mean(lr_mcc))
# print('fg2_sensivity:', np.mean(lr_sensivity))
# print('fg2_specificity:', np.mean(lr_specificity))
# print('fg3_accuracy:', np.mean(rf_accuracy))
# print('fg3_mcc:', np.mean(rf_mcc))
# print('fg3_sensivity:', np.mean(rf_sensivity))
# print('fg3_specificity:', np.mean(rf_specificity))
# print('fg4_accuracy:', np.mean(ab_accuracy))
# print('fg4_mcc:', np.mean(ab_mcc))
# print('fg4_sensivity:', np.mean(ab_sensivity))
# print('fg4_specificity:', np.mean(ab_specificity))