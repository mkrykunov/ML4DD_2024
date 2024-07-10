from rdkit import Chem
from rdkit.Chem import Descriptors as RDescriptors
from mordred import Calculator, descriptors
import polaris as po
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn import preprocessing
import random


def getMolDescriptors(mol, missingVal=None):
    ''' calculate the full list of descriptors for a molecule
        missingVal is used if the descriptor cannot be calculated
    '''
    res = {}
    for nm,fn in RDescriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            # print the error message:
            # import traceback
            # traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res


def XGBoost_r2(train_X, train_y, test_X, n_est, max_d, learn_r, rand_st):
    model = XGBRegressor(n_estimators=n_est, max_depth=max_d, learning_rate=learn_r, random_state=rand_st)
    scaler_X = preprocessing.StandardScaler().fit(train_X)

    X_train_scaled = scaler_X.transform(train_X)
    X_test_scaled = scaler_X.transform(test_X)
    
    model.fit(X_train_scaled, train_y)

    y_pred = model.predict(X_test_scaled)

    results = benchmark1.evaluate(y_pred)

    for k, item in enumerate(results.results['Metric']):
        if str(item) == 'Metric.r2':
            return results, results.results['Score'][k]
    return results, None


def generate_population(parent1, parent2, limits, npop):
    population = []

    for k in range(npop):
        p = random.random()
        mated_val = parent1 * p + parent2 * (1 - p) # mating
            
        if random.random() < 0.2:
            mated_val = mated_val * (0.7 + 0.6 * random.random()) # mutation
                
        if mated_val < limits[0]:
            mated_val = limits[0]
        elif mated_val > limits[1]:
            mated_val = limits[1]

        if isinstance(parent1, int) and isinstance(parent2, int):
            mated_val = int(mated_val)
                
        child = mated_val

        population.append(child)
       
    return population


def GA_for_XGBoost_r2(train_X, train_y, test_X, population_size = 32, n_steps = 10, to_print = False):
    # hyperparameters
    npop = population_size
    n_estimators_1 = 15
    n_estimators_2 = 150
    max_depth_1 = 3
    max_depth_2 = 12
    learning_rate_1 = -0.6 # it will be used as an exponent
    learning_rate_2 = -1.6 # it will be used as an exponent

    estimators_limits = (5, 200)
    depth_limits = (2, 20)
    learning_rate_limits = (-2.0, 0.0)

    # calculate initial score
    _, score1 = XGBoost_r2(train_X, train_y, test_X, n_estimators_1, max_depth_1, 10**learning_rate_1, 777)
    _, score2 = XGBoost_r2(train_X, train_y, test_X, n_estimators_2, max_depth_2, 10**learning_rate_2, 777)

    best_value = max(score1, score1)
    r2_list = list()

    if score1 > score2:
        r2_list.append((score2, max_depth_2, n_estimators_2, learning_rate_2))
        r2_list.append((score1, max_depth_1, n_estimators_1, learning_rate_1))
    else:
        r2_list.append((score1, max_depth_1, n_estimators_1, learning_rate_1))
        r2_list.append((score2, max_depth_2, n_estimators_2, learning_rate_2))

    # start the optimization
    for i in range(n_steps):
        print('iter %d. reward: %f' % (i, best_value))

        pop_estimators = generate_population(n_estimators_1, n_estimators_2, estimators_limits, npop)
        pop_depth = generate_population(max_depth_1, max_depth_2, depth_limits, npop)
        pop_learning_rate = generate_population(learning_rate_1, learning_rate_2, learning_rate_limits, npop)

        R = np.zeros(npop)
        for j in range(npop):
            _, R[j] = XGBoost_r2(train_X, train_y, test_X, pop_estimators[j], pop_depth[j], 10**pop_learning_rate[j], 777)
            if to_print: print (j, pop_estimators[j], pop_depth[j], 10**pop_learning_rate[j], R[j])

        Z = [(x, y, z) for _, x, y, z in sorted(zip(R, pop_estimators, pop_depth, pop_learning_rate), key=lambda pair: pair[0])]

        n_estimators_1 = Z[-1][0]
        n_estimators_2 = Z[-2][0]

        max_depth_1 = Z[-1][1]
        max_depth_2 = Z[-2][1]

        learning_rate_1 = Z[-1][2]
        learning_rate_2 = Z[-2][2]

        if np.max(R) > best_value:
            best_value = np.max(R)
            r2_list.append((best_value, n_estimators_1, max_depth_1, learning_rate_1))
        else:
            n_estimators_2 = r2_list[-1][1]
            max_depth_2 = r2_list[-1][2]
            learning_rate_2 = r2_list[-1][3]

    return best_value, r2_list


benchmark1 = po.load_benchmark("polaris/adme-fang-solu-1")

train, test = benchmark1.get_train_test_split()
print (train.inputs[0])
print (train.targets[0])


# Create a calculator with all mordred descriptors

calc_mordred = Calculator(descriptors, ignore_3D=True)

train_mols = [Chem.MolFromSmiles(smi) for smi in train.inputs]
train_descriptors = [calc_mordred(m).asdict() for m in train_mols]


is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))


train_df = pd.DataFrame(train_descriptors)

nan_train = is_number(train_df.dtypes)
nan_idx_train = set(np.where(nan_train == False)[0])

print ("\nnan_idx_train for mordred")
print (nan_idx_train)
print (len(nan_idx_train))


test_mols = [Chem.MolFromSmiles(smi) for smi in test.inputs]
test_descriptors = [calc_mordred(m).asdict() for m in test_mols]


test_df = pd.DataFrame(test_descriptors)

nan_test = is_number(test_df.dtypes)
nan_idx_test = set(np.where(nan_test == False)[0])

print ("\nnan_idx_test for mordred")
print (nan_idx_test)
print (len(nan_idx_test))


nan_idx_train_test = list(nan_idx_train.union(nan_idx_test))
print ("len(nan_idx_train_test) =", len(nan_idx_train_test))


df_train_clean = train_df.drop(train_df.columns[nan_idx_train_test], axis=1)
df_test_clean = test_df.drop(test_df.columns[nan_idx_train_test], axis=1)


train_X = df_train_clean.to_numpy()
test_X = df_test_clean.to_numpy()

print ("train_X.shape =", train_X.shape)
print ("test_X.shape =", test_X.shape)


# RDKit part

from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')


train_mols = [Chem.MolFromSmiles(smi) for smi in train.inputs]
train_descriptors = [getMolDescriptors(m) for m in train_mols]

test_mols = [Chem.MolFromSmiles(smi) for smi in test.inputs]
test_descriptors = [getMolDescriptors(m) for m in test_mols]


df_train = pd.DataFrame(train_descriptors)
df_test = pd.DataFrame(test_descriptors)

train_features = df_train.to_numpy()
test_features = df_test.to_numpy()


idx_train = np.argwhere(np.all(train_features[..., :] == 0, axis=0))
print ("\nidx_train.T for RDKit")
print (idx_train.T)

idx_test = np.argwhere(np.all(test_features[..., :] == 0, axis=0))
print ("\nidx_test.T for RDKit")
print (idx_test.T)

common_idx = list(set(idx_train.flatten()) & set(idx_test.flatten()))
print (common_idx)
print ("\nlen(common_idx) =", len(common_idx))


train_RDKit = np.delete(train_features, common_idx, axis=1)
print ("\ntrain_RDKit.shape =", train_RDKit.shape)

test_RDKit = np.delete(test_features, common_idx, axis=1)
print ("test_RDKit.shape =", test_RDKit.shape)


train_all_X = np.concatenate([train_X, train_RDKit], axis = 1)
print ("train_all_X.shape =", train_all_X.shape)

test_all_X = np.concatenate([test_X, test_RDKit], axis = 1)
print ("test_all_X.shape =", test_all_X.shape)


# Here is an example for 100 estimators, max depth of 6 and learning rate 0.0762324367284232.

mrxb, r2_score_mrxb = XGBoost_r2(train_all_X, train.targets, test_all_X, 100, 6, 0.0762324367284232, 777)
print (100, 6, 0.0762324367284232, r2_score_mrxb)


# Let's run GA for 10 steps.

best_value, r2score_list = GA_for_XGBoost_r2(train_all_X, train.targets, test_all_X, to_print = False)


print (r2score_list[-1])


mrxb, r2_score_mrxb = XGBoost_r2(train_all_X, train.targets, test_all_X, r2score_list[-1][1], r2score_list[-1][2], 10**r2score_list[-1][3], 777)
print (r2score_list[-1][1], r2score_list[-1][2], 10**r2score_list[-1][3], r2_score_mrxb)


print (mrxb)


'''
(0.36083719273172943, 100, 6, -1.1178602)
100 6 0.0762324367284232 0.36083719273172943
name=None description='' tags=[] user_attributes={} owner=None polaris_version='dev' results=  Test set    Target label                      Metric     Score
0     test  LOG_SOLUBILITY  Metric.mean_absolute_error  0.409750
1     test  LOG_SOLUBILITY   Metric.mean_squared_error  0.346540
2     test  LOG_SOLUBILITY                   Metric.r2  0.360837
3     test  LOG_SOLUBILITY            Metric.spearmanr  0.496059
4     test  LOG_SOLUBILITY             Metric.pearsonr  0.602040
5     test  LOG_SOLUBILITY        Metric.explained_var  0.361843
'''
