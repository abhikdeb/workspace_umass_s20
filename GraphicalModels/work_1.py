import pandas as pd
import numpy as np

file_path = "D:/UMass/Spring20/688/Homework/Assignment1/Assignment1/Data/"
file_pre = "data"
file_ext = ".txt"
train_str = 'train'
test_str = 'test'

Nodes = ['A', 'G', 'BP', 'CH', 'HD', 'CP', 'EIA', 'ECG', 'HR']
Edges = [('G', 'BP'), ('G', 'CH'), ('A', 'CH'), ('A', 'HR'),
         ('BP', 'HD'), ('CH', 'HD'), ('HD', 'CP'), ('HD', 'EIA'),
         ('HD', 'ECG'), ('HD', 'HR')]

Pos = {'A': 0, 'G': 1, 'BP': 3, 'CH': 4, 'CP': 2, 'EIA': 7, 'ECG': 5, 'HR': 6, 'HD': 8}


def read_file(is_train, i):
    file_name = file_path+'data-'+(train_str if is_train else test_str)+'-'+str(i)+file_ext
    cols = ['A', 'G', 'CP', 'BP', 'CH', 'ECG', 'HR', 'EIA', 'HD']
    df = pd.read_csv(file_name, delimiter=',', header=None)
    df.columns = cols
    return df


def get_cpt(inp, cpt):
    ser = None
    if '~' not in cpt:
        ser = inp.groupby(cpt).size()/len(inp)
    else:
        nd, par = cpt.split('~')
        pars = par.split(',') if ',' in par else [par]
        ser = inp.groupby([nd]+pars).size()/(len(inp)*(inp.groupby(pars).size()/len(inp)))
    return ser


def get_counts(inp, cpt):
    ser = None
    if '~' not in cpt:
        ser = inp.groupby(cpt).size()
    else:
        nd, par = cpt.split('~')
        pars = par.split(',') if ',' in par else [par]
        ser = inp.groupby([nd] + pars).size()
    return ser


def initialize(nodes, edges):
    parents = {}
    cpt_list = []
    for n in nodes:
        if n not in parents.keys():
            parents[n] = []
    for e in edges:
        parents[e[1]].append(e[0])
    for p in parents.keys():
        pt = parents[p]
        if len(pt) == 0:
            cpt_list.append(p)
        else:
            cpt_list.append(p + '~' + (",".join(pt)))
    return parents, cpt_list


def fetch_params(df, cpt_list):
    params = {}
    for cs in cpt_list:
        params[cs] = get_cpt(df, cs)
    return params


def fetch_counts(df, cpt_list):
    count = {}
    for cs in cpt_list:
        count[cs] = get_counts(df, cs)
    return count


def prob_4():
    print("Running for Problem 4 .....")
    parents, cpt_list = initialize(Nodes, Edges)
    inp = read_file(True, 1)
    parameters = fetch_params(inp, cpt_list)
    print(parameters)
    return


def prob_5():
    print("Running for Problem 5 .....")
    parents, cpt_list = initialize(Nodes, Edges)
    inp = read_file(True, 1)
    parameters = fetch_params(inp, cpt_list)
    print("For part a, ")
    p1 = parameters['HD~BP,CH']
    p2 = parameters['CH~G,A']
    val = (p1.reorder_levels([0, 2, 1])[1][1])*(p2[1][2]) / \
          ((p1.reorder_levels([0, 2, 1])[1][1][1])*(p2[1][2][1]) + \
           (p1.reorder_levels([0, 2, 1])[1][1][2])*(p2[1][2][2]))
    print(val)
    # print("For part b, ")
    # p1 = parameters['G']
    # p2 = parameters['BP~G']
    # p3 = parameters['HD~BP,CH']
    # p4 = parameters['CH~G,A']
    #
    # # val = (p1.reorder_levels([0, 2, 1])[1][1]) * (p2[1][2]) / \
    # #       ((p1.reorder_levels([0, 2, 1])[1][1][1]) * (p2[1][2][1]) + \
    # #        (p1.reorder_levels([0, 2, 1])[1][1][2]) * (p2[1][2][2]))
    # print(val)
    return


def get_hd(line, params):
    dct = {}
    num = None
    den = None
    for n in Nodes:
        if n != 'HD':
            dct[n] = line[Pos[n]]
    p1 = params['HD~BP,CH']
    p2 = params['CP~HD']
    p3 = params['EIA~HD']
    p4 = params['ECG~HD']
    p5 = params['HR~A,HD']

    num = p1[dct['BP']][dct['CH']] * \
          (p2[dct['CP']]) * \
          (p3[dct['EIA']]) * \
          (p4[dct['ECG']]) * \
          ((p5.reorder_levels([0, 2, 1]))[dct['A']][dct['HR']])

    den = p1[dct['BP']][dct['CH']].sum() * \
          (p2[dct['CP']].sum()) * \
          (p3[dct['EIA']].sum()) * \
          (p4[dct['ECG']].sum()) * \
          ((p5.reorder_levels([0, 2, 1]))[dct['A']][dct['HR']].sum())

    val = num/den

    return val.idxmax()


def predict(params, df):
    hd = []
    for line in df.values:
        hd.append(get_hd(line, params))
    return hd


def prob_6():
    print("Running for Problem 6 ....")
    parents, cpt_list = initialize(Nodes, Edges)
    train_set = []
    test_set = []
    for case in range(1, 6):
        f = None
        for run in range(1, 6):
            if run == case:
                test_set.append(read_file(True, run))
            else:
                if f is None:
                    f = read_file(True, run)
                else:
                    f = pd.concat([f, read_file(True, run)], ignore_index=True)
        train_set.append(f)
    best_acc = -1
    best_params = None
    # Running Cross Validation
    for i in range(len(test_set)):
        inp_df = train_set[i]
        test_df = test_set[i]
        params = fetch_params(inp_df, cpt_list)
        # test_df = read_file(True, 5)
        pred_hd = predict(params, test_df)
        act_hd = test_df['HD']
        acc = sum(1 for x, y in zip(act_hd, pred_hd) if x == y) / len(act_hd)
        # print(acc)/
        if best_acc < acc:
            best_acc = acc
            best_params = params
    inp_df = None
    print('Best Accuracy achieved : ', best_acc)
    # Run on Test Data
    accs = []
    for i in range(1, 6):
        t_df = read_file(False, i)
        pred_hd = predict(best_params, t_df)
        act_hd = t_df['HD']
        acc = sum(1 for x, y in zip(act_hd, pred_hd) if x == y) / len(act_hd)
        print('Accuracy in part ', i, ' : ', acc)
        accs.append(acc)
    print('Mean : ', np.mean(np.array(accs)))
    print('Std Dev : ', np.std(np.array(accs)))
    return


def get_hd_7(line, params):
    dct = {}
    num = None
    den = None
    for n in Nodes:
        if n != 'HD':
            dct[n] = line[Pos[n]]
    p1 = params['HD~G,BP,CH']
    p2 = params['CP~HD,EIA']
    p3 = params['EIA~HD,HR']
    p4 = params['ECG~HD,HR']
    p5 = params['HR~CH,HD']

    num = (p1[dct['G']][dct['BP']][dct['CH']]) * \
          (p2.reorder_levels([1, 2, 0])[dct['EIA']][dct['CP']]) * \
          (p3.reorder_levels([1, 2, 0])[dct['HR']][dct['EIA']]) * \
          (p4.reorder_levels([1, 2, 0])[dct['HR']][dct['ECG']]) * \
          ((p5.reorder_levels([0, 2, 1]))[dct['CH']][dct['HR']])

    den = p1[dct['G']][dct['BP']][dct['CH']].sum() * \
          (p2.reorder_levels([1, 2, 0])[dct['EIA']][dct['CP']].sum()) * \
          (p3.reorder_levels([1, 2, 0])[dct['HR']][dct['EIA']].sum()) * \
          (p4.reorder_levels([1, 2, 0])[dct['HR']][dct['ECG']].sum()) * \
          ((p5.reorder_levels([0, 2, 1]))[dct['CH']][dct['HR']].sum())

    val = num/den

    return val.idxmax()


def predict_7(params, df):
    hd = []
    for line in df.values:
        hd.append(get_hd_7(line, params))
    return hd


def prob_7():
    print("Running for Problem 7 ....")
    nodes = Nodes
    edges = [('G', 'BP'), ('G', 'CH'), ('G', 'HD'), ('A', 'BP'), \
             ('A', 'CH'), ('BP', 'HD'), ('CH', 'HD'), ('CH', 'HR'), \
             ('HD', 'CP'), ('HD', 'EIA'), ('HD', 'ECG'), ('HD', 'HR'), \
             ('EIA', 'CP'), ('HR', 'ECG'), ('HR', 'EIA')]
    parents, cpt_list = initialize(nodes, edges)
    train_set = []
    test_set = []
    for case in range(1, 6):
        f = None
        for run in range(1, 6):
            if run == case:
                test_set.append(read_file(True, run))
            else:
                if f is None:
                    f = read_file(True, run)
                else:
                    f = pd.concat([f, read_file(True, run)], ignore_index=True)
        train_set.append(f)
    best_acc = -1
    best_params = None
    # Running Cross Validation
    for i in range(len(test_set)):
        inp_df = train_set[i]
        test_df = test_set[i]
        params = fetch_params(inp_df, cpt_list)
        # print(params)
        # test_df = read_file(True, 5)
        pred_hd = predict_7(params, test_df)
        act_hd = test_df['HD']
        acc = sum(1 for x, y in zip(act_hd, pred_hd) if x == y) / len(act_hd)
        print(acc)
        if best_acc < acc:
            best_acc = acc
            best_params = params
    inp_df = None
    print('Best Accuracy achieved : ', best_acc)
    # Run on Test Data
    accs = []
    for i in range(1, 6):
        t_df = read_file(False, i)
        pred_hd = predict_7(best_params, t_df)
        act_hd = t_df['HD']
        acc = sum(1 for x, y in zip(act_hd, pred_hd) if x == y) / len(act_hd)
        print('Accuracy in part ', i, ' : ', acc)
        accs.append(acc)
    print('Mean : ', np.mean(np.array(accs)))
    print('Std Dev : ', np.std(np.array(accs)))
    return


def main():
    # prob_4()
    # prob_6()
    prob_7()

    exit(0)


if __name__ == "__main__":
    main()
