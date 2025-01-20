from pulp import LpProblem, LpVariable, LpStatus, value, LpMinimize
import pandas as pd
import time
import csv
import numpy as np

def find_index(input_list, target):
    """リストの中から該当する要素のインデックスを出力する関数
    
    Parameters:
    input_list:検索対象のリスト
    target:検索したい要素
    
    Returns:
    int: 該当する要素のインデックス。見つからなければ-1を返す
    """

    try:
        return input_list.index(target)
    except ValueError:
        return -1
    
def sum_columns(matrix):
    """二次元リストの各列の合計を計算する関数
    
    Parameters:
    matrix(list of list):入力となる二次元配列
    
    Returns:
    list: 各列の合計値
    """

    if not matrix or not matrix[0]:
        return []

    # 列数
    num_cols = len(matrix[0])

    #各コラム用に初期化
    columns_sums = [0]*num_cols

    for row in matrix:
        for i in range(num_cols):
            columns_sums[i] += row[i]
    
    return columns_sums

def save_2d_array_to_csv(array, filename):
    """リストの中から該当する要素のインデックスを出力する関数
    
    Parameters:
    array(list of lists or np.array):出力したい二次元配列
    filename(str):出力先のファイル名
    
    """

    # Numpy配列の場合はリストに変換
    if isinstance(array, (list, tuple)):
        pass #すでにリストまたはタプルの場合はそのまま使用  
    elif hasattr(array, 'tolist'):
        array = array.tolist() #Numpy配列からリスト変換
    else:
        raise ValueError("array はリストまたはNumpyである必要があります")

    # CSVファイルへの書き込み
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(array)


if __name__ == '__main__':
    setubi = pd.read_csv("./csv/保有設備DB_10月-1.csv", encoding='utf-8')

    #重複データの確認
    setubi.drop_duplicates(subset=['設備'], inplace=False)

    #横軸
    setubi_list = setubi['設備'].tolist()

    ability = pd.read_csv("./csv/設備生産能力_10月.csv", encoding='utf-8')

    ability['H50+工程'] = ability['H50'] + '_' + ability['工程']

    items = ability.drop_duplicates(subset=['H50+工程'], inplace=False)

    #縦軸
    items_list = items['H50+工程'].tolist()

    spec_setubi = [[0 for _ in range(int(len(setubi_list)))] for _ in range(int(len(items_list)))]

    for index, row in ability.iterrows():
        spec_setubi[int(find_index(items_list, row['H50'] + '_' + row['工程']))][int(find_index(setubi_list, row['設備']))] = row['設備生産能力']

    PRODUCT_NUM = int(len(items_list))
    MACHINE_NUM = int(len(setubi_list))

    #確認のため
    check = sum_columns(spec_setubi)

    #入力
    mount = pd.read_csv("./csv/生産量_26F1.csv", encoding='utf-8')
    product_mount = mount['生産量'].to_numpy()

    #問題の定義（最小化問題）
    problem = LpProblem("Integer_Programming_Example", LpMinimize)

    #決定変数の定義（整数変数）
    tmps = [f"{item}__{setubi}" for setubi in setubi_list for item in items_list]

    objective_function = None

    #決定変数の定義
    for k in range(MACHINE_NUM):
        for l in range(PRODUCT_NUM):
            tmps[PRODUCT_NUM*k+l] = LpVariable(tmps[PRODUCT_NUM*k+l], lowBound=0)
            objective_function += tmps[PRODUCT_NUM*k+l]


    #目的関数の設定
    problem += objective_function, "Objective"

    tmps_2d = [tmps[i * PRODUCT_NUM:(i+1)*PRODUCT_NUM]for i in range(MACHINE_NUM)]
    transposed_tmps_2d = [list(row) for row in zip(*tmps_2d)]

    hoyuu_setubi = setubi['保有台数'].to_numpy()
    hoyuu_setubi_list = setubi['保有台数'].to_list()
    transposed_spec_setubi = [list(row) for row in zip(*spec_setubi)]

    max_indexs = []

    last_index = 0
    constraints = [None]*PRODUCT_NUM
    for i_index, product_constraint in enumerate(product_mount):
        for j_index in range(len(spec_setubi[i_index])):
            #設備生産能力 spec_setubi[i_index][j_index]
            #決定変数　transposed_tmps_2d[i_index][j_index]
            print(type(transposed_tmps_2d[i_index][j_index]))
            print(type(spec_setubi[i_index][j_index]))
            constraints[i_index] += transposed_tmps_2d[i_index][j_index] * spec_setubi[i_index][j_index]
        
        # 制約条件の追加1（能力指標に対する制約）
        problem += constraints[i_index] >= product_mount[i_index], f"Constraint_{last_index+1}"
        last_index = last_index + 1

        counter = 0
        max_index = -1
        max_value = -1
        #（ステップ1）各品種で設備生産能力が1番高い装置を探す
        for j_index in range(MACHINE_NUM):
            #設備生産能力がある
            if spec_setubi[i_index][j_index] > 0:
                counter = counter + 1
                if spec_setubi[i_index][j_index] > max_value:
                    max_value = spec_setubi[i_index][j_index]
                    max_index = j_index

        max_indexs.append(max_index)

    #（ステップ2）準備
    constraints2 = [None]*MACHINE_NUM
    for k in range(MACHINE_NUM):
        counter=0
        for l in range(PRODUCT_NUM):
            #設備生産能力がある
            if transposed_spec_setubi[k][l] > 0:
                counter = counter + 1
        for l in range(PRODUCT_NUM):
            if transposed_spec_setubi[k][l] > 0:
                constraints2[k] += tmps[PRODUCT_NUM*k+l]

    condition_list = []

    #（ステップ2）各品種で設備生産能力が一番高い装置を探す
    for i_index, product_constraint in enumerate(product_mount):
        for j_index in range(MACHINE_NUM):
            count=0
            #品種探索（該当品種の最大設備以外の行に、その他の品種の最大設備がないか探索）
            if j_index != max_indexs[i_index]:
                for k_index in range(PRODUCT_NUM):
                    #同じ品種ではないこと
                    if k_index != i_index:
                        #最大設備か判定
                        #if j_index == max_indexs[k_index]:
                        max_value=max([row[k_index] for row in transposed_spec_setubi])
                        #if transposed_spec_setubi[j_index][i_index] == transposed_spec_setubi[max_indexs[k_index]][k_index]:
                        if transposed_spec_setubi[j_index][k_index] == max_value:
                            count = count + 1
                #制約条件の格納条件
                #if count==0 and transposed_spec_setubi[j_index][i_index] != transposed_spec_setubi[max_indexs[k_index][i_index]]:
                if count == 0:
                    #condition_list.append(hoyuu_setubi[j_index] <= constraints2[j_index])
                    condition_list.append([j_index, constraints2[j_index]])
    
    #重複の削除　制約条件の追加２（保有台数に対する制約）
    """unique_list = []
    [unique_list.append(x) for x in condition_list if x not in unique_list]
    for condition in condition_list:
        problem += condition, f'Constraint_{last_index}'
        last_index += last_index + 1"""


    unique_list = []
    [unique_list.append(x) for x in condition_list if x not in unique_list]
    for index, constraint in unique_list:
        print(hoyuu_setubi[index])
        print(constraint)
        problem += constraint >= hoyuu_setubi[index], f"Constraint_{last_index+1}"
        last_index += last_index + 1


    #計測開始
    start_time = time.time()

    #問題を解く
    problem.solve()

    #計算終了
    end_time = time.time()

    #計算
    execution_time = end_time - start_time
    print(f'処理の実行時間：{execution_time}秒')

    #結果表示
    print("Status:", LpStatus[problem.status])
    for index, tmp in enumerate(tmps):
        if value(tmps[index])>0:
            print(f"Optimal value for {tmps[index]}:", value(tmps[index]))

    print("Minimum objective function value:", value(problem.objective))

    #CSV出力
    csv_output = [[0 for _ in range(int(len(items_list)))]for _ in range(int(len(setubi_list)))]
    for i_index in range(MACHINE_NUM):
        for j_index in range(PRODUCT_NUM):
            if value(tmps[PRODUCT_NUM*i_index+j_index])>0:
                csv_output[i_index][j_index] = value(tmps[PRODUCT_NUM*i_index+j_index])
    

    #データをCSVファイルとして出力
    save_2d_array_to_csv(csv_output, 'output.csv')

    check=0