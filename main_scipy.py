from lib.mathmatical_optimization import Mathmatical
from scipy.optimize import linprog
import pandas as pd
import numpy as np
import time

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

class Mathmatical:
    def __init__(self, products, machines, specs, mounts, number_of_units_owned):
        #メンバ変数の初期化
        self.products = products
        self.machines = machines
        self.products_nums = int(len(self.products))
        self.machines_nums = int(len(self.machines)) 
        self.specs = specs
        self.mounts = mounts
        self.number_of_units_owned = number_of_units_owned
        self.decision_variable = [f"{product}__{machine}" for machine in self.machines for product in self.products]
        self.last_index = 0
        self.number_of_calc_detail = []
        self.c = [1]*len(self.decision_variable)
    
    def __del__(self):
        print(f"オブジェクトを削除しました")
        
    # 問題の定義（最小化問題）をする関数
    def problem_definition(self):

        #数理最適化の制約条件
        self.constraint_mount_definition()
        self.constraint_units_owned_definition()

        #等式制約がない場合は空のリストを指定
        self.A_eq = None
        self.b_eq = None

        self.bounds = []

        variables = {}
        for variable in self.decision_variable:
            #動的に変数名と値を追加
            variables[variable]=(0,None)
            self.bounds.append(variables[variable])
    
    # 問題の定義（最小化問題）をする関数
    def problem_definition2(self):

        #数理最適化の制約条件
        self.constraint_mount_definition()

        #等式制約がない場合は空のリストを指定
        self.A_eq = None
        self.b_eq = None

        self.bounds = []

        variables = {}
        for variable in self.decision_variable:
            #動的に変数名と値を追加
            variables[variable]=(0,None)
            self.bounds.append(variables[variable])

    # 制約条件（能力指標に対する制約）の定義をする関数
    def constraint_mount_definition(self):
        
        #self.specsをNumpy配列に変換し転置する
        specs_arr = np.array(self.specs, dtype=np.float32)
        transposed_specs_array = specs_arr.T
        self.transposed_specs = transposed_specs_array.tolist()

        #転置済みのspecsをflattenして各要素に-1をかける
        tmp_array = -transposed_specs_array.flatten()

        #A_ubは製品数（products_nums）x（機械数x製品数）の行列
        total_vars = self.machines_nums * self.products_nums
        products_nums = self.products_nums
        machines_nums = self.machines_nums

        #ゼロで初期化
        A_ub = np.zeros((products_nums, total_vars), dtype=np.float32)

        #各製品jについて各機械iの対応する箇所（index=products_nums * i + j）に
        #tmp_arrayの値を代入（その箇所は0のまま）
        for j in range(products_nums):
            idx = np.arange(machines_nums) * products_nums + j
            A_ub[j,idx] = tmp_array[idx]
        
        self.A_ub = A_ub

        #b_ubはmountsの各要素に-1を掛けたものとする
        self.b_ub = [-m for m in self.mounts]


    # 制約条件
    def constraint_units_owned_definition(self):
        
        P = self.products_nums
        M = self.machines_nums

        # 各品種で設備生産能力が>0の中から最大値を持つ装置のindexを求める
        specs_arr = np.array(self.specs, dtype=np.float32)
        mask = specs_arr > 0

        #正でない部分は-infに置換してargmaxの対象外にする
        masked_specs = np.where(mask, specs_arr, -np.inf)
        max_indices = np.argmax(masked_specs, axis=1)
        #全行が負なら-1とする
        no_positive = ~mask.any(axis=1)
        max_indices[no_positive] = -1
        max_indexs = max_indices.tolist()

        #各装置ごとに該当する決定変数（＝その装置で能力がある品種）をリストとして保持
        constraints = [
            [P * k + l for l in range(P) if self.transposed_specs[k][l]>0]
            for k in range(M)
        ]

        #各品種iについて最大装置（max_indices[i]）以外のjについて、
        # ほかの品種k（kはi以外）の中で、装置jの能力がその品種kでの全装置の最大値と一致しないかチェック
        trans_specs_arr = np.array(self.transposed_specs, dtype=np.float32)
        col_maxes = np.max(trans_specs_arr,axis=0)

        # 条件チェックは「各品種iで装置j（max_indices[i]以外のj）について
        # すべての品種k（kはi以外）でtrans_spec_arr[j,k] != colmaxes[k]ならOK
        condition_list = []

        # 外側は品種i,内側は装置j
        for i in range(P):
            for j in range(M):
                if j == max_indexs[i]:
                    continue
                # チェック対象；品種kがiでない部分
                # np.deleteでi番目を除いた行jとcolumn_maxの該当要素を取り出す
                spec_j_without_i = np.delete(trans_specs_arr[j], i)
                colmax_without_i = np.delete(col_maxes,i)
                # どの品種についても装置jの能力がその列の最大値でないなら条件成立
                if not np.any(spec_j_without_i == colmax_without_i):
                    condition_list.append([j, constraints[j]])
        
        #重複削除（同一の（machine_index, list）ペアが複数現れる場合）
        unique_list = []
        for item in condition_list:
            if item not in unique_list:
                unique_list.append(item)
        
        #各条件候補に対して決定変数のindicator vectorを作成して制約式を追加
        for machine_index, cons_indices in unique_list:
            outputs = [0] * len(self.decision_variable)
            for idx in cons_indices:
                outputs[idx] = 1
            #制約条件；-(indicator vector)*(decision_variable) >= - number_of_units_owned[machine_index]
            self.A_ub = np.vstack((self.A_ub,[-x for x in outputs]))
            self.b_ub.append(-1*self.number_of_units_owned[machine_index])
            self.last_index += 1

    
    # 問題を解く
    def solve(self):
        #計測開始
        start_time = time.time()

        #問題を解く
        result = linprog(self.c, A_ub=self.A_ub, b_ub=self.b_ub, A_eq=self.A_eq, b_eq=self.b_eq, bounds=self.bounds, method='highs')

        #計算終了
        end_time = time.time()

        #計算
        execution_time = end_time - start_time
        print(f'処理の実行時間：{execution_time}秒')

        #結果表示
        if result.success:
            print('最適解が見つかりました。')

            #CSV出力
            csv_output = [[0 for _ in range(self.products_nums)]for _ in range(self.machines_nums)]
            self.row_names = []
            self.column_names = []
            for i in range(self.machines_nums):
                #装置情報の取得
                row_name = self.decision_variable[self.products_nums*i]
                # 特定の文字列が存在する位置を検索
                index = row_name.find('__')
                self.row_names.append(row_name[index+2:])
                for j in range(self.products_nums):
                    if result.x[self.products_nums*i+j]>0:
                        csv_output[i][j] = result.x[self.products_nums*i+j]
                    if i==0:
                        column_name = self.decision_variable[self.products_nums*i]
                        column_index = column_name.find('__')
                        self.column_names.append(column_name[:column_index])
                    
                    #計算結果をメンバ変数に格納（小数点以下3桁目で四捨五入）
                    self.number_of_calc_detail.append(round(result.x[self.products_nums*i+j],2))

            #データをXLSファイルとして出力
            df = pd.DataFrame(np.array(csv_output), index=self.row_names, columns=self.column_names)
            df.to_excel('output.xlsx', index=True)

            #計算結果取得
            self.matrix = [self.number_of_calc_detail[i*self.products_nums:(i+1)*self.products_nums] for i in range(self.machines_nums)]

            o=0
            # 
        else:
            print('最適解が見つかりませんでした。')
            print(f'メッセージ：{result.message}')

    # 設備比較（保有台数と計算台数）
    def compare_machines(self):
        
        # 行方向に合計
        self.number_of_calc = [sum(row) for row in self.matrix]
        
        # 各要素の差分を計算
        self.diffs = [a - b for a, b in zip(self.number_of_calc, self.number_of_units_owned)]

        # 1回の数理最適化で良ければTrue
        return all(x <= 0 for x in self.diffs)

    # 残生産量を計算
    def calc_remaining_mount(self):
        #
        def subtract_until_target(lst, subtract_value, target):
            # 元のリストをNumpy配列に変換（float型）
            A = np.array(lst, dtype=np.float32)

            # 目標の引き算回数
            T_int = int(round(target/subtract_value))

            # 各要素が引ける最大回数（容量）
            cap = np.floor(A / subtract_value).astype(np.int64)
            n = len(A)

            #割り当てを計算するため、容量が小さい順にソート
            order = np.argsort(cap)
            cap_sorted = cap[order]
            allocation = np.zeros(n, dtype=np.int64)
            T_remaining = T_int
            current_level = 0

            #ソート済み順に各要素にいくつ引くかを決定
            for i in range(n):
                active = n-i #現在対象となる（余力のある）要素数
                #次の要素までに増加可能な量
                increment = cap_sorted[i] - current_level
                if increment <= 0:
                    allocation[order[i]] = cap_sorted[i]
                    continue
                if T_remaining >= increment * active:
                    #全アクティブ要素にincrement回分引く
                    allocation[order[i:]] += increment
                    current_level = cap_sorted[i]
                    T_remaining -= increment * active
                else:
                    #全アクティブ要素に均等に割り当て、残りは先頭から1個ずつ配分
                    add = T_remaining // active
                    rem = T_remaining % active
                    allocation[order[i:]] += add
                    if rem > 0:
                        #ソート順にrem個だけ1追加
                        allocation[order[i:]][:rem] += 1
                    
                    T_remaining = 0
                    break
            
            #各要素からallcation回分subtract_valueを引く
            result = np.maximum(0, A-allocation*subtract_value)

            return result.tolist()
        
        subtract_value = 0.01

        # 残装置台数
        remaining_machines = []

        for i, diff in enumerate(self.diffs):
            if diff > 0:
                result = subtract_until_target(self.matrix[i], subtract_value, self.number_of_units_owned[i])
                remaining_machines.append(result)
            else:
                remaining_machines.append([0.0]*self.products_nums)
        
        # 残生産量,装置台数（保有台数分）
        remaining_mount = []
        self.owned_matrix = []
        for row1, row2, row3 in zip(self.transposed_specs, remaining_machines, self.matrix):
            result_row = [a * b for a, b in zip(row1, row2)]
            remaining_mount.append(result_row)
            result_row2 = [round(c - b,2) for b, c in zip(row2, row3)]
            self.owned_matrix.append(result_row2)
        
        # 品種毎に残生産量を合計
        self.remaining_mount = [sum(column) for column in zip(*remaining_mount)]
        
        return self.remaining_mount

def get_products(specs_info):
    """メンバ変数である品種を取得する関数
    
    Parameters:
    specs(pandas):設備生産能力に関連する情報
    
    Returns:
    list: メンバ変数である品種
    """
    specs = specs_info.copy()

    #カラムを作成
    specs['H50+工程'] = specs['H50'] + '_' + specs['工程']
    
    #重複を削除
    specs.drop_duplicates(subset=['H50+工程'], inplace=True)    
    
    return specs['H50+工程'].tolist()

def get_machines(specs_info):
    """メンバ変数である設備を取得する関数
    
    Parameters:
    specs(pandas):保有設備に関連する情報
    
    Returns:
    list: メンバ変数である設備
    """
    specs = specs_info.copy()
    
    #重複を削除
    specs.drop_duplicates(subset=['設備'], inplace=True)    
    

    return specs['設備'].tolist()

def get_specs(products, machines, specs_info):
    """メンバ変数である設備生産能力を取得する関数
    
    Parameters:
    products(list):品種に関連する情報
    machines(list):設備に関連する情報
    
    Returns:
    list: メンバ変数である設備生産能力
    """
    # 事前に各リストからインデックス用の辞書を作成
    product_index = {p: i for i,p in enumerate(products)}
    machine_index = {m: j for j,m in enumerate(machines)}

    # 対象設備のみに絞る（辞書に存在するものだけ）
    specs_info = specs_info[specs_info['設備'].isin(machines)]

    # 製品キーを連結
    prod_keys = specs_info['H50'] + '_' + specs_info['工程']

    # 各行の製品・設備のインデックスを辞書を用いて」map
    row_indices = prod_keys.map(product_index)
    col_indices = specs_info['設備'].map(machine_index)

    # 対応する設備生産能力の値
    capacities = specs_info['設備生産能力'].values

    # Numpy配列でゼロ初期化
    specs_array = np.zeros((len(products), len(machines)))

    # ベクトル化されたインデックス代入
    specs_array[row_indices.values, col_indices.values] = capacities

    return specs_array.tolist()
    
def get_mounts(mounts_info):
    """メンバ変数である生産量を取得する関数
    
    Parameters:
    mounts_info(list):生産量に関連する情報
    
    Returns:
    list: メンバ変数である生産量
    """
    mounts = mounts_info['生産量'].tolist()

    return mounts

def get_mounts_all(mounts_info):
    """メンバ変数である生産量を取得する関数
    
    Parameters:
    mounts_info(list):生産量に関連する情報
    
    Returns:
    list: メンバ変数である生産量
    """
    filtered_list = [word for word in mounts_info.columns if '生産量' in word]
    mounts = []
    for col_name in filtered_list:
        mounts.append(mounts_info[col_name].tolist())

    return mounts

def get_number_of_units_owned(units_owned_info):
    """メンバ変数である生産量を取得する関数
    
    Parameters:
    mounts_info(list):生産量に関連する情報
    
    Returns:
    list: メンバ変数である生産量
    """
    number_of_units_owned = units_owned_info['保有台数'].tolist()

    return number_of_units_owned

def pad_with_zeros(smaller_list, larger_list):
    # 大きいリストの行数と列数を取得
    max_rows = len(larger_list)
    max_cols = max(len(row) for row in larger_list)
    # 行数を大きいリストと同じにする
    while len(smaller_list) < max_rows:
        smaller_list.append([0] * max_cols)
        # 各行の列数を大きいリストと同じにする
        for row in smaller_list:
            while len(row) < max_cols:
                row.append(0)
    return smaller_list

def update_machines_number_of_units_owned(result_matrix, number_of_units_owned, machines):
    # 保有台数の更新（各行の和を計算）
    number_of_units_owned = [sum(row) for row in result_matrix]

    # 各要素を切り上げて整数に変換
    number_of_units_owned = [int(x) + (x > int(x)) for x in number_of_units_owned]

    # メンバ変数（設備/保有台数）の更新
    value = 0
    indexs_to_remove=[index for index, element in enumerate(number_of_units_owned) if element == value]

    # インデックスのセットを作成
    indexs_to_remove_set = set(indexs_to_remove)

    # リスト内包表現で新しいリストを作成
    number_of_units_owned = [item for idx, item in enumerate(number_of_units_owned) if idx not in indexs_to_remove_set]
    machines = [item for idx, item in enumerate(machines) if idx not in indexs_to_remove_set]

    return number_of_units_owned, machines

def remove_duplicates_keep_first(words):
    seen = set()
    result = []
    for word in words:
        if word not in seen:
            seen.add(word)
            result.append(word)
    return result

def round_up_if_greater_than_one(number):
    # 小数第一位を取得
    decimal_part = number - int(number)
    if int(number) == 0:
        return int(number) + 1 #繰り上げ
    else:
        # 小数第一位が1以上の場合は繰り上げ
        if decimal_part >= 0.1:
            return int(number) + 1
        else:
            return int(number)




if __name__ == '__main__':
    #データ読込み
    specs_info = pd.read_csv("./test_patran/test0_設備生産能力.csv", encoding='utf-8')
    units_owned_info = pd.read_csv("./test_patran/test0_保有設備.csv", encoding='utf-8')
    mounts_info = pd.read_csv("./test_patran/test0_生産量.csv", encoding='utf-8')

    #計測開始
    start_time = time.time()

    #メンバ変数（品種/設備/設備生産能力）
    products = get_products(specs_info)
    machines = get_machines(units_owned_info)
    specs = get_specs(products, machines, specs_info)

    #メンバ変数（保有台数）
    number_of_units_owned = get_number_of_units_owned(units_owned_info)

    #メンバ変数（生産量）
    production_planning = get_mounts_all(mounts_info)

    for mounts in production_planning:

        #数理最適化のインスタンス作成
        math_matical = Mathmatical(products, machines, specs, mounts, number_of_units_owned)

        #問題の定義をする関数
        math_matical.problem_definition()

        #数理最適化の問題を解く
        math_matical.solve()

        # 保有台数比較
        flag = math_matical.compare_machines()

        # 保有台数以上の生産能力が必要と判断
        if flag == False:
            # 残生産量を計算
            mounts = math_matical.calc_remaining_mount()

            number_of_units_owned_copy = number_of_units_owned
            number_of_units_owned = constraints = [0]*math_matical.machines_nums

            old_calc_2d = math_matical.owned_matrix
            
            # オブジェクトの削除
            del math_matical

            #数理最適化のインスタンス作成
            math_matical = Mathmatical(products, machines, specs, mounts, number_of_units_owned)

           #問題の定義をする関数
            math_matical.problem_definition2()
            
            #数理最適化の問題を解く
            math_matical.solve()

            new_calc_2d = math_matical.matrix

            # サイズが一致しない場合、修正
            if len(old_calc_2d) != len(new_calc_2d):
                old_calc_2d = pad_with_zeros(old_calc_2d, new_calc_2d)

            # 2次元リストの要素同士を足す
            result_matrix = [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(old_calc_2d, new_calc_2d)]

            #データをXLSファイルとして出力
            #df = pd.DataFrame(np.array(result_matrix), index=math_matical.row_names, columns=math_matical.column_names)
            #df.to_excel('output2.xlsx', index=True)

            # 必要な装置台数
            calc_machines_sums = [sum(row) for row in result_matrix]

            # 必要な装置台数
            calc_machines_sums = [round_up_if_greater_than_one(num) for num in calc_machines_sums]

            #不足装置台数
            shortage_machines = [a-b for a,b in zip(calc_machines_sums, number_of_units_owned_copy)]

            # メンバ変数（設備/保有台数）の更新
            number_of_units_owned, machines = update_machines_number_of_units_owned(result_matrix ,number_of_units_owned, machines)

    #入力
    check=0

    #計算終了
    end_time = time.time()

    #計算
    execution_time = end_time - start_time
    print(f'処理の実行時間:{execution_time}秒')
    