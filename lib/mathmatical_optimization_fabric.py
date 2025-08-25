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
        self.c = [1] * len(self.decision_variable)
    
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
        # 空の辞書を作成
        variables = {}
        for variable in self.decision_variable:
            #動的に変数名と値を追加
            variables[variable]=(0, None)
            self.bounds.append(variables[variable])
    
    # 問題の定義（最小化問題）をする関数
    def problem_definition2(self):

        #数理最適化の制約条件
        self.constraint_mount_definition()

        #等式制約がない場合は空のリストを指定
        self.A_eq = None
        self.b_eq = None

        self.bounds = []
        # 空の辞書を作成
        variables = {}
        for variable in self.decision_variable:
            #動的に変数名と値を追加
            variables[variable]=(0, None)
            self.bounds.append(variables[variable])
    
    # 制約条件（能力指標に対する制約）の定義をする関数
    def constraint_mount_definition(self):
        # 制約条件➀
        self.transposed_specs = [list(row) for row in zip(*self.specs)]
        tmp = [item for sublist in self.transposed_specs for item in sublist]
        tmp = list(map(lambda x: x * -1, tmp))

        self.A_ub = []
        for j in range(self.products_nums):
            tmps2 = [0]*len(self.decision_variable)
            for i in range(self.machines_nums):
                tmps2[self.products_nums*i+j] = tmp[self.products_nums*i+j]
            self.A_ub.append(tmps2)

        self.b_ub = list(map(lambda x: x * -1, self.mounts))

    # 制約条件（設備生産能力）
    def constraint_specs_definition(self, machine_lsit):
        for machine_name in machine_lsit:
            #決定変数の定義
            for k in range(self.machines_nums):
                for l in range(self.products_nums):
                    if machine_name in self.machines[k]:
                        # 制約条件の追加
                        self.problem += self.decision_variable[self.products_nums*k+l] == 0, f"Constraint_{self.last_index+1}"
                        self.last_index += 1

    # 制約条件
    def constraint_units_owned_definition(self):
        #制約条件➁
        max_indexs = []

        for i in range(self.products_nums):
            counter = 0
            max_index = -1
            max_value = -1
            #（ステップ1）各品種で設備生産能力が1番高い装置を探す
            for j in range(self.machines_nums):
                #設備生産能力がある
                if self.specs[i][j] > 0:
                    counter += 1
                    if self.specs[i][j] > max_value:
                        max_value = self.specs[i][j]
                        max_index = j

            max_indexs.append(max_index)

        #（ステップ2）準備
        constraints = [None]*self.machines_nums
        for k in range(self.machines_nums):
            counter = 0
            tmps = []
            for l in range(self.products_nums):
                #設備生産能力がある
                if self.transposed_specs[k][l] > 0:
                    counter += 1
                    tmps.append(self.decision_variable[self.products_nums*k+l])
            constraints[k] = tmps

        condition_list = []

        #（ステップ2）各品種で設備生産能力が一番高い装置を探す
        for i in range(self.products_nums):
            for j in range(self.machines_nums):
                counter = 0
                #品種探索（該当品種の最大設備以外の行に、その他の品種の最大設備がないか探索）
                if j != max_indexs[i]:
                    for k in range(self.products_nums):
                        #同じ品種ではないこと
                        if k != i:
                            #最大設備か判定
                            max_value=max([row[k] for row in self.transposed_specs])
                            if self.transposed_specs[j][k] == max_value:
                                counter += 1
                    #制約条件の格納条件
                    if counter == 0:
                        condition_list.append([j, constraints[j]])

        #重複の削除　制約条件の追加２（保有台数に対する制約）
        unique_list = []
        [unique_list.append(x) for x in condition_list if x not in unique_list and x[1] is not None]
        for index, constraint in unique_list:
            outputs = [0] * len(self.decision_variable)
            for target in constraint:
                if target in self.decision_variable:
                    idx = find_index(self.decision_variable, target)
                    outputs[idx] = 1
            # リストに追加
            self.A_ub.append(list(map(lambda x: x * -1, outputs)))
            self.b_ub.append(-1*self.number_of_units_owned[index])

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
                # 装置情報の取得
                row_name = self.decision_variable[self.products_nums*i]
                # 特定の文字列が存在する位置を検索
                row_index = row_name.find('__')
                self.row_names.append(row_name[row_index+2:])
                for j in range(self.products_nums):
                    if result.x[self.products_nums*i+j] > 0:
                        csv_output[i][j] = result.x[self.products_nums*i+j]
                    if i==0:
                        column_name = self.decision_variable[self.products_nums*i+j]
                        column_index = column_name.find('__')
                        self.column_names.append(column_name[:column_index])

                    #計算結果をメンバ変数に格納（小数点以下3桁目で四捨五入）
                    self.number_of_calc_detail.append(round(result.x[self.products_nums*i+j],2))  

            #データをXLSファイルとして出力
            df = pd.DataFrame(np.array(csv_output), index=self.row_names, columns=self.column_names)
            df.to_excel('output.xlsx', index=True)

            #計算結果取得
            self.matrix = [self.number_of_calc_detail[i*self.products_nums:(i+1)*self.products_nums] for i in range(self.machines_nums)]
        else:
            print("最適解が見つかりませんでした")
            print(f"メッセージ: {result.message}")

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
            # 引いた値の合計
            total = 0
            # 処理中のリスト
            result = lst.copy()
            # リストの各要素を引く
            while total < target:
                for i in range(len(result)):
                    if total == target:
                        break

                    old_result = result[i]
                    result[i] = max(0, result[i] - subtract_value)
                    total += subtract_value if result[i]-old_result != 0 else 0
            return result
        
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