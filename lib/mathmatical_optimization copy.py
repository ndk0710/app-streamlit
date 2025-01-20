from pulp import LpProblem, LpVariable, LpStatus, value, LpMinimize
import pandas as pd
import time
import csv

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

        #メンバ関数の初期化
        self.problem_definition()
        self.constraint_mount_definition()
        self.constraint_units_owned_definition()
        
    # 問題の定義（最小化問題）をする関数
    def problem_definition(self):

        self.problem = LpProblem('Integer_Programing_Example', LpMinimize)

        objective_function = None

        #決定変数の定義
        for k in range(self.machines_nums):
            for l in range(self.products_nums):
                self.decision_variable[self.products_nums*k+l] = LpVariable(self.decision_variable[self.products_nums*k+l], lowBound=0)
                objective_function += self.decision_variable[self.products_nums*k+l]

        #目的関数の設定
        self.problem += objective_function, "Objective"
    
    # 制約条件（能力指標に対する制約）の定義をする関数
    def constraint_mount_definition(self):
        
        self.decision_variable_2d = [self.decision_variable[i * self.products_nums:(i+1)*self.products_nums]for i in range(self.machines_nums)]
        self.transposed_decision_variable_2d= [list(row) for row in zip(*self.decision_variable_2d)]

        constraints = [None]*self.products_nums
        for i in range(self.products_nums):
            for j in range(self.machines_nums):
                constraints[i] += self.transposed_decision_variable_2d[i][j] * self.specs[i][j]
            
            # 制約条件の追加
            self.problem += constraints[i] >= self.mounts[i], f"Constraint_{self.last_index+1}"
            self.last_index += 1

    # 制約条件（生産量が多くなった場合、制約条件を満たさずにエラー⇒間違い）
    '''def constraint_units_owned_definition(self):
        
        self.transposed_specs = [list(row) for row in zip(*self.specs)]

        constraints = [None]*self.machines_nums
        for k in range(self.machines_nums):
            for l in range(self.products_nums):
                if self.transposed_specs[k][l] > 0:
                    constraints[k] += self.decision_variable[self.products_nums*k+l]
            self.problem += self.number_of_units_owned[k] >=constraints[k] , f"Constraint_{self.last_index+1}"
            self.last_index += 1'''

    # 制約条件
    '''def constraint_units_owned_definition(self):

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

            #if counter > 1:
            #    self.problem += self.number_of_units_owned[max_index] >= self.transposed_decision_variable_2d[i][max_index], f"Constraint_{self.last_index+1}"
            #    self.last_index += 1
        
        constraints = [None]*self.machines_nums
        for k in range(self.machines_nums):
            for l in range(self.products_nums):
                if max_indexs[l] == k:
                    constraints[k] += self.decision_variable[self.products_nums*k+l]
            if constraints[k] != None:
                self.problem += self.number_of_units_owned[k] >=constraints[k] , f"Constraint_{self.last_index+1}"
                self.last_index += 1'''



    # 制約条件
    def constraint_units_owned_definition(self):
        
        self.transposed_specs = [list(row) for row in zip(*self.specs)]

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
            for l in range(self.products_nums):
                #設備生産能力がある
                if self.transposed_specs[k][l] > 0:
                    counter += 1
            for l in range(self.products_nums):
                if self.transposed_specs[k][l] > 0:
                    constraints[k] += self.decision_variable[self.products_nums*k+l]

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
        [unique_list.append(x) for x in condition_list if x not in unique_list]
        for index, constraint in unique_list:
            self.problem += constraint >= self.number_of_units_owned[index], f"Constraint_{self.last_index+1}"
            self.last_index += 1
    
    # 問題を解く
    def solve(self):
        #計測開始
        start_time = time.time()

        #問題を解く
        self.problem.solve()

        #計算終了
        end_time = time.time()

        #計算
        execution_time = end_time - start_time
        print(f'処理の実行時間：{execution_time}秒')

        #結果表示
        print("Status:", LpStatus[self.problem.status])
        for index, _ in enumerate(self.decision_variable):
            if value(self.decision_variable[index])>0:
                print(f"Optimal value for {self.decision_variable[index]}:", value(self.decision_variable[index]))

        print("Minimum objective function value:", value(self.problem.objective))

        #CSV出力
        csv_output = [[0 for _ in range(self.products_nums)]for _ in range(self.machines_nums)]
        for i in range(self.machines_nums):
            for j in range(self.products_nums):
                if value(self.decision_variable[self.products_nums*i+j])>0:
                    csv_output[i][j] = value(self.decision_variable[self.products_nums*i+j])
                #計算結果をメンバ変数に格納
                self.number_of_calc_detail.append(value(self.decision_variable[self.products_nums*i+j]))
    

        #データをCSVファイルとして出力
        save_2d_array_to_csv(csv_output, 'output.csv')

    # 設備比較（保有台数と計算台数）
    def compare_machines(self):
        #計算結果取得
        self.matrix = [self.number_of_calc_detail[i*self.products_nums:(i+1)*self.products_nums] for i in range(self.machines_nums)]
        
        # 行方向に合計
        self.number_of_calc = [sum(row) for row in self.matrix]
        
        # 各要素の差分を計算
        self.diffs = [a - b for a, b in zip(self.number_of_calc, self.number_of_units_owned)]

        # 1回の数理最適化で良ければTrue
        return all(x <= 0 for x in self.diffs)

    # 残生産量を計算
    def calc_remaining_mount(self):

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
        
        subtract_value = 1

        # 残装置台数
        remaining_machines = []

        for i, diff in enumerate(self.diffs):
            if diff != 0:
                result = subtract_until_target(self.matrix[i], subtract_value, diff)
                remaining_machines.append(result)
            else:
                remaining_machines.append([0.0]*self.products_nums)
        
        remaining_mount = []
        for row1, row2 in zip(self.transposed_specs, remaining_machines):
            result_row = [a * b for a, b in zip(row1, row2)]
            remaining_mount.append(result_row)
        
        self.remaining_mount = [sum(row) for row in remaining_mount]
        
        oo=0