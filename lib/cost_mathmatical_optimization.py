from pulp import LpProblem, LpVariable, LpStatus, value, LpMinimize
import pandas as pd
import numpy as np
import time


def get_costs(cost_info, cost_mount):

    bases = list(dict.fromkeys(cost_info['拠点前'].tolist()))

    #output = []
    for index, base in enumerate(bases):
        #特定の拠点に絞る
        filter_cost_info = cost_info[cost_info['拠点前'] == base].copy()
        filter_cost_mount = cost_mount[cost_mount['拠点'] == base].copy()

        filter_cost_info = filter_cost_info.loc[filter_cost_info["変更前"].isin(filter_cost_mount["設備"].tolist())].reset_index(drop=True)
        if index == 0:
            output = filter_cost_info
        else:
            # 縦方向に結合
            output = pd.concat([output, filter_cost_info], ignore_index=True)

    return output

class Cost_Mathmatical:
    def __init__(self, products, cost_info, cost_mount_info, short_machine_info, matrix):
        #メンバ変数の初期化
        self.products = products
        self.products_nums = int(len(self.products))
        self.matrix = matrix
        self.cost_mount = cost_mount_info.loc[cost_mount_info["保有台数"] >= 1]
        sum_short_machine_info = short_machine_info.groupby("設備", as_index=False)["不足台数"].sum()

        self.short_machine = sum_short_machine_info
        self.cost_info = get_costs(cost_info, self.cost_mount)

        self.decision_variable = [f"{row['変更前']}__{row['変更後']}" for _, row in self.cost_info.iterrows()]
        '''self.decision_variable = [f"{row['拠点前']}_{row['変更前']}__{row['拠点後']}_{row['変更後']}" for _, row in self.cost_info.iterrows()]'''

        self.last_index = 0
        self.constraint_condition = []

        #メンバ関数の初期化
        self.problem_definition()
    
    def __del__(self):
        print(f"オブジェクトを削除しました")
        
    # 問題の定義（最小化問題）をする関数
    def problem_definition(self):

        self.problem = LpProblem('Integer_Programing_Example', LpMinimize)

        objective_function = None

        #決定変数の定義
        for index, row in self.cost_info.iterrows():
            self.decision_variable[index] = LpVariable(self.decision_variable[index], lowBound=0)
            objective_function += self.decision_variable[index] * row["コスト"]
            
        #目的関数の設定
        self.problem += objective_function, "Objective"
    
    # 制約条件 (余剰設備に対する制約)の定義をする関数
    def constraint_excess_facilities_definition(self):
        cost= self.cost_info["コスト"].tolist()
        self.col_size = len(self.cost_mount)
        self.machines_nums = int(len(cost)/self.col_size)

        cost_2d = [cost[i:i + self.machines_nums] for i in range(0, len(cost), self.machines_nums)]
        for i in range(self.col_size):
            tmp=0
            for j in range(self.machines_nums):
                if cost_2d[i][j] > 0:
                    tmp+=self.decision_variable[i*self.machines_nums+j]
            name = self.decision_variable[i*self.machines_nums+j].name.split("__")[0]
            '''name = name.split("_")[1]'''
            mount_number = self.cost_mount["保有台数"].loc[self.cost_mount["設備"] == name].iloc[0]
            self.constraint_condition.append(tmp == mount_number)

     # 制約条件 (不足設備に対する制約)の定義をする関数
    def constraint_lack_of_facilities_definition(self):
        for index, row in self.short_machine.iterrows():
            target_machine = row["設備"]
            short_number = row["不足台数"]
            filtered_cost_info = self.cost_info.loc[(self.cost_info["変更後"] == target_machine) & (self.cost_info["コスト"] >= 10)]
            tmp=0
            for index, row in filtered_cost_info.iterrows():
                tmp+=self.decision_variable[index]
            self.constraint_condition.append(tmp <= short_number)

    def set(self):
        for constraint in self.constraint_condition:
            #制約条件の追加
            self.problem += constraint, f'Constraint_{self.last_index}'
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

        ans_machine = [0 for _ in range(self.machines_nums)]

        #結果表示
        print("Status:", LpStatus[self.problem.status])
        for index, _ in enumerate(self.decision_variable):
            if value(self.decision_variable[index]) is not None and value(self.decision_variable[index]) > 0:
                print(f"Optimal value for {self.decision_variable[index]}:", value(self.decision_variable[index]))
                remainder = index % self.machines_nums
                ans_machine[remainder] += int(value(self.decision_variable[index]))

        print("Minimum objective function value:", value(self.problem.objective))


        def subtract_until_zero(a, b):
            if a > b:  # AがBより大きい場合のみ処理を開始
                while b > 0:
                    a -= b
                    #print(f"A: {a}, B: {b}")
                    b -= 1
            return a


        #CSV出力
        self.row_names = []
        for i in range(self.machines_nums):
            
            #装置情報の取得
            row_name = self.decision_variable[i].name
            # 特定の文字列が存在する位置を検索
            index = row_name.find('__')
            self.row_names.append(row_name[index+2:])
            
            for j in range(self.products_nums):
                if ans_machine[i] == 0:
                    break
                else:
                    if self.matrix[i][j] > 0:
                        #輸送・改造コストによる振り分け
                        self.matrix[i][j] = subtract_until_zero(self.matrix[i][j], ans_machine[i])
                        break    
                    
        #データをXLSファイルとして出力
        df = pd.DataFrame(np.array(self.matrix), index=self.row_names, columns=self.products)
        df.to_excel('output_cost.xlsx', index=True)