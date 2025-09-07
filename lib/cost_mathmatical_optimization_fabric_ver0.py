from pulp import LpProblem, LpVariable, LpStatus, value, LpMinimize, lpSum
import pandas as pd
import numpy as np
import time
import re


class Cost_Mathmatical:
    def __init__(self, products, cost_info, cost_mount_info, short_machine_info, matrix):
        #メンバ変数の初期化
        self.products = products
        self.products_nums = int(len(self.products))
        self.matrix = matrix
        self.cost_mount_info = cost_mount_info
        self.machines_nums = int(len(self.matrix))
        sum_short_machine_info = short_machine_info.groupby("設備", as_index=False)["不足台数"].sum()

        self.short_machine = sum_short_machine_info
        self.cost_info = cost_info
        #self.decision_variable = [f"{row['変更前']}__{row['変更後']}" for _, row in self.cost_info.iterrows()]
        self.decision_variable = [f"{row['拠点前']}_{row['変更前']}__{row['拠点後']}_{row['変更後']}" for _, row in self.cost_info.iterrows()]

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
        # 各 decision_variable に対応する (拠点前, 変更前, コスト) を取得
        before_sites = self.cost_info["拠点前"].tolist()
        before_machines = self.cost_info["変更前"].tolist()
        costs = self.cost_info["コスト"].tolist()

        # 全ての (拠点, 設備) を走査
        for _, row in self.cost_mount_info.iterrows():
            site = row["拠点"]
            machine = row["設備"]
            mount_number = row["保有台数"]

            # decision_variable の中で (拠点前, 変更前) が一致し、
            # かつコスト > 0 のものだけを集める
            vars_for_machine = [
                var for var, s, m, c in zip(self.decision_variable, before_sites, before_machines, costs)
                if (s == site and m == machine and c > 0)
            ]

            if vars_for_machine:  # 対応関係が存在する場合のみ制約を追加
                self.constraint_condition.append(lpSum(vars_for_machine) == mount_number)
    
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

    def update_cost_mount_info(self):
        # DataFrame: self.cost_mount_info
        # columns = ["拠点", "設備", "保有台数"]

        # 行に素早くアクセスできるように index を (拠点, 設備) に設定
        self.cost_mount_info = self.cost_mount_info.set_index(["拠点", "設備"])

        for var, (_, row) in zip(self.decision_variable, self.cost_info.iterrows()):
            value = var.value()
            if value is None or value == 0:
                continue

            from_site, from_machine = row["拠点前"], row["変更前"]
            to_site, to_machine = row["拠点後"], row["変更後"]

            # 移動元から減算
            self.cost_mount_info.at[(from_site, from_machine), "保有台数"] -= value
            # 移動先へ加算
            self.cost_mount_info.at[(to_site, to_machine), "保有台数"] += value

        # index を戻す
        self.cost_mount_info = self.cost_mount_info.reset_index()
    
    def extract_results(self):
        records = []

        # self.decision_variable は LpVariable のリスト
        for var in self.decision_variable:
            name = var.name
            value = var.varValue

            if value is None or value <= 0:  # 未割当 or 0 はスキップ
                continue

            # 変数名を分解（例: "TOM_MN0001__FMC_MN0001"）
            match = re.match(r"(\w+)_(\w+)__([\w]+)_(\w+)", name)
            if not match:
                continue

            pre_site, pre_machine, post_site, post_machine = match.groups()

            # コスト情報を検索
            cost_row = self.cost_info[
                (self.cost_info["拠点前"] == pre_site) &
                (self.cost_info["変更前"] == pre_machine) &
                (self.cost_info["拠点後"] == post_site) &
                (self.cost_info["変更後"] == post_machine)
            ]

            if cost_row.empty:
                continue

            cost = cost_row["コスト"].iloc[0]
            category = cost_row["分類"].iloc[0]

            records.append([
                pre_site, pre_machine, post_site, post_machine,
                cost, int(value), category
            ])

        return records
    

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

        #余剰設備の更新
        self.update_cost_mount_info()

        return self.extract_results()