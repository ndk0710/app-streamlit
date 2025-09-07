from pulp import LpProblem, LpVariable, LpStatus, value, LpMinimize, lpSum
import time
import re


class Cost_Mathmatical:
    def __init__(self, cost_info, cost_mount_info, short_machine_info):
        # メンバ変数の初期化
        self.cost_mount_info = cost_mount_info
        sum_short_machine_info = short_machine_info.groupby("設備", as_index=False)["不足台数"].sum()

        self.short_machine = sum_short_machine_info
        self.cost_info = cost_info
        self.decision_variable = [
            f"{row['拠点前']}_{row['変更前']}__{row['拠点後']}_{row['変更後']}"
            for _, row in self.cost_info.iterrows()
        ]

        self.last_index = 0
        self.constraint_condition = []

        # メンバ関数の初期化
        self.problem_definition()
    
    def __del__(self):
        print(f"オブジェクトを削除しました")
        
    # 問題の定義（最小化問題）
    def problem_definition(self):
        self.problem = LpProblem('Integer_Programing_Example', LpMinimize)

        objective_function = None

        # 決定変数の定義
        for index, row in self.cost_info.iterrows():
            self.decision_variable[index] = LpVariable(self.decision_variable[index], lowBound=0)
            objective_function = (
                self.decision_variable[index] * row["コスト"]
                if objective_function is None
                else objective_function + self.decision_variable[index] * row["コスト"]
            )
            
        # 目的関数の設定
        self.problem += objective_function, "Objective"
    
    # 制約条件 (余剰設備に対する制約)
    def constraint_excess_facilities_definition(self):
        before_sites = self.cost_info["拠点前"].tolist()
        before_machines = self.cost_info["変更前"].tolist()
        costs = self.cost_info["コスト"].tolist()

        for _, row in self.cost_mount_info.iterrows():
            site = row["拠点"]
            machine = row["設備"]
            mount_number = row["保有台数"]

            vars_for_machine = [
                var for var, s, m, c in zip(self.decision_variable, before_sites, before_machines, costs)
                if (s == site and m == machine and c > 0)
            ]

            if vars_for_machine:
                self.constraint_condition.append(lpSum(vars_for_machine) == mount_number)
    
    # 制約条件 (不足設備に対する制約)
    def constraint_lack_of_facilities_definition(self):
        for _, row in self.short_machine.iterrows():
            target_machine = row["設備"]
            short_number = row["不足台数"]
            filtered_cost_info = self.cost_info.loc[
                (self.cost_info["変更後"] == target_machine) & (self.cost_info["コスト"] >= 10)
            ]
            tmp = 0
            for index, _ in filtered_cost_info.iterrows():
                tmp += self.decision_variable[index]
            self.constraint_condition.append(tmp <= short_number)

    def set(self):
        for constraint in self.constraint_condition:
            self.problem += constraint, f'Constraint_{self.last_index}'
            self.last_index += 1

    def update_cost_mount_info(self):
        self.cost_mount_info = self.cost_mount_info.set_index(["拠点", "設備"])

        for var, (_, row) in zip(self.decision_variable, self.cost_info.iterrows()):
            value = var.value()
            if value is None or value == 0:
                continue

            from_site, from_machine = row["拠点前"], row["変更前"]
            to_site, to_machine = row["拠点後"], row["変更後"]

            self.cost_mount_info.at[(from_site, from_machine), "保有台数"] -= value
            self.cost_mount_info.at[(to_site, to_machine), "保有台数"] += value

        self.cost_mount_info = self.cost_mount_info.reset_index()
    
    def extract_results(self):
        records = []
        for var in self.decision_variable:
            name = var.name
            value = var.varValue
            if value is None or value <= 0:
                continue

            match = re.match(r"(\w+)_(\w+)__([\w]+)_(\w+)", name)
            if not match:
                continue

            pre_site, pre_machine, post_site, post_machine = match.groups()
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

            records.append([pre_site, pre_machine, post_site, post_machine, cost, int(value), category])

        return records
    
    def solve(self):
        start_time = time.time()
        self.problem.solve()
        end_time = time.time()
        print(f'処理の実行時間：{end_time - start_time}秒')

        print("Status:", LpStatus[self.problem.status])
        for index, _ in enumerate(self.decision_variable):
            if value(self.decision_variable[index]) is not None and value(self.decision_variable[index]) > 0:
                print(f"Optimal value for {self.decision_variable[index]}:", value(self.decision_variable[index]))

        print("Minimum objective function value:", value(self.problem.objective))
        self.update_cost_mount_info()
        return self.extract_results()
