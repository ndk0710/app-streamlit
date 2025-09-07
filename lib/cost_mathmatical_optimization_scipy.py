import pandas as pd
import numpy as np
import re
from scipy.optimize import linprog
import time

class Cost_Mathmatical_Scipy:
    def __init__(self, cost_info, cost_mount_info, short_machine_info):
        self.cost_info = cost_info.copy()
        self.cost_mount_info = cost_mount_info.copy()
        self.short_machine = short_machine_info.groupby("設備", as_index=False)["不足台数"].sum()
        
        # 決定変数の名前
        self.var_names = [
            f"{row['拠点前']}_{row['変更前']}__{row['拠点後']}_{row['変更後']}"
            for _, row in self.cost_info.iterrows()
        ]
        self.n_vars = len(self.var_names)

    def build_problem(self):
        """線形計画の係数行列を作成"""
        # 目的関数
        self.c = self.cost_info["コスト"].values.astype(float)

        # 余剰設備制約: sum(x_ij for j where (拠点前, 変更前) == (site, machine) and cost>0) == mount_number
        A_eq = []
        b_eq = []

        for _, row in self.cost_mount_info.iterrows():
            site = row["拠点"]
            machine = row["設備"]
            mount_number = row["保有台数"]

            mask = [
                (row_c["拠点前"] == site and row_c["変更前"] == machine and row_c["コスト"] > 0)
                for _, row_c in self.cost_info.iterrows()
            ]
            if any(mask):
                A_row = [1 if m else 0 for m in mask]
                A_eq.append(A_row)
                b_eq.append(mount_number)

        self.A_eq = np.array(A_eq) if A_eq else None
        self.b_eq = np.array(b_eq) if b_eq else None

        # 不足設備制約: sum(x_ij for j where 変更後==target_machine and コスト>=10) <= short_number
        A_ub = []
        b_ub = []

        for _, row in self.short_machine.iterrows():
            target_machine = row["設備"]
            short_number = row["不足台数"]

            mask = [
                (row_c["変更後"] == target_machine and row_c["コスト"] >= 10)
                for _, row_c in self.cost_info.iterrows()
            ]
            if any(mask):
                A_row = [1 if m else 0 for m in mask]
                A_ub.append(A_row)
                b_ub.append(short_number)

        self.A_ub = np.array(A_ub) if A_ub else None
        self.b_ub = np.array(b_ub) if b_ub else None

        # 下限0
        self.bounds = [(0, None) for _ in range(self.n_vars)]

    def solve(self):
        self.build_problem()
        start_time = time.time()
        res = linprog(
            c=self.c,
            A_ub=self.A_ub, b_ub=self.b_ub,
            A_eq=self.A_eq, b_eq=self.b_eq,
            bounds=self.bounds,
            method='highs'
        )
        end_time = time.time()
        print(f"処理時間: {end_time - start_time:.2f}秒")
        if not res.success:
            print("Warning: 最適化に失敗しました:", res.message)
        
        self.solution = res.x
        return self.extract_results()

    def extract_results(self):
        records = []
        for i, val in enumerate(self.solution):
            if val <= 0:
                continue

            name = self.var_names[i]
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

            records.append([pre_site, pre_machine, post_site, post_machine, cost, int(np.round(val)), category])

        return records
