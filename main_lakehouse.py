import pandas as pd
import math
from lib.mathmatical_optimization_fabric import Mathmatical
from lib.cost_mathmatical_optimization_fabric import Cost_Mathmatical
#from lib.cost_mathmatical_optimization_scipy import Cost_Mathmatical_Scipy
from lib.lakehouse_utils import (
    get_table_data,
    get_number_of_units_owned,
    update_lakehouse_datas,
    generate_lakehouse_outputs
)
from lib.optimization_utils import (
    create_cost_mount_info,
    update_shortage_output,
    update_lakehouse
)


def main():
    # データ読込
    lakehouse_datas = pd.read_csv("./lakehouse/table_data_lakehouse3.csv", encoding="utf-8")
    lakehouse_datas2 = pd.read_csv("./lakehouse/table_data_lakehouse2.csv", encoding="utf-8")
    cost_info = pd.read_csv("./lakehouse/table_data_lakehouse_cost.csv", encoding="utf-8")

    shortage_output = []
    arrangement_output = []
    repurposing_modification_output = []

    # 年月度ごとに処理
    for ym in sorted(lakehouse_datas["年月度"].unique()):
        locations = sorted(lakehouse_datas[lakehouse_datas["年月度"] == ym]["拠点"].unique().tolist())

        total_short_machines, total_machine = [], []

        for location in locations:
            # データ絞る
            lakehouse_input_data = lakehouse_datas[(lakehouse_datas['年月度'] == ym) & (lakehouse_datas['拠点'] == location)]
            
            # 初期化1（数理最適化）
            products, machines, specs, mounts, number_of_units_owned_planing, number_of_units_discardec = get_table_data(lakehouse_input_data)
            
            # 初期化2（数理最適化）
            number_of_units_owned = get_number_of_units_owned(lakehouse_datas2[lakehouse_datas2['拠点'] == location], machines)

            # 意思入れの反映
            number_of_units_owned = [a + b - c for a, b, c in zip(number_of_units_owned, number_of_units_owned_planing, number_of_units_discardec)]

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
                result_matrix = [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(old_calc_2d, new_calc_2d)]
                calc_machines_sums = [sum(row) for row in result_matrix]
                calc_machines_sums = [math.ceil(num) for num in calc_machines_sums]
                shortage_machines = [a-b for a,b in zip(calc_machines_sums, number_of_units_owned_copy)]
            else:
                result_matrix = math_matical.matrix
                calc_machines_sums = [sum(row) for row in math_matical.matrix]
                calc_machines_sums = [math.ceil(num) for num in calc_machines_sums]
                shortage_machines = [a-b for a,b in zip(calc_machines_sums, number_of_units_owned)]

            arr_records, shor_records = generate_lakehouse_outputs(
                ym, location, machines, products,
                result_matrix, shortage_machines,
                number_of_units_owned_planing,
                number_of_units_discardec
            )

            arrangement_output.extend(arr_records)
            shortage_output.extend(shor_records)

            # レイクハウス用テーブル（保有台数更新）
            lakehouse_datas2 = update_lakehouse_datas(lakehouse_datas2, location, machines, calc_machines_sums)

        # 投資台数合算
        investment_sums = [row[3] for row in shortage_output if row[0] == ym]
        has_surplus = any(v < 0 for v in investment_sums)

        if has_surplus:
            total_short_mount_info = pd.DataFrame({"設備": total_machine, "不足台数": total_short_machines})
            cost_mount_info = create_cost_mount_info(shortage_output, ym)

            cost_math_matical = Cost_Mathmatical(cost_info, cost_mount_info, total_short_mount_info)
            cost_math_matical.constraint_excess_facilities_definition()
            cost_math_matical.constraint_lack_of_facilities_definition()
            cost_math_matical.set()

            repurposing_modification = cost_math_matical.solve()
            #cost_math_scipy = Cost_Mathmatical_Scipy(cost_info, cost_mount_info, total_short_mount_info)
            #repurposing_modification = cost_math_scipy.solve()
            #repurposing_modification_output.extend(repurposing_modification)

            shortage_output = update_shortage_output(shortage_output, cost_math_matical.cost_mount_info, ym)
            #shortage_output = update_shortage_output(shortage_output, cost_math_scipy.cost_mount_info, ym)
            lakehouse_datas2 = update_lakehouse(lakehouse_datas2, repurposing_modification)
    oo=0


if __name__ == "__main__":
    main()
