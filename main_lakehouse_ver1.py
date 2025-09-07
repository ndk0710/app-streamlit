from lib.mathmatical_optimization_fabric import Mathmatical
from lib.cost_mathmatical_optimization_fabric import Cost_Mathmatical
import pandas as pd
import copy
import math

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


def get_table_data(lakehouse_input_data):
    """レイクハウスのテーブルデータから数理最適化クラスのメンバ変数を取得する関数
    
    Parameters:
    lakehouse_input_data（pandas形式）:テーブルデータ
    
    Returns:
    list: メンバ変数であるH50x工程コード

    """
    #products
    lakehouse_tmp_data = lakehouse_input_data.copy()
    lakehouse_tmp_data['H50+工程コード'] = lakehouse_tmp_data['H50'] + '_' + lakehouse_tmp_data['工程コード']
    lakehouse_tmp_data.drop_duplicates(subset=['H50+工程コード'], inplace=True)    
    products = lakehouse_tmp_data['H50+工程コード'].tolist()
    
    #machines
    lakehouse_tmp_data = lakehouse_input_data.copy()
    lakehouse_tmp_data.drop_duplicates(subset=['設備コード'], inplace=True)    
    machines = lakehouse_tmp_data['設備コード'].tolist()

    #specs/mounts/number_of_units_owned_planing/number_of_units_discardec
    specs = [[0 for _ in range(int(len(machines)))] for _ in range(int(len(products)))]
    mounts = [0 for _ in range(int(len(products)))]
    number_of_units_owned_planing = [0 for _ in range(int(len(machines)))]
    number_of_units_discardec = [0 for _ in range(int(len(machines)))]

    lakehouse_tmp_data = lakehouse_input_data.copy()
    
    for _, row in lakehouse_tmp_data.iterrows():
        if row["設備コード"] in machines:
            specs[int(find_index(products, row['H50'] + '_' + row['工程コード']))][int(find_index(machines, row['設備コード']))] = row['設備生産能力']
            mounts[int(find_index(products, row['H50'] + '_' + row['工程コード']))] = row['能力指標']
            number_of_units_owned_planing[int(find_index(machines, row['設備コード']))] = row['投資台数']
            number_of_units_discardec[int(find_index(machines, row['設備コード']))] = row['廃棄台数']

    return products, machines, specs, mounts, number_of_units_owned_planing, number_of_units_discardec


def get_number_of_units_owned(lakehouse_input_data2, machines):
    # 保有台数を辞書化（設備コード → 保有台数）
    code_to_count = lakehouse_input_data2.set_index('設備コード')['保有台数'].to_dict()

    # 出力リスト（存在しないコードは 0 を埋める）
    number_of_units_owned = [code_to_count.get(code, 0) for code in machines]

    return number_of_units_owned


def update_shoratge_output(shoratge_output, cost_mount_info, target_period):
    """
    shoratge_output: list of lists
    cost_mount_info: pandas.DataFrame (columns=["拠点","設備","保有台数"])
    target_period: 更新対象の年月度 (str, e.g. "2025F1")
    """

    updated = []
    for row in shoratge_output:
        # 元データを壊さないようにコピーを作る
        new_row = row.copy()

        ym, site, machine = new_row[0], new_row[1], new_row[2]

        if ym == target_period:
            mount_number = cost_mount_info.loc[
                (cost_mount_info["拠点"] == site) & (cost_mount_info["設備"] == machine),
                "保有台数"
            ].iloc[0]

            if mount_number == 0:
                new_row[3] = 0
            else:
                new_row[3] = new_row[3] - mount_number

            new_row[5] = new_row[3] + new_row[4]

        updated.append(new_row)

    return updated

def update_lakehouse(lakehouse_datas, repurposing_modification):
    df = lakehouse_datas.copy()

    for row in repurposing_modification:
        pre_site, pre_machine, post_site, post_machine, _, value, _ = row

        # 転用先に加算
        df.loc[
            (df["拠点"] == post_site) & (df["設備コード"] == post_machine),
            "保有台数"
        ] += value

        # 転用元から減算
        df.loc[
            (df["拠点"] == pre_site) & (df["設備コード"] == pre_machine),
            "保有台数"
        ] -= value

    return df

def update_lakehouse_datas(lakehouse_datas2, location, machines, calc_machines_sums):
    """
    lakehouse_datas2 を、計算結果に基づいて更新する関数
    - 新規設備は追加
    - 保有台数が増加した場合は更新
    - 重複は保有台数が大きいものを優先
    """
    # 今回計算した拠点・設備の台数を DataFrame に変換
    df_update = pd.DataFrame({
        "拠点": location,
        "設備コード": machines,
        "保有台数": calc_machines_sums
    })

    # 拠点ごとの既存データとその他を分離
    df_target = lakehouse_datas2[lakehouse_datas2["拠点"] == location]
    df_others = lakehouse_datas2[lakehouse_datas2["拠点"] != location]

    def should_update(row):
        """更新条件: 新規設備 または 保有台数が既存より多い"""
        existing = df_target[df_target["設備コード"] == row["設備コード"]]
        if existing.empty:
            return True
        return row["保有台数"] > existing["保有台数"].values[0]

    # 更新対象のみ抽出
    df_update_filtered = df_update[df_update.apply(should_update, axis=1)]

    # データ結合（対象拠点のデータを更新）
    updated = pd.concat(
        [df_others, df_target, df_update_filtered],
        ignore_index=True
    )

    # 重複解消: 保有台数が大きいものを残す
    updated = (
        updated
        .sort_values("保有台数", ascending=False)
        .drop_duplicates(subset=["拠点", "設備コード"], keep="first")
        .reset_index(drop=True)
    )

    return updated

def create_cost_mount_info(shortage_output, ym):
    records = []
    for row in shortage_output:
        if row[0] != ym:
            continue
        # 不足分のみを正の保有台数に変換
        shortage = -row[3] if row[3] < 0 else 0
        records.append([row[1], row[2], shortage])
    return pd.DataFrame(records, columns=['拠点', '設備', '保有台数'])

if __name__ == '__main__':
    #加工したテーブルデータ読込み
    lakehouse_datas = pd.read_csv("./lakehouse/table_data_lakehouse3.csv", encoding='utf-8')
    lakehouse_datas2 = pd.read_csv("./lakehouse/table_data_lakehouse2.csv", encoding='utf-8')
    cost_info = pd.read_csv("./lakehouse/table_data_lakehouse_cost.csv", encoding='utf-8')

    shoratge_output = []
    arrangement_output = []

    #年月度
    target_ym = sorted(lakehouse_datas['年月度'].unique())

    for index, ym in enumerate(target_ym):
        # 拠点
        locations = sorted(lakehouse_datas[lakehouse_datas['年月度'] == ym]['拠点'].unique().tolist())
        
        #拠点間用の変数
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
            
            # 拠点間データ取りまとめ
            total_short_machines.extend([int(sum(row)) for row in result_matrix])
            total_machine.extend(machines)

            # レイクハウス用テーブル1（不足台数算出）
            shortage_records = [
                [ym, location, m, s + n, n, s, d]
                for m, s, n, d in zip(
                    machines,
                    shortage_machines,
                    number_of_units_owned_planing,
                    number_of_units_discardec
                )
            ]
            shoratge_output.extend(shortage_records)

            # レイクハウス用テーブル2（配台算出）
            arrangement_records = []
            for i, m in enumerate(machines):
                for j, p in enumerate(products):
                    h50, process = p.split('_')
                    count = result_matrix[i][j]
                    if count != 0:  # 0 は除外
                        arrangement_records.append([ym, location, m, h50, process, count])

            arrangement_output.extend(arrangement_records)

            # レイクハウス用テーブル3（保有台数更新）
            lakehouse_datas2 = update_lakehouse_datas(lakehouse_datas2, location, machines, calc_machines_sums)

        # 対象年月の投資台数合算を抽出
        investment_sums = [row[3] for row in shoratge_output if row[0] == ym]
        
        # 設備が余っている（マイナスが含まれる）かどうか
        has_surplus = any(value < 0 for value in investment_sums)

        if has_surplus:
            #コスト最適化処理
            total_short_mount_info = pd.DataFrame({"設備": total_machine, "不足台数": total_short_machines})
            cost_mount_info = create_cost_mount_info(shoratge_output, ym)

            #コスト数理最適化のインスタンス作成
            cost_math_matical = Cost_Mathmatical(cost_info, cost_mount_info, total_short_mount_info)
            
            #コスト数理最適化の制約条件
            cost_math_matical.constraint_excess_facilities_definition()
            cost_math_matical.constraint_lack_of_facilities_definition()

            #コスト数理最適化の制約条件のセット
            cost_math_matical.set()

            #コスト数理最適化の問題を解く(保有台数の更新含む)
            repurposing_modification = cost_math_matical.solve()

            # メンバ変数（保有台数）の更新
            shoratge_output = update_shoratge_output(shoratge_output, cost_math_matical.cost_mount_info, ym)
            lakehouse_datas2 = update_lakehouse(lakehouse_datas2, repurposing_modification)
            ooo=0

    
    check=0

    

       