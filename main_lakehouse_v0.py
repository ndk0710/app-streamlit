from lib.mathmatical_optimization_fabric import Mathmatical
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
        

if __name__ == '__main__':
    #加工したテーブルデータ読込み
    lakehouse_datas = pd.read_csv("./lakehouse/table_data_lakehouse.csv", encoding='utf-8')
    lakehouse_datas2 = pd.read_csv("./lakehouse/table_data_lakehouse2.csv", encoding='utf-8')
    #lakehouse_datas2 = pd.DataFrame(columns=["拠点", "設備コード", "保有台数"])

    seen_locations = set()  # 登場した拠点記録用

    shoratge_output = []

    #年月度
    target_ym = sorted(lakehouse_datas['年月度'].unique())

    for index, ym in enumerate(target_ym):
        # 拠点
        locations = sorted(lakehouse_datas[lakehouse_datas['年月度'] == ym]['拠点'].unique().tolist())
        
        for location in locations:
            # データ絞る
            lakehouse_input_data = lakehouse_datas[(lakehouse_datas['年月度'] == ym) & (lakehouse_datas['拠点'] == location)]
            
            # 初期化1（数理最適化）
            products, machines, specs, mounts, number_of_units_owned_planing, number_of_units_discardec = get_table_data(lakehouse_input_data)
            
            # 初期化2（数理最適化）
            #if location not in seen_locations:
            lakehouse_input_data2 = lakehouse_datas2[lakehouse_datas2['拠点'] == location]
            number_of_units_owned = get_number_of_units_owned(lakehouse_input_data2, machines)

            # 拠点を履歴追加
            #seen_locations.add(location)

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

                # 2次元リストの要素同士を足す
                result_matrix = [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(old_calc_2d, new_calc_2d)]

                # 必要な装置台数
                calc_machines_sums = [sum(row) for row in result_matrix]

                # 必要な装置台数
                calc_machines_sums = [math.ceil(num) for num in calc_machines_sums]

                #不足装置台数
                shortage_machines = [a-b for a,b in zip(calc_machines_sums, number_of_units_owned_copy)]

            else:
                # 必要な装置台数
                calc_machines_sums = [sum(row) for row in math_matical.matrix]

                # 必要な装置台数
                calc_machines_sums = [math.ceil(num) for num in calc_machines_sums]

                #不足装置台数
                shortage_machines = [a-b for a,b in zip(calc_machines_sums, number_of_units_owned)]
                

            # レイクハウス用のテーブル（不足台数算出）
            answer = [
                [ym, location, m, s+n, n, s, d]
                for m, s, n, d in zip(machines, shortage_machines, number_of_units_owned_planing, number_of_units_discardec)
            ]
            # データ格納
            shoratge_output.extend(answer)

            table = [
                [location, m, c]
                for m, c in zip(machines, calc_machines_sums)
            ]

            df_update = pd.DataFrame(table, columns=["拠点", "設備コード", "保有台数"])
            # 元データから対象拠点以外のデータを抽出
            df_filtered = lakehouse_datas2[~lakehouse_datas2["拠点"].isin([location])]
            # 更新データと結合（対象拠点は上書き）
            lakehouse_datas2 = pd.concat([df_filtered, df_update], ignore_index=True)
            """

            # 新規拠点の場合（追加）
            if location not in seen_locations:
                table_df = pd.DataFrame(table, columns=["拠点", "設備コード", "保有台数"])
                lakehouse_datas2 = pd.concat([lakehouse_datas2, table_df], ignore_index=True)
            else:
                df_update = pd.DataFrame(table, columns=["拠点", "設備コード", "保有台数"])
                # 元データから対象拠点以外のデータを抽出
                df_filtered = lakehouse_datas2[~lakehouse_datas2["拠点"].isin([location])]
                # 更新データと結合（対象拠点は上書き）
                lakehouse_datas2 = pd.concat([df_filtered, df_update], ignore_index=True)
            
            # 拠点を履歴追加
            seen_locations.add(location)"""




    
    check=0

    

       