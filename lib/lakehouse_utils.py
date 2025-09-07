import pandas as pd
from typing import List, Any


def find_index(input_list: List[Any], target: Any) -> int:
    """リストの中から該当する要素のインデックスを返す

    Args:
        input_list (List[Any]): 検索対象のリスト
        target (Any): 検索したい要素

    Returns:
        int: 該当する要素のインデックス。見つからなければ -1
    """
    try:
        return input_list.index(target)
    except ValueError:
        return -1


def get_table_data(lakehouse_input_data: pd.DataFrame):
    """レイクハウスのテーブルデータから数理最適化クラスのメンバ変数を生成

    Args:
        lakehouse_input_data (pd.DataFrame): テーブルデータ

    Returns:
        tuple: (products, machines, specs, mounts, number_of_units_owned_planing, number_of_units_discarded)
    """
    # products
    df_tmp = lakehouse_input_data.copy()
    df_tmp['H50+工程コード'] = df_tmp['H50'] + '_' + df_tmp['工程コード']
    products = df_tmp.drop_duplicates(subset=['H50+工程コード'])['H50+工程コード'].tolist()

    # machines
    machines = lakehouse_input_data.drop_duplicates(subset=['設備コード'])['設備コード'].tolist()

    # 初期化
    specs = [[0 for _ in machines] for _ in products]
    mounts = [0 for _ in products]
    number_of_units_owned_planing = [0 for _ in machines]
    number_of_units_discarded = [0 for _ in machines]

    for _, row in lakehouse_input_data.iterrows():
        product_idx = find_index(products, row['H50'] + '_' + row['工程コード'])
        machine_idx = find_index(machines, row['設備コード'])
        if machine_idx != -1 and product_idx != -1:
            specs[product_idx][machine_idx] = row['設備生産能力']
            mounts[product_idx] = row['能力指標']
            number_of_units_owned_planing[machine_idx] = row['投資台数']
            number_of_units_discarded[machine_idx] = row['廃棄台数']

    return products, machines, specs, mounts, number_of_units_owned_planing, number_of_units_discarded


def get_number_of_units_owned(lakehouse_input_data2: pd.DataFrame, machines: List[str]) -> List[int]:
    """保有台数を設備コードごとのリストに変換

    Args:
        lakehouse_input_data2 (pd.DataFrame): 保有台数テーブル
        machines (List[str]): 対象設備コード

    Returns:
        List[int]: 保有台数リスト
    """
    code_to_count = lakehouse_input_data2.set_index('設備コード')['保有台数'].to_dict()
    return [code_to_count.get(code, 0) for code in machines]


def update_lakehouse_datas(
    lakehouse_datas2: pd.DataFrame,
    location: str,
    machines: List[str],
    calc_machines_sums: List[int]
) -> pd.DataFrame:
    """lakehouse_datas2 を更新

    - 新規設備は追加
    - 保有台数が増加した場合は更新
    - 重複は保有台数が大きいものを優先

    Args:
        lakehouse_datas2 (pd.DataFrame): 更新対象データ
        location (str): 拠点
        machines (List[str]): 設備コード一覧
        calc_machines_sums (List[int]): 計算された保有台数

    Returns:
        pd.DataFrame: 更新済み lakehouse_datas2
    """
    df_update = pd.DataFrame({
        "拠点": location,
        "設備コード": machines,
        "保有台数": calc_machines_sums
    })

    df_target = lakehouse_datas2[lakehouse_datas2["拠点"] == location]
    df_others = lakehouse_datas2[lakehouse_datas2["拠点"] != location]

    def should_update(row):
        existing = df_target[df_target["設備コード"] == row["設備コード"]]
        if existing.empty:
            return True
        return row["保有台数"] > existing["保有台数"].values[0]

    df_update_filtered = df_update[df_update.apply(should_update, axis=1)]

    updated = pd.concat([df_others, df_target, df_update_filtered], ignore_index=True)
    updated = (
        updated
        .sort_values("保有台数", ascending=False)
        .drop_duplicates(subset=["拠点", "設備コード"], keep="first")
        .reset_index(drop=True)
    )
    return updated


def generate_lakehouse_outputs(ym, location, machines, products, result_matrix,
                               shortage_machines, number_of_units_owned_planing,
                               number_of_units_discardec):
    """
    arrangement_output と shortage_output を生成する関数
    """
    # 配台情報
    arrangement_records = []
    for i, m in enumerate(machines):
        for j, p in enumerate(products):
            h50, process = p.split('_')
            count = result_matrix[i][j]
            if count != 0:
                arrangement_records.append([ym, location, m, h50, process, count])

    # 不足台数情報
    shortage_records = [
        [ym, location, m, s + n, n, s, d]
        for m, s, n, d in zip(
            machines,
            shortage_machines,
            number_of_units_owned_planing,
            number_of_units_discardec
        )
    ]

    return arrangement_records, shortage_records