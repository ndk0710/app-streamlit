import pandas as pd
from typing import List


def create_cost_mount_info(shortage_output: List[List], ym: str) -> pd.DataFrame:
    """不足台数からコスト最適化用のDataFrameを作成

    Args:
        shortage_output (List[List]): 不足台数リスト
        ym (str): 対象年月

    Returns:
        pd.DataFrame: DataFrame(columns=['拠点', '設備', '保有台数'])
    """
    records = []
    for row in shortage_output:
        if row[0] != ym:
            continue
        shortage = -row[3] if row[3] < 0 else 0
        records.append([row[1], row[2], shortage])
    return pd.DataFrame(records, columns=['拠点', '設備', '保有台数'])


def update_shortage_output(
    shortage_output: List[List],
    cost_mount_info: pd.DataFrame,
    target_period: str
) -> List[List]:
    """不足台数リストを保有台数で更新

    Args:
        shortage_output (List[List]): 不足台数リスト
        cost_mount_info (pd.DataFrame): 保有台数情報
        target_period (str): 年月度

    Returns:
        List[List]: 更新済み不足台数リスト
    """
    updated = []
    for row in shortage_output:
        new_row = row.copy()
        ym, site, machine = new_row[0], new_row[1], new_row[2]

        if ym == target_period:
            mount_number = cost_mount_info.loc[
                (cost_mount_info["拠点"] == site) & (cost_mount_info["設備"] == machine),
                "保有台数"
            ].iloc[0]

            new_row[3] = new_row[3] - mount_number if mount_number else 0
            new_row[5] = new_row[3] + new_row[4]

        updated.append(new_row)
    return updated


def update_lakehouse(lakehouse_datas, repurposing_modification):
    """
    転用結果を lakehouse_datas に反映する
    """
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
