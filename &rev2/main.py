from lib.mathmatical_optimization import Mathmatical
import pandas as pd
import numpy as np

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

def get_products(specs_info):
    """メンバ変数である品種を取得する関数
    
    Parameters:
    specs(pandas):設備生産能力に関連する情報
    
    Returns:
    list: メンバ変数である品種
    """
    specs = specs_info.copy()

    #カラムを作成
    specs['H50+工程'] = specs['H50'] + '_' + specs['工程']
    
    #重複を削除
    specs.drop_duplicates(subset=['H50+工程'], inplace=True)    
    
    return specs['H50+工程'].tolist()

def get_machines(specs_info):
    """メンバ変数である設備を取得する関数
    
    Parameters:
    specs(pandas):保有設備に関連する情報
    
    Returns:
    list: メンバ変数である設備
    """
    specs = specs_info.copy()
    
    #重複を削除
    specs.drop_duplicates(subset=['設備'], inplace=True)    
    

    return specs['設備'].tolist()

def get_specs(products, machines, specs_info):
    """メンバ変数である設備生産能力を取得する関数
    
    Parameters:
    products(list):品種に関連する情報
    machines(list):設備に関連する情報
    
    Returns:
    list: メンバ変数である設備生産能力
    """
    specs = [[0 for _ in range(int(len(machines)))] for _ in range(int(len(products)))]

    for _, row in specs_info.iterrows():
        if row['設備'] in machines:
            specs[int(find_index(products, row['H50'] + '_' + row['工程']))][int(find_index(machines, row['設備']))] = row['設備生産能力']

    return specs
    
def get_mounts(mounts_info):
    """メンバ変数である生産量を取得する関数
    
    Parameters:
    mounts_info(list):生産量に関連する情報
    
    Returns:
    list: メンバ変数である生産量
    """
    mounts = mounts_info['生産量'].tolist()

    return mounts

def get_number_of_units_owned(units_owned_info):
    """メンバ変数である生産量を取得する関数
    
    Parameters:
    mounts_info(list):生産量に関連する情報
    
    Returns:
    list: メンバ変数である生産量
    """
    number_of_units_owned = units_owned_info['保有台数'].tolist()

    return number_of_units_owned

def pad_with_zeros(smaller_list, larger_list):
    # 大きいリストの行数と列数を取得
    max_rows = len(larger_list)
    max_cols = max(len(row) for row in larger_list)
    # 行数を大きいリストと同じにする
    while len(smaller_list) < max_rows:
        smaller_list.append([0] * max_cols)
        # 各行の列数を大きいリストと同じにする
        for row in smaller_list:
            while len(row) < max_cols:
                row.append(0)
        return smaller_list

if __name__ == '__main__':
    #データ読込み
    specs_info = pd.read_csv("./csv/設備生産能力_26F1.csv", encoding='utf-8')
    mounts_info = pd.read_csv("./csv/生産量_25F1.csv", encoding='utf-8')
    units_owned_info = pd.read_csv("./csv/保有設備DB_24F2.csv", encoding='utf-8')

    #メンバ変数（品種/設備/設備生産能力）
    products = get_products(specs_info)
    machines = get_machines(units_owned_info)
    specs = get_specs(products, machines, specs_info)

    #メンバ変数（生産量）
    mounts = get_mounts(mounts_info)

    #メンバ変数（保有台数）
    number_of_units_owned = get_number_of_units_owned(units_owned_info)

    #数理最適化のインスタンス作成
    math_matical = Mathmatical(products, machines, specs, mounts, number_of_units_owned)

    #数理最適化の制約条件
    math_matical.constraint_mount_definition()
    math_matical.constraint_units_owned_definition()

    #数理最適化の問題を解く
    math_matical.solve()

    # 保有台数比較
    flag = math_matical.compare_machines()

    if flag == False:
        # 残生産量を計算
        mounts = math_matical.calc_remaining_mount()

        number_of_units_owned = constraints = [0]*math_matical.machines_nums

        old_calc_2d = math_matical.owned_matrix
        
        # オブジェクトの削除
        del math_matical

        #メンバ変数（設備/設備生産能力）
        machines.extend(['SCF-28（DLW32共用）'])
        specs = get_specs(products, machines, specs_info)

        #数理最適化のインスタンス作成
        math_matical = Mathmatical(products, machines, specs, mounts, number_of_units_owned)

        #数理最適化の制約条件
        math_matical.constraint_mount_definition()
        math_matical.constraint_specs_definition(['YCWM-11*（8連）イメセン有'])
        
        #数理最適化の問題を解く
        math_matical.solve()

        new_calc_2d = math_matical.matrix

        # サイズが一致しない場合、修正
        if len(old_calc_2d) != len(new_calc_2d):
            old_calc_2d = pad_with_zeros(old_calc_2d, new_calc_2d)

        # 2次元リストの要素同士を足す
        result_matrix = [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(old_calc_2d, new_calc_2d)]

        #データをXLSファイルとして出力
        df = pd.DataFrame(np.array(result_matrix), index=math_matical.row_names, columns=math_matical.column_names)
        df.to_excel('output2.xlsx', index=True)

        kk=0

    #入力
    check=0
    