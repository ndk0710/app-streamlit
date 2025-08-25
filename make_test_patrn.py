import pandas as pd
import numpy as np
import copy
import math
import random

if __name__ == '__main__':

    random.seed(42)

    MACHINE_NUM = 10
    PRODUCT_NUM = 2

    #非ゼロ要素の最大数を設定
    max_non_zeros_elements = 3

    def generate_random_list(rows, cols):
        result = []
        for _ in range(rows):
            # 非ゼロ要素の個数を決定
            num_non_zeros = random.randint(1, max_non_zeros_elements)

            # ランダムな非ゼロ要素を生成
            non_zero_values = [random.randint(0, 2000) for _ in range(num_non_zeros)]

            # ゼロ埋めし最終的な配列サイズとする
            row_with_zeros = non_zero_values + [0] * (cols - num_non_zeros)

            # 行内でシャッフルしてランダムに配置
            random.shuffle(row_with_zeros)

            result.append(row_with_zeros)
        
        return result
    
    #二次元リスト関数を呼び出し
    randomized_list = generate_random_list(MACHINE_NUM, PRODUCT_NUM)
    one_d_list = [item for sublist in randomized_list for item in sublist]

    machine_list = [f'machine{i}' for i in range(1, MACHINE_NUM+1)]
    product_list = [f'product{i}' for i in range(1, PRODUCT_NUM+1)]

    sheet1 = []
    sheet1_columns = ['H50', '工程', '設備', '設備生産能力']

    # 設備生産能力シート作成
    for i, product in enumerate(product_list):
        for j, machine in enumerate(machine_list):
            spec = one_d_list[i*MACHINE_NUM+j]
            sheet1.append([product, 'テスト工程', machine, spec])
    
    # DataFrameの作成
    df1 = pd.DataFrame(sheet1, columns=sheet1_columns)

    # CSVファイルとして出力
    output_file = './test_patran/test0_設備生産能力.csv'
    df1.to_csv(output_file, index=False)

    sheet2 = []
    sheet2_columns = ['H50+工程','生産量']

    # 生産量シート作成
    for product in product_list:
        mount = random.randint(500,2000)
        sheet2.append([f'{product}_テスト工程', mount])
    
    # DataFrameの作成
    df2 = pd.DataFrame(sheet2, columns=sheet2_columns)
    
    # CSVファイルとして出力
    output_file = './test_patran/test0_生産量.csv'
    df2.to_csv(output_file, index=False)

    sheet3 = []
    sheet3_columns = ['設備','保有台数']

    # 生産量シート作成
    for machine in machine_list:
        number = random.randint(1,5)
        sheet3.append([machine, number])
    
    # DataFrameの作成
    df3 = pd.DataFrame(sheet3, columns=sheet3_columns)
    
    # CSVファイルとして出力
    output_file = './test_patran/test0_保有設備.csv'
    df3.to_csv(output_file, index=False)
