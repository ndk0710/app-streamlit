import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time

def helloworld():
    st.header("Hello World!")

if __name__ == "__main__":
    #タイトル入力
    st.title("Streamlit 超入門")

    #プログレスバーの表示
    st.write('プログレスバーの表示')
    st.write('Start!!')

    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        latest_iteration.text(f'Iteration{i+1}')
        bar.progress(i+1)
        time.sleep(0.1)
    
    st.write('Done!!')
    
    #2カラム
    left_column, right_column = st.columns(2)
    button = left_column.button('右カラムに文字を表示')
    if button:
        right_column.write('ここは右カラム')
    
    #expander
    expander1 = st.expander('問い合わせ1')
    expander1.write('問い合わせ1の回答')


    #テキストボックス
    #text = st.text_input('あなたの趣味を教えてください。')
    #st.write(f'わたしの趣味は{text}です')

    #スライダー
    #condition = st.slider('あなたの今の調子は？', 0, 100, 50)
    #st.write(f"コンディション：{condition}")