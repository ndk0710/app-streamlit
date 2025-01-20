import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

def helloworld():
    st.header("Hello World!")

if __name__ == "__main__":
    #helloworld()

    #タイトル入力
    st.title("Streamlit 超入門")

    #テキスト表示
    st.write("DataFrame")

    #データ設定
    #df = pd.DataFrame({
    #    "1列目": [1, 2, 3, 4],
    #    "2列目": [10, 20, 30, 40]
    #})

    #テーブル表示
    #st.dataframe(df.style.highlight_max(axis=0))

    #テキスト表示
    #"""
    ## 章
    ## 節
    ### 項

    #```python
    #import streamlit as st
    #import pandas as pd
    #```
    #"""

    """
    #グラフ描画
    df = pd.DataFrame(
        np.random.rand(20, 3),
        columns=['a', 'b', 'c']
    )

    st.line_chart(df)
    st.area_chart(df)
    st.bar_chart(df)
    """
    """
    #マッピング
    df = pd.DataFrame(
        np.random.rand(100, 2)/[50, 50] + [35.69, 139.70],
        columns=['lat', 'lon']
    )

    st.map(df)
    """
    """
    #チェックボックス
    if st.checkbox('Show Image'):
        #画像表示
        img = Image.open('パウパト.jpg')
        st.image(img, caption='パウパト', use_column_width=True)
    """

    """option = st.selectbox(
        'あなたの好きな数字は何ですか',
        list(range(1,11))
    )

    st.write(f'あなたの好きな数字は{option}です')"""
    """
    #テキストボックス
    text = st.text_input('あなたの趣味を教えてください。')

    st.write(f'わたしの趣味は{text}です')
    """

    #スライダー
    condition = st.slider('あなたの今の調子は？', 0, 100, 50)
    st.write(f"コンディション：{condition}")