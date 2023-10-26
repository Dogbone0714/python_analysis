from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/')
def index():
    # 從網路上獲取CSV資料並進行迴歸分析
    url = 'https://od.cdc.gov.tw/eic/covid19/covid19_global_stats.csv'
    data = pd.read_csv(url)
    
    # 提取所需的資料欄位
    x = data['Confirmed']
    y = data['Deaths']
    
    # 將資料重新塑形以符合迴歸模型的要求
    x = x.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)
    
    # 建立並訓練線性迴歸模型
    model = LinearRegression()
    model.fit(x, y)
    
    # 獲取迴歸模型的係數和截距
    coef = round(model.coef_[0][0], 2)
    intercept = round(model.intercept_[0], 2)
    
    # 將結果傳遞給網頁模板
    return render_template('index.html', coef=coef, intercept=intercept)

if __name__ == '__main__':
    app.run()