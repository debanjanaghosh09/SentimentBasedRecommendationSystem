from flask import Flask,render_template,request,jsonify
from model import *

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/predict",methods=['POST'])
def predict():
    userName = request.form['username'];
    items = get_recommendations(userName)
    return render_template("index.html", column_names=items.columns.values, row_data=list(items.values.tolist()), zip=zip)

if __name__=="__main__":
    app.run(debug=True)


# In[ ]:




