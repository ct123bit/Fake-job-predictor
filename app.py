from flask import Flask, render_template,request
from joblib import load
pipeline = load("model.joblib")
app= Flask(__name__)

@app.route('/',methods=['Get','POST'])
def hello():
    if request.method == 'POST':
        text1 = request.form['text1']
        text2 = request.form['text2']
        text3 = request.form['text3']
        text4 = request.form['text4']
        text5 = request.form['text5']
        text6 = request.form['text6']
        text7 = request.form['text7']
        text=" ".join([text1,text2,text3,text4,text5,text6,text7])
        text=[text]
        prediction = pipeline.predict(text)
        if(prediction==0):
            mk="Real Job Posting"
        else:
            mk="Fraudlent Job Posting"
        return render_template('index.html',prediction=mk)
        
    return render_template('index.html')

# @app.route('/sub',methods=['POST'])
# def submit():
#     # getting stuf from html to python
#     if request.method=="POST":
#         name=request.form["username"]
#     # returning to html from python
#     return render_template('sub.html',n=name)

if __name__ == '__main__':
    app.run(debug=True)

