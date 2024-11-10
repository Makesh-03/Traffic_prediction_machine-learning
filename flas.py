from flask import Flask, render_template, request
import os
import decisiontree as d  # Assuming these files contain your prediction functions
import Randomforest as r
import SVM as s

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    predictionR = None
    predictionS = None
    
    if request.method == 'POST':
        if 'decision_tree_submit' in request.form:
            prediction = d.predictD()
            print(prediction)
        elif 'random_forest_submit' in request.form:
            predictionR = r.predictR()
            print(predictionR)
        elif 'svm_submit' in request.form:
            predictionS = s.predictS()
            print(predictionS)

    return render_template('index.html', prediction=prediction, predictionR=predictionR, predictionS=predictionS)

'''if __name__ == "__main__":
    # Ensure the static directory exists
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)

    app.run(debug=True)
    #app.run(host='127.0.0.1', port=5000, debug=True)
'''
if __name__ == "__main__":
    app.run(debug=True)
  