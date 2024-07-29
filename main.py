from flask import Flask, render_template, request, redirect, url_for,jsonify
from predictv2 import predict_video
app = Flask(__name__)

@app.route("/")
def homePage():
    return render_template("homePage.html")

@app.route("/howworking")
def howworkingPage():
    return render_template("howWorking.html")

@app.route("/howdo")
def howdoPage():
    return render_template("howdo.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part in the request!'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file!'
    
    file.save('static/uploads/' + file.filename)
    predict_result = predict_video("./static/uploads/"+file.filename)
    return predict_result



if __name__ == "__main__":
    app.run(debug=True)