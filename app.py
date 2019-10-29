# -*- coding: utf-8 -*-
import os
from flask import Flask, request, url_for, send_from_directory, render_template
from werkzeug import secure_filename
from pypinyin import lazy_pinyin
import prediction.nsfw_predict as nsfw
import prediction.filter as filter

import prediction.replac_background.image_semantic_segmentation as iss

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()+"/imgs"
app.config['PREDS_FOLDER']=os.getcwd()+'/pred'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>图片上传</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=file>
         <input type=submit value=上传>
    </form>
    '''


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/preds/<filename>')
def preds_file(filename):
    return send_from_directory(app.config['PREDS_FOLDER'],
                               filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(''.join(lazy_pinyin(file.filename)))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            result = nsfw.predict(os.path.join(app.config['UPLOAD_FOLDER'], filename),os.getcwd()+'/models/1547856517')
            print(result)
            return html + '<br><img src=' + file_url + '><p>预测结果如下：' + str(result) + '</p>'
    return html


@app.route('/filter/', methods=['POST', 'GET'])
def txtFilter():
    if request.method == 'GET':
        return render_template('filter.html')
    else:
        txt = request.form.get('txt')
        gfw = filter.DFAFilter()
        gfw.parse(os.getcwd() + "/filter/keywords")
        txted=gfw.filter(txt, "*")
        return 'txt={}'.format(txted)

#orc识别 https://github.com/breezedeus/cnocr/blob/master/README_cn.md
@app.route('/ocr/txt/', methods=['GET', 'POST'])
def txtOcr():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(''.join(lazy_pinyin(file.filename)))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            import mxnet as mx
            from cnocr import CnOcr
            ocr = CnOcr()
            img_fp = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img = mx.image.imread(img_fp, 1)
            res = ocr.ocr(img)
            print("Predicted Chars:", res)
            return html + '<br><img src=' + file_url + '><p>预测结果如下：' +  str(res) + '</p>'
    return html

@app.route('/predict/replacebg/', methods=['GET', 'POST'])
def replaceBg():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(''.join(lazy_pinyin(file.filename)))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_url = url_for('uploaded_file', filename=filename)
            resultPic = iss.run_inference_on_image(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('resultpci path:'+resultPic)
            result_url = url_for('preds_file', filename=os.path.split(resultPic)[1])
            print('====='+result_url)
            return html + '<br><img src=' + file_url + '><p>处理结果如下：<img src=' + result_url + '>' +'</p>'
    return html

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')