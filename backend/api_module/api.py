import functools

from flask import (
    Blueprint,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
    make_response,
    send_file
)

from flask_restful import (
    Resource,
    Api,
    abort
)

from werkzeug.utils import secure_filename
from flasgger import Swagger
from io import BytesIO

import api_module
from api_module import utils
from api_module import functions as Func

import joblib
import pandas
import time
import zipfile
import os
import glob
import json
import numpy as np

from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    jwt_required,
    jwt_refresh_token_required,
    get_jwt_identity, get_raw_jwt
)
import xlsxwriter


bp = Blueprint('api', __name__, url_prefix='/api')
api = Api(bp)


class FileUpload(Resource):

    @jwt_refresh_token_required
    def post(self):
        """
                return 
                        status - upload status
                        filename - generate data from upload csv files.
                        token - jwt token per files.
                ---
                parameters:
                - name: file
                        in: formData
                        type: file
                        required: true
        """
        try:

            if 'file' not in request.files:
                return utils.output_json('No file part', 422)

            file = request.files['file']
            if file.filename == '':
                return utils.output_json('No selected file', 422)

            if file and utils.allow_filename(file.filename):
                # session.clear()
                # read file correctly
                upsvFilePath = utils.make_rnd_filepath()
                file.save(upsvFilePath)

                # save file data
                parser_f = pandas.read_csv(upsvFilePath)
                dataFileName = utils.make_rnd_filepath('datafilename', 'pkl')
                joblib.dump(parser_f, dataFileName)
                session['datafilename'] = dataFileName
                return utils.output_json('upload success', 200)
            return utils.output_json('Not Support file type', 415)
        except Exception as ex:
            raise(ex)
            return utils.output_json('Intenal Server Error', 500)


class PredictModel(Resource):

    @jwt_refresh_token_required
    def post(self):
        """
        Return predicted classes for and create data to visualize the results,
        and output an excel file if requested with class labels in the original dataframe
        ---
        parameters:
        - name: perplexity
                in: query
                type: int
                required: false
        - name: learningRate
                in: query
                type: number
                required: true
        - name: numberOfClasses
                in: query
                type: number
                required: true
        """

        try:
            reqData = utils.getDataFromRequest(request)

            modelFilePath = Func.getModelPath('clf_rf.pkl')
            if not modelFilePath:
                raise('model is not found')

            clf_model = joblib.load(modelFilePath)
            dataFile = session.get('datafilename', None)
            if dataFile == None:
                return utils.output_json('No data correct', 422)
            data = joblib.load(dataFile)

            probabilities = clf_model.predict_proba(data)
            probabilities = pandas.DataFrame(clf_model.predict_proba(data), 
                                             columns=['Probability of class 0',
                                                      'Probability of class 1'])
            
            data['prediction'] = clf_model.predict(data)
            new_data = pandas.concat([data, probabilities], axis=1)

            classes_number = len(data['prediction'].unique())

            clf_df = pandas.concat([new_data[new_data.columns[0:2]], new_data[new_data.columns[-3]], new_data[new_data.columns[-1]]], axis=1)

            clf_df['prediction'] = np.random.randint(1, 100, clf_df.shape[0])

            jsonObj = Func.plot_json(clf_df)
            # print(jsonObj)
            class_data = utils.make_rnd_filepath('clf_data', 'pkl')
            # print(class_data)
            joblib.dump(new_data, class_data)
            session['clf_data'] = class_data
            color = Func.getLinearColor(classes_number)
            
            # this is just to check if it works. Will remove
            return utils.output_json({'status': 'success', 'color': json.dumps(color), 'body': json.dumps(jsonObj), 'classCount': classes_number}, 200)
        except Exception as Ex:
            raise(Ex)
            return utils.output_json('Internal Server Error', 500)



class FileDownload(Resource):

    @jwt_refresh_token_required
    def get(self):
        try:
            """
            Download dataframe for users
            """
            class_data = session.get('clf_data', None)
            if class_data is None:
                return utils.output_json('No data file', 422)

            df = joblib.load(class_data)
            # create and output csv files
            output = BytesIO()
            writer = pandas.ExcelWriter(output, engine='xlsxwriter')
            df.to_excel(writer, sheet_name='Clusters',
                        encoding='utf-8', index=False)
            writer.save()

            # zipfile an dowload results locally
            memory_file = BytesIO()
            with zipfile.ZipFile(memory_file, 'w') as zf:
                names = ['loan_clf.xlsx']
                files = [output]
                for i in range(len(files)):
                    data = zipfile.ZipInfo(names[i])
                    data.date_time = time.localtime(time.time())[:6]
                    data.compress_type = zipfile.ZIP_DEFLATED
                    zf.writestr(data, files[i].getvalue())
            memory_file.seek(0)

            # remove pickle files from directory
            utils.remove_file(session.get('datafilename', None))
            utils.remove_file(session.get('class_data', None))
            utils.remove_file(session.get('mylogfile', None))
            session.pop('datafilename', None)
            session.pop('class_data', None)
            session.pop('mylogfile', None)

            response = make_response(send_file(
                memory_file, attachment_filename='cluster_output.zip', as_attachment=True))
            response.headers['Content-Disposition'] = 'attachment;filename=cluster_output.zip'
            return response
        except Exception as ex:
            raise(ex)
            return utils.output_json('Intenal Server Error', 500)


class GenToken(Resource):

    def get(self):
        return create_refresh_token(identity=request.remote_addr)

    def post(self):
        language_path = api_module.RESOURCES_FOLDER + 'languages.json'
        return utils.output_json({'token': create_refresh_token(identity=request.remote_addr), 'languages': utils.get_LanguageList(language_path)}, 200)


class DownloadImage(Resource):

    @jwt_refresh_token_required
    def get(self):
        imgFile = session.get('imgFilePath', None)
        try:
            if not imgFile:
                response = make_response(
                    send_file(imgFile, attachment_filename='result.png', as_attachment=True))
                response.headers['Content-Disposition'] = 'attachment;filename=result.png'
                return response
            raise(ValueError('imageFileNotFound'))
        except Exception:
            return utils.output_json('Intenal Server Error', 500)


api.add_resource(FileUpload, '/upload')
api.add_resource(PredictModel, '/predict')
api.add_resource(FileDownload, '/download')
api.add_resource(GenToken, '/')
