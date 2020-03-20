import datetime
import os
import json
import glob

from api_module import (
    ALLOWED_EXTENSIONS,
    UPLOAD_FOLDER
)

from flask import (
    session,
    make_response
)


def allow_filename(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def output_json(data, code, cookie = None,headers=None):
    """Makes a Flask response with a JSON encoded body"""
    if isinstance(data,str):
        status = '' 
        if code >= 200 and code <= 202 :
            status = 'ok'
        else :
            status = 'no'
        data = {'body':json.dumps(data),'status': status}
    
    resp = make_response(json.dumps(data), code)
    if cookie:
        resp.set_cookie('token', cookie)
    # resp.headers.extend(headers or {})
    return resp


def make_rnd_filepath(basename='mylogfile', ext='csv'):    
    try:
        result = session.get(basename, None)
        if result:
            return result
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        filename = "".join([basename,'_', suffix, '.', ext])
        return os.path.join(UPLOAD_FOLDER, filename)
    except Exception as ex:
        raise(ex)
        return False


def get_filename(path):
    try:
        head,tail = os.path.split(path)
        return tail
    except Exception as ex:
        raise(ex)
        return False


def remove_file(filename):
    try :
        if filename:
            os.remove(filename)
            return True
        return False
    except Exception as ex:
        return False


def clean_folder():
    files = glob.glob('uploads/*')
    for f in files:
        os.remove(f)


def getDataFromRequest(request):
    req_data = ''
    if request.data:
        req_data = json.loads(request.data)			
    
    if not req_data:
        req_data = request.args				
    return req_data


def get_LanguageList(filepath):
    lang_array = [
        {'name': 'English', 'code': 'en'},
    ]
    try:
        with open(filepath, 'r') as file:
            languages = file.read();
            lang_array = json.loads(languages)

    except Exception as ex:
        print(ex)

    finally:
        return lang_array


def readBlackListFromFile(filepath, bLists):
    words = bLists

    try:
        with open(filepath, 'r', encoding="utf-8") as file:
            str = file.read()
            words = words + str.split('\n')

    except FileNotFoundError as Ex:
        print('file not found: in', filepath)

    except Exception as Ex:
        raise(Ex)

    finally:
        return words
