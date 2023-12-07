#from werkzeug.utils import secure_filename
from enlace import enlace
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

#from rvcgui import vc_single, get_output_path

app = Flask(__name__)
cors = CORS(app, resources={r"/api-voice/*": {"origins": "*"}})
UPLOAD_FOLDER = 'output'  # Directorio donde se guardarán los archivos

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global modelo
modelo="Bruno Mars (RVC) 250 Epoch"
#from rvcgui import vc_single, get_output_path

@app.route('/api-voice/modelos')
def modelos():
    ruta_modelos = './models'  # Reemplaza con la ruta de tu carpeta 'modelos'

    if os.path.exists(ruta_modelos) and os.path.isdir(ruta_modelos):
        carpetas = [nombre for nombre in os.listdir(ruta_modelos) if os.path.isdir(os.path.join(ruta_modelos, nombre))]
        return jsonify({"carpetas": carpetas})
    else:
        return "La ruta no existe o no es una carpeta."
    
@app.route('/api-voice/descargar/<nombre_archivo>')
def descargar_archivo(nombre_archivo):
    # Ruta a la carpeta donde se encuentran los archivos
    ruta_carpeta = '/home/ia-voice/IA-voice/output'
    print(ruta_carpeta)

    # Verificar si el archivo existe en la ruta especificada
    ruta_archivo = os.path.join(ruta_carpeta, nombre_archivo)
    print(ruta_archivo)
    print(os.path.exists(ruta_archivo))
    print(os.path.isfile(ruta_archivo))
    if os.path.exists(ruta_archivo) and os.path.isfile(ruta_archivo):
        return send_file(ruta_archivo, as_attachment=True)
    else:
        return "El archivo no existe en la ruta especificada."


@app.route('/api-voice/upload-audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return 'No se ha enviado ningún archivo de audio', 400

    file = request.files['audio']
    
    print(request.form.get('modelo'))
    modelo = request.form.get('modelo')
    #print(modelo)

    if file.filename == '':
        return 'Archivo no seleccionado', 400

    if file:
        # Asegurar el nombre del archivo para evitar problemas de seguridad
        #filename = secure_filename(file.filename)
        filename = file.filename
        # Guardar el archivo en el directorio de uploads
        ruta = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        path = "/home/ia-voice/IA-voice"
        ruta = os.path.join(path, ruta)
        print(ruta)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        sid = 0
        input_audio = ruta
        f0_pitch = 0
        f0_file = "None"
        f0_method = "crepe"
        file_index = ""
        file_big_npy = ""
        index_rate = 0.4        
        # print("sid: ", sid, "input_audio: ", input_audio, "f0_pitch: ", f0_pitch, "f0_file: ", f0_file, "f0_method: ", f0_method,
        #   "file_index: ", file_index, "file_big_npy: ", "index_rate: ", index_rate, "output_file: ")
        
        enlace(sid=sid, input_audio=input_audio,f0_pitch=f0_pitch, f0_file=f0_file, f0_method=f0_method, file_index=file_index, file_big_npy=file_big_npy, index_rate=index_rate, modelo= modelo)
        
        return 'Archivo de audio subido exitosamente!', 200

def create_app():
   return app