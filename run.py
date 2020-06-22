#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, session, render_template
from flask_socketio import SocketIO, emit, send, disconnect
from flask_swagger_ui import get_swaggerui_blueprint
from tools import ASR, Audio, SpeakerDiarization, SttStandelone
import yaml, os, sox, logging
from time import gmtime, strftime
import numpy as np
import scipy.io.wavfile
import threading
import time


app = Flask("__stt-standelone-worker__")
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, binary=True, ping_timeout=10, ping_interval=5, max_http_buffer_size=100000000, cookie="test")

# Set logger config
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

# Main parameters
AM_PATH = '/opt/models/AM'
LM_PATH = '/opt/models/LM'
TEMP_FILE_PATH = '/opt/tmp'
CONFIG_FILES_PATH = '/opt/config'
NBR_PROCESSES = 1
SAVE_AUDIO = False
SERVICE_PORT = 80
SWAGGER_URL = '/api-doc'
asr = ASR(AM_PATH,LM_PATH, CONFIG_FILES_PATH)

if not os.path.isdir(TEMP_FILE_PATH):
    os.mkdir(TEMP_FILE_PATH)
if not os.path.isdir(CONFIG_FILES_PATH):
    os.mkdir(CONFIG_FILES_PATH)

# Environment parameters
if 'SERVICE_PORT' in os.environ:
    SERVICE_PORT = os.environ['SERVICE_PORT']
if 'SAVE_AUDIO' in os.environ:
    SAVE_AUDIO = os.environ['SAVE_AUDIO']
if 'NBR_PROCESSES' in os.environ:
    if int(os.environ['NBR_PROCESSES']) > 0:
        NBR_PROCESSES = int(os.environ['NBR_PROCESSES'])
    else:
        exit("You must to provide a positif number of processes 'NBR_PROCESSES'")
if 'SWAGGER_PATH' not in os.environ:
    exit("You have to provide a 'SWAGGER_PATH'")
SWAGGER_PATH = os.environ['SWAGGER_PATH']

def swaggerUI():
    ### swagger specific ###
    swagger_yml = yaml.load(open(SWAGGER_PATH, 'r'), Loader=yaml.Loader)
    swaggerui = get_swaggerui_blueprint(
        SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
        SWAGGER_PATH,
        config={  # Swagger UI config overrides
            'app_name': "STT API Documentation",
            'spec': swagger_yml
        }
    )
    app.register_blueprint(swaggerui, url_prefix=SWAGGER_URL)
    ### end swagger specific ###

def getAudio(file,audio):
    file_path = TEMP_FILE_PATH+"/"+file.filename.lower()
    file.save(file_path)
    audio.transform(file_path)
    if not SAVE_AUDIO:
        os.remove(file_path)
    
threads = {}
def decodeThread(audio,asr,socket,client):
    last = False
    chunk_size=1024
    current_position = 0
    t = threading.currentThread()
    while getattr(t, "do_run", True): # or len(audio) >= current_position
        if current_position + chunk_size <= len(audio):
            text, bool = asr.decoderOnlineSil(audio[current_position:current_position + chunk_size],last)
            current_position += len(audio[current_position:current_position + chunk_size])
            if text != None:
                if bool:
                    socket.emit('final',text, namespace='/transcribe', room=client)
                else:
                    socket.emit('partial',text, namespace='/transcribe', room=client)
    app.logger.info("Number of decoded samples vs audio sample: (%d,%d)" % (current_position, len(audio)))

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        app.logger.info('[%s] New user entry on /transcribe' % (strftime("%d/%b/%d %H:%M:%S", gmtime())))
        # create main objects
        spk = SpeakerDiarization()
        audio = Audio(asr.get_sample_rate())
        
        #get response content type
        metadata = False
        if request.headers.get('accept').lower() == 'application/json':
            metadata = True
        elif request.headers.get('accept').lower() == 'text/plain':
            metadata = False
        else:
            raise ValueError('Not accepted header')

        #get speaker parameter
        spkDiarization = False
        if request.form.get('speaker') != None and (request.form.get('speaker').lower() == 'yes' or request.form.get('speaker').lower() == 'no'):
            spkDiarization = True if request.form.get('speaker').lower() == 'yes' else False
            #get number of speakers parameter
            try:
                if request.form.get('nbrSpeaker') != None and spkDiarization and int(request.form.get('nbrSpeaker')) > 0:
                    spk.set_maxNrSpeakers(int(request.form.get('nbrSpeaker')))
                elif request.form.get('nbrSpeaker') != None and spkDiarization:
                    raise ValueError('Not accepted "nbrSpeaker" field value (nbrSpeaker>0)')
            except Exception as e:
                app.logger.error(e)
                raise ValueError('Not accepted "nbrSpeaker" field value (nbrSpeaker>0)')
        else:
            raise ValueError('Not accepted "speaker" field value (yes|no)')

        stt = SttStandelone(metadata,spkDiarization)
        
        #get input file
        if 'file' in request.files.keys():
            file = request.files['file']
            getAudio(file,audio)
            #last_chunk = False
            #chunk_size = 1024
            #for i in range(0, len(audio.data), chunk_size):
            #    if i + chunk_size >= len(audio.data):
            #        last_chunk = True
            #    text = asr.decoderOnline(audio.data[i:i + chunk_size],last_chunk)
            #    app.logger.info(text)
            #    output = text
            output = stt.run(audio,asr,spk)
        else:
            raise ValueError('No audio file was uploaded')

        return output, 200
    except ValueError as error:
        return str(error), 400
    except Exception as e:
        app.logger.error(e)
        return 'Server Error', 500

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('disconnect_request', namespace='/transcribe')
def disconnect_request():
    disconnect()

@socketio.on('connect', namespace='/transcribe')
def test_connect():
    session['audio'] = []
    asr.init_decoderOnline()
    emit('partial','decode started')
    threads[str(request.sid)] = threading.Thread(target=decodeThread,args=[session['audio'], asr, socketio, request.sid])
    threads[str(request.sid)].start()
    app.logger.info(len(threads))

@socketio.on('sample_rate', namespace='/transcribe')
def handle_my_sample_rate(sampleRate):
    session['sample_rate'] = sampleRate

@socketio.on('audio', namespace='/transcribe')
def handle_my_custom_event(audio):
    values = np.frombuffer(audio, dtype=np.int16)
    session['audio'] += values.tolist()

@socketio.on('disconnect', namespace='/transcribe')
def test_disconnect():
    if len(session['audio']) > 0:
        my_audio = np.array(session['audio'], np.int16)
        scipy.io.wavfile.write(AM_PATH+"/"+str(request.sid)+'.wav', session['sample_rate'], my_audio.view('int16'))
    session['audio'] = None
    threads[str(request.sid)].do_run = False
    #time.sleep(2) #use this when the second while condition is used
    threads[str(request.sid)].join()
    del threads[str(request.sid)]
    app.logger.info('Client disconnected - '+str(request.sid))

@app.route('/healthcheck', methods=['GET'])
def check():
    return '', 200

# Rejected request handlers
@app.errorhandler(405)
def method_not_allowed(error):
    return 'The method is not allowed for the requested URL', 405

@app.errorhandler(404)
def page_not_found(error):
    return 'The requested URL was not found', 404

@app.errorhandler(500)
def server_error(error):
    app.logger.error(error)
    return 'Server Error', 500

if __name__ == '__main__':
    #start SwaggerUI
    swaggerUI()
    #Run ASR engine
    asr.run()
    #asr.init_decoderOnline()
    
    #Run server
    socketio.run(app, host='0.0.0.0', port=SERVICE_PORT)
    #app.run(host='0.0.0.0', port=SERVICE_PORT, debug=False, threaded=False, processes=NBR_PROCESSES)