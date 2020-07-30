#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import pathlib
import websockets
import concurrent.futures
import logging
import os
from tools import ASR, Audio, SpeakerDiarization, SttStandelone, ASROnline
from time import gmtime, strftime
import scipy.io.wavfile



pool = concurrent.futures.ThreadPoolExecutor((os.cpu_count() or 1))
loop = asyncio.get_event_loop()


# Set logger config
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Main parameters
AM_PATH = '/opt/models/AM'
LM_PATH = '/opt/models/LM'
TEMP_FILE_PATH = '/opt/tmp'
CONFIG_FILES_PATH = '/opt/config'
NBR_PROCESSES = 1
SILENCE=True
SAVE_AUDIO = False
SERVICE_PORT = 80
SWAGGER_URL = '/api-doc'
SAMPLE_RATE=16000
asr = ASR(AM_PATH,LM_PATH, CONFIG_FILES_PATH)

if not os.path.isdir(TEMP_FILE_PATH):
    os.mkdir(TEMP_FILE_PATH)
if not os.path.isdir(CONFIG_FILES_PATH):
    os.mkdir(CONFIG_FILES_PATH)

# Environment parameters
if 'SILENCE' in os.environ:
    SILENCE = os.environ['SILENCE']
if 'SERVICE_PORT' in os.environ:
    SERVICE_PORT = os.environ['SERVICE_PORT']
if 'SAVE_AUDIO' in os.environ:
    SAVE_AUDIO = os.environ['SAVE_AUDIO']
if 'NBR_PROCESSES' in os.environ:
    if int(os.environ['NBR_PROCESSES']) > 0:
        NBR_PROCESSES = int(os.environ['NBR_PROCESSES'])
    else:
        exit("You must to provide a positif number of processes 'NBR_PROCESSES'")


def process_chunk(_asr, audio):
    if audio == '{"eof" : 1}':
        if SILENCE:
            text, bool = _asr.decoderOnlineSil(audio,True)
        else:
            text, bool = _asr.decoderOnline(audio,True)
        return text, True
    else:
        if SILENCE:
            text, bool = _asr.decoderOnlineSil(audio,False)
        else:
            text, bool = _asr.decoderOnline(audio,False)
        return text, False


async def recognize(websocket, path):

    _asr = None

    while True:

        audio = await websocket.recv()

        if not _asr:
            _asr = ASROnline(asr)
        response, stop = await loop.run_in_executor(pool, process_chunk, _asr, audio)

        await websocket.send(response)
        if stop:
            break

if __name__ == '__main__':
    #Run ASR engine
    asr.run()
    #asr.init_decoderOnline()
    logger.info('Server is successfully started. Waiting for clients')

    #Run websockets
    start_server = websockets.serve(recognize, '0.0.0.0', SERVICE_PORT)

    loop.run_until_complete(start_server)
    loop.run_forever()
