import os
from omegaconf import OmegaConf
from argparse import ArgumentParser
import multiprocessing as mp
import threading
import socket
from time import sleep
from tqdm import tqdm

import protocol

'''
python tools/cmd_runner.py -p 7005
python tools/cmd_runner.py -p 7006
python tools/cmd_runner.py -p 7007
'''
parser = ArgumentParser('Command Runner')
parser.add_argument('--config', '-c', type=str, nargs='*', default=[])
parser.add_argument('--port', '-p', type=int)
parser.add_argument('--buffer-size', '-b', type=int)
parser.add_argument('--delay', '-d', type=int, nargs=   '*')
args = parser.parse_args()
args = OmegaConf.merge(OmegaConf.load('./cfg/cmd/run/default.yaml'),
                       *[OmegaConf.load(f'./cfg/cmd/run/{name}.yaml') for name in args.config], 
                       {key: val for key, val in vars(args).items() if val is not None})

PORT = args.port
BUF_SIZE = args.buffer_size

if args.delay is not None:
    delay = ([0] * (3 - len(args.delay))) + args.delay
    delay[1] += delay[2] // 60
    delay[2] = delay[2] % 60
    delay[0] += delay[1] // 60
    delay[1] = delay[1] % 60
    
    delay_secs = delay[0] * 3600 + delay[1] * 60 + delay[2]
    for _ in tqdm(range(delay_secs), desc=f'DELAY: {delay[0]}H_{delay[1]}M_{delay[2]}S'):
        sleep(1)

client = socket.socket(socket.AF_INET, socket.SocketKind.SOCK_STREAM)
client.connect(('localhost', PORT))

signal_queue = mp.Queue()
quit_flag = mp.Value('b', False)

def recv_daemon(client, signal_queue, quit_flag):
    while not quit_flag.value:
        signal = protocol.recv(client, BUF_SIZE)
        if signal != '':
            signal_queue.put(signal)
            
thread = threading.Thread(target=recv_daemon, args=[client, signal_queue, quit_flag])
thread.start()
            
def wait_signal(request: str):
    global signal_queue
    
    protocol.send(client, request)
    while signal_queue.qsize() == 0:
        sleep(0.2)
    signal = signal_queue.get()
    return signal

try:
    got_quit = False
    while True:
        signal = wait_signal(protocol.CMD_ACQUIRE)
        if signal == protocol.CMD_QUIT:
            got_quit = True
            break
        else:
            print('COMMAND:', f'"{signal}"')
            os.system(signal)
            print()
        
    if not got_quit:
        wait_signal(protocol.CMD_QUIT)
    else:
        print('[Server-side Interrupt]')
except KeyboardInterrupt:
    wait_signal(protocol.CMD_QUIT)
    print('[Keyboard Interrupt]')
finally:
    quit_flag.value = True
    thread.join()
