import os
from pathlib import Path
from omegaconf import OmegaConf
from argparse import ArgumentParser

import threading
import multiprocessing
import socket

import re
import numpy as np
import torch
from tqdm import tqdm

import protocol

# Arguments
value_pattern = re.compile(r'^[+-]?\d+(\.\d?)?$')
range_pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+:[-+]?[0-9]*\.?[0-9]+(:[-+]?[0-9]*\.?[0-9]+)?$')
linspace_pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+_[-+]?[0-9]*\.?[0-9]+_[-+]?[0-9]*\.?[0-9]+$')
list_pattern = re.compile(r'^\[([+-]?\d+(\.\d?)?,)*[+-]?\d+(\.\d?)?\,?\]$')

'''
python tools/cmd_streamer.py -r _temp.sh -v -p 7005
'''
parser = ArgumentParser('Command Streaming Center')
parser.add_argument('--config', '-c', type=str, nargs='*', default=[])
parser.add_argument('--devices', '-d', type=int, nargs='*', default=list(range(torch.cuda.device_count())))
parser.add_argument('--port', '-p', type=int)
parser.add_argument('--buffer-size', '-b', type=int)
parser.add_argument('--recipe', '-r', type=str, required=True, nargs='+')
parser.add_argument('--head', '-H', type=int)
parser.add_argument('--run-immediately', '-R', action='store_true', default=None)
parser.add_argument('--verbose', '-v', action='store_true', default=None)
args = parser.parse_args()
args = OmegaConf.merge(OmegaConf.load('./cfg/cmd/stream/default.yaml'),
                       *[OmegaConf.load(f'./cfg/cmd/stream/{name}.yaml') for name in args.config], 
                       {key: val for key, val in vars(args).items() if val is not None})
if args.verbose:
    print(args)

PORT = args.port
BUF_SIZE = args.buffer_size
if args.head == None:
    HEAD = np.inf
elif args.head <= 0:
    raise ValueError('Head must be a positive integer.')
else:
    HEAD = args.head

commands = []
for filename in args.recipe:
    filename = os.path.join('./recipe', filename)
    with open(filename, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if line != '' and not line.startswith('#'):
                commands.append(line)
                    
                if (HEAD >= 0) and (len(commands) >= HEAD):
                    break

if args.verbose:
    max_commands_len = int(np.floor(np.log10(len(commands))) + 1)
    for i, cmd in enumerate(commands, start=1):
        print(f'{i:0{max_commands_len}d}. "{cmd}"')
print()

if not args.run_immediately:
    query = '\0'
    while True:
        query = input(f'Start {len(commands)} commands? [Y/n] ').strip().lower()
        if query in {'', 'y'}:
            break
        elif query == 'n':
            exit()

# Hosting
with socket.socket(socket.AddressFamily.AF_INET, socket.SocketKind.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('', PORT))
    server.listen()
    server.settimeout(1)

    num_clients = 0
    print_lock = threading.Lock()
    def s_print(*args, sep=' ', end='\n'):
        global print_lock
        print_lock.acquire()
        print(*args, sep=sep, end=end)
        print_lock.release()
            
    threads, clients = {}, {}
    try:
        with tqdm(total=len(commands), desc=f'{args.port}', leave=False) as progress_bar:
            def update_postfix_str():
                progress_bar.set_postfix_str(f'# clients={num_clients}')
            
            cmd_index = multiprocessing.Value('i', 0)
            progress_lock = threading.Lock()
            interrupt_flag = multiprocessing.Value('b', False)
            def handle_client(client: socket.socket, idx: int):
                global num_clients, cmd_index, progress_lock
                global threads, clients
                
                progress_lock.acquire()
                num_clients += 1
                update_postfix_str()
                progress_lock.release()
                
                # Start transactions
                request = ''
                while request != protocol.CMD_QUIT:
                    if interrupt_flag.value:
                        break
                    
                    request = protocol.recv(client, BUF_SIZE)
                    if request == protocol.CMD_ACQUIRE:
                        if cmd_index.value < len(commands):
                            cmd = commands[cmd_index.value].replace('[ID]', str(args.devices[idx % len(args.devices)]))
                            cmd_index.value += 1
                            
                            protocol.send(client, cmd)
                            progress_lock.acquire()
                            progress_bar.update()
                            progress_lock.release()
                        else:
                            break

                protocol.send(client, protocol.CMD_QUIT)
                client.close()
                progress_lock.acquire()
                num_clients -= 1
                update_postfix_str()
                del clients[idx]
                progress_lock.release()
            
            while cmd_index.value < len(commands):
                try:
                    client, _ = server.accept()
                    thread = threading.Thread(target=handle_client, args=(client, num_clients))
                    threads[num_clients] = thread
                    clients[num_clients] = client
                    thread.start()
                except socket.timeout:
                    pass
            
            for client in clients.values():
                protocol.send(client, protocol.CMD_QUIT)
            print('All commands have been exectued.')
    except KeyboardInterrupt: 
        interrupt_flag.value = True
        print('[Keyboard Interrupt]')
    finally:
        for thread in threads.values():
            thread.join()
        for client in clients.values():
            client.close()
        server.close()
