from os import getpid, kill
from time import sleep
import re
import signal

from notebook.notebookapp import list_running_servers
from requests import get
from requests.compat import urljoin
import ipykernel
import json
import psutil


# adapted from from https://stackoverflow.com/a/52977180
# tweaked to catch more exceptions, per getListOfProcessSortedByMemory
def get_active_kernels(cpu_threshold=0.0001):
    """Get a list of active jupyter kernels."""
    active_kernels = []
    pids = psutil.pids()
    my_pid = getpid()

    for pid in pids:
        if pid == my_pid:
            continue
        try:
            p = psutil.Process(pid)
            cmd = p.cmdline()
            for arg in cmd:
                if arg.count('ipykernel'):
                    cpu = p.cpu_percent(interval=0.1)
                    if cpu > cpu_threshold:
                        active_kernels.append((cpu, pid, cmd))
#         except psutil.AccessDenied:
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return active_kernels


# taken from https://stackoverflow.com/a/52977180, along with the function and imports above
# doesn't quite currently work; see https://github.com/jupyter/notebook/issues/1000#issuecomment-359875246
# def interrupt_bad_notebooks(cpu_threshold=0.2):
#     """Interrupt active jupyter kernels. Prompts the user for each kernel."""

#     active_kernels = sorted(get_active_kernels(cpu_threshold), reverse=True)

#     servers = list(list_running_servers())
#     print(f"servers = {servers}")
#     for ss in servers:
#         #this doesn't behave as expected / as it used to
#         response = get(urljoin(ss['url'].replace('localhost', '127.0.0.1'), 'api/sessions'),
#                        params={'token': ss.get('token', '')})
#         print(f"response = {response}")
#         for nn in json.loads(response.text): #<<<exception gets raised here
#             for kernel in active_kernels:
#                 for arg in kernel[-1]:
#                     if arg.count(nn['kernel']['id']):
#                         pid = kernel[1]
#                         cpu = kernel[0]
#                         interrupt = input(
#                             'Interrupt kernel {}; PID: {}; CPU: {}%? (y/n) '.format(nn['notebook']['path'], pid, cpu))
#                         if interrupt.lower() == 'y':
#                             p = psutil.Process(pid)
#                             while p.cpu_percent(interval=0.1) > cpu_threshold:
#                                 kill(pid, signal.SIGINT)
#                                 sleep(0.5)


#adapted slightly from https://thispointer.com/python-get-list-of-all-running-processes-and-sort-by-highest-memory-usage/
# def getListOfProcessSortedByMemory():
#     '''
#     Get list of running process sorted by Memory Usage
#     '''
#     listOfProcObjects = []
#     # Iterate over the list
#     for proc in psutil.process_iter():
#        try:
#            # Fetch process details as dict
#            pinfo = proc.as_dict(attrs=['pid', 'name', 'username', 'vms'])
#            pinfo['vms'] = proc.memory_info().vms / (1024 * 1024)
#            # Append dict to list
#            listOfProcObjects.append(pinfo);
#        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
#            pass
 
#     # Sort list of dict by key vms i.e. memory usage
#     listOfProcObjects = sorted(listOfProcObjects, key=lambda procObj: procObj['vms'], reverse=True)
 
#     return listOfProcObjects



# adapted from https://stackoverflow.com/a/32009595
def toHuman(size, precision=2, asString=True):
    suffixes=['B','KB','MB','GB','TB']
    suffixIndex = 0
    while size > 1024 and suffixIndex < 4:
        suffixIndex += 1 #increment the index of the suffix
        size = size/1024.0 #apply the division
    if asString:
        return "%.*f%s"%(precision,size,suffixes[suffixIndex])
    return size

# adapted from https://stackoverflow.com/a/32009595
def bytesTo(size_bytes, scale='GB'):
    scales = ('B', 'KB', 'MB', 'GB', 'TB')
    assert scale in scales, f'scale must be one of {scales}, got {scale} instead.'
    scaleIndex = scales.index(scale)
    return size_bytes / (1024 ** scaleIndex)


def memTotal(units='GB'):
    return bytesTo(psutil.virtual_memory().total, units)


def memAvailable(units='GB'):
    return bytesTo(psutil.virtual_memory().available, units)


def memUsed(units='GB'):
    return bytesTo(psutil.virtual_memory().used, units)




def wait_and_check_for_active_kernels(cpu_threshold=0.2, waiting_time_m=1, timeout_m=600):
    print(f"Watching for active kernels; timeout in {timeout_m}m.")
    minutes_since_first_activity_check = 0

    #check: are there any active jupyter kernels?
    active_kernels = sorted(get_active_kernels(cpu_threshold), reverse=True)

    while len(active_kernels) == 0 and minutes_since_first_activity_check < timeout_m:
        print(f"\t{minutes_since_first_activity_check}m since watch period started. No active kernels found. Sleeping for {waiting_time_m}m...")
        
        #wait waiting_time_m minutes
        sleep(60*waiting_time_m) 
        
        minutes_since_first_activity_check+=waiting_time_m
        active_kernels = sorted(get_active_kernels(cpu_threshold), reverse=True)

    print(f"{minutes_since_first_activity_check}m since watch period started.")
    if len(active_kernels) > 0:
        print(f"{len(active_kernels)} kernels found!")
    else:
        print(f"No kernels found.")

    # whenever we either find active kernels or reach the timeout point, 
    # return the current list of active kernels
    return active_kernels


def guard_memory_usage(cpu_threshold=0.000001, threshold_available_GB=2.0, time_between_checks_s=5):
    print(">Monitoring active kernels...")
    active_kernels = sorted(get_active_kernels(cpu_threshold=cpu_threshold), reverse=True)
    available_memory = memAvailable()
    print(f"\t|active kernels| = {len(active_kernels)}; available memory = {toHuman(psutil.virtual_memory().available)}; total memory = {toHuman(psutil.virtual_memory().total)}.")

    while len(active_kernels) > 0 and available_memory > threshold_available_GB:
        
        sleep(time_between_checks_s)
        
        active_kernels = sorted(get_active_kernels(cpu_threshold=cpu_threshold), reverse=True)
        available_memory = memAvailable()
    
    while len(active_kernels) > 0 and available_memory < threshold_available_GB:
        print(f">Available memory {available_memory}GB is below trigger threshold {threshold_available_GB}GB!")
        print(f">{len(active_kernels)} active kernels.")
        
        for kernel in active_kernels:
            pid = kernel[1]
            
            print(f">Attempting to kill pid {pid}!")
            try:
                p = psutil.Process(pid)
                
                for child in p.children(recursive=True):  # or parent.children() for recursive=False
                    child.kill()
                while p.cpu_percent(interval=0.1) > cpu_threshold:
                    p.kill()
                    sleep(0.5)
                
#                 while p.cpu_percent(interval=0.1) > cpu_threshold:
#                     kill(pid, signal.SIGINT)
#                     sleep(0.5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
               break
    
    if len(active_kernels) == 0:
        print(f"No more active kernels.")
    else:
        raise Exception("wtf")
    print("Guard period has ended.")

    
# import jupyter_client
# my_KernelSpecManager = jupyter_client.kernelspec.KernelSpecManager()
# my_KernelSpecManager.find_kernel_specs()
# my_ks = my_KernelSpecManager.get_kernel_spec('python3')

if __name__ == '__main__':
    
#     print("Watching for active kernels...")
    #check: are there any active jupyter kernels?
    active_kernels = wait_and_check_for_active_kernels(cpu_threshold=0.000001)
    print("Initial watch period ended.")

    while len(active_kernels) > 0:
#         print("Monitoring active kernels...")
        guard_memory_usage()
        
        active_kernels = wait_and_check_for_active_kernels(cpu_threshold=0.000001)
    
    print("Watch period timed out. No more active kernels.")
    print("Exiting.")
