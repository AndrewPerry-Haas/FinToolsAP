#import os
#import inspect
#import pathlib
import datetime
import functools
#import knockknock
import tracemalloc
#import configparser

#SLACK_NOTIF_NAME = 'SlackNotificationSettings'

def Performance(func):
    """
    Decorator that measures and prints the performance of a function, including execution time and peak memory usage.

    This decorator uses tracemalloc to track memory allocation and datetime to record start and end timestamps.
    When the decorated function completes, it prints:
        - Function name
        - Start time
        - End time
        - Elapsed time
        - Peak memory usage

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: A wrapped version of the original function that executes it and prints performance metrics.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        s = datetime.datetime.now()
        res = func(*args, **kwargs)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        e = datetime.datetime.now()
        print('-' * 50)
        print(f'Function: {func.__name__}')
        print('-' * 50)
        print(f'Start Time:\t\t{s}')
        print(f'End Time:\t\t{e}')
        print(f'Elapsed Time:\t\t{e - s}')
        print(f'Peak Memory Usage:\t{humanbytes(peak)}')
        print('-' * 50)
        return(res)
    return(wrapper)
    
""" Commented out until aohttps is out of beta for python 3.12.0
def SlackNotify(func):


    user_config_dir = os.path.expanduser('~') + '/.config/fintoolsap'
    user_config_ini = user_config_dir + f'/slacknotify.ini'

    os.makedirs(os.path.dirname(user_config_ini), exist_ok = True)
    
    try: 
        config = configparser.ConfigParser()
        config.read(user_config_ini)
        webhook_url = config[SLACK_NOTIF_NAME]['workspace_url']
        channel = config[SLACK_NOTIF_NAME]['channel']
        user_mentions = config[SLACK_NOTIF_NAME]['user_mentions']
        user_mentions = user_mentions.strip('][')
        user_mentions = ''.join(user_mentions.split())
        user_mentions = user_mentions.split(',')
    except Exception as ex:
        config_file = configparser.ConfigParser()
        config_file.add_section(SLACK_NOTIF_NAME)
        config_file.set(SLACK_NOTIF_NAME, 'workspace_url', '<workspace_url>')
        config_file.set(SLACK_NOTIF_NAME, 'channel', '<channel_name>')
        config_file.set(SLACK_NOTIF_NAME, 'user_mentions', '<[user_ids]>')

        with open(user_config_ini, 'w') as configfileObj:
            config_file.write(configfileObj)
            configfileObj.flush()
            configfileObj.close()

        raise Exception(f'Slack Notification ini file created in {user_config_ini}. Please update fields.')

    @functools.wraps(func)
    @knockknock.slack_sender(webhook_url = webhook_url, 
                             channel = channel, 
                             user_mentions = user_mentions)
    def SlackNotification(*args, **kwargs):
        res = func(*args, **kwargs)
        return(res)
    return(SlackNotification)
    """

              
def humanbytes(B):
    """
    Convert a byte value to a human-readable string using decimal (SI) units.

    Parameters:
        B (int or float): The size in bytes.

    Returns:
        str: A human‐friendly string with two decimal places (for KB and above) and the appropriate unit:
             - Bytes for values < 1 000
             - KB    for values ≥ 1 000 and < 1 000 000
             - MB    for values ≥ 1 000 000 and < 1 000 000 000
             - GB    for values ≥ 1 000 000 000 and < 1 000 000 000 000
             - TB    for values ≥ 1 000 000 000 000

    Examples:
        >>> humanbytes(0)
        '0.0 Byte'
        >>> humanbytes(512)
        '512.0 Byte'
        >>> humanbytes(2048)
        '2.05 KB'
        >>> humanbytes(5_242_880)
        '5.24 MB'
    """
    B = float(B)
    KB = float(1000)
    MB = float(KB ** 2) 
    GB = float(KB ** 3)
    TB = float(KB ** 4)
    PB = float(KB ** 5)
    if(B < KB):
        return('{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte'))
    elif(KB <= B < MB):
        return('{0:.2f} KB'.format(B / KB))
    elif(MB <= B < GB):
        return('{0:.2f} MB'.format(B / MB))
    elif(GB <= B < TB):
        return('{0:.2f} GB'.format(B / GB))
    elif(TB <= B < PB):
        return('{0:.2f} TB'.format(B / TB))
    elif(B >= PB):
        return('{0:.2f} TB'.format(B / PB))