import os
import sys
import time

import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr


my_sender = '1601631842@qq.com'  # sender email address (leave value unchanged)
my_pass = 'pmwzyyyxcpgshhdi'  # sender email authorization code (leave value unchanged)
# my_user = 'iruizewu@petalmail.com'  # recipient email address
my_user = 'iruizewu@petalmail.com'  # recipient email address

def send_msg(message):
    ret = True
    try:
        msg = MIMEText(message, 'plain', 'utf-8')  # email body
        msg['From'] = formataddr(["wrz GPU monitor", my_sender])  # (display name, sender email)
        msg['To'] = formataddr(["Primary Admin", my_user])  # (display name, recipient email)
        msg['Subject'] = "Lab NVIDIA GPU Monitor"  # email subject/title

        server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # SMTP server for sender email
        server.login(my_sender, my_pass)  # login using sender email and authorization code
        server.sendmail(my_sender, [my_user, ], msg.as_string())  # send the email
        server.quit()  # close connection
    except Exception:  # 如果 try 中的语句没有执行，则会执行下面的 ret=False
        ret = False
    return ret

def gpu_info(gpu_index=2):
    info = os.popen('nvidia-smi|grep %').read().split('\n')[gpu_index].split('|')

    power = int(info[1].split()[-3][:-1])
    memory = int(info[2].split('/')[0].strip()[:-3])
    Utilization = float(info[3].split()[0][:-1])
    return power, memory

def launcher(main, config, interval=2, can_use_list=[], max_use_num=2, accelerate=None):
    if len(can_use_list) == 0:
        gpu_power, gpu_memory = gpu_info(gpu_index=0)
        i = 0
        threshold = 3000 # memory usage threshold in MiB
        can_use_list = []
        waiting_start_time = time.time()
        while 1:  # set waiting condition
            for idx in range(8):
                gpu_power, gpu_memory = gpu_info(gpu_index=idx)
                if gpu_memory < threshold:
                    can_use_list.append(idx)
            i = i % 5
            symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|' + " wait " + f"{(time.time() - waiting_start_time)/60:.2f}" + " min"
            gpu_power_str = 'gpu power:%d W |' % gpu_power
            gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
            now_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            # sys.stdout.write('\r ' + now_time  + ' no gpu, ' + symbol)
            # sys.stdout.flush()
            time.sleep(interval)
            i += 1
            if len(can_use_list) > 0:
                start_time = time.time()
                while time.time() - start_time < 5 * 60:
                    op = 1
                # print(f"\nwaiting time: {time.time() - start_time}")
                break
        # print()
        if len(can_use_list) > max_use_num:
            can_use_list = can_use_list[:max_use_num]

    gpu_str = ""
    for idx in can_use_list:
        gpu_str += str(idx) + ','
    # remove the trailing comma
    gpu_str = gpu_str[:-1]
    print("select gpu: ", gpu_str)

    # cmd = 'python ~/hehe.py'
    # print('\n' + cmd)
    # os.system(cmd)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    # import tensorflow as tf
    # a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    if len(can_use_list) == 1:
        main(config)
    else:
        from accelerate import notebook_launcher
        notebook_launcher(main, config, num_processes=len(can_use_list))

# narrow_setup()
# send_msg("Other notifications...")