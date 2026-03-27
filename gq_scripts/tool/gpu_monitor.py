"""
GPU 监控脚本 - 监控显存与利用率，按条件发送邮件通知

逻辑说明：
1. 空闲判定：剩余显存 > FREE_MEM_NEED 且 利用率 < UTIL_THRESHOLD 时视为空闲
2. busy -> free：满足空闲条件时，发送「GPU x 现已空闲」邮件
3. free -> busy：若之前发过「已空闲」，之后卡被占用，发送「GPU x 已被占用」邮件
4. free -> free：经过 STABLE_RUNTIME 后卡仍空闲，不重复发邮件
5. 额外：监控 MONITOR_PIDS 中的进程，结束时发送邮件
"""
import time
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import pynvml  # pip install pynvml

# nohup python -u gq_scripts/tool/gpu_monitor.py > monitor.log 2>&1 &
# ps -ef | grep gpu_monitor.py | grep -v grep
# pkill -f gpu_monitor.py

# --- 配置区域 ---
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465
SENDER_MAIL = "765584187@qq.com"
AUTH_CODE = "vydxiygzqdaxbehc"  # 注意：是授权码，不是QQ密码
RECEIVER_MAIL = "765584187@qq.com"

MONITOR_PIDS = [3162449, 2738088, 3145330,2738089,3154887]  # 你要监控的进程ID

# --- 修改配置区 ---
FREE_MEM_NEED = 30720  # 只要剩余显存大于 30G 就提醒我
UTIL_THRESHOLD = 30    # 利用率低于 30% 就算比较空闲
STABLE_RUNTIME = 300    # 检查频率 5 分钟

# --- 邮件发送函数 ---
def send_email(subject, content):
    try:
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['From'] = SENDER_MAIL
        msg['To'] = RECEIVER_MAIL
        msg['Subject'] = Header(subject, 'utf-8')
        
        server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT)
        server.login(SENDER_MAIL, AUTH_CODE)
        server.sendmail(SENDER_MAIL, [RECEIVER_MAIL], msg.as_string())
        server.quit()
        print(f"邮件已发送: {subject}")
    except Exception as e:
        print(f"邮件发送失败: {e}")

# --- 主监控逻辑 ---
def monitor():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    # 初始化状态
    gpu_status = {i: "busy" for i in range(device_count)}
    watched_pids = {pid: True for pid in MONITOR_PIDS}
    
    print("监控启动...")
    
    try:
        while True:
            # 1. 监控 GPU 状态
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                free_mem = mem_info.free / 1024**2
                util = util_info.gpu
                
                # 判断当前是否满足空闲条件：剩余显存足够且利用率低
                is_currently_free = free_mem > FREE_MEM_NEED and util < UTIL_THRESHOLD
                
                if is_currently_free and gpu_status[i] == "busy":
                    # 发现从忙碌转为空闲，发送邮件
                    gpu_status[i] = "free"
                    send_email(f"通知: GPU {i} 现已空闲", 
                               f"GPU {i} 状态更新：\n剩余显存: {free_mem:.1f} MiB\n利用率: {util}%")
                elif not is_currently_free and gpu_status[i] == "free":
                    # 之前发过「已空闲」，现在卡被占用，也发邮件通知
                    gpu_status[i] = "busy"
                    send_email(f"通知: GPU {i} 已被占用", 
                               f"GPU {i} 已被他人占用。\n剩余显存: {free_mem:.1f} MiB\n利用率: {util}%")

            # 2. 监控特定进程
            import os
            for pid in list(watched_pids.keys()):
                # 检查 PID 是否还在运行
                if not os.path.exists(f"/proc/{pid}"):
                    send_email(f"通知: 进程 {pid} 已结束", f"您监控的进程 {pid} 已经在服务器上运行完毕。")
                    del watched_pids[pid]

            time.sleep(STABLE_RUNTIME) # 轮询间隔
            
    except KeyboardInterrupt:
        print("监控停止")
    finally:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    monitor()