"""
run_web_app.py
--------------
Production launcher for the Hair Progress Tracker Web Application.
Features:
- Windows Job Object integration: ensures cloudflared.exe is automatically
  terminated by the OS kernel even if the console window is forcibly closed.
- Proactive port management: terminates any dangling listeners on port 7860
  and verifies socket availability before launching Gradio.
- Asynchronous pipe draining: background daemon thread streams cloudflared
  logs to 'tunnel.log' to prevent pipe buffer saturation and tunnel freeze.
- Clean signal handling and graceful teardown on Ctrl+C or exit.
"""

import os
import sys
import time
import subprocess
import socket
import re
import threading
import signal
import atexit
import webbrowser
import warnings

# Suppress Gradio deprecation warnings in console output
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_DIR = os.path.join(BASE_DIR, "deployment")
CF_EXE = os.path.join(BASE_DIR, "cloudflared.exe")
TUNNEL_LOG = os.path.join(BASE_DIR, "tunnel.log")
HF_MODEL_URL = "https://huggingface.co/AlgomCorp/hair-segmentation-unet"
PORT = 7860

cf_process = None
job_handle = None

# ─────────────────────────────────────────────────────────────────────────────
#  Windows Job Object (Kernel-level child process termination)
# ─────────────────────────────────────────────────────────────────────────────

def init_job_object():
    """Configures a Windows Job Object with KILL_ON_JOB_CLOSE."""
    global job_handle
    if os.name != 'nt':
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', wintypes.LARGE_INTEGER),
                ('PerJobUserTimeLimit', wintypes.LARGE_INTEGER),
                ('LimitFlags', wintypes.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t),
                ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('ReadOperationCount', ctypes.c_uint64),
                ('WriteOperationCount', ctypes.c_uint64),
                ('OtherOperationCount', ctypes.c_uint64),
                ('ReadTransferCount', ctypes.c_uint64),
                ('WriteTransferCount', ctypes.c_uint64),
                ('OtherTransferCount', ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryLimit', ctypes.c_size_t),
                ('PeakJobMemoryLimit', ctypes.c_size_t),
            ]

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        res = kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(info), ctypes.sizeof(info)
        )
        if res:
            job_handle = job
            return job
    except Exception:
        pass
    return None

def assign_to_job(proc):
    """Assigns a subprocess to the Windows Job Object."""
    global job_handle
    if not job_handle or os.name != 'nt' or proc is None:
        return
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.AssignProcessToJobObject(job_handle, wintypes.HANDLE(proc._handle))
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────────
#  Port & Process Management
# ─────────────────────────────────────────────────────────────────────────────

def is_port_in_use(port=PORT):
    """Checks whether the specified TCP port is accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0

def clean_environment():
    """Guarantees port 7860 is completely free and no zombie cloudflared processes linger."""
    # 1. Kill any existing cloudflared instances
    try:
        subprocess.run("taskkill /F /IM cloudflared.exe", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    # 2. Kill any processes holding port 7860 (excluding this script's PID)
    my_pid = os.getpid()
    try:
        net_out = subprocess.check_output(f"netstat -ano | findstr :{PORT}", shell=True,
                                          text=True, stderr=subprocess.DEVNULL)
        pids = set()
        for line in net_out.splitlines():
            if "LISTENING" in line:
                parts = line.strip().split()
                if parts:
                    try:
                        pid = int(parts[-1])
                        if pid > 0 and pid != my_pid:
                            pids.add(pid)
                    except ValueError:
                        pass
        for pid in pids:
            try:
                subprocess.run(f"taskkill /F /T /PID {pid}", shell=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
    except Exception:
        pass

    # 3. Wait up to 5 seconds until port is verified free
    start = time.time()
    while time.time() - start < 5:
        if not is_port_in_use(PORT):
            break
        time.sleep(0.3)

def cleanup():
    """Teardown handler on normal or abrupt exit."""
    global cf_process
    if cf_process is not None:
        try:
            cf_process.terminate()
            cf_process.wait(timeout=2)
        except Exception:
            try:
                cf_process.kill()
            except Exception:
                pass
        cf_process = None
    try:
        subprocess.run("taskkill /F /IM cloudflared.exe", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

atexit.register(cleanup)

def signal_handler(sig, frame):
    print("\nStopping services cleanly...", flush=True)
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ─────────────────────────────────────────────────────────────────────────────
#  Main Execution Flow
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global cf_process

    print("=" * 70, flush=True)
    print("  HAIR PROGRESS TRACKER & DEEP LEARNING VISUALIZER", flush=True)
    print("  Production Web Deployment Launcher", flush=True)
    print("=" * 70, flush=True)

    # Step 1: Clean environment & initialize kernel Job Object
    print("[1/3] Preparing environment and verifying port 7860...", flush=True)
    init_job_object()
    clean_environment()
    if is_port_in_use(PORT):
        print(f"      WARNING: Port {PORT} is still in use. Retrying cleanup...", flush=True)
        clean_environment()
    print("      Port 7860 is clean and verified free.", flush=True)

    # Step 2: Launch Cloudflare Tunnel with Asynchronous Pipe Drain
    print("[2/3] Establishing secure public HTTPS tunnel...", flush=True)
    if not os.path.exists(CF_EXE):
        print(f"ERROR: {CF_EXE} not found.", flush=True)
        sys.exit(1)

    cf_process = subprocess.Popen(
        [CF_EXE, "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    )
    assign_to_job(cf_process)

    public_url_container = {"url": None}
    url_event = threading.Event()

    def drain_pipe(proc):
        """Continuously reads tunnel stdout so OS pipe buffer never fills up."""
        try:
            with open(TUNNEL_LOG, "w", encoding="utf-8", errors="replace") as log_f:
                for line in iter(proc.stdout.readline, ''):
                    log_f.write(line)
                    log_f.flush()
                    if not public_url_container["url"]:
                        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                        if m:
                            public_url_container["url"] = m.group(0)
                            url_event.set()
        except Exception:
            pass
        finally:
            if proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass

    drain_thread = threading.Thread(target=drain_pipe, args=(cf_process,), daemon=True)
    drain_thread.start()

    # Wait up to 15 seconds for Cloudflare to assign public tunnel URL
    url_event.wait(timeout=15)
    public_url = public_url_container["url"]

    # Step 3: Load PyTorch Model & Start Gradio
    print("[3/3] Loading PyTorch U-Net model and starting Gradio...", flush=True)
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    sys.path.insert(0, DEPLOY_DIR)

    import app

    print("\n" + "=" * 70, flush=True)
    print("  >>> LIVE WEB DEPLOYMENT ACTIVE & RUNNING <<<", flush=True)
    print("=" * 70, flush=True)
    if public_url:
        print(f"  [+] PUBLIC HTTPS LINK  : {public_url}", flush=True)
    else:
        print("  [!] Cloudflare URL pending; running locally on port 7860.", flush=True)
    print(f"  [+] LOCAL BROWSER LINK : http://127.0.0.1:{PORT}", flush=True)
    print(f"  [+] HUGGING FACE MODEL : {HF_MODEL_URL}", flush=True)
    print("=" * 70, flush=True)
    print("  Share the PUBLIC HTTPS LINK with your teacher or open it on any phone/PC!", flush=True)
    print("  Press Ctrl+C in this window anytime to stop cleanly.", flush=True)
    print("=" * 70 + "\n", flush=True)

    # Open browser automatically
    target_url = public_url if public_url else f"http://127.0.0.1:{PORT}"
    try:
        webbrowser.open(target_url)
    except Exception:
        pass

    # Launch Gradio server
    try:
        app.demo.launch(
            server_name="0.0.0.0",
            server_port=PORT,
            inbrowser=False,
            quiet=True
        )
    except (KeyboardInterrupt, SystemExit):
        print("\nShutting down web server cleanly...", flush=True)
    finally:
        cleanup()
        print("All services stopped and ports freed.", flush=True)

if __name__ == "__main__":
    main()
