"""
firewall.py — Simple Firewall + Response System for HybridIDS
Place this file in: hybrid_ids/backend/firewall.py
"""

import os
import datetime

# ── Port knocking state ──────────────────────────────────────────
KNOCK_SEQUENCE = [5000, 6000, 7000]
knock_history = []   # stores recent knock ports per IP
# In production you'd key this by IP; for simplicity, one global list


# ─────────────────────────────────────────────────────────────────
# 1. BLOCK / UNBLOCK IP
# ─────────────────────────────────────────────────────────────────

def block_ip(ip: str):
    """Block all incoming traffic from a given IP address."""
    cmd = f"iptables -I INPUT -s {ip} -j DROP"
    os.system(cmd)
    log_action(ip, f"BLOCKED IP via iptables: {cmd}")


def unblock_ip(ip: str):
    """Remove the block rule for an IP address."""
    cmd = f"iptables -D INPUT -s {ip} -j DROP"
    os.system(cmd)
    log_action(ip, f"UNBLOCKED IP: {cmd}")


# ─────────────────────────────────────────────────────────────────
# 2. BLOCK / OPEN PORT
# ─────────────────────────────────────────────────────────────────

def block_port(port: int):
    """Block incoming TCP traffic on a specific port."""
    cmd = f"iptables -I INPUT -p tcp --dport {port} -j DROP"
    os.system(cmd)
    log_action("system", f"BLOCKED PORT {port} via iptables: {cmd}")


def open_port(port: int):
    """Remove the block rule for a specific port."""
    cmd = f"iptables -D INPUT -p tcp --dport {port} -j DROP"
    os.system(cmd)
    log_action("system", f"OPENED PORT {port}: {cmd}")


# ─────────────────────────────────────────────────────────────────
# 3. PORT KNOCKING (simple fixed sequence)
# ─────────────────────────────────────────────────────────────────

def check_knock(port: int) -> bool:
    """
    Update knock history and check if sequence matches.
    Sequence: 5000 → 6000 → 7000
    If matched → open port 22 and reset history.
    If wrong knock → reset history.
    Returns True if sequence completed successfully.
    """
    global knock_history

    expected = KNOCK_SEQUENCE[len(knock_history)]

    if port == expected:
        knock_history.append(port)
        log_action("knock", f"Valid knock: {port} ({len(knock_history)}/{len(KNOCK_SEQUENCE)})")

        if knock_history == KNOCK_SEQUENCE:
            # Sequence complete — open SSH
            open_port(22)
            log_action("knock", "Full sequence matched — SSH port 22 opened")
            knock_history = []   # reset for next round
            return True
    else:
        # Wrong knock — reset
        log_action("knock", f"Wrong knock: {port}, expected {expected} — sequence reset")
        knock_history = []

    return False


# ─────────────────────────────────────────────────────────────────
# 4. AUTOMATED RESPONSE
# ─────────────────────────────────────────────────────────────────

def respond(attack_type: str, ip: str):
    """
    Take automated action based on detected attack type.

    Attack Type → Action
    ───────────────────────────────────────
    DoS         → Block IP
    Probe       → Block port 80
    R2L         → Block IP
    U2R         → Block IP + Block port 22
    """
    attack_type = attack_type.strip().upper()

    log_attack(ip, attack_type)

    if attack_type == "DOS":
        block_ip(ip)

    elif attack_type == "PROBE":
        block_port(80)

    elif attack_type == "R2L":
        block_ip(ip)

    elif attack_type == "U2R":
        block_ip(ip)
        block_port(22)

    else:
        log_action(ip, f"Unknown attack type: {attack_type} — no action taken")


# ─────────────────────────────────────────────────────────────────
# 5. LOGGING
# ─────────────────────────────────────────────────────────────────

LOG_FILE = "logs.txt"

def log_attack(ip: str, attack_type: str):
    """Log detected attack in format: <IP> -> <ATTACK_TYPE>"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {ip} -> {attack_type}\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)
    print(line.strip())


def log_action(ip: str, message: str):
    """Log a firewall action with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] [{ip}] {message}\n"
    with open(LOG_FILE, "a") as f:
        f.write(line)
    print(line.strip())


def get_logs(n: int = 50) -> list:
    """Return the last n lines from logs.txt."""
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        return [l.strip() for l in lines[-n:] if l.strip()]
    except FileNotFoundError:
        return []