#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# ==============================================================================
# Jetson Hardware Clocks & Performance Setup Script
# Maximizes GPU/CPU frequencies (MAXN) and installs persistent clock lock service.
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}[+] Configuring NVIDIA Jetson for Maximum Performance...${NC}"

if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}[!] This script must be run as root (use sudo).${NC}"
   exit 1
fi

# 1. Set MAXN Power Profile
echo -e "${YELLOW}[*] Setting power profile to MAXN (Mode 0)...${NC}"
if command -v nvpmodel &> /dev/null; then
    nvpmodel -m 0
    echo -e "${GREEN}[✔] Power mode set to MAXN.${NC}"
else
    echo -e "${RED}[!] nvpmodel utility not found. Are you running on an NVIDIA Jetson?${NC}"
fi

# 2. Lock Clocks to Maximum Frequency
echo -e "${YELLOW}[*] Locking CPU, GPU, and EMC clocks via jetson_clocks...${NC}"
if command -v jetson_clocks &> /dev/null; then
    jetson_clocks
    jetson_clocks --show
    echo -e "${GREEN}[✔] jetson_clocks successfully applied.${NC}"
fi

# 3. Optimize Network Socket Buffers for Low-Latency RTSP/SRT Streaming
echo -e "${YELLOW}[*] Tuning kernel network receive buffers...${NC}"
sysctl -w net.core.rmem_max=26214400
sysctl -w net.core.rmem_default=26214400
sysctl -w net.core.wmem_max=26214400
sysctl -w net.core.wmem_default=26214400

# 4. Install Persistent Systemd Service
echo -e "${YELLOW}[*] Installing jetson-clocks systemd service for persistence across reboots...${NC}"
cat << 'EOF' > /etc/systemd/system/jetson-clocks-lock.service
[Unit]
Description=Lock Jetson Clocks to Maximum Frequency on Boot
After=nvpmodel.service
Wants=nvpmodel.service

[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable jetson-clocks-lock.service
echo -e "${GREEN}[✔] Systemd service 'jetson-clocks-lock' enabled.${NC}"

echo -e "\n${GREEN}================================================================${NC}"
echo -e "${GREEN} Jetson Performance Tuning Complete!                            ${NC}"
echo -e "${GREEN} Mode: MAXN (1020 MHz GPU) | Clocks: Locked                     ${NC}"
echo -e "${GREEN}================================================================${NC}\n"
