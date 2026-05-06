import os
import subprocess
import sys
import re

def run_cmd(cmd, sudo=True, capture=True):
    """Executes shell commands with safety and output capture."""
    if sudo and os.geteuid() != 0:
        cmd = ["sudo"] + cmd
    try:
        result = subprocess.run(cmd, check=True, capture_output=capture, text=True)
        return result.stdout.strip() if capture else ""
    except subprocess.CalledProcessError:
        return None

def print_bios_prerequisites():
    """Prints clear BIOS/UEFI instructions."""
    instructions = [
        "--- BIOS/UEFI PREREQUISITES ---",
        "1. CPU VIRTUALIZATION: Find 'SVM Mode' or 'AMD-V' and set to ENABLED.",
        "2. IOMMU: Find 'IOMMU' or 'AMD-IOMMU' and set to ENABLED (Do not use Auto).",
        "3. PRIMARY DISPLAY: Set to 'IGFX', 'Internal Graphics', or 'Onboard'.",
        "4. SAVE AND EXIT: Ensure these are set before proceeding with Step 2."
    ]
    print("\n" + "\n".join(instructions) + "\n")

def setup_ssh():
    """Idempotently installs SSH and configures UFW firewall."""
    print("\n--- Configuring Remote Access (SSH) ---")
    
    # 1. Install OpenSSH Server
    check_ssh = run_cmd(["dpkg", "-s", "openssh-server"])
    if not check_ssh or "Status: install ok installed" not in check_ssh:
        print("Installing OpenSSH Server...")
        run_cmd(["apt", "update"])
        run_cmd(["apt", "install", "-y", "openssh-server"])
    
    # 2. Enable and Start Service
    run_cmd(["systemctl", "enable", "ssh"])
    run_cmd(["systemctl", "start", "ssh"])
    
    # 3. Configure Firewall (UFW)
    print("Configuring UFW to allow SSH...")
    run_cmd(["ufw", "allow", "ssh"])
    run_cmd(["ufw", "--force", "enable"]) # Force to avoid interactive prompt
    
    status = run_cmd(["ufw", "status"])
    print(f"Firewall Status:\n{status}")
    print("[OK] SSH is now active and firewall is open.")

def setup_physical_bridge():
    """Configures a Linux Bridge (br0) for external VM access using NetworkManager."""
    print("\n--- Configuring Physical Network Bridge (br0) ---")
    
    # Check if br0 already exists
    check_br0 = run_cmd(["nmcli", "con", "show", "br0"])
    if check_br0:
        print("Bridge br0 already exists. Skipping.")
        return

    # Find the active physical ethernet interface
    route_info = run_cmd(["ip", "-o", "-4", "route", "show", "to", "default"])
    if not route_info:
        print("Could not detect active internet interface.")
        return
    
    iface = route_info.split()[4]
    print(f"Detected physical interface: {iface}")

    # Create the bridge and attach the interface
    print("Creating bridge br0 (this may briefly disrupt connection)...")
    run_cmd(["nmcli", "con", "add", "ifname", "br0", "type", "bridge", "con-name", "br0"])
    run_cmd(["nmcli", "con", "add", "type", "bridge-slave", "ifname", iface, "master", "br0", "con-name", f"br0-slave-{iface}"])
    run_cmd(["nmcli", "con", "up", "br0"])
    
    print("[OK] Bridge br0 is active. VMs should use 'br0' for networking.")

def manage_nvidia_drivers():
    """Blacklists Nvidia drivers for isolated cards."""
    print("\n--- Nvidia Driver Management ---")
    chk = run_cmd(["modinfo", "nvidia"])
    if not chk:
        print("Nvidia drivers not detected. Skipping.")
        return

    blacklist_content = (
        "blacklist nvidia\n"
        "blacklist nvidia-drm\n"
        "blacklist nvidia-modeset\n"
        "blacklist nvidia-uvm\n"
        "blacklist nouveau\n"
    )
    conf_path = "/etc/modprobe.d/idg-nvidia-blacklist.conf"
    if not os.path.exists(conf_path):
        with open("nv.tmp", "w") as f: f.write(blacklist_content)
        run_cmd(["mv", "nv.tmp", conf_path])
        print("Nvidia drivers blacklisted.")
    else:
        print("Nvidia blacklist already present.")

def install_stack():
    """Installs the KVM/Libvirt stack and guest tools."""
    print("\n--- Syncing Virtualization Stack ---")
    pkgs = [
        "qemu-kvm", "libvirt-daemon-system", "libvirt-clients", 
        "bridge-utils", "ovmf", "cpu-checker", "libguestfs-tools"
    ]
    to_install = [p for p in pkgs if not run_cmd(["dpkg", "-s", p])]
    if to_install:
        run_cmd(["apt", "update"])
        run_cmd(["apt", "install", "-y"] + to_install)
    print("Virtualization stack ready.")

def configure_kernel():
    """Configures kernel parameters via kernelstub."""
    print("\n--- Configuring Kernel stub ---")
    current = run_cmd(["kernelstub", "-p"]) or ""
    required = ["amd_iommu=on", "iommu=pt", "kvm_amd.npt=1"]
    missing = [p for p in required if p not in current]
    if missing:
        run_cmd(["kernelstub", "-a", " ".join(missing)])
        print(f"Added: {' '.join(missing)}")
    else:
        print("Kernel parameters already up to date.")

def isolate_gpus():
    """Sets up VFIO isolation."""
    print("\n--- GPU Isolation Check ---")
    lspci = run_cmd(["lspci", "-nn"], sudo=False)
    gpu_ids = []
    for line in lspci.splitlines():
        if ("VGA" in line or "Audio device" in line) and not any(x in line for x in ["Internal", "Integrated", "iGPU"]):
            match = re.search(r"\[([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\]", line)
            if match: gpu_ids.append(match.group(1))
    
    if not gpu_ids:
        print("No discrete GPUs found for isolation.")
        return

    unique_ids = ",".join(sorted(list(set(gpu_ids))))
    conf_content = f"options vfio-pci ids={unique_ids}\nsoftdep nvidia pre: vfio-pci\n"
    conf_path = "/etc/modprobe.d/vfio.conf"
    
    if not os.path.exists(conf_path) or open(conf_path).read() != conf_content:
        with open("vfio.tmp", "w") as f: f.write(conf_content)
        run_cmd(["mv", "vfio.tmp", conf_path])
        run_cmd(["update-initramfs", "-u"])
        print(f"VFIO IDs {unique_ids} applied.")
    else:
        print("VFIO isolation already configured.")

def main_menu():
    while True:
        print("\n" + "="*40)
        print(" IDG NODE PROVISIONER - MAIN MENU ")
        print("="*40)
        print("1. VIEW BIOS PREREQUISITES")
        print("2. RUN AUTOMATED SYSTEM SETUP (Apt, Kernel, VFIO)")
        print("3. CONFIGURE NETWORK BRIDGE (br0)")
        print("4. CONFIGURE NVIDIA DRIVER BLACKLIST")
        print("5. ENABLE SSH AND FIREWALL")
        print("6. VERIFY IOMMU GROUPS")
        print("7. EXIT")
        
        choice = input("\nSelect an option (1-7): ")
        
        if choice == '1': print_bios_prerequisites()
        elif choice == '2':
            install_stack()
            configure_kernel()
            isolate_gpus()
            print("\nInitial Setup complete. Reboot recommended.")
        elif choice == '3':
            setup_physical_bridge()
        elif choice == '4':
            manage_nvidia_drivers()
            run_cmd(["update-initramfs", "-u"])
        elif choice == '5': setup_ssh()
        elif choice == '6':
            g_path = "/sys/kernel/iommu_groups/"
            if not os.path.exists(g_path):
                print("\nIOMMU not active. Reboot first.")
            else:
                for g in sorted(os.listdir(g_path), key=int):
                    print(f"Group {g}:")
                    for d in os.listdir(f"{g_path}{g}/devices"):
                        print(f"  {run_cmd(['lspci', '-nns', d], sudo=False)}")
        elif choice == '7': break

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Please run with sudo.")
        sys.exit(1)
    main_menu()
