import os
import json
import subprocess
import re
import sys

# --- GLOBAL DEFAULTS ---
CONFIG_FILE = os.path.expanduser("~/idg_infra_config.json")
STORAGE_POOL = "/var/lib/libvirt/images"
CONFIG_DIR = os.path.expanduser("~/idg_configs")

def initialize_config():
    """Builds the initial JSON config with defaults if it doesn't exist."""
    defaults = {
        "paths": {
            "storage_pool": STORAGE_POOL,
            "config_dir": CONFIG_DIR,
            "network_share": "192.168.1.100:/mnt/idg_data"
        },
        "os_profiles": {
            "ubuntu": {
                "variant": "ubuntu22.04",
                "extra_args": []
            },
            "debian-trixie": {
                "variant": "debian12",
                "extra_args": []
            },
            "windows": {
                "variant": "win11",
                "extra_args": [
                    "--features", "hyperv_relaxed=on,hyperv_vapic=on,hyperv_spinlocks=on",
                    "--clock", "hypervclock_present=yes",
                    "--boot", "loader=/usr/share/OVMF/OVMF_CODE.fd,loader_ro=yes,loader_type=pflash"
                ]
            },
            "hackintosh": {
                "variant": "macos13",
                "extra_args": [
                    "--cpu", "host-passthrough,cache.mode=passthrough",
                    "--machine", "q35",
                    "--boot", "loader=/usr/share/OVMF/OVMF_CODE.fd"
                ]
            }
        }
    }
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(defaults, f, indent=4)
        print(f"Created default config at {CONFIG_FILE}")
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

def get_pci_ids():
    """Identifies discrete GPUs, skipping host integrated graphics."""
    try:
        output = subprocess.check_output(["lspci", "-nn"]).decode()
        gpus = []
        for line in output.splitlines():
            if "VGA compatible controller" in line or "Audio device" in line:
                if any(x in line for x in ["Internal", "Integrated", "iGPU"]):
                    continue
                pci_match = re.search(r'^([0-9a-fA-F:\.]+)', line)
                if pci_match:
                    gpus.append(pci_match.group(1))
        return gpus
    except Exception as e:
        print(f"Error scanning PCI bus: {e}")
        return []

def convert_and_inject_kernel():
    """Converts Docker .tar to QCOW2 and injects kernel + SSH."""
    print("\n--- Docker Tarball to Bootable VM ---")
    tar_path = input("Enter path to .tar file: ").strip()
    if not os.path.exists(tar_path):
        print("Error: File not found.")
        return

    output_name = input("Enter desired VM name (e.g., idg-trainer): ").strip()
    qcow_path = os.path.join(STORAGE_POOL, f"{output_name}.qcow2")

    print(f"Step 1: Converting {tar_path} to filesystem image...")
    subprocess.run([
        "virt-make-fs", "--format=qcow2", "--type=ext4", 
        "--size=+2G", tar_path, qcow_path
    ], check=True)

    print("Step 2: Injecting kernel, SSH, and networking tools...")
    subprocess.run([
        "virt-customize", "-a", qcow_path,
        "--install", "linux-image-amd64,grub-pc,virtio-guest-utils,openssh-server,curl",
        "--run-command", "systemctl enable ssh",
        "--root-password", "password:idg-admin",
        "--run-command", "ssh-keygen -A",
        "--update-grub"
    ], check=True)
    
    print(f"\n[SUCCESS] Bootable image created at: {qcow_path}")
    print("Default credentials: root / idg-admin")

def build_gpu_xml(pci_addr, name, cfg_dir):
    """Generates XML snippet for passthrough."""
    bus, slot_func = pci_addr.split(":")
    slot, func = slot_func.split(".")
    xml = f"""<hostdev mode='subsystem' type='pci' managed='yes'>
  <source><address domain='0x0000' bus='0x{bus}' slot='0x{slot}' function='0x{func}'/></source>
</hostdev>"""
    path = os.path.join(cfg_dir, f"{name}.xml")
    with open(path, "w") as f: f.write(xml)
    return path

def deploy_vm_interactive(config, pci_addrs):
    """Interactive prompt to deploy a new VM and configure persistence/autostart."""
    print("\n--- Deploy New VM ---")
    name = input("VM Name: ").strip()
    os_type = input("OS Profile (ubuntu/debian-trixie/windows/hackintosh): ").strip()
    min_ram = int(input("Min RAM (GB): "))
    max_ram = int(input("Max RAM (GB): "))
    min_cpu = int(input("Min Cores: "))
    max_cpu = int(input("Max Cores: "))
    disk = input("Path to QCOW2 image: ").strip()
    
    gpu_xml = None
    if pci_addrs:
        use_gpu = input(f"Assign a GPU? Found {len(pci_addrs)} (y/n): ").lower()
        if use_gpu == 'y':
            gpu_idx = int(input(f"Enter GPU Index (0-{len(pci_addrs)-1}): "))
            gpu_xml = build_gpu_xml(pci_addrs[gpu_idx], f"temp_gpu_{name}", config["paths"]["config_dir"])

    profile = config["os_profiles"].get(os_type, config["os_profiles"]["ubuntu"])
    
    cmd = [
        "virt-install", "--name", name,
        "--memory", f"memory={max_ram*1024},currentMemory={min_ram*1024}",
        "--vcpus", f"{min_cpu},maxvcpus={max_cpu}",
        "--import", "--disk", f"path={disk},format=qcow2",
        "--network", "bridge=br0,model=virtio",
        "--os-variant", profile["variant"],
        "--filesystem", f"{config['paths']['network_share']},idg_share,mode=mapped",
        "--nographics", "--noautoconsole"
    ]
    cmd.extend(profile["extra_args"])
    if gpu_xml: cmd.extend(["--hostdev", gpu_xml])
    
    print(f"Deploying {name}...")
    subprocess.run(cmd, check=True)

    # Persistence: Autostart configuration
    auto = input(f"Should {name} start automatically on host boot? (y/n): ").lower()
    if auto == 'y':
        subprocess.run(["virsh", "autostart", name], check=True)
        print(f"[OK] {name} configured to autostart.")
    else:
        print(f"[NOTE] {name} deployed. Manual start required after host reboot.")

def main_menu():
    config = initialize_config()
    os.makedirs(config["paths"]["config_dir"], exist_ok=True)
    os.makedirs(config["paths"]["storage_pool"], exist_ok=True)
    
    while True:
        pci_addrs = get_pci_ids()
        print("\n" + "="*40)
        print(" IDG INFRASTRUCTURE MANAGER ")
        print("="*40)
        print(f"Detected GPUs: {len(pci_addrs)}")
        print("1. Convert Docker .tar to Bootable QCOW2")
        print("2. Deploy New VM")
        print("3. List Existing VMs (virsh)")
        print("4. Exit")
        
        choice = input("\nSelect Option: ")
        
        if choice == '1':
            convert_and_inject_kernel()
        elif choice == '2':
            deploy_vm_interactive(config, pci_addrs)
        elif choice == '3':
            subprocess.run(["virsh", "list", "--all"])
        elif choice == '4':
            break

if __name__ == "__main__":
    main_menu()
