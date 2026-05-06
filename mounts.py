import os
import subprocess
import json
import sys
import re

def run_cmd(cmd, sudo=True):
    if sudo and os.geteuid() != 0:
        cmd = ["sudo"] + cmd
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None

def get_block_devices():
    """Scans for physical drives, excluding the OS partition."""
    output = run_cmd(["lsblk", "-J", "-o", "NAME,SIZE,TYPE,MOUNTPOINT,UUID,MODEL,FSTYPE"])
    if not output: return []
    data = json.loads(output)
    devices = []
    for dev in data.get('blockdevices', []):
        if dev['type'] == 'disk':
            parts = dev.get('children', [])
            if not parts:
                devices.append(dev)
            else:
                for p in parts:
                    if p['mountpoint'] != '/':
                        devices.append(p)
    return devices

def get_existing_fstab_records():
    """Parses /etc/fstab into a dictionary of {UUID: (mount_point, line_index)}."""
    records = {}
    try:
        with open("/etc/fstab", "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip() and not line.startswith("#"):
                    match = re.search(r"UUID=([a-zA-Z0-9-]+)\s+([^\s]+)", line)
                    if match:
                        records[match.group(1)] = (match.group(2), i)
        return records, lines
    except Exception as e:
        print(f"Error reading fstab: {e}")
        return {}, []

def manage_storage():
    print("\n" + "="*40)
    print(" IDG STORAGE & MOUNT MANAGER ")
    print("="*40)
    
    devices = get_block_devices()
    fstab_records, original_lines = get_existing_fstab_records()
    
    print(f"{'#':<3} {'NAME':<10} {'SIZE':<8} {'FSTYPE':<8} {'ID / PREV MOUNT'}")
    for i, dev in enumerate(devices):
        uuid = dev.get('uuid', 'N/A')
        # Hint at your preferred naming convention
        hint = ""
        if dev['size'].startswith('11') or dev['size'].startswith('12'):
            hint = "[Potential /data (HDD)]"
        elif "nvme" in dev['name'].lower():
            hint = "[Potential /models (NVMe)]"
            
        status = fstab_records.get(uuid, (hint, "New"))[0]
        print(f"{i:<3} {dev['name']:<10} {dev['size']:<8} {dev['fstype'] or 'RAW':<8} {status}")

    choice = input("\nSelect device index to manage (or 'q' to quit): ")
    if choice.lower() == 'q': return

    try:
        selected = devices[int(choice)]
        uuid = selected['uuid']
        if not uuid:
            print("Error: Device has no UUID. Format it first.")
            return

        mount_path = input(f"Enter mount path (suggested: /data or /models): ").strip()
        
        # Check if this mount path is already taken by another UUID
        for u, (path, idx) in fstab_records.items():
            if path == mount_path and u != uuid:
                print(f"WARNING: {mount_path} is already assigned to UUID {u} in fstab!")
                if input("Override this mount point? (y/n): ").lower() != 'y': return

        # Prep the new fstab entry
        fstype = selected['fstype'] or "ext4"
        new_entry = f"UUID={uuid}  {mount_path}  {fstype}  defaults,noatime,nofail  0  2"

        if uuid in fstab_records:
            print(f"Found existing record for this UUID at {fstab_records[uuid][0]}.")
            action = "update"
        else:
            action = "append"

        confirm = input(f"Confirm {action} to /etc/fstab? (y/n): ").lower()
        if confirm == 'y':
            # Create the dir and set ACLs first
            run_cmd(["mkdir", "-p", mount_path])
            
            if action == "update":
                line_idx = fstab_records[uuid][1]
                original_lines[line_idx] = new_entry + "\n"
            else:
                original_lines.append(f"\n# IDG Managed Storage\n{new_entry}\n")

            with open("/etc/fstab", "w") as f:
                f.writelines(original_lines)
            
            # Apply permissions
            run_cmd(["mount", "-a"])
            run_cmd(["setfacl", "-R", "-m", "u:libvirt-qemu:rwx", mount_path])
            print(f"[SUCCESS] {mount_path} is live and recorded.")

    except (ValueError, IndexError):
        print("Invalid selection.")

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("Please run with sudo.")
        sys.exit(1)
    manage_storage()
