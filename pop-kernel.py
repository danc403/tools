#!/usr/bin/env python3

import subprocess
import os
import sys
import re

def run_cmd(cmd, sudo=False):
    if sudo and os.geteuid() != 0:
        cmd = ["sudo"] + cmd
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return None

def get_running_kernel():
    """Returns the version of the currently running kernel."""
    return run_cmd(['uname', '-r'])

def get_installed_kernels():
    """Returns a list of installed kernel package names using dpkg."""
    output = run_cmd(['dpkg', '--list', 'linux-image-[0-9]*'])
    if not output:
        return []
    
    kernels = []
    for line in output.splitlines():
        if line.startswith("ii"):
            parts = line.split()
            if len(parts) >= 2:
                kernels.append(parts[1])
    return kernels

def parse_kernel_versions(kernels):
    """Groups packages by their version string."""
    kernel_map = {}
    for pkg in kernels:
        # Extract version like 6.8.0-76060800-generic
        version = pkg.replace("linux-image-", "")
        if version not in kernel_map:
            kernel_map[version] = []
        kernel_map[version].append(pkg)
        # Also include associated headers and modules if present
        header_pkg = pkg.replace("linux-image-", "linux-headers-")
        kernel_map[version].append(header_pkg)
    return sorted(kernel_map.keys()), kernel_map

def list_kernels(unique_versions, running_kernel):
    print("\nInstalled Kernels:")
    for i, version in enumerate(unique_versions):
        mark = "*" if version in running_kernel else ""
        print(f"{i+1}. {version} {mark}")

def remove_kernels(version, kernel_map, running_kernel):
    if version in running_kernel:
        print("\nCRITICAL: You are trying to remove the running kernel!")
        confirm = input("This is highly dangerous. Proceed anyway? (y/N): ").lower()
        if confirm != 'y': return

    packages = kernel_map.get(version, [])
    print(f"\nPreparing to remove: {', '.join(packages)}")
    
    confirm = input("Confirm removal via APT? (y/N): ").lower()
    if confirm == 'y':
        # Pop!_OS automatically triggers kernelstub and initramfs updates on APT remove
        run_cmd(['apt', 'purge', '-y'] + packages, sudo=True)
        print("Kernel removed. Boot entries updated via kernelstub.")

def set_default_kernel(version):
    """Sets the default kernel using Pop!_OS kernelstub."""
    print(f"Setting {version} as default...")
    # Pop!_OS uses kernelstub to manage the systemd-boot entries
    # We point to the vmlinuz image associated with the version
    run_cmd(['kernelstub', '-v', '-k', f'/boot/vmlinuz-{version}'], sudo=True)
    print("Default boot configuration updated.")

def main():
    if os.geteuid() != 0:
        print("This script requires sudo privileges to manage kernels.")
        sys.exit(1)

    while True:
        running = get_running_kernel()
        installed = get_installed_kernels()
        unique_versions, kernel_map = parse_kernel_versions(installed)

        print("\n" + "="*30)
        print(f" POP!_OS KERNEL TOOL (Running: {running})")
        print("="*30)
        print("1. List/Remove Kernels")
        print("2. Set Default Kernel (kernelstub)")
        print("q. Quit")

        choice = input("\nChoice: ").lower()

        if choice == '1':
            list_kernels(unique_versions, running)
            idx = input("\nSelect number to remove (or 'b' to go back): ")
            if idx.isdigit() and 0 < int(idx) <= len(unique_versions):
                remove_kernels(unique_versions[int(idx)-1], kernel_map, running)
        
        elif choice == '2':
            list_kernels(unique_versions, running)
            idx = input("\nSelect number to set as default (or 'b' to go back): ")
            if idx.isdigit() and 0 < int(idx) <= len(unique_versions):
                set_default_kernel(unique_versions[int(idx)-1])

        elif choice == 'q':
            break

if __name__ == "__main__":
    main()
