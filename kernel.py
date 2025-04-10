#!/usr/bin/env python3

import subprocess
import os

def get_running_kernel():
    """Returns the version of the currently running kernel."""
    result = subprocess.run(['uname', '-r'], capture_output=True, text=True, check=True)
    return result.stdout.strip()

def get_installed_kernels():
    """Returns a list of installed kernel package names."""
    result = subprocess.run(['rpm', '-qa', 'kernel*'], capture_output=True, text=True, check=True)
    return result.stdout.strip().split('\n')

def list_kernels_with_numbers(kernels, running_kernel):
    """Prints a numbered list of kernel versions, marking the running kernel with '*'."""
    print("Installed Kernels:")
    kernel_versions = {}
    count = 1
    for package in kernels:
        parts = package.split('-')
        version_parts = []
        for part in parts:
            if part.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')):
                version_parts.append(part)
        if version_parts:
            version = '-'.join(version_parts)
            if version not in kernel_versions:
                kernel_versions[version] = []
            kernel_versions[version].append(package)

    unique_versions = sorted(kernel_versions.keys())
    for i, version in enumerate(unique_versions):
        running_mark = "*" if version == running_kernel else ""
        print(f"{i+1}. Kernel Version: {version} {running_mark}")
        for pkg in kernel_versions[version]:
            print(f"   - {pkg}")
    return unique_versions, kernel_versions

def get_kernel_packages_to_remove(selected_version, kernel_versions):
    """Returns a list of package names to remove for the selected version."""
    if selected_version in kernel_versions:
        return kernel_versions[selected_version]
    return []

def remove_kernels(packages_to_remove, removing_running_kernel):
    """Removes the specified kernel packages using dnf with confirmation and running kernel check."""
    if not packages_to_remove:
        print("No packages selected for removal.")
        return

    print("Packages to be removed:")
    for pkg in packages_to_remove:
        print(f"- {pkg}")

    if removing_running_kernel:
        print("\nWARNING: You are about to remove the currently running kernel.")
        print("This will require a reboot, and you will need to select a different kernel from the GRUB menu if a new default is not set.")

    confirmation = input("Are you sure you want to proceed with the removal? (y/N): ").lower()
    if confirmation == 'y':
        try:
            subprocess.run(['sudo', 'dnf', 'remove'] + packages_to_remove, check=True)
            print("Kernel packages removed successfully.")
            update_grub()
        except subprocess.CalledProcessError as e:
            print(f"Error removing kernel packages: {e}")
    else:
        print("Removal cancelled.")

def update_grub():
    """Updates the GRUB configuration."""
    try:
        subprocess.run(['sudo', 'grub2-mkconfig', '-o', '/boot/grub2/grub.cfg'], check=True)
        print("GRUB configuration updated.")
    except subprocess.CalledProcessError as e:
        print(f"Error updating GRUB configuration: {e}")

def set_default_kernel(kernel_version):
    """Sets the specified kernel version as the default GRUB kernel."""
    try:
        subprocess.run(['sudo', 'grubby', '--set-default', f'/boot/vmlinuz-{kernel_version}'], check=True)
        print(f"Default boot kernel set to: {kernel_version}")
    except subprocess.CalledProcessError as e:
        print(f"Error setting default kernel: {e}")

def change_default_kernel_prompt(unique_versions, running_kernel):
    """Prompts the user to select a new default kernel."""
    print("\nAvailable kernels to set as default:")
    for i, version in enumerate(unique_versions):
        print(f"{i+1}. {version}")

    while True:
        default_choice = input("Enter the number of the kernel version to set as default (or 's' to skip): ")
        if default_choice.lower() == 's':
            break
        try:
            default_index = int(default_choice) - 1
            if 0 <= default_index < len(unique_versions):
                set_default_kernel(unique_versions[default_index])
                break
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")

if __name__ == "__main__":
    running_kernel = get_running_kernel()
    installed_kernels = get_installed_kernels()

    if not installed_kernels:
        print("No kernel packages found.")
        exit()

    unique_versions, kernel_versions_map = list_kernels_with_numbers(installed_kernels, running_kernel)

    while True:
        print("\nOptions:")
        print("1. Remove a kernel")
        print("2. Change default boot kernel")
        print("q. Quit")

        choice = input("Enter your choice: ").lower()

        if choice == '1':
            while True:
                print("\nSelect kernel to remove:")
                unique_versions_remove, _ = list_kernels_with_numbers(installed_kernels, running_kernel)
                try:
                    remove_choice = input("Enter the number of the kernel version to remove (or 'q' to go back): ")
                    if remove_choice.lower() == 'q':
                        break
                    selection_index = int(remove_choice) - 1
                    if 0 <= selection_index < len(unique_versions_remove):
                        selected_version = unique_versions_remove[selection_index]
                        packages_to_remove = get_kernel_packages_to_remove(selected_version, kernel_versions_map)
                        removing_running_kernel = selected_version == running_kernel
                        remove_kernels(packages_to_remove, removing_running_kernel)
                        break
                    else:
                        print("Invalid selection. Please enter a number from the list.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'q'.")

        elif choice == '2':
            print("\nSelect kernel to set as default:")
            change_default_kernel_prompt(unique_versions, running_kernel)

        elif choice == 'q':
            break

        else:
            print("Invalid choice. Please enter 1, 2, or q.")
