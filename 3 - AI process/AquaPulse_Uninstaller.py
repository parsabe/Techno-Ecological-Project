import os
import sys
import shutil
import subprocess
import winreg
import time

def remove_registry_keys():
    """Removes AquaPulse from Windows Control Panel (Add or Remove Programs)."""
    reg_paths = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\AquaPulse"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\AquaPulse")
    ]
    for root, key_path in reg_paths:
        try:
            winreg.DeleteKey(root, key_path)
            print(f"✅ Removed registry key: {key_path}")
        except Exception:
            pass

def remove_shortcuts():
    """Removes Desktop and Start Menu shortcuts."""
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    start_menu = os.path.join(os.environ.get("APPDATA", r"C:\Users\Public"), r"Microsoft\Windows\Start Menu\Programs")
    
    shortcuts = [
        os.path.join(desktop, "AquaPulse AI Vision.lnk"),
        os.path.join(desktop, "AquaPulse.lnk"),
        os.path.join(start_menu, "AquaPulse AI Vision.lnk"),
        os.path.join(start_menu, "AquaPulse.lnk")
    ]
    for sc in shortcuts:
        if os.path.exists(sc):
            try:
                os.remove(sc)
                print(f"✅ Removed shortcut: {sc}")
            except Exception as e:
                print(f"Notice removing shortcut {sc}: {e}")

def run_gui_uninstall():
    import tkinter as tk
    from tkinter import messagebox
    
    root = tk.Tk()
    root.title("AquaPulse System Uninstaller")
    root.geometry("480x220")
    root.resizable(False, False)
    root.attributes('-topmost', True)
    
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = (screen_w - 480) // 2
    y = (screen_h - 220) // 2
    root.geometry(f"480x220+{x}+{y}")
    root.configure(bg="#1e222d")
    
    header = tk.Label(root, text="🗑️ AquaPulse System Uninstaller", font=("Segoe UI", 14, "bold"), fg="#ef4444", bg="#1e222d")
    header.pack(pady=(20, 5))
    
    msg = tk.Label(root, text="Are you sure you want to remove AquaPulse AI Neural Vision\nand all associated components from C:\\AquaPulse?", font=("Segoe UI", 10), fg="#e2e8f0", bg="#1e222d")
    msg.pack(pady=10)
    
    btn_frame = tk.Frame(root, bg="#1e222d")
    btn_frame.pack(pady=15)
    
    def confirm_uninstall():
        root.withdraw()
        perform_uninstall()
        messagebox.showinfo("Uninstall Complete", "AquaPulse System has been successfully uninstalled from your computer.")
        root.destroy()
        sys.exit(0)
        
    def cancel_uninstall():
        root.destroy()
        sys.exit(0)
        
    yes_btn = tk.Button(btn_frame, text="Yes, Uninstall", font=("Segoe UI", 10, "bold"), bg="#dc2626", fg="white", activebackground="#b91c1c", relief="flat", padx=15, pady=6, cursor="hand2", command=confirm_uninstall)
    yes_btn.pack(side="left", padx=10)
    
    no_btn = tk.Button(btn_frame, text="Cancel", font=("Segoe UI", 10), bg="#475569", fg="white", activebackground="#334155", relief="flat", padx=15, pady=6, cursor="hand2", command=cancel_uninstall)
    no_btn.pack(side="left", padx=10)
    
    root.mainloop()

def perform_uninstall():
    remove_registry_keys()
    remove_shortcuts()
    
    install_target = r"C:\AquaPulse"
    
    # Schedule post-exit cleanup of install directory
    cleanup_cmd = f'cmd.exe /c "timeout /t 2 /nobreak >NUL & rmdir /s /q \"{install_target}\""'
    try:
        subprocess.Popen(cleanup_cmd, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
    except Exception as e:
        print(f"Notice scheduling directory removal: {e}")

def main():
    is_silent = "/quiet" in sys.argv or "/silent" in sys.argv or "/S" in sys.argv
    if is_silent:
        perform_uninstall()
    else:
        try:
            run_gui_uninstall()
        except Exception:
            perform_uninstall()

if __name__ == "__main__":
    main()
