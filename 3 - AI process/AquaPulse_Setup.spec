# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None
script_dir = os.path.abspath(r"C:\Users\parsa\Desktop\Code\3 - AI process")

a = Analysis(
    [os.path.join(script_dir, 'AquaPulse_Setup.py')],
    pathex=[script_dir],
    binaries=[],
    datas=[],
    hiddenimports=['winreg', 'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox', 'urllib.request'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AquaPulse_Setup',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
