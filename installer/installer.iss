; Inno Setup script for Flamingo Stitcher (Windows installer)
;
; Compile (from the repo root, after PyInstaller has produced dist\FlamingoStitcher\):
;     iscc /DAppVersion=0.1.0 installer\installer.iss
;
; Produces: installer\Output\FlamingoStitcher-Setup-<AppVersion>.exe
; Per-user install (no admin rights required).

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define AppName "Flamingo Stitcher"
#define AppPublisher "UW-Madison LOCI"
#define AppExeName "FlamingoStitcher.exe"

[Setup]
AppId={{8F3C5A1E-2B7D-4E9A-9C1F-FLAMINGOSTITCH}}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/uw-loci/flamingo-stitcher
DefaultDirName={localappdata}\Programs\FlamingoStitcher
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=Output
OutputBaseFilename=FlamingoStitcher-Setup-{#AppVersion}
SetupIconFile=..\src\flamingo_stitcher\gui\flamingo_icon.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[InstallDelete]
; Clear the previous bundle BEFORE copying the new one.
;
; [Files] only adds and overwrites — it never removes a file the new bundle
; no longer contains. Python package metadata lives in VERSIONED directory
; names (multiview_stitcher-0.1.44.dist-info), so upgrading in place leaves
; every old release's dist-info sitting alongside the new one. importlib.
; metadata then answers with whichever it happens to find first, which is how
; a rig running the 0.1.59 we bundle reported multiview-stitcher 0.1.44 and
; tripped the correctness guard on 2026-08-08.
;
; _internal is pure PyInstaller payload — the interpreter, the stdlib, and
; every third-party package — all of it rewritten by this install. Nothing
; user-owned lives there (settings are in the registry via QSettings), so
; wiping it is safe and makes each upgrade a clean slate.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
; The entire PyInstaller one-folder bundle.
Source: "..\dist\FlamingoStitcher\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent
