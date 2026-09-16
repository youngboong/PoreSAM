; Compile with Inno Setup 6 after building dist/PoreSAM.
#define AppVersion "0.1.0"
[Setup]
AppId={{AE406D27-A102-4874-A44B-CA279B669D42}
AppName=PoreSAM
AppVersion={#AppVersion}
AppPublisher=PoreSAM
DefaultDirName={localappdata}\Programs\PoreSAM
DefaultGroupName=PoreSAM
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=PoreSAM-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\PoreSAM.exe
[Languages]
Name: "default"; MessagesFile: "compiler:Default.isl"
[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked
[Files]
Source: "..\dist\PoreSAM\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{group}\PoreSAM"; Filename: "{app}\PoreSAM.exe"
Name: "{autodesktop}\PoreSAM"; Filename: "{app}\PoreSAM.exe"; Tasks: desktopicon
[Run]
Filename: "{app}\PoreSAM.exe"; Description: "Launch PoreSAM"; Flags: nowait postinstall skipifsilent
