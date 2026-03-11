' launch_gui.vbs — Запуск GUI без консольного окна (как .exe)
' Можно поместить этот файл на Рабочий стол

Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Определяем путь к папке проекта (где лежит этот файл)
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Используем pythonw.exe (без консоли) если есть, иначе python.exe через bat
pythonw = scriptDir & "\.venv\Scripts\pythonw.exe"
appPy = scriptDir & "\app.py"

WshShell.CurrentDirectory = scriptDir

If fso.FileExists(pythonw) Then
    ' pythonw запускает GUI без консольного окна
    WshShell.Run """" & pythonw & """ """ & appPy & """", 0, False
Else
    ' Fallback: запускаем через bat
    WshShell.Run """" & scriptDir & "\launch_gui.bat""", 0, False
End If
