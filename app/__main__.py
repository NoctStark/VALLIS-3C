from __future__ import annotations
import sys
from pathlib import Path
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication
from .main_window import MainWindow

ROOT = Path(__file__).resolve().parents[1]
ICON = ROOT / "assets" / "icons" / "VALLIS-3C.ico"

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VALLIS-3C")
    app.setOrganizationName("JDCA-IIUNAM")
    if ICON.is_file():
        app.setWindowIcon(QIcon(str(ICON)))
    w = MainWindow()
    w.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())
