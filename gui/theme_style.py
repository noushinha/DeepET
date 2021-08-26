from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QSizePolicy

def set_theme_style(app):
    app.setStyle("Fusion")
    qp = QPalette()
    qp.setColor(QPalette.ButtonText, Qt.black)
    qp.setColor(QPalette.WindowText, Qt.white)
    qp.setColor(QPalette.Window, Qt.black)
    qp.setColor(QPalette.Button, Qt.gray)
    app.setPalette(qp)


def display_message(message, flag=True):
    msg = QMessageBox()
    if flag:
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle('Warning')

    else:
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle('Successful')
    msg.setText(message)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()
