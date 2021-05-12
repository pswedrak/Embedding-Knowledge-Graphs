from PyQt5.QtWidgets import QWidget
from PyQt5 import uic


class FileManager(QWidget):
    def __init__(self, parent):
        super(FileManager, self).__init__()
        uic.loadUi('files.ui', self)
        self.parent = parent
