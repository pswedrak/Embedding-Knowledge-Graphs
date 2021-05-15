import sys
import gensim.downloader as api

from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QMainWindow

from gui.manage_files import FileManager
from gui.words_table import WordsTable


class MainWindow(QMainWindow):
    def __init__(self, ui_file):
        super(MainWindow, self).__init__()
        uic.loadUi(ui_file, self)

        self.wv_from_bin = api.load("glove-twitter-100")
        # self.wv_from_bin = None

        self.action_words = self.findChild(QtWidgets.QAction, 'action_words')
        self.action_words.triggered.connect(self.handle_action_words)

        self.action_files = self.findChild(QtWidgets.QAction, 'action_files')
        self.action_files.triggered.connect(self.handle_action_files)

        self.setCentralWidget(WordsTable(self, self.wv_from_bin))
        self.show()

    def handle_action_words(self):
        self.setCentralWidget(WordsTable(self, self.wv_from_bin))

    def handle_action_files(self):
        self.setCentralWidget(FileManager(self, self.wv_from_bin))


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow('design.ui')
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
