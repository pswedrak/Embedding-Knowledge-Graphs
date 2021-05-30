from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt5 import uic
from qtpy import QtWidgets

from semantic_similarity.graph_creator import build_graph
from semantic_similarity.main import compute_similarity


class FileManager(QWidget):
    def __init__(self, parent, wv_from_bin):
        super(FileManager, self).__init__()
        self.ui = uic.loadUi('files.ui', self)
        self.parent = parent
        self.actions_file_name = None
        self.wv_from_bin = wv_from_bin

        self.calculate_button = self.findChild(QtWidgets.QPushButton, 'calculate_button')
        self.calculate_button.clicked.connect(self.calculate)

        self.choose_file_button = self.findChild(QtWidgets.QPushButton, 'choose_file_button')
        self.choose_file_button.clicked.connect(self.choose_file)

        self.selected_file_text = self.findChild(QtWidgets.QTextBrowser, 'selected_file_text')

    @pyqtSlot()
    def calculate(self):
        with open(self.selected_file_text.toPlainText()) as file:
            with open('report.txt', 'w') as report:
                for line in file.readlines():
                    words = line.split(" ")
                    word1 = words[0]
                    word2 = words[1][:-1]

                    g, max_depth, root, dist1, dist2, lch_concept, max_lch_path_length = build_graph(word1, word2)
                    if max_lch_path_length != 0:
                        alpha_coef = (dist1 - dist2) / max_lch_path_length
                    else:
                        alpha_coef = 0
                    sim = compute_similarity(self.wv_from_bin, word1, word2)

                    report.write(word1 + ' ' + word2 + ' ' + str(alpha_coef) + ' ' + str(10*sim))
                    report.write('\n')

        self.show_success_popup()

    @pyqtSlot()
    def choose_file(self):
        self.actions_file_name, _ = QFileDialog.getOpenFileName()
        if self.actions_file_name:
            self.selected_file_text.setText(self.actions_file_name)

    @staticmethod
    def show_success_popup():
        msg = QMessageBox()
        msg.setWindowTitle("Success")
        msg.setText("The report has been successfully created!")
        _ = msg.exec_()
