import sys
import gensim.downloader as api

from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QTableWidgetItem

from semantic_similarity.graph_creator import build_graph
from semantic_similarity.graph_drawer import draw_graph
from semantic_similarity.main import compute_similarity


class Main(QMainWindow):
    def __init__(self, ui_file):
        super(Main, self).__init__()
        uic.loadUi(ui_file, self)

        self.wv_from_bin = api.load("glove-twitter-100")

        self.table = self.findChild(QtWidgets.QTableWidget, 'table_widget')

        self.add_row_button = self.findChild(QtWidgets.QPushButton, 'add_row_button')
        self.add_row_button.clicked.connect(self.add_row)

        self.delete_row_button = self.findChild(QtWidgets.QPushButton, 'delete_row_button')
        self.delete_row_button.clicked.connect(self.delete_row)

        self.calculate_button = self.findChild(QtWidgets.QPushButton, 'calculate_button')
        self.calculate_button.clicked.connect(self.calculate)

        self.visualize_checkbox = self.findChild(QtWidgets.QCheckBox, 'visualize_checkbox')

        self.show()

    @pyqtSlot()
    def add_row(self):
        row_count = self.table.rowCount()
        self.table.setRowCount(row_count + 1)

    @pyqtSlot()
    def delete_row(self):
        row_count = self.table.rowCount()
        if row_count > 0:
            self.table.setRowCount(row_count - 1)

    @pyqtSlot()
    def calculate(self):
        row_count = self.table.rowCount()
        if row_count > 0:
            self.table.setRowCount(row_count - 1)

    @pyqtSlot()
    def calculate(self):
        for row in range(self.table.rowCount()):
            word1_item = self.table.item(row, 0)
            word2_item = self.table.item(row, 1)
            if (word1_item is not None) & (word2_item is not None):
                word1 = word1_item.text()
                word2 = word2_item.text()
                g, max_depth, root, dist1, dist2, lch_concept, max_lch_path_length = build_graph(word1, word2)

                if self.visualize_checkbox.isChecked():
                    draw_graph(g, word1, word2, dist1, dist2, lch_concept, max_lch_path_length)

                sim = compute_similarity(self.wv_from_bin, word1, word2)
                self.table.setItem(row, 2, QTableWidgetItem(str(round(10*sim, 2))))

                alpha_coef = (dist1 - dist2)/max_lch_path_length
                self.table.setItem(row, 3, QTableWidgetItem(str(alpha_coef)))


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Main('design.ui')
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
