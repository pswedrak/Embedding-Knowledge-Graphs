import sys

from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow


class Main(QMainWindow):
    def __init__(self, ui_file):
        super(Main, self).__init__()
        uic.loadUi(ui_file, self)

        self.table = self.findChild(QtWidgets.QTableWidget, 'table_widget')

        self.add_row_button = self.findChild(QtWidgets.QPushButton, 'add_row_button')
        self.add_row_button.clicked.connect(self.add_row)

        self.delete_row_button = self.findChild(QtWidgets.QPushButton, 'delete_row_button')
        self.delete_row_button.clicked.connect(self.delete_row)

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


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = Main('design.ui')
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
