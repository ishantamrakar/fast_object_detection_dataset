from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton, QLabel, QMessageBox
from PyQt6.QtCore import Qt
import sys

# Predefined class mapping
CLASS_NAME_TO_INT = {
    "bottle": 1,
    "champagne": 2,
    "espresso": 3,
    "fork": 4,
    "hammer": 5,
    "knife_bread": 6,
    "knife_cleaver": 7,
    "knife_coreing": 8,
    "knife_paring": 9,
    "knife_steak": 10,
    "ladle": 11,
    "masher": 12,
    "mug": 13,
    "pliers": 14,
    "screwdriver": 15,
    "shot": 16,
    "spatula": 17,
    "spoon": 18,
    "whisk": 19,
    "wine": 20,
    "wrench": 21,
}

class LabelSelector(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Object Label")
        self.selected_label = None
        self.class_mapping = CLASS_NAME_TO_INT.copy()
        self.current_mapping = {}

        layout = QVBoxLayout()

        self.label = QLabel("Select or enter object label:")
        layout.addWidget(self.label)

        self.combo_box = QComboBox()
        self.combo_box.setEditable(True)
        self.combo_box.addItems(list(self.class_mapping.keys()))
        self.combo_box.setCurrentIndex(-1)
        self.combo_box.lineEdit().returnPressed.connect(self.add_label)
        layout.addWidget(self.combo_box)

        self.add_button = QPushButton("Select label for current object")
        self.add_button.clicked.connect(self.add_label)
        layout.addWidget(self.add_button)

        self.setLayout(layout)

    def add_label(self):
        text = self.combo_box.currentText().strip()
        if not text:
            QMessageBox.warning(self, "Error", "Please enter or select a label.")
            return
        # If new label, add it to the mapping
        if text not in self.class_mapping:
            new_id = max(self.class_mapping.values()) + 1
            self.class_mapping[text] = new_id
            self.combo_box.addItem(text)
            QMessageBox.information(self, "New Class Added", f"Added new class '{text}' with ID {new_id}")
            
        # Update current mapping by adding the selected label 
        self.current_mapping[text] = self.class_mapping[text]
        self.selected_label = text
        self.combo_box.setCurrentIndex(-1)
        self.combo_box.setEditText("")
        self.close()

_app = None
_selector = None

def get_label():
    global _app, _selector
    if _app is None:
        _app = QApplication(sys.argv)
    if _selector is None:
        _selector = LabelSelector()
    # Reset selected_label before showing
    _selector.selected_label = None
    _selector.show()
    _app.exec()
    return _selector.selected_label, _selector.current_mapping

# Example usage:
if __name__ == "__main__":
    label, mapping = get_label()
    print("Selected label:", label)
    print("Current mapping:", mapping)