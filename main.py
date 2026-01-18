"""
Main entry point for VietFood Detection Application
"""
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from PyQt5.QtWidgets import QApplication
from app.ui.main_window import MainWindow


def main():
    """
    Main function to start the application
    """
    # Create QApplication instance
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("VietFood Detection")
    app.setOrganizationName("VietFood Detection Team")
    app.setApplicationVersion("1.0.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
