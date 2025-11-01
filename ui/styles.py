class ModernStyles:
    @staticmethod
    def get_stylesheet():
        return """
        QMainWindow {
            background-color: #1e1e2e;
        }

        QGroupBox {
            background-color: #2d2d44;
            border: 2px solid #3d3d5c;
            border-radius: 10px;
            margin-top: 15px;
            padding: 15px;
            font-size: 14px;
            font-weight: bold;
            color: #ffffff;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 5px 10px;
            background-color: #3d3d5c;
            border-radius: 5px;
            color: #89b4fa;
        }

        QPushButton {
            background-color: #89b4fa;
            color: #1e1e2e;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 13px;
            font-weight: bold;
            min-height: 40px;
        }

        QPushButton:hover {
            background-color: #a6c8ff;
        }

        QPushButton:pressed {
            background-color: #6c9ddb;
        }

        QPushButton:disabled {
            background-color: #45475a;
            color: #6c6f85;
        }

        QPushButton#criticalBtn {
            background-color: #f38ba8;
        }

        QPushButton#criticalBtn:hover {
            background-color: #f5a3bb;
        }

        QTextEdit {
            background-color: #181825;
            color: #cdd6f4;
            border: 2px solid #3d3d5c;
            border-radius: 8px;
            padding: 10px;
            font-size: 13px;
            font-family: 'Consolas', 'Courier New', monospace;
        }

        QLabel {
            color: #cdd6f4;
            font-size: 13px;
        }

        QLabel#titleLabel {
            font-size: 24px;
            font-weight: bold;
            color: #89b4fa;
            padding: 10px;
        }

        QLabel#statsLabel {
            font-size: 16px;
            color: #a6e3a1;
            font-weight: bold;
        }

        QComboBox {
            background-color: #2d2d44;
            color: #cdd6f4;
            border: 2px solid #3d3d5c;
            border-radius: 6px;
            padding: 8px;
            min-height: 35px;
        }

        QComboBox:hover {
            border-color: #89b4fa;
        }

        QComboBox::drop-down {
            border: none;
            width: 30px;
        }

        QComboBox::down-arrow {
            image: url(none);
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 8px solid #89b4fa;
        }

        QProgressBar {
            border: 2px solid #3d3d5c;
            border-radius: 8px;
            background-color: #181825;
            text-align: center;
            color: #cdd6f4;
            font-weight: bold;
        }

        QProgressBar::chunk {
            background-color: #89b4fa;
            border-radius: 6px;
        }

        QScrollBar:vertical {
            background-color: #181825;
            width: 12px;
            border-radius: 6px;
        }

        QScrollBar::handle:vertical {
            background-color: #45475a;
            border-radius: 6px;
            min-height: 20px;
        }

        QScrollBar::handle:vertical:hover {
            background-color: #585b70;
        }
        """