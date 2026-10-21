"""PyQt5 桌面端动物识别界面。"""
from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


def _setup_qt_plugin_path():
    import PyQt5

    candidates = []
    try:
        from PyQt5.QtCore import QLibraryInfo

        info_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        if info_path:
            candidates.append(info_path)
    except Exception:
        pass

    pyqt_root = os.path.dirname(PyQt5.__file__)
    candidates.extend([
        os.path.join(pyqt_root, "Qt5", "plugins"),
        os.path.join(pyqt_root, "Qt", "plugins"),
        os.path.join(pyqt_root, "plugins"),
    ])

    for plugin_path in candidates:
        qwindows = os.path.join(plugin_path, "platforms", "qwindows.dll")
        if os.path.isfile(qwindows):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
            os.environ.setdefault("QT_PLUGIN_PATH", plugin_path)
            return

    raise RuntimeError(
        "未找到 Qt windows 平台插件（qwindows.dll）。\n"
        "请重装: pip install --force-reinstall PyQt5 PyQt5-Qt5"
    )


_setup_qt_plugin_path()

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from config import CLASS_NAMES, NUM_CLASSES, ROOT, get_class_description
from predict import load_model, predict

HISTORY_DIR = ROOT / "images" / "history"
BACKGROUND_PATH = ROOT / "background.jpg"

APP_STYLE = """
QWidget#Root {
    background: transparent;
    color: #f3eee4;
    font-family: "Microsoft YaHei UI";
}
QLabel#Brand {
    color: #f3eee4;
    font-size: 42px;
    font-weight: 700;
    letter-spacing: 1px;
    background: transparent;
}
QLabel#Sub {
    color: #d7e4db;
    font-size: 18px;
    background: transparent;
}
QLabel#Preview {
    background-color: #ffffff;
    border: 2px solid #ffffff;
    border-radius: 8px;
    min-width: 560px;
    min-height: 520px;
    font-size: 20px;
    color: #4a5560;
}
QLabel#ResultTitle, QLabel#HistoryTitle {
    color: #e7c56a;
    font-size: 18px;
    background: transparent;
}
QLabel#ResultName {
    color: #f4d38a;
    font-size: 56px;
    font-weight: 700;
    background: transparent;
}
QLabel#ResultConf {
    color: #f3eee4;
    font-size: 24px;
    background: transparent;
}
QLabel#ResultDesc {
    color: #2c3338;
    font-size: 17px;
    line-height: 1.45;
    background: #ffffff;
    border: 1px solid #ffffff;
    border-radius: 8px;
    padding: 10px;
}
QLabel#Hint {
    color: #c5d2c8;
    font-size: 15px;
    background: transparent;
}
QPushButton#Upload {
    background-color: #e7c56a;
    color: #10241c;
    border: 0;
    border-radius: 8px;
    padding: 14px 28px;
    font-size: 20px;
    font-weight: 700;
    min-width: 160px;
}
QPushButton#Upload:hover { background-color: #f0d484; }
QPushButton#Upload:disabled { background-color: #8a7a4a; color: #3a3420; }
QPushButton#Ghost {
    background-color: transparent;
    color: #ffffff;
    border: 1px solid #ffffff;
    border-radius: 8px;
    padding: 12px 18px;
    font-size: 16px;
    font-weight: 600;
}
QPushButton#Ghost:hover { background-color: rgba(255,255,255,0.12); }
QPushButton#Danger {
    background-color: transparent;
    color: #ffb4a8;
    border: 1px solid #ffb4a8;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 15px;
    font-weight: 600;
}
QPushButton#Danger:hover { background-color: rgba(255, 120, 100, 0.18); }
QListWidget {
    background: #ffffff;
    border: 2px solid #ffffff;
    border-radius: 8px;
    padding: 6px;
    font-size: 16px;
    outline: 0;
    color: #1f2933;
}
QListWidget::item {
    padding: 10px 8px;
    border-bottom: 1px solid #e5e7eb;
    color: #1f2933;
}
QListWidget::item:selected {
    background: #e8f0ec;
    color: #10241c;
}
"""


class PredictWorker(QThread):
    finished_ok = pyqtSignal(str, float)
    finished_err = pyqtSignal(str)

    def __init__(self, img_path: str, parent=None):
        super().__init__(parent)
        self.img_path = img_path

    def run(self):
        try:
            _, name, conf = predict(self.img_path)
            self.finished_ok.emit(name, conf)
        except Exception as exc:
            self.finished_err.emit(str(exc))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("Root")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowIcon(QIcon("images/logo.jpg"))
        self.setWindowTitle("动物识别系统")
        self.to_predict_name = ""
        self._worker = None
        self._history = []
        self._bg = QPixmap(str(BACKGROUND_PATH)) if BACKGROUND_PATH.exists() else QPixmap()
        self.resize(1400, 900)
        self.setMinimumSize(1200, 800)

        HISTORY_DIR.mkdir(parents=True, exist_ok=True)

        try:
            load_model()
        except FileNotFoundError as exc:
            QMessageBox.warning(self, "模型未找到", str(exc))

        self._build_ui()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        if not self._bg.isNull():
            scaled = self._bg.scaled(
                self.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            x = (scaled.width() - self.width()) // 2
            y = (scaled.height() - self.height()) // 2
            painter.drawPixmap(0, 0, scaled, x, y, self.width(), self.height())
            painter.fillRect(self.rect(), QColor(10, 22, 16, 150))
        else:
            painter.fillRect(self.rect(), QColor(16, 36, 28))
        super().paintEvent(event)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(40, 36, 40, 36)
        root.setSpacing(36)

        left = QVBoxLayout()
        left.setSpacing(16)

        brand = QLabel("动物识别")
        brand.setObjectName("Brand")
        sub = QLabel(
            f"ResNet18 · Animals-10 · 共 {NUM_CLASSES} 类"
            "（整图单类别识别，不支持同图多种动物）"
        )
        sub.setObjectName("Sub")
        sub.setWordWrap(True)
        left.addWidget(brand)
        left.addWidget(sub)

        self.img_label = QLabel("请上传一张动物图片")
        self.img_label.setObjectName("Preview")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.img_label, 1)

        btns = QHBoxLayout()
        btns.setSpacing(12)
        self.btn_upload = QPushButton("上传并识别")
        self.btn_upload.setObjectName("Upload")
        self.btn_upload.clicked.connect(self.change_img)
        self.btn_clear = QPushButton("清除当前识别")
        self.btn_clear.setObjectName("Ghost")
        self.btn_clear.clicked.connect(self.clear_current)
        btns.addWidget(self.btn_upload)
        btns.addWidget(self.btn_clear)
        btns.addStretch()
        left.addLayout(btns)

        right = QVBoxLayout()
        right.setSpacing(12)

        title = QLabel("识别结果")
        title.setObjectName("ResultTitle")
        self.result_name = QLabel("—")
        self.result_name.setObjectName("ResultName")
        self.result_conf = QLabel("上传图片后自动识别")
        self.result_conf.setObjectName("ResultConf")
        self.result_conf.setWordWrap(True)
        self.result_desc = QLabel("")
        self.result_desc.setObjectName("ResultDesc")
        self.result_desc.setWordWrap(True)

        hint = QLabel("可识别：" + "、".join(CLASS_NAMES))
        hint.setObjectName("Hint")
        hint.setWordWrap(True)

        hist_title = QLabel("识别历史（可多选）")
        hist_title.setObjectName("HistoryTitle")
        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(220)
        self.history_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.history_list.itemClicked.connect(self._on_history_clicked)

        hist_btns = QHBoxLayout()
        hist_btns.setSpacing(8)
        self.btn_del_selected = QPushButton("删除选中")
        self.btn_del_selected.setObjectName("Danger")
        self.btn_del_selected.clicked.connect(self.delete_selected_history)
        self.btn_del_all = QPushButton("一键清空")
        self.btn_del_all.setObjectName("Danger")
        self.btn_del_all.clicked.connect(self.clear_all_history)
        hist_btns.addWidget(self.btn_del_selected)
        hist_btns.addWidget(self.btn_del_all)
        hist_btns.addStretch()

        right.addWidget(title)
        right.addWidget(self.result_name)
        right.addWidget(self.result_conf)
        right.addWidget(self.result_desc)
        right.addSpacing(8)
        right.addWidget(hint)
        right.addSpacing(16)
        right.addWidget(hist_title)
        right.addWidget(self.history_list, 1)
        right.addLayout(hist_btns)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

    def _refresh_preview(self, img_path: str):
        pix = QPixmap(img_path)
        if pix.isNull():
            self.img_label.setText("无法读取图片")
            return
        w = max(self.img_label.width(), 560)
        h = max(self.img_label.height(), 480)
        scaled = pix.scaled(w - 20, h - 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()
        if self.to_predict_name and os.path.exists(self.to_predict_name):
            self._refresh_preview(self.to_predict_name)

    def _save_history_image(self, src_path: str) -> str:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(src_path).suffix or ".jpg"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dst = HISTORY_DIR / f"{stamp}{suffix}"
        shutil.copy2(src_path, dst)
        return str(dst)

    def _remove_history_file(self, path: str):
        try:
            p = Path(path)
            if p.exists() and HISTORY_DIR in p.resolve().parents:
                p.unlink()
        except Exception:
            pass

    def clear_current(self):
        self.to_predict_name = ""
        self.img_label.clear()
        self.img_label.setText("请上传一张动物图片")
        self.result_name.setText("—")
        self.result_conf.setText("上传图片后自动识别")
        self.result_desc.setText("")
        self.history_list.clearSelection()

    def change_img(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "Image files(*.jpg *.png *.jpeg *.bmp *.webp)",
        )
        if not path:
            return

        suffix = path.rsplit(".", 1)[-1]
        os.makedirs("images", exist_ok=True)
        target = f"images/tmp_up.{suffix}"
        shutil.copy(path, target)
        self.to_predict_name = target
        self._refresh_preview(target)
        self.result_name.setText("识别中…")
        self.result_conf.setText("")
        self.result_desc.setText("")
        self._start_predict()

    def _start_predict(self):
        if not self.to_predict_name or not os.path.exists(self.to_predict_name):
            return
        if self._worker and self._worker.isRunning():
            return

        self.btn_upload.setEnabled(False)
        self._worker = PredictWorker(self.to_predict_name, self)
        self._worker.finished_ok.connect(self._on_ok)
        self._worker.finished_err.connect(self._on_err)
        self._worker.finished.connect(lambda: self.btn_upload.setEnabled(True))
        self._worker.start()

    def _on_ok(self, name: str, conf: float):
        desc = get_class_description(name)
        self.result_name.setText(name)
        self.result_conf.setText(f"置信度  {conf * 100:.1f}%")
        self.result_desc.setText(desc)

        history_path = self._save_history_image(self.to_predict_name)
        stamp = datetime.now().strftime("%H:%M:%S")
        record = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "time": stamp,
            "name": name,
            "conf": conf,
            "desc": desc,
            "path": history_path,
        }
        self._history.insert(0, record)
        self._history = self._history[:50]
        self._refresh_history()

    def _refresh_history(self):
        self.history_list.clear()
        for item in self._history:
            text = f"{item['time']}  {item['name']}  {item['conf'] * 100:.1f}%"
            row = QListWidgetItem(text)
            row.setData(Qt.UserRole, item)
            self.history_list.addItem(row)

    def _on_history_clicked(self, row: QListWidgetItem):
        item = row.data(Qt.UserRole)
        if not item:
            return
        self.result_name.setText(item["name"])
        self.result_conf.setText(f"置信度  {item['conf'] * 100:.1f}%")
        self.result_desc.setText(item["desc"])
        path = item.get("path")
        if path and os.path.exists(path):
            self.to_predict_name = path
            self._refresh_preview(path)
        else:
            self.img_label.setText("历史图片已丢失")

    def delete_selected_history(self):
        selected = self.history_list.selectedItems()
        if not selected:
            QMessageBox.information(self, "提示", "请先选中要删除的历史记录（可按住 Ctrl 多选）")
            return

        ids = {item.data(Qt.UserRole).get("id") for item in selected if item.data(Qt.UserRole)}
        keep = []
        for rec in self._history:
            if rec.get("id") in ids:
                self._remove_history_file(rec.get("path", ""))
            else:
                keep.append(rec)
        self._history = keep
        self._refresh_history()

    def clear_all_history(self):
        if not self._history:
            return
        reply = QMessageBox.question(
            self,
            "确认清空",
            "确定清空全部识别历史吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        for rec in self._history:
            self._remove_history_file(rec.get("path", ""))
        self._history = []
        self._refresh_history()

    def _on_err(self, message: str):
        self.result_name.setText("失败")
        self.result_conf.setText(message)
        self.result_desc.setText("")
        QMessageBox.critical(self, "识别失败", message)

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "退出",
            "是否要退出程序？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 14))
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
