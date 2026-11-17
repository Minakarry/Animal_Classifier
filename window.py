"""PyQt5 desktop animal recognition UI (English) with Top-3, uncertainty, Grad-CAM."""
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
        "Qt Windows platform plugin (qwindows.dll) not found.\n"
        "Please reinstall: pip install --force-reinstall PyQt5 PyQt5-Qt5"
    )


_setup_qt_plugin_path()

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from config import CLASS_NAMES, CONFIDENCE_THRESHOLD, NUM_CLASSES, ROOT
from predict import load_model, predict_full

HISTORY_DIR = ROOT / "images" / "history"
BACKGROUND_PATH = ROOT / "background.jpg"

APP_STYLE = """
QWidget#Root { background: transparent; color: #f3eee4; font-family: "Segoe UI", "Arial"; }
QLabel#Brand { color: #f3eee4; font-size: 40px; font-weight: 700; background: transparent; }
QLabel#Sub { color: #d7e4db; font-size: 16px; background: transparent; }
QLabel#Preview, QLabel#Cam {
    background-color: #ffffff; border: 2px solid #ffffff; border-radius: 8px;
    font-size: 15px; color: #4a5560; padding: 4px;
}
QLabel#ResultTitle, QLabel#HistoryTitle, QLabel#TopTitle, QLabel#CamTitle {
    color: #e7c56a; font-size: 17px; background: transparent;
}
QLabel#ResultName { color: #f4d38a; font-size: 48px; font-weight: 700; background: transparent; }
QLabel#ResultConf { color: #f3eee4; font-size: 20px; background: transparent; }
QLabel#Warn {
    color: #7a2e0e; font-size: 14px; background: #ffe2b8; border-radius: 8px; padding: 8px;
}
QLabel#ResultDesc {
    color: #2c3338; font-size: 15px; background: #ffffff; border: 1px solid #ffffff;
    border-radius: 8px; padding: 10px;
}
QLabel#Hint { color: #c5d2c8; font-size: 14px; background: transparent; }
QPushButton#Upload {
    background-color: #e7c56a; color: #10241c; border: 0; border-radius: 8px;
    padding: 12px 22px; font-size: 18px; font-weight: 700;
}
QPushButton#Upload:hover { background-color: #f0d484; }
QPushButton#Upload:disabled { background-color: #8a7a4a; color: #3a3420; }
QPushButton#Ghost {
    background-color: transparent; color: #ffffff; border: 1px solid #ffffff;
    border-radius: 8px; padding: 10px 16px; font-size: 15px; font-weight: 600;
}
QPushButton#Ghost:hover { background-color: rgba(255,255,255,0.12); }
QPushButton#Danger {
    background-color: transparent; color: #ffb4a8; border: 1px solid #ffb4a8;
    border-radius: 8px; padding: 8px 12px; font-size: 14px; font-weight: 600;
}
QPushButton#Danger:hover { background-color: rgba(255, 120, 100, 0.18); }
QListWidget {
    background: #ffffff; border: 2px solid #ffffff; border-radius: 8px;
    padding: 6px; font-size: 14px; outline: 0; color: #1f2933;
}
QListWidget::item { padding: 8px; border-bottom: 1px solid #e5e7eb; }
QListWidget::item:selected { background: #e8f0ec; color: #10241c; }
QProgressBar {
    border: 0; background: rgba(255,255,255,0.25); border-radius: 5px; max-height: 10px;
}
QProgressBar::chunk {
    border-radius: 5px;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1f4d3d, stop:1 #e7c56a);
}
QLabel#TopName { color: #f3eee4; font-size: 13px; background: transparent; }
"""


def pil_to_pixmap(pil_img) -> QPixmap:
    rgb = pil_img.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimg = QImage(data, rgb.width, rgb.height, 3 * rgb.width, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class PredictWorker(QThread):
    finished_ok = pyqtSignal(dict)
    finished_err = pyqtSignal(str)

    def __init__(self, img_path: str, parent=None):
        super().__init__(parent)
        self.img_path = img_path

    def run(self):
        try:
            result = predict_full(self.img_path, topk=3, with_cam=True)
            # QPixmap can't cross threads reliably; send PIL then convert in UI thread
            self.finished_ok.emit(result)
        except Exception as exc:
            self.finished_err.emit(str(exc))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("Root")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setWindowIcon(QIcon("images/logo.jpg"))
        self.setWindowTitle("Animal Recognition System")
        self.to_predict_name = ""
        self._worker = None
        self._history = []
        self._preview_src = QPixmap()
        self._cam_src = QPixmap()
        self._bg = QPixmap(str(BACKGROUND_PATH)) if BACKGROUND_PATH.exists() else QPixmap()
        self.resize(1480, 920)
        self.setMinimumSize(1280, 820)
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)

        try:
            load_model()
        except FileNotFoundError as exc:
            QMessageBox.warning(self, "Model Not Found", str(exc))

        self._build_ui()
        self._reset_image_frames()
        self._fit_history_height()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        if not self._bg.isNull():
            scaled = self._bg.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            x = (scaled.width() - self.width()) // 2
            y = (scaled.height() - self.height()) // 2
            painter.drawPixmap(0, 0, scaled, x, y, self.width(), self.height())
            painter.fillRect(self.rect(), QColor(10, 22, 16, 150))
        else:
            painter.fillRect(self.rect(), QColor(16, 36, 28))
        super().paintEvent(event)

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(24)

        left = QVBoxLayout()
        left.setSpacing(10)
        brand = QLabel("Animal Recognition")
        brand.setObjectName("Brand")
        sub = QLabel(
            f"ResNet18 · Animals-10 · {NUM_CLASSES} classes · Top-3 + Grad-CAM\n"
            f"Low-confidence warning if < {CONFIDENCE_THRESHOLD * 100:.0f}%"
        )
        sub.setObjectName("Sub")
        sub.setWordWrap(True)
        left.addWidget(brand)
        left.addWidget(sub)

        imgs = QHBoxLayout()
        imgs.setSpacing(12)
        self.img_label = QLabel("Please upload an animal image")
        self.img_label.setObjectName("Preview")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(280, 180)
        self.img_label.setMaximumHeight(420)
        self.img_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        cam_wrap = QVBoxLayout()
        cam_wrap.setSpacing(4)
        cam_title = QLabel("Grad-CAM")
        cam_title.setObjectName("CamTitle")
        self.cam_label = QLabel("Grad-CAM will appear here")
        self.cam_label.setObjectName("Cam")
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setMinimumSize(200, 150)
        self.cam_label.setMaximumHeight(420)
        self.cam_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        cam_wrap.addWidget(cam_title)
        cam_wrap.addWidget(self.cam_label, 0)
        cam_wrap.addStretch(1)

        imgs.addWidget(self.img_label, 3)
        imgs.addLayout(cam_wrap, 2)
        left.addLayout(imgs, 0)
        left.addStretch(1)

        btns = QHBoxLayout()
        self.btn_upload = QPushButton("Upload & Recognize")
        self.btn_upload.setObjectName("Upload")
        self.btn_upload.clicked.connect(self.change_img)
        self.btn_clear = QPushButton("Clear Current Result")
        self.btn_clear.setObjectName("Ghost")
        self.btn_clear.clicked.connect(self.clear_current)
        btns.addWidget(self.btn_upload)
        btns.addWidget(self.btn_clear)
        btns.addStretch()
        left.addLayout(btns)

        right = QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(self._label("Recognition Result", "ResultTitle"))
        self.result_name = QLabel("—")
        self.result_name.setObjectName("ResultName")
        self.result_conf = QLabel("Upload an image to recognize automatically")
        self.result_conf.setObjectName("ResultConf")
        self.result_conf.setWordWrap(True)
        self.warn_label = QLabel("")
        self.warn_label.setObjectName("Warn")
        self.warn_label.setWordWrap(True)
        self.warn_label.hide()
        self.result_desc = QLabel("")
        self.result_desc.setObjectName("ResultDesc")
        self.result_desc.setWordWrap(True)
        self.result_desc.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        right.addWidget(self.result_name)
        right.addWidget(self.result_conf)
        right.addWidget(self.warn_label)
        right.addWidget(self.result_desc)

        right.addWidget(self._label("Top-3 Predictions", "TopTitle"))
        self.topk_rows = []
        for _ in range(3):
            name = QLabel("-")
            name.setObjectName("TopName")
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            right.addWidget(name)
            right.addWidget(bar)
            self.topk_rows.append((name, bar))

        hint = QLabel("Supported: " + ", ".join(CLASS_NAMES))
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        right.addWidget(hint)

        right.addWidget(self._label("History (multi-select)", "HistoryTitle"))
        self.history_list = QListWidget()
        self.history_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.history_list.itemClicked.connect(self._on_history_clicked)
        self.history_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.history_list.setMinimumHeight(72)
        self.history_list.setMaximumHeight(220)
        right.addWidget(self.history_list, 0)

        hist_btns = QHBoxLayout()
        self.btn_del_selected = QPushButton("Delete Selected")
        self.btn_del_selected.setObjectName("Danger")
        self.btn_del_selected.clicked.connect(self.delete_selected_history)
        self.btn_del_all = QPushButton("Clear All")
        self.btn_del_all.setObjectName("Danger")
        self.btn_del_all.clicked.connect(self.clear_all_history)
        hist_btns.addWidget(self.btn_del_selected)
        hist_btns.addWidget(self.btn_del_all)
        hist_btns.addStretch()
        right.addLayout(hist_btns)
        right.addStretch(1)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

    def _label(self, text, obj):
        lab = QLabel(text)
        lab.setObjectName(obj)
        return lab

    def _preview_max_size(self, label: QLabel):
        """Max box size so landscape photos do not leave huge white padding."""
        if label is self.img_label:
            max_w = max(320, int(self.width() * 0.42))
            max_h = min(420, max(200, int(self.height() * 0.48)))
        else:
            max_w = max(220, int(self.width() * 0.28))
            max_h = min(420, max(180, int(self.height() * 0.48)))
        return max_w, max_h

    def _set_pixmap(self, label: QLabel, pix: QPixmap):
        if pix.isNull():
            return
        if label is self.img_label:
            self._preview_src = QPixmap(pix)
        elif label is self.cam_label:
            self._cam_src = QPixmap(pix)
        max_w, max_h = self._preview_max_size(label)
        scaled = pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        # Shrink white frame to the image (adaptive), not a tall empty panel.
        label.setFixedSize(scaled.width() + 8, scaled.height() + 8)

    def _reset_image_frames(self):
        self.img_label.setMinimumSize(280, 180)
        self.img_label.setMaximumHeight(420)
        self.img_label.setFixedSize(320, 200)
        self.cam_label.setMinimumSize(200, 150)
        self.cam_label.setMaximumHeight(420)
        self.cam_label.setFixedSize(240, 170)

    def _fit_history_height(self):
        n = self.history_list.count()
        if n <= 0:
            self.history_list.setFixedHeight(72)
            return
        row_h = max(28, self.history_list.sizeHintForRow(0))
        height = min(220, max(72, n * (row_h + 2) + 14))
        self.history_list.setFixedHeight(height)

    def _refresh_preview(self, img_path: str):
        pix = QPixmap(img_path)
        if pix.isNull():
            self.img_label.setText("Failed to load image")
            self._reset_image_frames()
            return
        self._set_pixmap(self.img_label, pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()
        if not self._preview_src.isNull():
            self._set_pixmap(self.img_label, self._preview_src)
        if not self._cam_src.isNull():
            self._set_pixmap(self.cam_label, self._cam_src)

    def clear_current(self):
        self.to_predict_name = ""
        self._preview_src = QPixmap()
        self._cam_src = QPixmap()
        self.img_label.clear()
        self.img_label.setText("Please upload an animal image")
        self.cam_label.clear()
        self.cam_label.setText("Grad-CAM will appear here")
        self._reset_image_frames()
        self.result_name.setText("—")
        self.result_conf.setText("Upload an image to recognize automatically")
        self.result_desc.setText("")
        self.warn_label.hide()
        for name, bar in self.topk_rows:
            name.setText("-")
            bar.setValue(0)
        self.history_list.clearSelection()
        self._fit_history_height()

    def change_img(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image files(*.jpg *.png *.jpeg *.bmp *.webp)"
        )
        if not path:
            return
        suffix = path.rsplit(".", 1)[-1]
        os.makedirs("images", exist_ok=True)
        target = f"images/tmp_up.{suffix}"
        shutil.copy(path, target)
        self.to_predict_name = target
        self._refresh_preview(target)
        self.result_name.setText("Recognizing…")
        self.result_conf.setText("")
        self.result_desc.setText("")
        self.warn_label.hide()
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

    def _on_ok(self, result: dict):
        top1 = result["top1"]
        self.result_name.setText(top1["name"])
        self.result_conf.setText(f"Confidence  {top1['confidence'] * 100:.1f}%")
        self.result_desc.setText(result.get("message", ""))
        # Keep description from config if available via message + optional longer text
        from config import get_class_description
        desc = get_class_description(top1["name"])
        if result.get("uncertain"):
            self.warn_label.setText("⚠ " + result["message"])
            self.warn_label.show()
            self.result_desc.setText(desc)
        else:
            self.warn_label.hide()
            self.result_desc.setText(desc)

        for i, (name_label, bar) in enumerate(self.topk_rows):
            if i < len(result["topk"]):
                item = result["topk"][i]
                name_label.setText(f"{i + 1}. {item['name']}  {item['confidence'] * 100:.1f}%")
                bar.setValue(int(item["confidence"] * 1000))
            else:
                name_label.setText("-")
                bar.setValue(0)

        cam = result.get("gradcam_image")
        if cam is not None:
            self._set_pixmap(self.cam_label, pil_to_pixmap(cam))

        history_path = self._save_history_image(self.to_predict_name)
        cam_path = ""
        if cam is not None:
            cam_path = str(HISTORY_DIR / f"cam_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
            cam.save(cam_path, quality=90)

        record = {
            "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "name": top1["name"],
            "conf": top1["confidence"],
            "desc": desc,
            "path": history_path,
            "cam_path": cam_path,
            "topk": result["topk"],
            "uncertain": result.get("uncertain", False),
            "message": result.get("message", ""),
        }
        self._history.insert(0, record)
        self._history = self._history[:50]
        self._refresh_history()

    def _save_history_image(self, src_path: str) -> str:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(src_path).suffix or ".jpg"
        dst = HISTORY_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{suffix}"
        shutil.copy2(src_path, dst)
        return str(dst)

    def _remove_history_file(self, path: str):
        try:
            p = Path(path)
            if p.exists() and HISTORY_DIR in p.resolve().parents:
                p.unlink()
        except Exception:
            pass

    def _refresh_history(self):
        self.history_list.clear()
        for item in self._history:
            flag = " ?" if item.get("uncertain") else ""
            text = f"{item['time']}  {item['name']}{flag}  {item['conf'] * 100:.1f}%"
            row = QListWidgetItem(text)
            row.setData(Qt.UserRole, item)
            self.history_list.addItem(row)
        self._fit_history_height()

    def _on_history_clicked(self, row: QListWidgetItem):
        item = row.data(Qt.UserRole)
        if not item:
            return
        self.result_name.setText(item["name"])
        self.result_conf.setText(f"Confidence  {item['conf'] * 100:.1f}%")
        self.result_desc.setText(item["desc"])
        if item.get("uncertain"):
            self.warn_label.setText("⚠ " + item.get("message", "Low confidence"))
            self.warn_label.show()
        else:
            self.warn_label.hide()
        for i, (name_label, bar) in enumerate(self.topk_rows):
            topk = item.get("topk") or []
            if i < len(topk):
                t = topk[i]
                name_label.setText(f"{i + 1}. {t['name']}  {t['confidence'] * 100:.1f}%")
                bar.setValue(int(t["confidence"] * 1000))
            else:
                name_label.setText("-")
                bar.setValue(0)
        path = item.get("path")
        if path and os.path.exists(path):
            self.to_predict_name = path
            self._refresh_preview(path)
        cam_path = item.get("cam_path")
        if cam_path and os.path.exists(cam_path):
            self._set_pixmap(self.cam_label, QPixmap(cam_path))

    def delete_selected_history(self):
        selected = self.history_list.selectedItems()
        if not selected:
            QMessageBox.information(
                self, "Notice",
                "Please select history records to delete (Ctrl/Shift for multi-select).",
            )
            return
        ids = {item.data(Qt.UserRole).get("id") for item in selected if item.data(Qt.UserRole)}
        keep = []
        for rec in self._history:
            if rec.get("id") in ids:
                self._remove_history_file(rec.get("path", ""))
                self._remove_history_file(rec.get("cam_path", ""))
            else:
                keep.append(rec)
        self._history = keep
        self._refresh_history()

    def clear_all_history(self):
        if not self._history:
            return
        reply = QMessageBox.question(
            self, "Confirm", "Clear all recognition history?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        for rec in self._history:
            self._remove_history_file(rec.get("path", ""))
            self._remove_history_file(rec.get("cam_path", ""))
        self._history = []
        self._refresh_history()

    def _on_err(self, message: str):
        self.result_name.setText("Failed")
        self.result_conf.setText(message)
        self.result_desc.setText("")
        self.warn_label.hide()
        QMessageBox.critical(self, "Recognition Failed", message)

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, "Exit", "Do you want to exit?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 13))
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
