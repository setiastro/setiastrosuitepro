# src/setiastro/saspro/ops/benchmark.py
from __future__ import annotations

import json, platform
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QProgressBar, QTextEdit, QMessageBox, QApplication
)

from setiastro.saspro.cosmicclarity_engines.benchmark_engine import (
    benchmark_image_path, download_benchmark_image, run_benchmark,  # keep your run_benchmark
)
from setiastro.saspro.cosmicclarity_engines.benchmark_engine import BENCHMARK_FITS_URL  # or define here

BENCHMARK_FITS_URL = "https://github.com/setiastro/setiastrosuitepro/releases/download/benchmarkFIT/benchmarkimage.fit"

class _BenchWorker(QObject):
    log = pyqtSignal(str)
    prog = pyqtSignal(int, int)          # done,total
    done = pyqtSignal(bool, dict, str)   # ok, results, err

    def __init__(self, mode: str, use_gpu: bool):
        super().__init__()
        self.mode = mode
        self.use_gpu = use_gpu

    def run(self):
        try:
            def status_cb(s: str):
                self.log.emit(str(s))

            def progress_cb(done: int, total: int) -> bool:
                self.prog.emit(int(done), int(total))
                return not QThread.currentThread().isInterruptionRequested()

            results = run_benchmark(
                mode=self.mode,
                use_gpu=self.use_gpu,
                status_cb=status_cb,
                progress_cb=progress_cb,
            )
            self.done.emit(True, results, "")
        except Exception as e:
            self.done.emit(False, {}, str(e))


class _DownloadWorker(QObject):
    log = pyqtSignal(str)
    prog = pyqtSignal(int, int)          # bytes_done, bytes_total
    done = pyqtSignal(bool, str)         # ok, message/path

    def __init__(self, url: str):
        super().__init__()
        self.url = url

    def run(self):
        try:
            def status_cb(s: str):
                self.log.emit(str(s))

            def progress_cb(done: int, total: int):
                self.prog.emit(int(done), int(total))

            def cancel_cb() -> bool:
                return QThread.currentThread().isInterruptionRequested()

            p = download_benchmark_image(
                self.url,
                status_cb=status_cb,
                progress_cb=progress_cb,
                cancel_cb=cancel_cb,
            )
            self.done.emit(True, str(p))
        except Exception as e:
            self.done.emit(False, str(e))


class BenchmarkDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Seti Astro Benchmark")
        self.setModal(False)
        self.setMinimumSize(560, 520)

        self._results = None
        self._thread = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # Top row: image status + download
        top = QHBoxLayout()
        self.lbl_img = QLabel(self)
        self.btn_dl = QPushButton("Download Benchmark Image…", self)
        self.btn_dl.clicked.connect(self._download_image)
        top.addWidget(self.lbl_img, 1)
        top.addWidget(self.btn_dl)
        outer.addLayout(top)

        # Mode row
        row = QHBoxLayout()
        row.addWidget(QLabel("Run:", self))
        self.cmb = QComboBox(self)
        self.cmb.addItems(["CPU", "GPU", "Both"])
        self.cmb.setCurrentText("Both")
        row.addWidget(self.cmb)

        self.btn_run = QPushButton("Run Benchmark", self)
        self.btn_run.clicked.connect(self._run_benchmark)
        row.addWidget(self.btn_run)
        row.addStretch(1)
        outer.addLayout(row)

        # Progress
        self.pbar = QProgressBar(self)
        self.pbar.setRange(0, 100)
        outer.addWidget(self.pbar)

        # Log / results
        self.txt = QTextEdit(self)
        self.txt.setReadOnly(True)
        outer.addWidget(self.txt, 1)

        # Bottom buttons
        bot = QHBoxLayout()
        self.btn_copy = QPushButton("Copy JSON", self)
        self.btn_copy.setEnabled(False)
        self.btn_copy.clicked.connect(self._copy_json)

        self.btn_save = QPushButton("Save Locally", self)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_local)

        self.btn_submit = QPushButton("Submit…", self)
        self.btn_submit.clicked.connect(self._submit)

        self.btn_close = QPushButton("Close", self)
        self.btn_close.clicked.connect(self.close)

        bot.addWidget(self.btn_copy)
        bot.addWidget(self.btn_save)
        bot.addStretch(1)
        bot.addWidget(self.btn_submit)
        bot.addWidget(self.btn_close)
        outer.addLayout(bot)

        self.refresh_ui()

    def refresh_ui(self):
        p = benchmark_image_path()
        if p.exists():
            self.lbl_img.setText(f"Benchmark image: Ready ({p.name})")
            self.btn_run.setEnabled(True)
        else:
            self.lbl_img.setText("Benchmark image: Not downloaded")
            self.btn_run.setEnabled(False)

        self.pbar.setValue(0)

    # ---------- download ----------
    def _download_image(self):
        self._stop_thread_if_any()

        self.txt.append("Starting download…")
        self.btn_dl.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.pbar.setValue(0)

        t = QThread(self)
        w = _DownloadWorker(BENCHMARK_FITS_URL)
        w.moveToThread(t)

        w.log.connect(self._log)
        w.prog.connect(self._dl_progress)
        w.done.connect(lambda ok, msg: self._dl_done(ok, msg, t, w))
        t.started.connect(w.run)
        t.start()
        self._thread = t

    def _dl_progress(self, done: int, total: int):
        if total > 0:
            pct = int(done * 100 / total)
            self.pbar.setValue(max(0, min(100, pct)))
        else:
            # unknown length
            self.pbar.setRange(0, 0)

    def _dl_done(self, ok: bool, msg: str, t: QThread, w: QObject):
        t.quit(); t.wait()
        self._thread = None

        self.pbar.setRange(0, 100)
        self.btn_dl.setEnabled(True)

        if ok:
            self._log(f"✅ Downloaded: {msg}")
            self.refresh_ui()
        else:
            self._log(f"❌ Download failed: {msg}")
            QMessageBox.warning(self, "Download failed", msg)
            self.refresh_ui()

    # ---------- run benchmark ----------
    def _run_benchmark(self):
        p = benchmark_image_path()
        if not p.exists():
            QMessageBox.information(self, "Benchmark image missing", "Please download the benchmark image first.")
            return

        self._stop_thread_if_any()

        self._results = None
        self.btn_copy.setEnabled(False)
        self.btn_save.setEnabled(False)

        self.txt.clear()
        self._log("Running benchmark…")
        self.pbar.setValue(0)

        mode = self.cmb.currentText()
        use_gpu = True  # benchmark engine will pick CPU if no CUDA/DML

        t = QThread(self)
        w = _BenchWorker(mode=mode, use_gpu=use_gpu)
        w.moveToThread(t)

        w.log.connect(self._log)
        w.prog.connect(self._bench_progress)
        w.done.connect(lambda ok, results, err: self._bench_done(ok, results, err, t, w))
        t.started.connect(w.run)
        t.start()
        self._thread = t

    def _bench_progress(self, done: int, total: int):
        if total > 0:
            self.pbar.setValue(int(done * 100 / total))
        QApplication.processEvents()

    def _bench_done(self, ok: bool, results: dict, err: str, t: QThread, w: QObject):
        t.quit(); t.wait()
        self._thread = None
        self.pbar.setValue(100 if ok else 0)

        if not ok:
            self._log(f"❌ Benchmark failed: {err}")
            QMessageBox.warning(self, "Benchmark failed", err)
            return

        self._results = results
        self._log("✅ Benchmark complete.\n")
        self._log(json.dumps([results], indent=2))

        self.btn_copy.setEnabled(True)
        self.btn_save.setEnabled(True)

    # ---------- actions ----------
    def _copy_json(self):
        if not self._results:
            return
        s = json.dumps([self._results], indent=4)
        QApplication.clipboard().setText(s)
        QMessageBox.information(self, "Copied", "Benchmark JSON copied to clipboard.")

    def _save_local(self):
        if not self._results:
            return
        # reuse your existing helper if you want; simplest local save:
        import os, time
        fn = "benchmark_results.json"
        try:
            if os.path.exists(fn):
                with open(fn, "r", encoding="utf-8") as f:
                    try:
                        allr = json.load(f)
                    except Exception:
                        allr = []
            else:
                allr = []
            allr.append(self._results)
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(allr, f, indent=4)
            self._log(f"\n✅ Saved to {fn}")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))

    def _submit(self):
        import webbrowser
        webbrowser.open("https://setiastro.com/benchmark-submit")

    def _log(self, s: str):
        self.txt.append(str(s))

    def _stop_thread_if_any(self):
        if self._thread is not None and self._thread.isRunning():
            self._thread.requestInterruption()
            self._thread.quit()
            self._thread.wait()
        self._thread = None
