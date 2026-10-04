from pathlib import Path
import sys,traceback
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'audit_20260922/deps'),str(HERE.parent/'jitter_shimmer_validation_20260927/recorder_deps')]
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')
if __name__=='__main__':
    try:
        if '--controlled' in sys.argv:
            from controlled_gui import main
        else:
            from main_gui import main
        main()
    except Exception:
        error=traceback.format_exc();(HERE/'startup_error.txt').write_text(error,encoding='utf-8')
        import tkinter.messagebox as mb
        mb.showerror('시작 오류',error)
        raise
