"""Run: python main_gui.py. All Tk operations stay on the main thread."""
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
from pathlib import Path
import os
import time
from comparison_model import ComparisonModel
from persistence import save_baseline, load_baseline, save_log


class SpeechApp:
    def __init__(self, root):
        from personal_baseline import ProgressiveBaseline
        self.root, self.baseline = root, ProgressiveBaseline()
        from experiment_log import ExperimentLog
        self.log = ExperimentLog()
        self.storage = Path(__file__).resolve().parent / 'records'
        self.storage.mkdir(parents=True, exist_ok=True)
        self.baseline_path = self.storage / 'baseline.json'
        self.autosave_path = self.storage / ('speech_compare_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+self.log.session_id[:6]+'.csv')
        self.model = ComparisonModel()
        self.awaiting_label = None
        self.restore_error = None
        try:
            self.baseline = load_baseline(self.baseline_path)
        except Exception as exc:
            self.restore_error = str(exc)
        self.row_records = {}
        self.session = None
        self.dialog = None
        self.pending_row = None
        self.closing = False
        self.counter = 0
        self.devices = [None]
        self.device_items = [None]
        root.title('민감 발화 비교 실험 · 내장 마이크 · 측정 개선 적용')
        root.geometry('1200x850')
        root.minsize(860, 600)
        root.configure(bg='#f3f5f9')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', font=('맑은 고딕', 10))
        style.configure('Title.TLabel', font=('맑은 고딕', 21, 'bold'))
        style.configure('Status.TLabel', font=('맑은 고딕', 12, 'bold'))
        frame = ttk.Frame(root, padding=24)
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text='민감 발화 비교 실험', style='Title.TLabel').pack(anchor='w')
        ttk.Label(frame, text='구간 보정·측정 안정성 검사 적용 / 기존 기준선·모델 결과는 참고용이며 정확성 미검증').pack(anchor='w', pady=(5,18))
        controls = ttk.Frame(frame)
        controls.pack(fill='x')
        ttk.Label(controls, text='입력 장치').pack(side='left')
        self.device_box = ttk.Combobox(controls, state='readonly', width=60, values=['— 실제 입력 장치를 선택하세요 —'])
        self.device_box.current(0)
        self.device_box.pack(side='left', padx=8)
        self.refresh_button = ttk.Button(controls, text='장치 새로고침', command=self.refresh)
        self.refresh_button.pack(side='left')
        self.start_button = ttk.Button(controls, text='녹음 시작', command=self.start)
        self.start_button.pack(side='left', padx=8)
        self.stop_button = ttk.Button(controls, text='중지', command=self.stop, state='disabled')
        self.stop_button.pack(side='left')
        self.reset_button = ttk.Button(controls, text='기준선 초기화', command=self.reset)
        self.reset_button.pack(side='right')
        sampling = ttk.Frame(frame)
        sampling.pack(fill='x', pady=(14,0))
        ttk.Label(sampling, text='기존 45개 기준선은 구간 보정 전 방식으로 만든 과거 참고값입니다. 현재 판정의 분류 정확성은 미검증입니다.\n입력 장치: 노트북 내장 마이크 항목을 직접 고르세요. 같은 마이크가 여러 번 보이면 [MME] 항목을 권장합니다.', wraplength=1100).pack(side='left')
        self.status = tk.StringVar(value='대기 중 · 내장 마이크 항목을 선택하고 녹음 시작을 누르세요.')
        ttk.Label(frame, textvariable=self.status, style='Status.TLabel', wraplength=940).pack(anchor='w', pady=(20,10))
        self.progress = ttk.Progressbar(frame, maximum=45)
        self.progress.pack(fill='x')
        self.baseline_text = tk.StringVar()
        ttk.Label(frame, textvariable=self.baseline_text).pack(anchor='w', pady=(5,12))
        self.summary_text = tk.StringVar()
        ttk.Label(frame, textvariable=self.summary_text, wraplength=1000).pack(anchor='w', pady=(0,8))
        self.update_baseline()
        self.values = tk.StringVar(value='발화속도 —    Jitter —    Shimmer —    F0 —    침묵 —')
        ttk.Label(frame, textvariable=self.values, wraplength=940).pack(anchor='w', pady=8)
        self.mode = tk.StringVar(value='판정 대기')
        ttk.Label(frame, textvariable=self.mode).pack(anchor='w', pady=(0,12))
        table_frame = ttk.Frame(frame)
        table_frame.pack(fill='both', expand=True)
        columns = ('time','judgment','votes','model_default','model_relaxed','model_score','label','latency')
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings', height=10)
        for c, title, width in zip(columns, ['시각','개인 기준선','득표','모델 기본','모델 완화','모델 점수','실제 정답','특징·개인 분석'], [75,155,45,85,85,75,75,80]):
            self.table.heading(c, text=title)
            self.table.column(c, width=width, anchor='center')
        scroll = ttk.Scrollbar(table_frame, command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        labelling = ttk.Frame(frame)
        labelling.pack(fill='x', pady=8)
        ttk.Label(labelling, text='결과 행 선택 후 실제 정답 지정:').pack(side='left')
        for title, value in [('일반', 'normal'), ('민감', 'sensitive'), ('모름', 'unknown')]:
            ttk.Button(labelling, text=title, command=lambda v=value: self.set_label(v)).pack(side='left', padx=3)
        ttk.Button(labelling, text='현재 성적 보기', command=self.show_metrics).pack(side='right', padx=3)
        ttk.Button(labelling, text='저장 폴더 열기', command=lambda: os.startfile(str(self.storage))).pack(side='right', padx=3)
        ttk.Button(labelling, text='수치 기록 CSV 저장', command=self.export_log).pack(side='right')
        self.root.bind('<Control-s>', lambda event: self.export_log())
        ttk.Label(frame, text='수치·정답·입력 장치는 records 폴더에 자동 저장 / 기준선은 재실행 시 복원 / 원본 음성·전사문 저장·전송 없음\n모델은 기존 상담 데이터로 사전 학습. 현재 정답 입력으로 재학습하지 않음. 모델 점수는 검증된 개인정보 확률이 아닙니다.', wraplength=940).pack(anchor='w', pady=(15,0))
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.refresh()
        if self.restore_error:
            messagebox.showwarning('기준선 복원 실패', self.restore_error+'\n새 기준선으로 시작합니다.', parent=root)
        elif self.baseline.n_samples:
            self.status.set(f'저장된 기준선 {self.baseline.n_samples}/45개 복원 · 같은 사람·마이크일 때 사용하세요.')
        root.after(80, self.poll)

    def autosave(self):
        try:
            save_log(self.log, self.autosave_path)
            save_baseline(self.baseline, self.baseline_path)
            return True
        except Exception as exc:
            self.stop()
            messagebox.showerror('자동 저장 실패', f'{exc}\n녹음을 중지합니다. CSV 저장 버튼으로 다른 위치에 저장하세요.', parent=self.root)
            return False

    def refresh(self):
        try:
            import sounddevice as sd
            from device_info import list_input_devices
            items = list_input_devices(sd)
            self.device_items = [None]+items
            self.devices = [None]+[d['index'] for d in items]
            self.device_box['values'] = ['— 실제 입력 장치를 선택하세요 —']+[d['label'] for d in items]
            self.device_box.current(0)
        except Exception as exc:
            self.status.set(f'장치 조회 실패: {exc}')

    def update_baseline(self):
        n = self.baseline.n_samples
        target = 45
        self.progress['maximum'] = target
        self.progress['value'] = n
        phase = '수집 중 · 민감 여부 판정 안 함' if self.baseline.collecting else '판정 단계 · 기준선 고정'
        self.baseline_text.set(f'{phase} · 누적 {n}/45개 · 15개 {"✓" if n >= 15 else "—"} / 30개 {"✓" if n >= 30 else "—"} / 45개 {"✓" if n >= 45 else "—"}')
        def describe(stats):
            return ' / '.join(f'{label} {stats[key]["median"]:.4f} (MAD {stats[key]["mad"]:.4f})'
                for key, label in [('speech_rate','속도'),('jitter','Jitter'),('shimmer','Shimmer')])
        lines = ['중앙값과 MAD · 수집량 증가가 정확도 향상을 의미하지는 않습니다.']
        if n:
            lines.append('현재: '+describe(self.baseline.summary()))
        for count, stats in self.baseline.checkpoints.items():
            lines.append(f'{count}개 시점: '+describe(stats))
        self.summary_text.set('\n'.join(lines))

    def start(self):
        if self.device_box.current()==0:
            messagebox.showinfo('실제 입력 선택', '목록에서 노트북 내장 마이크 항목을 직접 선택하세요.\n(기본 입력을 내장 마이크로 가정하지 않습니다.)', parent=self.root)
            return
        if self.session:
            return
        item = self.device_items[self.device_box.current()]
        try:
            import sounddevice as sd
            from device_info import check_device, log_fields
            check_device(sd, item)
        except Exception as exc:
            messagebox.showerror('입력 장치 열기 불가', str(exc), parent=self.root)
            return
        self.log.device_info = log_fields(item)
        from gui_runtime import LiveSession
        self.session = LiveSession(self.baseline, device=self.devices[self.device_box.current()])
        self.start_button['state'] = 'disabled'
        self.reset_button['state'] = 'disabled'
        self.refresh_button['state'] = 'disabled'
        self.device_box['state'] = 'disabled'
        self.stop_button['state'] = 'normal'
        self.status.set('마이크를 여는 중…')
        self.awaiting_label = None
        self.session.start()

    def stop(self):
        if self.session:
            self.session.stop()
            self.stop_button['state'] = 'disabled'
            self.status.set('중지 중…')
        if self.dialog:
            if self.pending_row:
                self.table.set(self.pending_row, 'consent', '취소 · 학습 미사용')
                self.pending_row = None
            self.dialog.destroy()
            self.dialog = None

    def reset(self):
        if self.session:
            return
        if messagebox.askyesno('개인 기준선 초기화',
                '기준선과 화면을 초기화할까요? 기존 수치 기록은 CSV 저장용으로 유지됩니다.', parent=self.root):
            from personal_baseline import ProgressiveBaseline
            self.baseline = ProgressiveBaseline()
            self.log.new_baseline()
            self.autosave()
            self.row_records.clear()
            for row in self.table.get_children():
                self.table.delete(row)
            self.values.set('발화속도 —    Jitter —    Shimmer —    F0 —    침묵 —')
            self.mode.set('일상 발화 수집 대기')
            self.update_baseline()
            self.status.set('초기화 완료 · 녹음 시작을 누르세요.')

    def show_collection(self, payload):
        f = payload['features']
        self.log.add(payload, 'baseline', self.baseline)
        if not self.autosave(): return
        self.update_baseline()
        self.values.set(f"음향 피크율 {f['speech_rate']:.3f}/s    Jitter {f['jitter']*100:.3f}%    Shimmer {f['shimmer']*100:.3f}%    F0 {f['f0']:.1f} Hz    침묵 {f['silence_duration']:.2f} 초")
        self.mode.set('기준선 수집용 일상 발화로 반영 · 내용의 일반/민감 여부는 확인하지 않음')
        if self.baseline.collecting:
            self.status.set('수집 중 · 평소처럼 일상적인 문장을 말해 주세요.')
        else:
            self.mode.set('45개 수집 완료 · 기준선 고정 · 다음 발화부터 민감 후보 판정')
            self.status.set('자동 전환 완료 · 다음 문장을 말해 주세요.')
        self.session.resume()

    def show_result(self, payload):
        f, r = payload['features'], payload['result']
        self.update_baseline()
        self.values.set(f"음향 피크율 {f['speech_rate']:.3f}/s    Jitter {f['jitter']*100:.3f}%    Shimmer {f['shimmer']*100:.3f}%    F0 {f['f0']:.1f} Hz    침묵 {f['silence_duration']:.2f} 초")
        mode = f'개인 {self.baseline.n_samples}개'
        label = '판정 보류 · 기준선 변동 부족' if r.is_sensitive is None else ('민감 발화 후보' if r.is_sensitive else '민감 후보 미검출')
        score_text = ' / '.join(f'{k}: {v:.2f}' for k,v in r.scores.items())
        self.mode.set(f'{score_text} · {mode} 기준 · {label} · {r.votes}/3 · 분석 시간은 발화 종료·정답 대기 및 모델 계산 시간을 제외합니다.')
        self.counter += 1
        row = str(self.counter)
        self.row_records[row] = self.log.add(payload, 'evaluation', self.baseline)
        try:
            model_started = time.perf_counter()
            comparison = self.model.predict(f)
            comparison['model_elapsed_ms'] = (time.perf_counter()-model_started)*1000
            self.log.rows[-1].update(comparison)
            names = ['미검출', '민감 후보']
            default = names[comparison['model_default']]
            relaxed = names[comparison['model_relaxed']]
            score = f"{comparison['model_score']:.3f}"
        except Exception as exc:
            default = relaxed = '계산 실패'; score = '—'
            self.log.rows[-1]['model_error'] = str(exc)
        self.table.insert('', 'end', iid=row, values=(datetime.now().strftime('%H:%M:%S'), label,
            f'{r.votes}/3', default, relaxed, score, '미지정', f"{payload['elapsed_ms']:.0f} ms"))
        children = self.table.get_children()
        if len(children)>200:
            self.row_records.pop(children[0], None)
            self.table.delete(children[0])
        self.table.see(row)
        self.table.selection_set(row)
        self.awaiting_label = row
        if not self.autosave(): return
        self.status.set('녹음 잠시 멈춤 · 방금 발화의 실제 정답(일반/민감/모름)을 누르면 다음 발화를 받습니다.')

    def show_withheld(self, payload):
        reason = payload.get('reason', '')
        f = payload.get('features')
        collecting = getattr(self.baseline, 'collecting', False)
        record = self.log.add_withheld(payload, self.baseline)
        if f:
            self.values.set(f"음향 피크율 {f['speech_rate']:.3f}/s    Jitter {f['jitter']*100:.3f}%    Shimmer {f['shimmer']*100:.3f}%    F0 {f['f0']:.1f} Hz    침묵 {f['silence_duration']:.2f} 초 (보류된 발화 · 판정 미사용)")
        else:
            self.values.set('발화속도 —    Jitter —    Shimmer —    F0 —    침묵 — (특징 계산 불가)')
        self.mode.set('측정 불안정 · 이번 발화의 개인·모델 판정을 모두 보류 · 기준선 반영 없음')
        if collecting:
            if not self.autosave(): return
            self.status.set(f'판정 보류: {reason} · 기준선 반영 없음 · 다음 문장을 말해 주세요.')
            self.session.resume()
            return
        self.counter += 1
        row = str(self.counter)
        self.row_records[row] = record
        self.table.insert('', 'end', iid=row, values=(datetime.now().strftime('%H:%M:%S'), '측정 보류', '—', '보류', '보류', '—', '미지정', f"{payload.get('elapsed_ms', 0):.0f} ms"))
        self.table.see(row)
        self.table.selection_set(row)
        self.awaiting_label = row
        if not self.autosave(): return
        # Label withheld utterances too, so the hold rate can be reported per true class.
        self.status.set(f'판정 보류: {reason} · 기준선 반영 없음 · 방금 발화의 실제 정답(일반/민감/모름)을 누르면 다음 발화를 받습니다.')

    def set_label(self, value):
        selected = self.table.selection()
        if not selected:
            messagebox.showinfo('정답 지정', '먼저 결과 표에서 발화를 선택하세요.', parent=self.root)
            return
        names = dict(normal='일반', sensitive='민감', unknown='모름')
        for row in selected:
            if row not in self.row_records: continue
            self.log.label(self.row_records[row], value)
            self.table.set(row, 'label', names[value])
        if not self.autosave(): return
        if self.awaiting_label in selected:
            self.awaiting_label = None
            if self.session and not self.session.stopping.is_set():
                self.session.resume()
                self.status.set('자동 저장 완료 · 다음 문장을 말해 주세요.')

    def show_metrics(self):
        rows = [r for r in self.log.rows if r['phase'] in ('evaluation','withheld') and r['label'] in ('normal','sensitive')]
        lines = [f'이번 실행에서 정답을 지정한 평가 발화(측정 보류 포함): {len(rows)}개', '기준선 수집·미지정·모름은 제외합니다. 같은 화자의 반복 발화이므로 독립 표본이 아닙니다.']
        for name, title in [('normal','일반'),('sensitive','민감')]:
            group = [r for r in rows if r['label']==name]
            held = sum(r['phase']=='withheld' for r in group)
            if group:
                lines.append(f'{title} 정답 {len(group)}개 중 측정 보류 {held}개 ({held/len(group):.1%}), 판정 가능 {len(group)-held}개')
        for title,key in [('개인 기준선','prediction'),('모델 기본','model_default'),('모델 완화','model_relaxed')]:
            valid = [r for r in rows if r.get(key) in (0,1,'normal','sensitive')]
            tp = sum(r['label']=='sensitive' and r[key] in (1,'sensitive') for r in valid)
            fn = sum(r['label']=='sensitive' and r[key] in (0,'normal') for r in valid)
            fp = sum(r['label']=='normal' and r[key] in (1,'sensitive') for r in valid)
            tn = sum(r['label']=='normal' and r[key] in (0,'normal') for r in valid)
            recall = f'{tp/(tp+fn):.1%}' if tp+fn else '계산 불가'
            fpr = f'{fp/(fp+tn):.1%}' if fp+tn else '계산 불가'
            lines.append(f'\n{title}: 민감 검출 {tp}/{tp+fn} ({recall}), 일반 오탐 {fp}/{fp+tn} ({fpr}) · 판정된 발화 기준\n판정 보류·계산 실패: {len(rows)-len(valid)}/{len(rows)}개')
        messagebox.showinfo('현재 성적 · 이번 실행만', '\n'.join(lines), parent=self.root)

    def export_log(self):
        if not self.log.rows:
            messagebox.showinfo('CSV 저장', '저장할 기록이 없습니다.', parent=self.root)
            return
        path = filedialog.asksaveasfilename(parent=self.root, defaultextension='.csv',
            initialfile='speech_experiment_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv',
            filetypes=[('CSV', '*.csv')])
        if path:
            try:
                self.log.export(path)
                messagebox.showinfo('CSV 저장', f'{len(self.log.rows)}개 수치 기록 저장 완료', parent=self.root)
            except Exception as exc:
                messagebox.showerror('저장 실패', str(exc), parent=self.root)

    def poll(self):
        if self.session:
            while True:
                try:
                    kind, payload = self.session.events.get_nowait()
                except queue.Empty:
                    break
                if kind == 'stopped':
                    self.session = None
                    self.start_button['state'] = 'normal'
                    self.reset_button['state'] = 'normal'
                    self.refresh_button['state'] = 'normal'
                    self.device_box['state'] = 'readonly'
                    self.stop_button['state'] = 'disabled'
                    if self.closing:
                        self.root.destroy()
                        return
                    self.update_baseline()
                    if not self.status.get().startswith('오류'):
                        self.status.set('중지됨 · 기준선은 유지됩니다. 다시 시작할 수 있습니다.')
                    break
                if self.session.stopping.is_set():
                    continue
                if kind == 'collection':
                    self.show_collection(payload)
                elif kind == 'result':
                    self.show_result(payload)
                elif kind == 'opened':
                    from device_info import log_fields
                    self.log.device_info = log_fields(self.device_items[self.device_box.current()], payload)
                elif kind == 'quality':
                    self.show_withheld(payload)
                elif kind == 'error':
                    self.status.set(f'오류: {payload}')
                    messagebox.showerror('실행 오류', f'{payload}\n\n설치 여부, 마이크 연결 및 Windows 마이크 권한을 확인해 주세요.', parent=self.root)
                elif kind == 'status':
                    self.status.set(payload)
        self.root.after(80, self.poll)

    def close(self):
        self.closing = True
        if self.session:
            self.stop()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    try:
        SpeechApp(root)
    except ImportError as exc:
        messagebox.showerror('필수 패키지 없음', f'{exc}\ninstall_windows.bat를 먼저 실행해 주세요.', parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == '__main__':
    main()
