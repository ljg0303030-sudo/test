"""Guided paired reading experiment with hidden predictions and atomic local logs."""
import queue,os,time,json
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk,messagebox
from personal_baseline import ProgressiveBaseline
from threshold_compare import AcousticFeatures
from experiment_log import ExperimentLog
from comparison_model import ComparisonModel
from persistence import load_baseline,save_baseline,save_log
from gui_runtime import LiveSession
from controlled_protocol import make_trials,summarize

class ControlledApp:
    def __init__(self,root):
        self.root=root;self.session=None;self.closing=False;self.pending=None
        self.trials=make_trials();self.index=0;self.log=ExperimentLog();self.model=ComparisonModel()
        self.folder=Path(__file__).resolve().parent/'records';self.folder.mkdir(exist_ok=True)
        self.baseline_path=self.folder/'baseline.json'
        self.path=self.folder/('controlled_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+self.log.session_id[:6]+'.csv')
        self.baseline=load_baseline(self.baseline_path)
        root.title('20문장 비교 실험 · 결과 숨김 · 자동 저장');root.geometry('1000x700');root.minsize(800,600)
        ttk.Style().configure('.',font=('맑은 고딕',11))
        frame=ttk.Frame(root,padding=22);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text='일반·민감 숫자 문장 비교',font=('맑은 고딕',21,'bold')).pack(anchor='w')
        ttk.Label(frame,text='같은 숫자와 비슷한 문장 길이로 비교합니다. 일부러 속도·목소리를 바꾸지 마세요.',wraplength=900).pack(anchor='w',pady=8)
        ttk.Label(frame,text='모든 번호는 실험용 가상 문자열입니다. 주문번호도 전화번호와 같은 형식으로 읽습니다.\n숫자는 한 자리씩, 0은 모두 “공”으로 읽으세요. /는 묶음 표시이며 길게 쉬라는 뜻이 아닙니다.',wraplength=900).pack(anchor='w')
        controls=ttk.Frame(frame);controls.pack(fill='x',pady=12)
        self.devices=[None]
        self.device=ttk.Combobox(controls,state='readonly',width=36,values=['시스템 기본 마이크']);self.device.current(0);self.device.pack(side='left')
        self.start_button=ttk.Button(controls,text='시작 / 계속',command=self.start);self.start_button.pack(side='left',padx=5)
        ttk.Button(controls,text='중지',command=self.stop).pack(side='left')
        self.reuse_button=ttk.Button(frame,text='지난 실험의 45개 기준선 불러오기',command=self.reuse);self.reuse_button.pack(anchor='w')
        self.reset_button=ttk.Button(frame,text='다른 사람·마이크: 기준선 새로 수집',command=self.reset);self.reset_button.pack(anchor='w',pady=4)
        self.progress=tk.StringVar();ttk.Label(frame,textvariable=self.progress).pack(anchor='w',pady=12)
        self.prompt=tk.StringVar();ttk.Label(frame,textvariable=self.prompt,font=('맑은 고딕',18,'bold'),wraplength=900).pack(anchor='w',pady=12)
        self.reading=tk.StringVar();ttk.Label(frame,textvariable=self.reading,wraplength=900).pack(anchor='w',pady=8)
        self.status=tk.StringVar(value='마이크를 고르고 시작을 누르세요. 기존 기준선은 같은 사람·마이크·환경에서만 재사용하세요.')
        ttk.Label(frame,textvariable=self.status,wraplength=900).pack(anchor='w',pady=10)
        buttons=ttk.Frame(frame);buttons.pack(fill='x')
        self.accept_button=ttk.Button(buttons,text='문장 끝까지 맞게 읽었음 → 다음',command=lambda:self.confirm(True),state='disabled');self.accept_button.pack(side='left')
        self.retry_button=ttk.Button(buttons,text='잘못 읽었음 / 잘림 → 다시 읽기',command=lambda:self.confirm(False),state='disabled');self.retry_button.pack(side='left',padx=8)
        self.results_button=ttk.Button(frame,text='20문장 결과 보기',command=self.show_results,state='disabled');self.results_button.pack(anchor='w',pady=12)
        ttk.Button(frame,text='저장 폴더 열기',command=lambda:os.startfile(str(self.folder))).pack(anchor='w')
        ttk.Label(frame,text='판정과 수치는 20문장 완료 전까지 숨깁니다. 모든 시도·확인·기준선은 자동 저장됩니다.\n원본 음성 저장·외부 전송·STT·실험 중 재학습 없음. 실험 문장과 정답 조건은 CSV에 기록됩니다.',wraplength=900).pack(anchor='w',pady=15)
        try:
            import sounddevice as sd
            items=[(i,d) for i,d in enumerate(sd.query_devices()) if d['max_input_channels']>0]
            self.devices+=[i for i,d in items];self.device['values']=['시스템 기본 마이크']+[f"{i}: {d['name']}" for i,d in items]
        except Exception as exc:self.status.set(f'마이크 목록 조회 실패: {exc}')
        self.update_prompt();root.protocol('WM_DELETE_WINDOW',self.close);root.after(80,self.poll)
    def update_prompt(self):
        self.progress.set(f'기준선 {self.baseline.n_samples}/45개 · 평가 {self.index}/20문장')
        if self.baseline.collecting:
            self.prompt.set('평소 말투로 일반 문장을 말해 주세요.')
            self.reading.set('오늘 한 일, 음식, 날씨 등 다양한 문장으로 수집합니다. 번호 문장은 기준선 완료 후 안내합니다.')
        elif self.index<20:
            t=self.trials[self.index];self.prompt.set(f"{self.index+1}/20  {t['prompt']}");self.reading.set('숫자 읽기: '+t['reading'])
        else:self.prompt.set('20문장 완료');self.reading.set('결과 보기 또는 저장 폴더 열기를 누르세요.');self.results_button['state']='normal'
    def save(self):
        try:save_log(self.log,self.path);save_baseline(self.baseline,self.baseline_path);return True
        except Exception as exc:
            self.stop();messagebox.showerror('저장 실패',str(exc)+'\n녹음을 중지했습니다. 폴더 쓰기 권한과 여유 공간을 확인하세요.',parent=self.root);return False
    def reuse(self):
        if self.session or self.index or self.log.rows:return
        if not messagebox.askyesno('기준선 재사용','9월 21일 16:47 실험의 45개 기준선입니다.\n같은 사람·마이크·환경인가요?',parent=self.root):return
        self.baseline=load_baseline(Path(__file__).with_name('previous_baseline.json'));self.save();self.update_prompt()
    def reset(self):
        if self.session or self.index or self.log.rows:return
        if messagebox.askyesno('기준선 초기화','기준선 45개를 새로 수집할까요?',parent=self.root):
            self.baseline=ProgressiveBaseline();self.save();self.update_prompt()
    def start(self):
        if self.session or self.index>=20:return
        self.pending=None;self.accept_button['state']=self.retry_button['state']='disabled'
        self.session=LiveSession(self.baseline,device=self.devices[self.device.current()]);self.session.start()
        self.start_button['state']=self.reuse_button['state']=self.reset_button['state']=self.device['state']='disabled'
    def stop(self):
        if self.session:self.session.stop()
        self.status.set('중지 중…')
    def confirm(self,valid):
        if self.pending is None:return
        row=self.log.rows[self.pending-1];row['trial_valid']=int(valid)
        self.log.label(self.pending,row['assigned_condition'] if valid else 'unknown')
        if not self.save():return
        self.pending=None;self.accept_button['state']=self.retry_button['state']='disabled'
        if valid:self.index+=1
        self.update_prompt()
        if self.index>=20:
            self.path.with_suffix('.txt').write_text(summarize(self.log.rows),encoding='utf-8');self.stop()
        elif self.session and not self.session.stopping.is_set():
            self.status.set('다음 안내 문장을 읽어 주세요.' if valid else '같은 문장을 다시 읽어 주세요.');self.session.resume()
    def process(self,kind,payload):
        if kind=='collection':
            self.log.add(payload,'baseline',self.baseline)
            if self.save():self.update_prompt();self.session.resume()
        elif kind=='result':
            rid=self.log.add(payload,'evaluation',self.baseline);row=self.log.rows[-1];t=self.trials[self.index]
            row.update(protocol='paired-numeric-v1',trial_order=t['trial_order'],pair_id=t['pair_id'],prompt=t['prompt'],digits=t['digits'],assigned_condition=t['condition'],trial_valid='')
            try:
                started=time.perf_counter();row.update(self.model.predict(payload['features']));row['model_elapsed_ms']=(time.perf_counter()-started)*1000
            except Exception as exc:row['model_error']=str(exc)
            self.pending=rid
            if self.save():
                self.accept_button['state']=self.retry_button['state']='normal'
                self.status.set('녹음 잠시 멈춤 · 안내 문장을 끝까지 맞게 읽었나요? 아래 버튼으로 확인하세요.')
        elif kind=='quality':self.status.set('측정 보류: '+str(payload)+' · 같은 문장을 다시 읽어 주세요.');self.session.resume()
        elif kind=='status':self.status.set(payload)
        elif kind=='error':messagebox.showerror('실행 오류',str(payload),parent=self.root)
    def poll(self):
        if self.session:
            while True:
                try:kind,payload=self.session.events.get_nowait()
                except queue.Empty:break
                if kind=='stopped':
                    self.session=None;self.start_button['state']='normal';self.device['state']='readonly'
                    if not self.log.rows:self.reuse_button['state']=self.reset_button['state']='normal'
                    self.status.set('20문장 완료 · 결과를 확인하세요.' if self.index>=20 else '중지됨 · 계속하려면 시작을 누르세요.')
                    if self.closing:self.root.destroy();return
                    break
                if not self.session.stopping.is_set():self.process(kind,payload)
        self.root.after(80,self.poll)
    def show_results(self):
        if self.index<20:return
        messagebox.showinfo('문장 쌍 비교 결과',summarize(self.log.rows),parent=self.root)
    def close(self):
        self.closing=True
        if self.session:self.stop()
        else:self.root.destroy()

def main():
    root=tk.Tk()
    try:ControlledApp(root)
    except Exception as exc:messagebox.showerror('실행 실패',str(exc)+'\ninstall_windows.bat 설치 여부를 확인하세요.',parent=root);root.destroy();return
    root.mainloop()
if __name__=='__main__':main()
