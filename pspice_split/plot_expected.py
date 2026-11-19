import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"NanumGothic","axes.unicode_minus":False,"font.size":10})
S="/tmp/claude-0/-home-user-test/5034b632-429f-5fb8-8486-29d2b08e23e8/scratchpad"
stages=[("stage1_U1A","A","REF1","OUT1","D1",9.6),("stage2_U1B","B","REF2","OUT7","D2",7.2),
        ("stage3_U1C","C","REF3","OUT8","D3",4.8),("stage4_U1D","D","REF4","OUT14","D4",2.4)]
C=["#2a78d6","#eb6834","#1baf7a","#e34948"]; ink="#0b0b0b"; mut="#52514e"; grid="#e4e3df"
fig,axs=plt.subplots(4,2,figsize=(13,14),dpi=150,gridspec_kw=dict(hspace=0.55,wspace=0.18))
fig.patch.set_facecolor("#fcfcfb")
for r,(f,L,ref,out,d,th) in enumerate(stages):
    for c,kind in enumerate(["tran","dc"]):
        a=np.loadtxt(f"{S}/{f}_{kind}.txt"); x=a[:,0]*(1e3 if kind=="tran" else 1)
        ys=[a[:,1],a[:,3],a[:,5],a[:,7]]
        names=["V(+입력) = VIN",f"V(-입력) = {th}V 기준",f"V(OUT) U1{L} 출력",f"V({d} 애노드)"]
        ax=axs[r,c]; ax.set_facecolor("#fcfcfb")
        for y,n,col in zip(ys,names,C): ax.plot(x,y,color=col,lw=2,label=n)
        ax.set_ylim(-0.5,12.5); ax.set_yticks(range(0,13,2))
        ax.grid(color=grid,lw=0.8); [s.set_visible(False) for s in (ax.spines["top"],ax.spines["right"])]
        for s in ("left","bottom"): ax.spines[s].set_color(mut)
        ax.tick_params(colors=mut); ax.set_ylabel("전압 (V)",color=mut)
        on = 6>th
        if kind=="tran":
            ax.set_xlabel("시간 (ms)",color=mut); ax.set_xlim(0,10)
            ax.axvline(1,color=mut,ls=":",lw=1); ax.text(1.1,11.8,"U2 닫힘 (1ms)",color=mut,fontsize=8)
            ax.set_title(f"U1{L}만 남김 — 과도해석 (VIN=6V)  →  {d} {'켜짐' if on else '꺼짐'}",
                         loc="left",color=ink,fontsize=11,fontweight="bold")
            if on:
                ax.text(9.9,a[-1,5]+0.35,f"출력 {a[-1,5]:.2f} V",ha="right",color=ink,fontsize=9)
                ax.text(9.9,a[-1,7]+0.35,f"{d} {a[-1,7]:.2f} V",ha="right",color=ink,fontsize=9)
            else:
                ax.text(9.9,0.45,f"출력 = {d} = {a[-1,5]:.2f} V",ha="right",color=ink,fontsize=9)
        else:
            ax.set_xlabel("VIN (V)",color=mut); ax.set_xlim(0,12)
            ax.axvline(6,color=mut,ls="--",lw=1); ax.text(6.15 if th<6 else 5.85,11.6,"현재 VIN=6V",ha="left" if th<6 else "right",color=mut,fontsize=8)
            left=(6<th<8.5); ax.text(th+0.15 if left else th-0.15,11.6,f"전환점 ≈ {th}V",ha="left" if left else "right",color=ink,fontsize=9)
            ax.set_title(f"U1{L}만 남김 — VIN 0→12V 스윕 (DC)",loc="left",color=ink,fontsize=11,fontweight="bold")
axs[0,0].legend(loc="upper center",bbox_to_anchor=(1.05,1.42),ncol=4,frameon=False,fontsize=10,
                labels=["V(+입력) = VIN","V(-입력) 기준전압","V(OUT) 비교기 출력","V(D 애노드) LED 전압"])
fig.suptitle("op-amp 섹션별 예상 시뮬레이션 결과 (LM324 모델, ngspice 계산)",x=0.07,y=0.965,ha="left",fontsize=15,fontweight="bold",color=ink)
fig.savefig("/home/user/test/pspice_split/expected_results.png",bbox_inches="tight",facecolor=fig.get_facecolor())
