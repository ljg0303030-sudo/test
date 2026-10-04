"""Fixed, paired numeric reading protocol. Simulated sensitivity, no real PII."""
import random

def make_trials():
    # Same digits, same digit-by-digit reading, and seven Hangul syllables before the number.
    suffixes=['1204','2836','3957','4618','5729','6841','7962','8173','9284','1395']
    blocks=[[],[]]
    for i,tail in enumerate(suffixes):
        number='010-0000-'+tail
        for condition in ['normal','sensitive']:
            prefix='이번 주문번호는' if condition=='normal' else '제 휴대폰번호는'
            trial=dict(pair_id=i+1,condition=condition,digits=number,
                       prompt=prefix+' '+number+'입니다.',
                       reading='공 일 공 / 공 공 공 공 / '+' '.join({'0':'공','1':'일','2':'이','3':'삼','4':'사','5':'오','6':'육','7':'칠','8':'팔','9':'구'}[c] for c in tail))
            first='normal' if i<5 else 'sensitive'
            blocks[0 if condition==first else 1].append(trial)
    rng=random.Random(20260921)
    for block in blocks:rng.shuffle(block)
    return [dict(t,trial_order=n+1) for n,t in enumerate(blocks[0]+blocks[1])]

def summarize(rows):
    rows=[r for r in rows if r.get('phase')=='evaluation' and r.get('trial_valid')==1]
    lines=[f'완료한 유효 발화: {len(rows)}/20',
           '가짜 번호를 읽는 1인 모의 실험입니다. 감정·실제 개인정보 노출 상황과 동일하지 않습니다.',
           'STT를 사용하지 않으므로 문장을 끝까지 읽었는지는 사용자 확인에 따릅니다.']
    paired={}
    for row in rows:paired.setdefault(row['pair_id'],{})[row['label']]=row
    pairs=[p for p in paired.values() if set(p)=={'normal','sensitive'}]
    lines.append(f'일반·민감이 모두 있는 문장 쌍: {len(pairs)}/10')
    for key,title in [('speech_rate','발화속도 추정'),('jitter','Jitter'),('shimmer','Shimmer'),('_duration_sec','구간 길이(초)')]:
        if not pairs:continue
        normal=[float(p['normal'][key]) for p in pairs];sensitive=[float(p['sensitive'][key]) for p in pairs]
        lower=sum(s<n for n,s in zip(normal,sensitive))
        lines.append(f'{title}: 일반 평균 {sum(normal)/len(normal):.5f}, 민감 평균 {sum(sensitive)/len(sensitive):.5f}; 민감이 낮은 쌍 {lower}/{len(pairs)}')
    if pairs:
        all_lower=sum(all(p['sensitive'][k]<p['normal'][k] for k in ['speech_rate','jitter','shimmer']) for p in pairs)
        lines.append(f'세 특징이 모두 낮은 쌍: {all_lower}/{len(pairs)}')
    lines.append('문장 쌍은 독립적인 참가자가 아닙니다. 이 결과만으로 감소 가정의 일반적 성립 여부를 결론내리지 마세요.')
    return '\n'.join(lines)
