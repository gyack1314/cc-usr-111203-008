"""整理不规则潮位时刻，拒绝同刻冲突。"""
from .io import number,positive,integer

def summarize(v):
    bytime={};duplicates=0
    for r in v['measurements']:
        t=number(r['hour']);y=number(r['level']);sigma=positive(r['sigma']);payload=(y,sigma)
        if t in bytime:
            if bytime[t]!=payload:raise ValueError('同刻潮位资料冲突')
            duplicates+=1
        bytime[t]=payload
    if not bytime:raise ValueError('缺少潮位')
    for w in v['frequencies']:positive(w)
    if len(set(v['frequencies']))!=len(v['frequencies']):raise ValueError('频率重复')
    positive(v['time_scale'])
    for p in v['candidates']:
        rho=number(p['rho']);ridge=number(p['ridge'])
        if not 0<=rho<1 or ridge<0:raise ValueError('核或惩罚参数不合法')
    return {'distinct_times':len(bytime),'duplicate_same_content':duplicates,'time_span_hours':max(bytime)-min(bytime),'frequencies':len(v['frequencies'])}
