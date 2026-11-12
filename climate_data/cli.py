"""显示已有原始摘要或调和分析，来源文件保持不动。"""
import argparse
import json
from .io import load
from .domain import summarize
from .harmonic import analyze

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['inspect','analyze']);parser.add_argument('file');a=parser.parse_args()
    v,digest=load(a.file)
    out={'source_sha256':digest}
    if a.command=='inspect':out['summary']=summarize(v)
    else:out['analysis']=analyze(v)
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
if __name__=='__main__':main()
