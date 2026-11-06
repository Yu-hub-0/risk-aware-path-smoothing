"""Render spatial path reveals from reproducible E1 curves (not flight motion)."""
from pathlib import Path
import argparse, csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
COLORS = {'P0':'#707780', 'P1':'#d18b45', 'P2':'#749bbf', 'P3':'#16776e'}
LABELS = {'P0':'B-A* polyline', 'P1':'Ordinary B-spline', 'P2':'Risk penalty', 'P3':'Risk penalty + repair'}

def prepare():
    from run_e1 import make_case, run_b_astar, grid_path_xy, solve_one
    rows = list(csv.DictReader((ROOT/'results_005/case_results.csv').open(encoding='utf-8-sig')))
    folder = ROOT/'data'; folder.mkdir(exist_ok=True)
    selection=[]
    for kind in ['corner','s_bend','narrow']:
        # Fixed, disclosed visual-example criterion; all 90 tasks remain in CSV.
        candidates=[]
        for i in range(30):
            rr={r['method']:r for r in rows if r['scenario']==kind and int(r['case'])==i}
            if rr['P3']['status']=='accepted' and float(rr['P1'].get('violation_length_m') or 0)>0:
                candidates.append(i)
        i=min(candidates) if candidates else 0
        risk,start,goal=make_case(kind,i,20261010)
        dense,sparse=run_b_astar(risk,start,goal,risk_threshold=.5,target_nodes=None)
        curves={};metrics={}
        for m in COLORS:
            metrics[m],curves[m],_=solve_one(m,grid_path_xy(sparse),grid_path_xy(dense),risk)
        np.savez_compressed(folder/f'{kind}.npz',risk=risk,**curves)
        selection.append(dict(scenario=kind,case=i,metrics=metrics))
    (folder/'demo_selection.json').write_text(json.dumps(dict(rule='lowest case index per family with accepted P3 and nonzero P1 risk violation; fallback index 0', examples=selection, playback='equal normalized arc-length progression; not physical time or optimization iterations'),indent=2),encoding='utf-8')

def resample(p, n=500):
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    keep=np.r_[True,np.diff(s)>1e-12]
    assert np.all(np.diff(s[keep]) > 0), 'Arc length must increase'
    return np.column_stack([np.interp(np.linspace(0,s[-1],n),s[keep],p[keep,j]) for j in range(2)])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frames',type=int,default=100);ap.add_argument('--prepare',action='store_true');args=ap.parse_args()
    if args.frames<20:ap.error('--frames must be at least 20')
    if args.prepare or not (ROOT/'data/corner.npz').exists():prepare()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(12,4.8));fig.subplots_adjust(left=.05,right=.99,bottom=.23,top=.80,wspace=.20)
    fig.suptitle('Risk-aware path smoothing',fontsize=20,fontweight='bold',y=.98)
    fig.text(.5,.88,'B-A* search → cubic B-spline → risk penalty → local repair',ha='center',color='#52606c')
    artists=[]
    records=json.loads((ROOT/'data/demo_selection.json').read_text())['examples']
    for ax,kind,title,record in zip(axes,['corner','s_bend','narrow'],['Corner','S-bend','Narrow passage'],records):
        z=np.load(ROOT/f'data/{kind}.npz')
        ax.imshow(z['risk']>.5,origin='lower',extent=(-.5,39.5,-.5,39.5),cmap=ListedColormap(['#f4f6f7','#e6b2ad']),interpolation='nearest')
        ax.plot(z['P0'][:,0],z['P0'][:,1],color=COLORS['P0'],lw=1,alpha=.5)
        for m in COLORS:
            p=resample(z[m]);line,=ax.plot([],[],color=COLORS[m],lw=2.4 if m=='P3' else 1.4,ls='--' if m=='P1' else '-',label=LABELS[m])
            dot,=ax.plot([],[],'.',color=COLORS[m],ms=7);artists.append((line,dot,p))
        ax.set(xlim=(0,39),ylim=(0,39),xlabel='X (m)',ylabel='Y (m)',title=f'{title} · case {record["case"]:02d}',aspect='equal')
        ax.text(.02,.02,'Shaded cells: risk > 0.5',transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.07),ncol=4,frameon=False,fontsize=10)
    fig.text(.5,.035,'Spatial reveal at equal normalized distance · illustrative cases; not a flight simulation',ha='center',fontsize=9,color='#52606c')
    def update(frame):
        k=max(1,round(500*min(frame/(args.frames-16),1)))
        for line,dot,p in artists:line.set_data(p[:k,0],p[:k,1]);dot.set_data([p[k-1,0]],[p[k-1,1]])
        return []
    update(args.frames-1);out=ROOT/'assets';out.mkdir(exist_ok=True)
    fig.savefig(out/'path_smoothing.png',dpi=150);fig.savefig(out/'path_smoothing.svg');fig.savefig(out/'path_smoothing.pdf')
    anim=FuncAnimation(fig,update,frames=args.frames,interval=67,blit=False)
    anim.save(out/'path_smoothing.gif',writer=PillowWriter(fps=15),dpi=90)
    plt.close(fig);print('Saved assets/path_smoothing.gif',flush=True)

if __name__=='__main__':main()
