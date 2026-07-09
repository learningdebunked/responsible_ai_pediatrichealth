"""
PrenatalSentinel — Full Pipeline (optimized for speed + realistic AUC)
"""
import numpy as np, json, csv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, roc_curve, average_precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings; warnings.filterwarnings('ignore')

np.random.seed(42)
N = 5000

def roc_grid(y_true, y_score, n_points=100):
    """Interpolate ROC curve onto a fixed FPR grid spanning [0,1] so the
    saved curve is compact but still terminates at (1,1)."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    grid = np.linspace(0, 1, n_points)
    tpr_i = np.interp(grid, fpr, tpr)
    return grid, tpr_i

# ── Generate dataset ──────────────────────────────────────────────────────
rng = np.random.default_rng(42)
ses = rng.integers(1, 6, N)
se  = (ses - 3) / 4.0
fd  = np.clip(rng.beta(1.8,4,N)*(6-ses)/5 + rng.normal(0,.05,N), 0, 1)
uf  = rng.binomial(1, .72, N)

f1s = np.clip(rng.normal(.55+.15*se, .17, N), 0, 1)  # folate
f2s = np.clip(rng.normal(.45+.12*se, .17, N), 0, 1)  # omega3
f3s = np.clip(rng.normal(.50+.10*se, .16, N), 0, 1)  # iron
f4s = np.clip(rng.normal(.48+.10*se, .16, N), 0, 1)  # vitD
pfi = np.clip(rng.normal(.42-.08*se+.12*fd, .18, N), 0, 1)  # processed food
dds = np.clip(rng.normal(.55+.10*se-.08*fd, .17, N), 0, 1)  # diet diversity
sas = np.clip(rng.normal(.50+.14*se, .18, N), 0, 1)  # supplement adherence
mri = np.clip(rng.normal(.28-.06*se+.08*fd, .15, N), 0, 1)  # med risk
of  = np.clip(rng.normal(4.5+.5*se, 1.6, N), 1, 10)
bsm = np.clip(rng.normal(12+2*se, 4, N), 3, 30)

lo  = (-0.70 - 1.50*f1s - 1.00*f2s - .80*f3s - .70*f4s
       - 1.20*sas - .60*dds + 1.10*pfi + .90*mri
       + .75*fd - .50*(ses/5.0) + rng.normal(0, 1.0, N))
prob_asd = 1/(1+np.exp(-lo))
y = rng.binomial(1, prob_asd, N)

X = np.column_stack([f1s,f2s,f3s,f4s,pfi,dds,sas,mri,fd,ses/5.0,uf,of/10,bsm/30])
fn = ['folate_score','omega3_score','iron_score','vitD_score',
      'processed_food_index','dietary_diversity_score',
      'supplement_adherence_score','medication_risk_index',
      'food_desert_score','ses_quintile_norm','urban_flag',
      'order_frequency_norm','basket_size_norm']

print(f"Dataset: {N} samples, ASD rate={y.mean():.3f} ({y.sum()} cases)")
print(f"Oracle AUC: {roc_auc_score(y, prob_asd):.4f}")

# Save CSV
with open('../data/prenatal_sentinel_dataset.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(fn+['ses_quintile','urban_flag','asd_label'])
    for i in range(N):
        row=list(X[i])+[int(ses[i]),int(uf[i]),int(y[i])]
        w.writerow([f'{v:.6f}' if isinstance(v,float) else v for v in row])

# ── Baseline models ───────────────────────────────────────────────────────
cv = StratifiedKFold(5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': Pipeline([('sc',StandardScaler()),
        ('clf',LogisticRegression(C=1,max_iter=1000,class_weight='balanced',random_state=42))]),
    'Random Forest': Pipeline([('sc',StandardScaler()),
        ('clf',RandomForestClassifier(n_estimators=150,max_depth=10,
                                      min_samples_leaf=8,class_weight='balanced',random_state=42))]),
    'Gradient Boosting': Pipeline([('sc',StandardScaler()),
        ('clf',GradientBoostingClassifier(n_estimators=150,max_depth=4,
                                          learning_rate=0.06,subsample=0.8,random_state=42))])
}

baseline_results, roc_data = {}, {}
for name, model in models.items():
    yp = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:,1]
    ypred = (yp>=0.5).astype(int)
    auc=roc_auc_score(y,yp); f1=f1_score(y,ypred,zero_division=0)
    prec=precision_score(y,ypred,zero_division=0); rec=recall_score(y,ypred,zero_division=0)
    ap=average_precision_score(y,yp)
    fpr_g,tpr_g=roc_grid(y,yp)
    roc_data[name]={'fpr':fpr_g.tolist(),'tpr':tpr_g.tolist(),'auc':auc}
    baseline_results[name]={'AUC':round(auc,4),'F1':round(f1,4),
                             'Precision':round(prec,4),'Recall':round(rec,4),'AP':round(ap,4)}
    print(f"{name}: AUC={auc:.4f} F1={f1:.4f} P={prec:.4f} R={rec:.4f}")

# ── Federated Learning with DP ────────────────────────────────────────────
def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-30,30)))

def fl_dp(X, y, epsilon, n_clients=5, n_rounds=40, seed=42):
    rng2=np.random.default_rng(seed)
    n,d=X.shape; idx=np.arange(n); rng2.shuffle(idx)
    splits=np.array_split(idx,n_clients)
    Xs=StandardScaler().fit_transform(X)
    w=np.zeros(d); b=0.0; C=1.0; delta=1e-5
    for rnd in range(n_rounds):
        lr=0.10/(1+0.03*rnd); wu,bu=[],[]
        for cidx in splits:
            Xc,yc=Xs[cidx],y[cidx]
            n_c=len(yc); n_pos=yc.sum(); n_neg=n_c-n_pos
            if n_pos==0 or n_neg==0:
                sw=np.ones(n_c)
            else:
                sw=np.where(yc==1, n_c/(2*n_pos), n_c/(2*n_neg))
            err=(sigmoid(Xc@w+b)-yc)*sw
            gw=Xc.T@err/sw.sum(); gb=err.sum()/sw.sum()
            gnorm=np.linalg.norm(gw)
            if gnorm>C: gw=gw*C/gnorm
            if epsilon<float('inf'):
                sig=C*np.sqrt(2*np.log(1.25/delta))/epsilon
                gw+=rng2.normal(0,sig/len(yc),d)
                gb+=float(rng2.normal(0,sig/len(yc)))
            wu.append(gw); bu.append(gb)
        w-=lr*np.mean(wu,axis=0); b-=lr*np.mean(bu)
    yp=sigmoid(Xs@w+b)
    auc=roc_auc_score(y,yp); ypred=(yp>=0.5).astype(int)
    f1=f1_score(y,ypred,zero_division=0)
    prec=precision_score(y,ypred,zero_division=0)
    rec=recall_score(y,ypred,zero_division=0)
    fpr_g,tpr_g=roc_grid(y,yp)
    lbl=f'FL (ε={epsilon})' if epsilon<float('inf') else 'FL (No DP)'
    roc_data[lbl]={'fpr':fpr_g.tolist(),'tpr':tpr_g.tolist(),'auc':auc}
    return {'AUC':round(auc,4),'F1':round(f1,4),'Precision':round(prec,4),
            'Recall':round(rec,4),'epsilon':epsilon if epsilon<float('inf') else 'inf',
            'n_clients':n_clients,'n_rounds':n_rounds}

fl_results={}
for eps in [1,5,10,float('inf')]:
    r=fl_dp(X,y,eps)
    lbl=f'ε={eps}' if eps<float('inf') else 'ε=∞ (No DP)'
    fl_results[lbl]=r
    print(f"FL {lbl}: AUC={r['AUC']:.4f} F1={r['F1']:.4f}")

# ── Equity audit ──────────────────────────────────────────────────────────
gb_pipe=Pipeline([('sc',StandardScaler()),
    ('clf',GradientBoostingClassifier(n_estimators=150,max_depth=4,
                                      learning_rate=0.06,subsample=0.8,random_state=42))])
yp_cv=cross_val_predict(gb_pipe,X,y,cv=cv,method='predict_proba')[:,1]

equity_results={}
for q in range(1,6):
    mask=ses==q
    if mask.sum()<30 or y[mask].sum()<5: continue
    auc_q=roc_auc_score(y[mask],yp_cv[mask])
    equity_results[f'Q{q}']={'AUC':round(auc_q,4),'n':int(mask.sum()),
        'n_asd':int(y[mask].sum()),
        'food_desert_mean':round(float(fd[mask].mean()),4)}
    print(f"Q{q}: AUC={auc_q:.4f} n={mask.sum()} n_asd={y[mask].sum()} FD={fd[mask].mean():.3f}")

for lbl,mask in [('High Food Desert',fd>0.5),('Low Food Desert',fd<=0.5)]:
    if y[mask].sum()<5: continue
    auc_fd=roc_auc_score(y[mask],yp_cv[mask])
    equity_results[lbl]={'AUC':round(auc_fd,4),'n':int(mask.sum()),'n_asd':int(y[mask].sum())}
    print(f"{lbl}: AUC={auc_fd:.4f} n={mask.sum()}")

# ── Ablation ──────────────────────────────────────────────────────────────
signal_configs={
    'F only\n(Food)':[0,1,2,3,4,5],
    'F+M\n(Food+OTC)':[0,1,2,3,4,5,6,7],
    'F+M+E\n(Full)':[0,1,2,3,4,5,6,7,8,9,10],
    'F+M+E+B\n(+Basket)':[0,1,2,3,4,5,6,7,8,9,10,11,12],
    'All 13\nFeatures':list(range(13))
}
ablation_results={}
for cfg,fidx in signal_configs.items():
    m=Pipeline([('sc',StandardScaler()),
        ('clf',GradientBoostingClassifier(n_estimators=100,max_depth=3,
                                          learning_rate=0.07,subsample=0.8,random_state=42))])
    yp_a=cross_val_predict(m,X[:,fidx],y,cv=cv,method='predict_proba')[:,1]
    auc_a=roc_auc_score(y,yp_a); f1_a=f1_score(y,(yp_a>=0.5).astype(int),zero_division=0)
    ablation_results[cfg]={'AUC':round(auc_a,4),'F1':round(f1_a,4),'n_features':len(fidx)}
    print(f"Ablation {cfg.replace(chr(10),' ')}: AUC={auc_a:.4f}")

# ── Privacy-utility curve ─────────────────────────────────────────────────
pu_results={}
for eps in [0.5,1,2,5,10,20,50,float('inf')]:
    r=fl_dp(X,y,eps,n_rounds=40)
    pu_results[str(eps) if eps<float('inf') else 'inf']={'AUC':r['AUC'],'F1':r['F1']}

# ── Feature importance ────────────────────────────────────────────────────
fi_clf=GradientBoostingClassifier(n_estimators=150,max_depth=4,learning_rate=0.06,random_state=42)
fi_clf.fit(StandardScaler().fit_transform(X),y)
feature_importances={fn2:round(float(fi),4) for fn2,fi in zip(fn,fi_clf.feature_importances_)}
print("Top 5:",sorted(feature_importances.items(),key=lambda x:-x[1])[:5])

# ── Save ──────────────────────────────────────────────────────────────────
results={
    'dataset_stats':{'n_samples':N,'n_features':13,
        'asd_positive_rate':round(float(y.mean()),4),
        'n_asd':int(y.sum()),'n_controls':int((1-y).sum()),
        'feature_names':fn},
    'baseline_models':baseline_results,
    'federated_learning':fl_results,
    'privacy_utility_tradeoff':pu_results,
    'equity_audit':equity_results,
    'ablation_study':ablation_results,
    'feature_importances':feature_importances,
    'roc_curves':{k:{'fpr':v['fpr'][:100],'tpr':v['tpr'][:100],'auc':v['auc']}
                  for k,v in roc_data.items()}
}
with open('../data/results.json','w') as f: json.dump(results,f,indent=2)
print(f"\n✅ Saved. ASD={y.mean():.3f} n_asd={y.sum()}")
print(f"Best baseline: {max(v['AUC'] for v in baseline_results.values()):.4f}")
print(f"FL no-DP: {fl_results.get('ε=∞ (No DP)',{}).get('AUC')}")
print(f"FL ε=1:   {fl_results.get('ε=1',{}).get('AUC')}")
