from __future__ import annotations

import ast, json, math, os, re, time, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
DEGRADATION_CONFIRM_S = 3
DEGRADATION_RECOVERY_S = 10

# Direct-run configuration (no argparse)
DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
FAST_MODE = False

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    with path.open('r', encoding='utf-8-sig', errors='ignore') as f:
        first = f.readline()
    sep = ';' if first.count(';') > first.count(',') else ','
    return pd.read_csv(path, sep=sep, low_memory=False)


def coerce_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    x = s.astype(str).str.strip().str.lower()
    return x.isin({'1','true','t','yes','y','on'})


def numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')


def safe_col(df: pd.DataFrame, *names: str, default=np.nan) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(default, index=df.index)


def parse_route(v: Any) -> List[int]:
    if isinstance(v, list):
        return [int(x) for x in v if pd.notna(x)]
    if pd.isna(v):
        return []
    s = str(v).strip()
    if not s or s in {'[]','nan','None'}:
        return []
    try:
        out = ast.literal_eval(s)
        if isinstance(out, (list, tuple)):
            return [int(float(x)) for x in out]
    except Exception:
        pass
    return [int(x) for x in re.findall(r'-?\d+', s)]


def parse_legacy_elapsed_time(series: pd.Series, base='2026-03-01') -> pd.Series:

    vals=[]; day=pd.Timestamp(base); prev=None; offset=0.0
    pat=re.compile(r'^\s*(\d{1,2}):(\d{2})(?:\.(\d+))?\s*$')
    for raw in series.astype(str):
        m=pat.match(raw)
        if not m:
            vals.append(pd.NaT); continue
        mm=int(m.group(1)); ss=int(m.group(2)); frac=float('0.'+(m.group(3) or '0'))
        sec=mm*60+ss+frac
        if prev is not None and sec+offset < prev-30:
            offset += 3600.0
        cur=sec+offset; prev=cur
        vals.append(day + pd.to_timedelta(cur, unit='s'))
    return pd.to_datetime(vals)


def mode_or_last(x: pd.Series):
    z=x.dropna()
    if z.empty: return np.nan
    m=z.mode()
    return m.iloc[0] if not m.empty else z.iloc[-1]


def last_non_null(x: pd.Series):
    z=x.dropna(); return z.iloc[-1] if len(z) else np.nan


def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)


COMMON = [
    'session','timestamp','x','y','target_node','speed_mps','requested_speed_mps',
    'power_w','energy_cumulative','current_ma','voltage_mv','soc','heading',
    'target_reached','wheel_r','wheel_l','segment','position_confidence',
    'snap_node_logged','edge_u_logged','edge_v_logged','recorded_route',
    'status','tms_action','hold_reason','brake_hold','scanner_hold','operator_hold',
    'deadlock_hold','emergency_hold','safety_circuit','bumper','collision',
    'telemetry_age_s','network_condition','source_annotation_quality'
]


def harmonize_modern(df: pd.DataFrame, session: str) -> pd.DataFrame:
    # Exact logger duplicates carry no additional physical observation.
    df=df.drop_duplicates().copy()
    out=pd.DataFrame(index=df.index)
    out['session']=session
    out['timestamp']=pd.to_datetime(safe_col(df,'local_time'), errors='coerce')
    out['x']=numeric(safe_col(df,'x'))
    out['y']=numeric(safe_col(df,'y'))
    out['target_node']=numeric(safe_col(df,'going_to_id','current_goal','goal'))
    out['speed_mps']=numeric(safe_col(df,'speed_mps','abs_speed'))
    out['requested_speed_mps']=numeric(safe_col(df,'requested_straight_speed_mps'))
    out['power_w']=numeric(safe_col(df,'power_W'))
    out['energy_cumulative']=numeric(safe_col(df,'cumulative_energy_Wh'))
    out['current_ma']=numeric(safe_col(df,'current_mA'))
    out['voltage_mv']=numeric(safe_col(df,'cell_voltage_mV'))
    out['soc']=numeric(safe_col(df,'battery_percent','battery'))
    out['heading']=numeric(safe_col(df,'heading','plc_heading_deg'))
    out['target_reached']=coerce_bool(safe_col(df,'target_reached',default=False))
    out['wheel_r']=numeric(safe_col(df,'actual_speed_r'))
    out['wheel_l']=numeric(safe_col(df,'actual_speed_l'))
    out['segment']=numeric(safe_col(df,'current_segment'))
    out['position_confidence']=numeric(safe_col(df,'position_confidence'))
    out['snap_node_logged']=numeric(safe_col(df,'snap_node'))
    out['edge_u_logged']=numeric(safe_col(df,'u'))
    out['edge_v_logged']=numeric(safe_col(df,'v'))
    out['recorded_route']=safe_col(df,'series_leg_route','route','path',default='[]').map(parse_route)
    out['status']=safe_col(df,'status','series_status',default='').astype(str)
    out['tms_action']=safe_col(df,'tms_action','tms_runtime_action',default='').astype(str)
    out['hold_reason']=safe_col(df,'hold_reason','reason',default='').astype(str)
    txt=(out['status']+' '+out['tms_action']+' '+out['hold_reason']).str.lower()
    out['brake_hold']=coerce_bool(safe_col(df,'brake_block',default=False)) | txt.str.contains('brake')
    out['scanner_hold']=coerce_bool(safe_col(df,'scanner_violation_flag',default=False)) | txt.str.contains('scanner')
    status_text=out['status'].str.lower()
    hold_text=out['hold_reason'].str.lower()
    operator_stop=(
        status_text.str.contains(
            r'operator.*(?:brake|hold|stop|blocked)',
            regex=True,
        )
        | hold_text.str.contains(r'op\d+.*close',regex=True)
    )
    operator_release=status_text.str.contains(
        r'operator.*(?:clear|release|resume)',
        regex=True,
    )
    out['operator_hold']=operator_stop & ~operator_release
    out['deadlock_hold']=txt.str.contains('deadlock')
    out['emergency_hold']=coerce_bool(safe_col(df,'emergency_hold',default=False)) | txt.str.contains('emergency')
    out['safety_circuit']=False; out['bumper']=False; out['collision']=False
    out['telemetry_age_s']=numeric(safe_col(df,'telemetry_age_s','stale_position_s'))
    out['network_condition']=safe_col(df,'network_condition',default='recorded').astype(str)
    out['source_annotation_quality']='rich'
    return out[COMMON]


def harmonize_legacy(df: pd.DataFrame, session: str, safety_rich: bool=False) -> pd.DataFrame:
    # S3/S4 contain repeated exporter rows with identical timestamps/values.
    # Collapse exact duplicates before within-second aggregation so they do not
    # receive artificial statistical weight.
    df=df.drop_duplicates().copy()
    out=pd.DataFrame(index=df.index)
    out['session']=session
    if safety_rich:
        out['timestamp']=parse_legacy_elapsed_time(safe_col(df,'timestamp'))
    else:
        out['timestamp']=parse_legacy_elapsed_time(safe_col(df,'timestamp'))
    out['x']=numeric(safe_col(df,'X-coordinate'))
    out['y']=numeric(safe_col(df,'Y-coordinate'))
    out['target_node']=numeric(safe_col(df,'Going to ID'))
    out['speed_mps']=numeric(safe_col(df,'Speed'))
    out['requested_speed_mps']=np.nan
    out['power_w']=numeric(safe_col(df,'power consumption'))
    out['energy_cumulative']=numeric(safe_col(df,'Cumulative energy consumption'))
    out['current_ma']=numeric(safe_col(df,'current consuption'))
    out['voltage_mv']=np.nan
    out['soc']=numeric(safe_col(df,'Battery value'))
    out['heading']=numeric(safe_col(df,'Heading'))
    out['target_reached']=coerce_bool(safe_col(df,'Target reached','Target reached2','Target reached.1',default=False))
    out['wheel_r']=numeric(safe_col(df,'RIGHT DRIVE SIGNALS.ActualSpeed_R'))
    out['wheel_l']=numeric(safe_col(df,'LEFT DRIVE SIGNALS.ActualSpeed_L'))
    out['segment']=numeric(safe_col(df,'Current segment'))
    out['position_confidence']=numeric(safe_col(df,'Position confidence'))
    out['snap_node_logged']=np.nan; out['edge_u_logged']=np.nan; out['edge_v_logged']=np.nan
    out['recorded_route']=[[] for _ in range(len(out))]
    out['status']=''; out['tms_action']=''; out['hold_reason']=''
    if safety_rich:
        out['safety_circuit']=coerce_bool(safe_col(df,'Safety - Circuit Opened',default=False))
        out['emergency_hold']=coerce_bool(safe_col(df,'Safety - Front Emergency Stops Active',default=False))
        out['bumper']=coerce_bool(safe_col(df,'Safety - Front Emergency Bumper Active',default=False))
        out['collision']=coerce_bool(safe_col(df,'CR collision detected',default=False))
        protective=coerce_bool(safe_col(df,'Safety - Front Scanner Protective Zone Active',default=False))
        warning=coerce_bool(safe_col(df,'Safety - Front Scanner Warning Zone Active',default=False))
        out['scanner_hold']=protective | warning
        out['source_annotation_quality']='safety_rich'
    else:
        out['safety_circuit']=False; out['emergency_hold']=False; out['bumper']=False; out['collision']=False; out['scanner_hold']=False
        out['source_annotation_quality']='telemetry_only'
    out['brake_hold']=False; out['operator_hold']=False; out['deadlock_hold']=False
    out['telemetry_age_s']=np.nan; out['network_condition']='unknown'
    return out[COMMON]


def resample_1hz(df: pd.DataFrame, max_ffill_s: int=2) -> pd.DataFrame:

    df=df.dropna(subset=['timestamp']).sort_values('timestamp').copy()
    if df.empty: return df
    session=str(df['session'].dropna().iloc[0])
    ann=str(df['source_annotation_quality'].dropna().iloc[0]) if df['source_annotation_quality'].notna().any() else 'unknown'
    net=str(df['network_condition'].dropna().iloc[0]) if df['network_condition'].notna().any() else 'unknown'
    df['_sec']=df['timestamp'].dt.floor('s')
    mean_cols={'x','y','speed_mps','requested_speed_mps','power_w','current_ma','voltage_mv','heading','wheel_r','wheel_l','position_confidence','telemetry_age_s'}
    bool_or={'target_reached','brake_hold','scanner_hold','operator_hold','deadlock_hold','emergency_hold','safety_circuit','bumper','collision'}
    aggs={}
    for c in df.columns:
        if c in {'timestamp','_sec','session'}: continue
        aggs[c]='max' if c in bool_or else ('mean' if c in mean_cols else last_non_null)
    obs=df.groupby('_sec',sort=True).agg(aggs); obs.index.name='timestamp'
    full=pd.date_range(obs.index.min(),obs.index.max(),freq='1s')
    aligned=obs.reindex(full)
    observed=pd.Series(aligned.notna().any(axis=1).to_numpy(),index=full)
    ages=[]; age=0
    for flag in observed:
        age=0 if flag else age+1; ages.append(age)
    age_s=pd.Series(ages,index=full,dtype='int64')
    filled=aligned.ffill(limit=max_ffill_s)
    unavailable=age_s.gt(max_ffill_s)
    filled.loc[unavailable,:]=np.nan
    # metadata must exist on every grid row
    filled['session']=session; filled['source_annotation_quality']=ann; filled['network_condition']=net
    filled['observed_sample']=observed.astype('int8').to_numpy()
    filled['seconds_since_observed']=age_s.to_numpy()
    filled['telemetry_available']=(~unavailable).astype('int8').to_numpy()
    filled['short_causal_hold']=((~observed)&(~unavailable)).astype('int8').to_numpy()
    valid=~unavailable; starts=valid & ~valid.shift(fill_value=False)
    seg=starts.cumsum().astype('Int64'); seg.loc[unavailable]=pd.NA
    filled['temporal_segment_id']=seg.to_numpy()
    filled.index.name='timestamp'
    return filled.reset_index()

def load_graph(nodes_path: Path, edges_path: Path):
    nodes=read_table(nodes_path); edges=read_table(edges_path)
    nodes=nodes.rename(columns={'X-coordinate':'x','Y-coordinate':'y'})
    edges=edges.rename(columns={'From':'u','To':'v'})
    G=nx.DiGraph()
    for _,r in nodes.iterrows():
        G.add_node(int(r['Node']), x=float(r['x']), y=float(r['y']),
                   degree=float(r.get('Node_Degree',np.nan)),
                   corridor=float(r.get('Type_Corridor',0)),
                   intersection=float(r.get('Type_Intersection',0)),
                   station=float(r.get('Type_Station',0)))
    edge_lookup={}
    for _,r in edges.iterrows():
        u,v=int(r['u']),int(r['v']); d=float(r['distance'])
        G.add_edge(u,v,weight=d)
        edge_lookup[(u,v)]={'distance':d,'ax':float(r['X_from']),'ay':float(r['Y_from']),'bx':float(r['X_to']),'by':float(r['Y_to'])}
    node_xy=np.array([[int(r['Node']),float(r['x']),float(r['y'])] for _,r in nodes.iterrows()])
    return G, nodes, edges, edge_lookup, node_xy


def nearest_node(x: float,y: float,node_xy: np.ndarray)->Tuple[float,float]:
    if not np.isfinite(x) or not np.isfinite(y): return np.nan,np.nan
    d=np.hypot(node_xy[:,1]-x,node_xy[:,2]-y); j=int(np.argmin(d)); return float(node_xy[j,0]),float(d[j])


def edge_projection(x,y,e):
    ax,ay,bx,by=e['ax'],e['ay'],e['bx'],e['by']; vx,vy=bx-ax,by-ay
    den=vx*vx+vy*vy
    if den<=1e-12 or not np.isfinite(x) or not np.isfinite(y): return np.nan,np.nan,np.nan
    lam=((x-ax)*vx+(y-ay)*vy)/den; lam=float(np.clip(lam,0,1))
    px,py=ax+lam*vx,ay+lam*vy
    lateral=float(math.hypot(x-px,y-py)); remain=(1-lam)*e['distance']
    return lam,remain,lateral


def choose_route(row,G,current_node,target)->Tuple[List[int],str]:
    rr=row.get('recorded_route',[])
    if isinstance(rr,list) and len(rr)>=2 and current_node in rr:
        return rr,'recorded'
    if pd.notna(current_node) and pd.notna(target) and int(current_node) in G and int(target) in G:
        try: return nx.shortest_path(G,int(current_node),int(target),weight='weight'),'reconstructed_shortest'
        except nx.NetworkXNoPath: pass
    return [],'unavailable'


def add_graph_features(df,G,nodes,edge_lookup,node_xy):
    rows=[]; node_attr={int(r['Node']):r for _,r in nodes.iterrows()}
    for _,r in df.iterrows():
        if int(r.get('telemetry_available',1) or 0)==0:
            rows.append({k:np.nan for k in ['snap_node','snap_distance_m','edge_u','edge_v','edge_progress_fraction','edge_remaining_m','edge_lateral_error_m','route_index','route_total_m','graph_remaining_m','euclidean_remaining_m','node_degree','node_corridor','node_intersection','node_station']} | {'route_nodes':'[]','route_source':'telemetry_unavailable'})
            continue
        x=float(r['x']) if pd.notna(r['x']) else np.nan; y=float(r['y']) if pd.notna(r['y']) else np.nan
        logged=r.get('snap_node_logged',np.nan)
        if pd.notna(logged) and int(logged) in G:
            sn=int(logged); sd=math.hypot(x-G.nodes[sn]['x'],y-G.nodes[sn]['y']) if np.isfinite(x) else np.nan
        else:
            snf,sd=nearest_node(x,y,node_xy); sn=int(snf) if pd.notna(snf) else np.nan
        target=r.get('target_node',np.nan); target=int(target) if pd.notna(target) and int(target) in G else np.nan
        route,source=choose_route(r,G,sn,target)
        u=r.get('edge_u_logged',np.nan); v=r.get('edge_v_logged',np.nan)
        if pd.notna(u) and pd.notna(v) and (int(u),int(v)) in edge_lookup: u,v=int(u),int(v)
        else:
            u=v=np.nan
            if route and sn in route:
                idx=max(i for i,n in enumerate(route) if n==sn)
                if idx<len(route)-1 and (route[idx],route[idx+1]) in edge_lookup: u,v=route[idx],route[idx+1]
        remain=frac=lateral=route_remaining=route_total=route_index=np.nan
        if route and len(route)>=2:
            route_total=sum(edge_lookup[(a,b)]['distance'] for a,b in zip(route[:-1],route[1:]) if (a,b) in edge_lookup)
            if pd.notna(u) and (u,v) in edge_lookup:
                frac,erem,lateral=edge_projection(x,y,edge_lookup[(u,v)])
                try:
                    idx=next(i for i,(a,b) in enumerate(zip(route[:-1],route[1:])) if a==u and b==v); route_index=idx
                    tail=sum(edge_lookup[(a,b)]['distance'] for a,b in zip(route[idx+1:-1],route[idx+2:]) if (a,b) in edge_lookup)
                    # Include the off-route connector. Without it, a graph
                    # path can be shorter than the straight-line distance.
                    connector=lateral if pd.notna(lateral) else 0.0
                    remain=erem; route_remaining=connector+erem+tail
                except StopIteration: pass
            if pd.isna(route_remaining) and sn in route:
                idx=route.index(sn); route_index=idx
                connector=sd if pd.notna(sd) else 0.0
                route_remaining=connector+sum(edge_lookup[(a,b)]['distance'] for a,b in zip(route[idx:-1],route[idx+1:]) if (a,b) in edge_lookup)
        elif route and len(route)==1:
            route_index=0
            route_total=sd if pd.notna(sd) else 0.0
            route_remaining=route_total
        euclid=math.hypot(x-G.nodes[target]['x'],y-G.nodes[target]['y']) if pd.notna(target) and np.isfinite(x) else np.nan
        attrs=node_attr.get(int(sn),{}) if pd.notna(sn) else {}
        rows.append({'snap_node':sn,'snap_distance_m':sd,'edge_u':u,'edge_v':v,'edge_progress_fraction':frac,'edge_remaining_m':remain,'edge_lateral_error_m':lateral,'route_nodes':json.dumps(route),'route_source':source,'route_index':route_index,'route_total_m':route_total,'graph_remaining_m':route_remaining,'euclidean_remaining_m':euclid,'node_degree':attrs.get('Node_Degree',np.nan),'node_corridor':attrs.get('Type_Corridor',0),'node_intersection':attrs.get('Type_Intersection',0),'node_station':attrs.get('Type_Station',0)})
    out=pd.concat([df.reset_index(drop=True),pd.DataFrame(rows)],axis=1)
    out['route_completion']=(1-out['graph_remaining_m']/out['route_total_m'].replace(0,np.nan)).clip(0,1)
    keys=['session','temporal_segment_id']
    for h in [1,3,5,10,20,30]:
        grp=out.groupby(keys,dropna=True)
        out[f'graph_progress_{h}s_m']=grp['graph_remaining_m'].shift(h)-out['graph_remaining_m']
        out[f'euclidean_progress_{h}s_m']=grp['euclidean_remaining_m'].shift(h)-out['euclidean_remaining_m']
        out[f'displacement_{h}s_m']=np.hypot(out['x']-grp['x'].shift(h),out['y']-grp['y'].shift(h))
    out['graph_progress_rate_mps']=out['graph_progress_5s_m']/5; out['euclidean_progress_rate_mps']=out['euclidean_progress_5s_m']/5
    out['wheel_mean']=out[['wheel_l','wheel_r']].mean(axis=1); out['wheel_abs_diff']=(out['wheel_l']-out['wheel_r']).abs()
    command_observed=(
        out['requested_speed_mps'].notna()
        & out['speed_mps'].notna()
    )
    out['command_mismatch']=np.where(
        command_observed,
        (
            out['requested_speed_mps'].gt(0.05)
            & out['speed_mps'].lt(0.03)
        ).astype(float),
        np.nan,
    )
    return out

def rolling_features(df):
    out=df.copy(); keys=['session','temporal_segment_id']; group=out.groupby(keys,dropna=True,group_keys=False)
    for w in [5,10,30]:
        out[f'speed_mean_{w}s']=group['speed_mps'].transform(lambda x:x.rolling(w,min_periods=max(2,w//3)).mean())
        out[f'speed_std_{w}s']=group['speed_mps'].transform(lambda x:x.rolling(w,min_periods=max(2,w//3)).std())
        out[f'stop_share_{w}s']=group['speed_mps'].transform(lambda x:(x<0.03).rolling(w,min_periods=max(2,w//3)).mean())
        out[f'power_mean_{w}s']=group['power_w'].transform(lambda x:x.rolling(w,min_periods=max(2,w//3)).mean())
        out[f'power_std_{w}s']=group['power_w'].transform(lambda x:x.rolling(w,min_periods=max(2,w//3)).std())
        out[f'soc_drop_{w}s']=group['soc'].transform(lambda x:x.shift(w)-x)
        out[f'command_mismatch_share_{w}s']=group['command_mismatch'].transform(lambda x:x.rolling(w,min_periods=max(2,w//3)).mean())
    out['edge_dwell_s']=out.groupby(['session','temporal_segment_id','edge_u','edge_v'],dropna=True).cumcount()+1
    return out

def construct_missions(df):
    out=df.copy(); target=out['target_node'].fillna(-1)
    previous_target=target.groupby(out['session']).shift()
    reached=out['target_reached'].fillna(False).astype(bool)
    reached_rise=reached & ~reached.groupby(out['session']).shift(fill_value=False)
    segment_changed=out['temporal_segment_id'].ne(
        out.groupby('session')['temporal_segment_id'].shift()
    )
    # Start a new leg only when the target changes, target_reached rises from
    # false to true, or a telemetry segment changes. A sustained true
    # target_reached flag must not create a new mission leg every second.
    changed=target.ne(previous_target) | reached_rise | segment_changed
    out['mission_leg_local']=changed.groupby(out['session']).cumsum().fillna(0).astype('int64')
    out['mission_leg_id']=out['session'].astype(str)+'_L'+out['mission_leg_local'].astype(str)

    # Progress must reset at a target/mission-leg boundary. Computing shifted
    # distances across two different targets creates artificial progress jumps.
    progress_keys=['session','temporal_segment_id','mission_leg_id']
    progress_group=out.groupby(progress_keys,dropna=True)
    for h in [1,3,5,10,20,30]:
        out[f'graph_progress_{h}s_m']=progress_group['graph_remaining_m'].shift(h)-out['graph_remaining_m']
        out[f'euclidean_progress_{h}s_m']=progress_group['euclidean_remaining_m'].shift(h)-out['euclidean_remaining_m']
        out[f'displacement_{h}s_m']=np.hypot(
            out['x']-progress_group['x'].shift(h),
            out['y']-progress_group['y'].shift(h),
        )
    out['graph_progress_rate_mps']=out['graph_progress_5s_m']/5
    out['euclidean_progress_rate_mps']=out['euclidean_progress_5s_m']/5

    # Route completion is relative to the first valid remaining distance in
    # the current mission leg. This remains meaningful for reconstructed
    # routes whose shortest path is refreshed at each sample.
    initial_remaining=progress_group['graph_remaining_m'].transform('first')
    out['route_completion']=(
        1-out['graph_remaining_m']/initial_remaining.replace(0,np.nan)
    ).clip(0,1)
    out['edge_dwell_s']=out.groupby(
        progress_keys+['edge_u','edge_v'],dropna=True
    ).cumcount()+1
    external=out[['brake_hold','scanner_hold','operator_hold','deadlock_hold','emergency_hold','safety_circuit','bumper','collision']].fillna(False).any(axis=1)
    active=(out['target_node'].fillna(0)>0)&(~out['target_reached'].fillna(False).astype(bool))&(out['telemetry_available']==1)
    fresh=(out['telemetry_available']==1)&(out['seconds_since_observed']<=2)
    out['external_hold']=external.astype(int); out['active_mission']=active.astype(int); out['telemetry_fresh']=fresh.astype(int)
    progress=out['graph_progress_5s_m'].fillna(out['euclidean_progress_5s_m'])
    stalled=(out['stop_share_10s']>=0.60)&(progress<0.03)
    degrading=((out['speed_mean_10s']<0.055)&(progress<0.03))|(out['command_mismatch_share_10s']>=0.5)
    state=np.full(len(out),'INACTIVE',object); state[active.values]='PROGRESS'; state[(active&external).values]='EXPECTED_HOLD'; state[(active&~external&degrading).values]='DEGRADING'; state[(active&~external&stalled).values]='STALLED'; state[(out['telemetry_available']==0).values]='TELEMETRY_UNAVAILABLE'
    prior_bad=pd.Series(np.isin(state,['DEGRADING','STALLED']),index=out.index).groupby([out['session'],out['temporal_segment_id']],dropna=True).transform(lambda x:x.shift(1).rolling(5,min_periods=1).max()).fillna(0).astype(bool)
    recovery=active&~external&prior_bad&(progress>=0.05)&(out['speed_mean_5s']>=0.05); state[recovery.values]='RECOVERING'
    out['mission_state']=state; return out


def _causal_state_for_group(
    group: pd.DataFrame,
    history_window: int,
    min_observed_samples: int,
    min_observed_span_s: int,
) -> pd.DataFrame:

    g=group.sort_values('timestamp').copy()
    observed=(
        g['observed_sample'].fillna(0).eq(1)
        & g['speed_mps'].notna()
        & g['x'].notna()
        & g['y'].notna()
    )
    speed_observed=g['speed_mps'].where(observed)
    stop_observed=(g['speed_mps']<0.03).where(observed)

    min_periods=max(1,min_observed_samples)
    observed_count=observed.astype(int).rolling(
        history_window,min_periods=1
    ).sum()
    speed_mean=speed_observed.rolling(
        history_window,min_periods=min_periods
    ).mean()
    stop_share=stop_observed.astype(float).rolling(
        history_window,min_periods=min_periods
    ).mean()
    active_share=g['active_mission'].astype(float).rolling(
        history_window,min_periods=history_window
    ).mean()
    external_share=g['external_hold'].astype(float).rolling(
        history_window,min_periods=history_window
    ).mean()

    x_values=g['x'].to_numpy(dtype=float)
    y_values=g['y'].to_numpy(dtype=float)
    observed_values=observed.to_numpy(dtype=bool)
    displacement=np.full(len(g),np.nan,dtype=float)
    observed_span=np.full(len(g),np.nan,dtype=float)

    for end in range(len(g)):
        start=max(0,end-history_window+1)
        positions=np.flatnonzero(observed_values[start:end+1])+start
        if len(positions)<min_observed_samples:
            continue
        first=int(positions[0]); last=int(positions[-1])
        observed_span[end]=float(last-first)
        if observed_span[end]<min_observed_span_s:
            continue
        displacement[end]=float(np.hypot(
            x_values[last]-x_values[first],
            y_values[last]-y_values[first],
        ))

    valid=(
        observed_count.ge(min_observed_samples)
        & pd.Series(observed_span,index=g.index).ge(min_observed_span_s)
        & speed_mean.notna()
        & stop_share.notna()
        & active_share.notna()
        & external_share.notna()
        & pd.Series(displacement,index=g.index).notna()
    )
    degraded=(
        active_share.ge(0.60)
        & external_share.lt(0.20)
        & (stop_share.ge(0.60)|speed_mean.lt(0.055))
        & pd.Series(displacement,index=g.index).lt(0.03)
    )

    result=pd.DataFrame(index=g.index)
    result['state_observed_count_10s']=observed_count.astype(float)
    result['state_observed_span_s']=observed_span
    result['state_speed_mean_10s']=speed_mean.astype(float)
    result['state_stop_share_10s']=stop_share.astype(float)
    result['state_cartesian_displacement_10s_m']=displacement
    result['current_state_valid']=valid.astype('int8')
    result['degradation_candidate']=np.where(valid,degraded.astype(int),np.nan)
    return result.reindex(group.index)


def _apply_degradation_hysteresis(
    group: pd.DataFrame,
    confirm_s: int,
    recovery_s: int,
) -> pd.DataFrame:
    """Convert noisy second-wise candidates into causal degradation episodes."""
    g=group.sort_values('timestamp')
    n=len(g)
    state=np.full(n,np.nan,dtype=float)
    onset=np.zeros(n,dtype=np.int8)
    candidate=g['degradation_candidate'].to_numpy(dtype=float)
    valid=g['current_state_valid'].eq(1).to_numpy(dtype=bool)
    active=g['active_mission'].eq(1).to_numpy(dtype=bool)
    external=g['external_hold'].eq(1).to_numpy(dtype=bool)
    available=g['telemetry_available'].eq(1).to_numpy(dtype=bool)

    latched=False
    bad_run=0
    recovery_run=0
    for i in range(n):
        if not valid[i]:
            # An unobservable second is not evidence of recovery. Preserve
            # the episode latch and require genuine valid recovered seconds
            # before re-arming; otherwise a short observation-quality break
            # can create a second onset only a few seconds later.
            bad_run=0
            recovery_run=0
            continue
        if not active[i] or external[i] or not available[i]:
            # External/inactive/unavailable intervals censor follow-up but do
            # not prove that a previously confirmed degradation recovered.
            bad_run=0
            recovery_run=0
            state[i]=0.0
            continue

        if candidate[i]==1.0:
            bad_run+=1
            recovery_run=0
            if not latched and bad_run>=confirm_s:
                latched=True
                onset[i]=1
        else:
            bad_run=0
            if latched:
                recovery_run+=1
                if recovery_run>=recovery_s:
                    latched=False
                    recovery_run=0
            else:
                recovery_run=0
        state[i]=1.0 if latched else 0.0

    return pd.DataFrame(
        {'current_degraded':state,'event_onset':onset},
        index=g.index,
    ).reindex(group.index)


def add_future_labels(
    df,
    horizons=(5,10,20),
    history_window=10,
    min_observed_samples=4,
    min_observed_span_s=6,
    degradation_confirm_s=DEGRADATION_CONFIRM_S,
    degradation_recovery_s=DEGRADATION_RECOVERY_S,
):

    out=df.sort_values(['session','timestamp']).copy()

    state_parts=[]
    state_keys=['session','temporal_segment_id','mission_leg_id']
    for _,group in out.groupby(state_keys,sort=False,dropna=True):
        state_parts.append(_causal_state_for_group(
            group,
            history_window=history_window,
            min_observed_samples=min_observed_samples,
            min_observed_span_s=min_observed_span_s,
        ))
    state_frame=pd.concat(state_parts).reindex(out.index) if state_parts else pd.DataFrame(index=out.index)
    for column in state_frame.columns:
        out[column]=state_frame[column]

    episode_parts=[]
    for _,group in out.groupby(state_keys,sort=False,dropna=True):
        episode_parts.append(_apply_degradation_hysteresis(
            group,
            confirm_s=degradation_confirm_s,
            recovery_s=degradation_recovery_s,
        ))
    episode_frame=(
        pd.concat(episode_parts).reindex(out.index)
        if episode_parts else pd.DataFrame(index=out.index)
    )
    out['current_degraded']=episode_frame['current_degraded']
    out['event_onset']=episode_frame['event_onset'].fillna(0).astype('int8')
    out['at_risk']=(
        out['current_state_valid'].eq(1)
        & out['current_degraded'].eq(0)
        & out['active_mission'].eq(1)
        & out['external_hold'].eq(0)
        & out['telemetry_fresh'].eq(1)
        & out['target_reached'].fillna(False).eq(False)
    ).astype('int8')

    out['time_to_onset_s']=np.nan
    out['followup_available_s']=0.0
    for horizon in horizons:
        out[f'label_h{horizon}']=np.nan

    group_keys=['session','temporal_segment_id','mission_leg_id']
    for _,group in out.groupby(group_keys,sort=False,dropna=True):
        g=group.sort_values('timestamp')
        indices=g.index.to_list()
        onset_values=g['event_onset'].to_numpy(dtype=int)
        active_values=g['active_mission'].to_numpy(dtype=int)
        external_values=g['external_hold'].to_numpy(dtype=int)
        available_values=g['telemetry_available'].to_numpy(dtype=int)
        reached_values=g['target_reached'].fillna(False).to_numpy(dtype=bool)
        at_risk_values=g['at_risk'].to_numpy(dtype=int)
        max_horizon=max(horizons)

        for local_i,index in enumerate(indices):
            if at_risk_values[local_i]!=1:
                continue

            event_time=None
            followup=0
            for step in range(1,max_horizon+1):
                local_j=local_i+step
                if local_j>=len(indices):
                    break
                # Exact 1-Hz continuity is required inside the mission leg.
                expected=g.iloc[local_i]['timestamp']+pd.Timedelta(seconds=step)
                if g.iloc[local_j]['timestamp']!=expected:
                    break
                if (
                    active_values[local_j]!=1
                    or external_values[local_j]==1
                    or available_values[local_j]!=1
                    or reached_values[local_j]
                ):
                    break
                followup=step
                if onset_values[local_j]==1:
                    event_time=step
                    break

            out.at[index,'followup_available_s']=float(followup)
            if event_time is not None:
                out.at[index,'time_to_onset_s']=float(event_time)

            for horizon in horizons:
                if event_time is not None and event_time<=horizon:
                    out.at[index,f'label_h{horizon}']=1.0
                elif followup>=horizon:
                    out.at[index,f'label_h{horizon}']=0.0

    # Retain a causal h=0 label for recognition and persistence baselines.
    out['label_h0']=out['current_degraded'].where(
        out['current_state_valid'].eq(1)
    ).astype(float)

    current_episode=(
        out['event_onset']
        .groupby(out['session'])
        .cumsum()
        .where(out['current_degraded'].eq(1),0)
    )
    out['event_id']=current_episode.astype('int64')
    out['event_id_global']=np.where(
        out['event_id']>0,
        out['session']+'_E'+out['event_id'].astype(str),
        ''
    )
    return out.sort_index()

NUM_FEATURES=[
 'speed_mps','speed_mean_5s','speed_mean_10s','speed_mean_30s','speed_std_10s','stop_share_5s','stop_share_10s','stop_share_30s',
 'power_w','power_mean_10s','power_std_10s','current_ma','voltage_mv','soc','soc_drop_10s','soc_drop_30s',
 'wheel_mean','wheel_abs_diff','command_mismatch_share_10s','command_mismatch_share_30s',
 'graph_remaining_m','route_completion','graph_progress_1s_m','graph_progress_3s_m','graph_progress_5s_m','graph_progress_10s_m','graph_progress_30s_m','graph_progress_rate_mps',
 'euclidean_remaining_m','euclidean_progress_5s_m','euclidean_progress_rate_mps','edge_progress_fraction','edge_remaining_m','edge_lateral_error_m','edge_dwell_s',
 'node_degree','node_corridor','node_intersection','node_station','position_confidence','seconds_since_observed'
]
CAT_FEATURES=['route_source']

def make_preprocessor(num,cat):
    return ColumnTransformer([
        ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)
    ])


def metrics(y,p,pred):
    out={'n':len(y),'positive_rate':float(np.mean(y)),'accuracy':accuracy_score(y,pred),
         'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
         'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),
         'f1_positive':f1_score(y,pred,zero_division=0),'brier':brier_score_loss(y,p)}
    out['roc_auc']=roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan
    out['pr_auc']=average_precision_score(y,p) if len(np.unique(y))>1 else np.nan
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); out.update({'tn':tn,'fp':fp,'fn':fn,'tp':tp})
    return out


def alert_policy(prob,on=0.60,off=0.35,k=2,n=3,cooldown=5):
    state=False; cool=0; hist=[]; arr=[]
    for p in prob:
        hist.append(float(p)); hist=hist[-n:]
        if cool>0: cool-=1
        if not state and cool==0 and sum(v>=on for v in hist)>=k: state=True
        elif state and p<=off: state=False; cool=cooldown
        arr.append(int(state))
    return np.array(arr)


def event_alert_metrics(g,y,alert):
    z=g.copy(); z['y']=y; z['alert']=alert
    starts=z['alert'].eq(1)&z['alert'].shift(fill_value=0).eq(0)
    hours=max((z['timestamp'].max()-z['timestamp'].min()).total_seconds()/3600,1e-9)
    false_starts=starts & z['y'].eq(0)
    event_ids=z.loc[z['y'].eq(1),'event_id_global'].replace('',np.nan).dropna().unique()
    warned=0; leads=[]
    for eid in event_ids:
        idx=z.index[z['event_id_global'].eq(eid)]
        onset=z.loc[idx,'timestamp'].min()
        before=z[(z['timestamp']<=onset)&(z['timestamp']>=onset-pd.Timedelta(seconds=30))&z['alert'].eq(1)]
        if len(before): warned+=1; leads.append((onset-before['timestamp'].min()).total_seconds())
    return {'alert_episodes':int(starts.sum()),'false_alert_episodes':int(false_starts.sum()),
            'false_alerts_per_hour':float(false_starts.sum()/hours),'events':int(len(event_ids)),
            'warned_events':int(warned),'event_recall':warned/len(event_ids) if len(event_ids) else np.nan,
            'median_lead_s':float(np.median(leads)) if leads else np.nan,
            'pct_events_lead_ge_2s':float(np.mean(np.array(leads)>=2)) if leads else np.nan,
            'pct_events_lead_ge_5s':float(np.mean(np.array(leads)>=5)) if leads else np.nan}


def run_loso(df,outdir,fast=False):
    eligible=df[(df['active_mission']==1)&(df['external_hold']==0)&(df['telemetry_fresh']==1)&df['label_h5'].notna()].copy()
    eligible=eligible[eligible['source_annotation_quality']!='telemetry_only'].copy() if (eligible['source_annotation_quality']=='telemetry_only').all()==False else eligible
    # retain telemetry-only S3 but mark; not silently exclude all. primary and sensitivity outputs
    variants={'all_sessions':df[(df['active_mission']==1)&(df['external_hold']==0)&(df['telemetry_fresh']==1)&df['label_h5'].notna()].copy(),
              'annotated_sessions':df[(df['active_mission']==1)&(df['external_hold']==0)&(df['telemetry_fresh']==1)&df['label_h5'].notna()&(df['source_annotation_quality']!='telemetry_only')].copy()}
    all_results=[]; all_preds=[]; all_alerts=[]; latency=[]
    for variant,data in variants.items():
        sessions=sorted(data['session'].unique())
        if len(sessions)<2: continue
        for test_s in sessions:
            tr=data[data.session!=test_s]; te=data[data.session==test_s].copy()
            ytr=tr['label_h5'].astype(int); yte=te['label_h5'].astype(int)
            if ytr.nunique()<2 or len(te)<20: continue
            num=[c for c in NUM_FEATURES if c in data.columns and tr[c].notna().any()]; cat=[c for c in CAT_FEATURES if c in data.columns and tr[c].notna().any()]
            models=({'ExtraTrees':ExtraTreesClassifier(n_estimators=40,min_samples_leaf=3,class_weight='balanced',n_jobs=-1,random_state=RANDOM_STATE)} if fast else {
              'LogisticRegression':LogisticRegression(max_iter=1000,class_weight='balanced',random_state=RANDOM_STATE),
              'ExtraTrees':ExtraTreesClassifier(n_estimators=300,min_samples_leaf=3,class_weight='balanced',n_jobs=-1,random_state=RANDOM_STATE),
              'RandomForest':RandomForestClassifier(n_estimators=250,min_samples_leaf=3,class_weight='balanced_subsample',n_jobs=-1,random_state=RANDOM_STATE),
            })
            for name,model in models.items():
                pipe=Pipeline([('prep',make_preprocessor(num,cat)),('model',model)])
                t0=time.perf_counter(); pipe.fit(tr[num+cat],ytr); fit=time.perf_counter()-t0
                t0=time.perf_counter(); p=pipe.predict_proba(te[num+cat])[:,1]; infer=time.perf_counter()-t0
                pred=(p>=0.5).astype(int); m=metrics(yte,p,pred); m.update({'variant':variant,'test_session':test_s,'model':name,'fit_s':fit,'infer_ms_per_row':infer*1000/len(te)})
                all_results.append(m)
                tmp=te[['session','timestamp','mission_leg_id','event_id_global','label_h5']].copy(); tmp['model']=name; tmp['variant']=variant; tmp['prob']=p; tmp['pred']=pred
                alert=alert_policy(p); tmp['alert']=alert; all_preds.append(tmp)
                am=event_alert_metrics(tmp,yte.to_numpy(),alert); am.update({'variant':variant,'test_session':test_s,'model':name}); all_alerts.append(am)
                # latency repeated mini-batches
                sample=te[num+cat].iloc[:min(500,len(te))]
                reps=3 if fast else 50; vals=[]
                for _ in range(reps):
                    s=time.perf_counter_ns(); pipe.predict_proba(sample); vals.append((time.perf_counter_ns()-s)/1e6/len(sample))
                latency.append({'variant':variant,'test_session':test_s,'model':name,'p50_ms_row':np.percentile(vals,50),'p95_ms_row':np.percentile(vals,95),'p99_ms_row':np.percentile(vals,99)})
            # rule baselines
            for name,pred in {
                'RuleGraph':((te['stop_share_10s']>=0.6)&(te['graph_progress_10s_m'].fillna(-1)<0.03)).astype(int),
                'RuleEuclidean':((te['stop_share_10s']>=0.6)&(te['euclidean_progress_10s_m'].fillna(-1)<0.03)).astype(int),
                'Persistence':((te['stop_share_5s']>=0.6)&(te['graph_progress_5s_m'].fillna(te['euclidean_progress_5s_m'])<0.03)).astype(int)
            }.items():
                p=pred.astype(float).to_numpy(); m=metrics(yte,p,pred.to_numpy()); m.update({'variant':variant,'test_session':test_s,'model':name,'fit_s':0,'infer_ms_per_row':0}); all_results.append(m)
                tmp=te[['session','timestamp','mission_leg_id','event_id_global','label_h5']].copy(); tmp['model']=name; tmp['variant']=variant; tmp['prob']=p; tmp['pred']=pred.to_numpy(); alert=alert_policy(p); tmp['alert']=alert; all_preds.append(tmp)
                am=event_alert_metrics(tmp,yte.to_numpy(),alert); am.update({'variant':variant,'test_session':test_s,'model':name}); all_alerts.append(am)
    pd.DataFrame(all_results).to_csv(outdir/'loso_results.csv',index=False)
    pd.concat(all_preds,ignore_index=True).to_csv(outdir/'loso_predictions.csv',index=False)
    pd.DataFrame(all_alerts).to_csv(outdir/'alert_episode_metrics.csv',index=False)
    pd.DataFrame(latency).to_csv(outdir/'latency_benchmark.csv',index=False)
    return pd.DataFrame(all_results),pd.concat(all_preds,ignore_index=True),pd.DataFrame(all_alerts)


def gap_stress(df,outdir):
    rows=[]
    base=df[(df.active_mission==1)&(df.external_hold==0)].copy()
    for gap in [1,2,3,5,10]:
        for s,g in base.groupby('session'):
            if len(g)<50: continue
            rng=np.random.default_rng(RANDOM_STATE+gap)
            mask=np.zeros(len(g),bool)
            starts=rng.choice(max(1,len(g)-gap),size=max(1,len(g)//200),replace=False)
            for st in starts: mask[st:st+gap]=True
            suppressed=(gap>2)
            rows.append({'session':s,'gap_s':gap,'injected_rows':int(mask.sum()),'policy':'suppress_after_2s' if suppressed else 'limited_ffill',
                         'expected_suppressed_rows':int(mask.sum()) if suppressed else 0})
    pd.DataFrame(rows).to_csv(outdir/'communication_gap_stress.csv',index=False)


def make_figures(df, results, preds, outdir):

    figdir = outdir / "figures"
    ensure_dir(figdir)

    # Plot SOC against elapsed session time, not unrelated synthetic dates.
    plt.figure(figsize=(8, 4.5))
    for session, group in df.groupby("session"):
        group = group.sort_values("timestamp")
        elapsed_min = (group["timestamp"] - group["timestamp"].min()).dt.total_seconds() / 60.0
        plt.plot(elapsed_min, group["soc"], label=session)
    plt.xlabel("Elapsed session time (min)")
    plt.ylabel("SOC (%)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(figdir / "session_soc_elapsed.png", dpi=250)
    plt.close()

    # Graph-route versus Euclidean remaining distance for one mission leg.
    example = df[df["active_mission"].eq(1)].dropna(
        subset=["graph_remaining_m", "euclidean_remaining_m"]
    )
    if len(example):
        leg = example["mission_leg_id"].value_counts().index[0]
        group = example[example["mission_leg_id"].eq(leg)].sort_values("timestamp")
        elapsed_s = (group["timestamp"] - group["timestamp"].min()).dt.total_seconds()
        plt.figure(figsize=(8, 4.5))
        plt.plot(elapsed_s, group["graph_remaining_m"], label="Graph-route remaining")
        plt.plot(elapsed_s, group["euclidean_remaining_m"], label="Euclidean remaining")
        plt.ylabel("Remaining distance (m)")
        plt.xlabel("Elapsed mission time (s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figdir / "graph_vs_euclidean.png", dpi=250)
        plt.close()


def audit(raws,harmonized,resampled,outdir):
    rows=[]
    for s,raw in raws.items():
        h=harmonized[s]; r=resampled[resampled.session==s]
        rows.append({'session':s,'raw_rows':len(raw),
                     'exact_duplicate_raw_rows':int(raw.duplicated().sum()),
                     'harmonized_rows_after_deduplication':len(h),
                     'valid_time_rows':h.timestamp.notna().sum(),'aligned_1hz_rows':len(r),
                     'start':h.timestamp.min(),'end':h.timestamp.max(),'duration_min':(h.timestamp.max()-h.timestamp.min()).total_seconds()/60 if h.timestamp.notna().any() else np.nan,
                     'soc_min':h.soc.min(),'soc_max':h.soc.max(),'targets':h.target_node.nunique(dropna=True),'annotation_quality':h.source_annotation_quality.mode().iloc[0],
                     'recorded_route_rows':sum(bool(x) for x in h.recorded_route),'safety_hold_rows':int(h[['scanner_hold','emergency_hold','safety_circuit','bumper','collision']].any(axis=1).sum())})
    pd.DataFrame(rows).to_csv(outdir/'session_audit.csv',index=False)
    # feature availability
    avail=[]
    for s,h in harmonized.items():
        for c in COMMON:
            avail.append({'session':s,'feature':c,'non_null_fraction':float(h[c].notna().mean()),'unique':int(h[c].nunique(dropna=True)) if c!='recorded_route' else np.nan})
    pd.DataFrame(avail).to_csv(outdir/'feature_availability.csv',index=False)


def main():
    data_dir = DATA_DIR
    out_dir = OUT_DIR
    ensure_dir(out_dir)
    files={
      'S1_LOW_SOC_STRESS':data_dir/'S1_LOW_SOC_STRESS.csv',
      'S2_HIGH_SOC_CONTROL':data_dir/'S2_HIGH_SOC_CONTROL.csv',
      'S3_MEDIUM_SOC_WHOLETESTING':data_dir/'S3_MEDIUM_SOC_WHOLETESTING.csv',
      'S4_SAFETY_RICH_NAVEEN12':data_dir/'S4_SAFETY_RICH_NAVEEN12.csv',
      'S5_REAL_TMS_JULY10':data_dir/'S5_REAL_TMS_JULY10.csv'}
    for p in list(files.values())+[data_dir/'Node_F3.csv',data_dir/'Edge_Distances3.csv']:
        if not p.exists(): raise FileNotFoundError(p)
    raws={s:read_table(p) for s,p in files.items()}
    hs={}
    for s,raw in raws.items():
        hs[s]=harmonize_legacy(raw,s,s.startswith('S4')) if s.startswith(('S3','S4')) else harmonize_modern(raw,s)
    res=pd.concat([resample_1hz(h) for h in hs.values()],ignore_index=True)
    G,nodes,edges,edge_lookup,node_xy=load_graph(data_dir/'Node_F3.csv',data_dir/'Edge_Distances3.csv')
    data=add_graph_features(res,G,nodes,edge_lookup,node_xy)
    data=rolling_features(data); data=construct_missions(data); data=add_future_labels(data)
    audit(raws,hs,data,out_dir)
    data.to_csv(out_dir/'harmonized_graph_mission_state.csv',index=False)
    # event table
    ev=data[data.event_onset==1][['session','timestamp','mission_leg_id','soc','graph_remaining_m','euclidean_remaining_m','mission_state','route_source']]
    ev.to_csv(out_dir/'event_onsets.csv',index=False)
    # Scientific data-quality figures only. Model experiments are intentionally
    # separated into 02_run_scientific_experiments.py to avoid mixing old and
    # final evaluation protocols.
    make_figures(data, pd.DataFrame(), pd.DataFrame(), out_dir)

    label_definition = {
        'causal_history_window_s': 10,
        'future_onset_horizons_s': [5, 10, 20],
        'speed_stop_threshold_mps': 0.03,
        'past_stop_share_threshold': 0.60,
        'past_mean_speed_threshold_mps': 0.055,
        'past_cartesian_displacement_threshold_m': 0.03,
        'minimum_active_mission_share': 0.60,
        'maximum_external_hold_share': 0.20,
        'minimum_genuine_observations': 4,
        'minimum_observed_span_s': 6,
        'degradation_confirmation_s': DEGRADATION_CONFIRM_S,
        'degradation_recovery_rearm_s': DEGRADATION_RECOVERY_S,
        'current_state_rule': 'active AND not externally explained AND ((past stop_share>=0.60 OR past mean_speed<0.055) AND past displacement<0.03)',
        'future_label_rule': 'first causally confirmed degradation-episode onset occurs within h seconds while the mission remains at risk',
        'censoring_events': [
            'mission completion',
            'external hold',
            'telemetry unavailable',
            'mission-leg end',
        ],
        'short_causal_holds_count_as_physical_observations': False,
        'graph_progress_used_in_target': False,
        'interpretation': 'causal operational proxy for onset of ineffective mission execution; not a mechanical-fault label',
    }
    (out_dir/'label_definition.json').write_text(
        json.dumps(label_definition, indent=2), encoding='utf-8'
    )

    event_distribution = (
        data.groupby('session', as_index=False)
            .agg(aligned_rows=('timestamp','size'),
                 eligible_h5=('label_h5', lambda x: int(x.notna().sum())),
                 positive_h5=('label_h5', lambda x: int((x==1).sum())),
                 event_onsets=('event_onset','sum'))
    )
    event_distribution.to_csv(out_dir/'event_distribution.csv', index=False)

    graph_pair=data.dropna(subset=['graph_remaining_m','euclidean_remaining_m'])
    graph_consistent=(
        graph_pair['graph_remaining_m']+1e-6
        >= graph_pair['euclidean_remaining_m']
    )
    event_legs=data.loc[data.event_onset.eq(1),['session','mission_leg_id']].drop_duplicates()
    summary={'sessions':int(data.session.nunique()),'aligned_rows':int(len(data)),
             'eligible_h5':int(data.label_h5.notna().sum()),
             'event_onsets':int(data.event_onset.sum()),
             'mission_legs_with_onsets':int(len(event_legs)),
             'recorded_route_fraction':float((data.route_source=='recorded').mean()),
             'reconstructed_route_fraction':float((data.route_source=='reconstructed_shortest').mean()),
             'graph_ge_euclidean_fraction':float(graph_consistent.mean()) if len(graph_consistent) else np.nan,
             'all_sessions_exact_1hz':bool(data.groupby('session')['timestamp'].apply(lambda x: x.sort_values().diff().dt.total_seconds().dropna().eq(1).all()).all()),
             'status':'dataset_completed'}
    (out_dir/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2)); print('Outputs:',out_dir.resolve())

if __name__=='__main__': main()