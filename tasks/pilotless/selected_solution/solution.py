# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pickle
import math
import torch
import torch.nn.functional as F

try:
    from hp import HP
except Exception:
    class HP:
        @staticmethod
        def get(name, default, **kwargs):
            return default

from link_config import NUM_BITS_PER_SYMBOL, NUM_OFDM_SYMBOLS, FFT_SIZE, NUM_GUARD_CARRIERS, DC_NULL

with open("constellation_points.pkl", "rb") as f:
    _points = pickle.load(f)

_CONST = torch.as_tensor(_points, dtype=torch.complex64)
_CR_CPU = _CONST.real.to(torch.float32).contiguous()
_CI_CPU = _CONST.imag.to(torch.float32).contiguous()
_M = int(2**NUM_BITS_PER_SYMBOL)
_idx = torch.arange(_M, dtype=torch.long)
_bp = torch.arange(NUM_BITS_PER_SYMBOL, dtype=torch.long)
_bits = ((_idx[:, None] >> (NUM_BITS_PER_SYMBOL-1-_bp)) & 1).to(torch.bool)
_BITS0_CPU = (~_bits).contiguous(); _BITS1_CPU = _bits.contiguous(); _BITSF_CPU = _bits.to(torch.float32).contiguous()

_active = torch.ones(int(FFT_SIZE), dtype=torch.float32)
_l = int(NUM_GUARD_CARRIERS[0]); _r = int(NUM_GUARD_CARRIERS[1])
if _l>0: _active[:_l] = 0.0
if _r>0: _active[int(FFT_SIZE)-_r:] = 0.0
if bool(DC_NULL): _active[int(FFT_SIZE)//2] = 0.0
_ACTIVE_CPU = _active.view(1, int(FFT_SIZE)).repeat(int(NUM_OFDM_SYMBOLS), 1).contiguous()
_MOMPOW_CPU = torch.mean(_CR_CPU.square()+_CI_CPU.square()).contiguous()
_LOGM = float(math.log(float(_M))); _LN2 = float(math.log(2.0))

def _npow(k):
    rr = torch.ones_like(_CR_CPU); ri = torch.zeros_like(_CI_CPU)
    for _ in range(k):
        rr, ri = rr*_CR_CPU-ri*_CI_CPU, rr*_CI_CPU+ri*_CR_CPU
    return torch.mean(rr).contiguous(), torch.mean(ri).contiguous()
_M2R_CPU,_M2I_CPU=_npow(2); _M3R_CPU,_M3I_CPU=_npow(3); _M4R_CPU,_M4I_CPU=_npow(4); _M6R_CPU,_M6I_CPU=_npow(6)

_SCALE_TF_T=int(HP.get("pyr_narrow_time",3,low=1,high=13)); _SCALE_TF_F=int(HP.get("pyr_wide_freq",17,low=5,high=45))
_SCALE_BAL_T=int(HP.get("pyr_bal_time",7,low=3,high=23)); _SCALE_BAL_F=int(HP.get("pyr_bal_freq",9,low=3,high=31))
_SCALE_FT_T=int(HP.get("pyr_wide_time",13,low=5,high=35)); _SCALE_FT_F=int(HP.get("pyr_narrow_freq",5,low=1,high=19))
_TOP_PER_SCALE=int(HP.get("top_h_per_scale",2,choices=[2])); _SCORE_T=int(HP.get("score_time_extra",1,low=1,high=9)); _SCORE_F=int(HP.get("score_freq_extra",1,low=1,high=9))
_EM_KT=int(HP.get("em_kernel_time",5,low=1,high=21)); _EM_KF=int(HP.get("em_kernel_freq",7,low=1,high=25)); _EM_ITERS=int(HP.get("em_iters_plus_one",4,low=2,high=9))
_WEIGHT_KT=int(HP.get("weight_kernel_time",5,low=1,high=21)); _WEIGHT_KF=int(HP.get("weight_kernel_freq",7,low=1,high=25))
_TEMP_FLOOR=HP.get("temp_floor",0.015,low=0.001,high=0.20,log=True); _TEMP_GAIN=HP.get("temp_gain",1.0,low=0.3,high=4.0)
_DEMAP_FLOOR=HP.get("demap_floor",0.0035,low=1e-5,high=0.5,log=True); _H_FLOOR=HP.get("h_floor",0.08,low=0.01,high=0.3)
_LLR_CLIP=HP.get("llr_clip",60.0,low=10.0,high=100.0); _MASK_EPS=HP.get("mask_eps",1e-6,low=1e-9,high=1e-3,log=True)
_CONF_POWER=HP.get("conf_power",1.0,low=0.0,high=3.0); _H_PRIOR_BLEND=HP.get("h_prior_blend",0.30,low=0.0,high=1.0)
_CAND_WEIGHT_TEMP=HP.get("cand_weight_temp",0.05,low=0.02,high=5.0,log=True); _ALL_CAND_TEMP=HP.get("all_cand_weight_temp",0.06,low=0.01,high=5.0,log=True)
_CRF_SWEEPS=int(HP.get("cand_crf_sweeps",2,low=0,high=8)); _FINAL_CRF_SWEEPS=int(HP.get("final_cand_crf_sweeps",1,low=0,high=8)); _ALL_CRF_SWEEPS=int(HP.get("all_final_cand_crf_sweeps",1,low=0,high=8))
_CRF_LAMBDA=HP.get("cand_crf_lambda",0.03,low=0.0,high=5.0); _FINAL_CRF_LAMBDA=HP.get("final_cand_crf_lambda",0.02,low=0.0,high=5.0); _ALL_CRF_LAMBDA=HP.get("all_final_cand_crf_lambda",0.025,low=0.0,high=5.0)
_CRF_TEMP=HP.get("cand_crf_temp",0.55,low=0.02,high=3.0,log=True); _CRF_KT=int(HP.get("cand_crf_kernel_time",3,low=1,high=11)); _CRF_KF=int(HP.get("cand_crf_kernel_freq",5,low=1,high=17)); _CRF_UNARY_SCALE=HP.get("cand_crf_unary_scale",1.0,low=0.05,high=8.0,log=True)
_EM_CAND_POWER=HP.get("em_candidate_weight_power",0.40,low=0.0,high=3.0); _FINAL_PRIOR_BLEND=HP.get("final_crf_prior_blend",0.10,low=0.0,high=1.0); _ALL_PRIOR_BLEND=HP.get("all_cand_prior_blend",0.06,low=0.0,high=1.0); _ALL_FINAL_PRIOR_BLEND=HP.get("all_final_crf_prior_blend",0.25,low=0.0,high=1.0)
_BLEND_GAP=HP.get("blend_score_gap",0.35,low=0.02,high=2.5,log=True); _BLEND_TEMP=HP.get("blend_score_temp",0.12,low=0.01,high=1.5,log=True); _BLEND_WITH_BEST=HP.get("blend_with_best",0.0,low=0.0,high=1.0)
_HVAR_SPREAD_GAIN=HP.get("hvar_spread_gain",0.15,low=0.0,high=8.0); _HVAR_RES_GAIN=HP.get("hvar_residual_gain",0.12,low=0.0,high=4.0); _HVAR_MAX=HP.get("hvar_max",2.0,low=0.05,high=10.0,log=True); _HVAR_SMOOTH_T=int(HP.get("hvar_smooth_time",5,low=1,high=21)); _HVAR_SMOOTH_F=int(HP.get("hvar_smooth_freq",7,low=1,high=25))
_RATIO_TOPK=int(HP.get("ratio_prior_symbol_topk",4,choices=[4])); _RATIO_BP_ITERS=int(HP.get("ratio_prior_bp_iters",1,low=0,high=1)); _RATIO_BP_W=HP.get("ratio_prior_neighbor_weight",0.85,low=0.0,high=1.5)
_RATIO_TEMP=HP.get("ratio_prior_temp",0.55,low=0.10,high=4.0,log=True); _RATIO_NOISE=HP.get("ratio_prior_noise_scale",1.4,low=0.1,high=10.0,log=True); _RATIO_HCHVAR=HP.get("ratio_prior_horiz_chvar",0.018,low=5e-4,high=0.4,log=True); _RATIO_VCHVAR=HP.get("ratio_prior_vert_chvar",0.045,low=5e-4,high=0.6,log=True)
_RATIO_DEN_FLOOR=HP.get("ratio_prior_den_floor",0.03,low=5e-4,high=0.5,log=True); _RATIO_QMAX=HP.get("ratio_prior_q_max",18.0,low=3.0,high=60.0); _RATIO_MSG_CLIP=HP.get("ratio_prior_msg_clip",28.0,low=4.0,high=80.0); _RATIO_PRIOR_CLIP=HP.get("ratio_prior_center_clip",5.0,low=0.2,high=20.0)
_RATIO_CAUCHY_C=HP.get("ratio_prior_cauchy_c",1.0,low=0.15,high=8.0,log=True); _RATIO_CAUCHY_GAIN=HP.get("ratio_prior_cauchy_gain",6.0,low=0.1,high=12.0,log=True); _RATIO_FADE_TOL_GAIN=HP.get("ratio_prior_fade_tol_gain",0.5,low=0.0,high=12.0); _RATIO_DEN_REL_POWER=HP.get("ratio_edge_den_rel_power",1.0,low=0.1,high=4.0,log=True)
_RATIO_SOURCE_W=HP.get("ratio_prior_source_weight",0.7,low=0.0,high=2.0); _RATIO_SCORE_W=HP.get("ratio_prior_final_score_weight",0.05,low=0.0,high=0.5); _RATIO_DEMAP_W=HP.get("ratio_prior_final_demap_weight",0.10,low=0.0,high=0.8); _RATIO_LLR_SUB_DAMP=HP.get("ratio_prior_llr_subtract_damp",0.40,low=0.0,high=1.5)
_RATIO_EM_BLEND=HP.get("ratio_prior_final_em_blend",0.30,low=0.0,high=1.0); _RATIO_EM_PRIOR_WEIGHT=HP.get("ratio_prior_final_em_weight",0.45,low=0.0,high=1.5); _RATIO_EM_CONF_POWER=HP.get("ratio_prior_final_em_conf_power",0.75,low=0.0,high=3.0)
_RATIO_REL_SNR_K=HP.get("ratio_edge_rel_snr_k",1.0,low=0.05,high=20.0,log=True); _RATIO_REL_ENT_POWER=HP.get("ratio_edge_rel_entropy_power",0.40,low=0.0,high=4.0); _RATIO_REL_HVAR_GAIN=HP.get("ratio_edge_rel_hspread_gain",0.5,low=0.0,high=20.0); _RATIO_REL_DEN_NOISE=HP.get("ratio_edge_rel_den_noise",1.0,low=0.0,high=20.0); _RATIO_REL_FLOOR=HP.get("ratio_edge_rel_floor",0.25,low=0.0,high=0.8); _RATIO_LLR_SUB_UNREL_GAIN=HP.get("ratio_llr_subtract_unrel_gain",0.50,low=0.0,high=4.0); _RATIO_LLR_SUB_MAX=HP.get("ratio_llr_subtract_max",0.60,low=0.0,high=2.0)
_DD_ENABLE=HP.get("dd_enable",1,choices=[0,1]); _DD_KT=int(HP.get("dd_kernel_time",7,low=1,high=25)); _DD_KF=int(HP.get("dd_kernel_freq",13,low=3,high=41)); _DD_SIGMA_T=HP.get("dd_sigma_time",2.2,low=0.4,high=12.0,log=True); _DD_SIGMA_F=HP.get("dd_sigma_freq",4.0,low=0.5,high=20.0,log=True)
_DD_SOFT_POWER=HP.get("dd_soft_power",0.30,low=0.0,high=4.0); _DD_MI_POWER=HP.get("dd_mi_power",0.70,low=0.0,high=4.0); _DD_COH_POWER=HP.get("dd_coh_power",0.30,low=0.0,high=4.0); _DD_SOFT_DEN_EPS=HP.get("dd_soft_den_eps",0.004,low=1e-5,high=0.2,log=True)
_DD_MASK_FLOOR=HP.get("dd_mask_floor",1e-4,low=0.0,high=0.2); _DD_WEIGHT_TEMP=HP.get("dd_weight_temp",0.12,low=0.02,high=5.0,log=True); _DD_WEIGHT_MASK_GAIN=HP.get("dd_weight_mask_gain",0.15,low=0.0,high=4.0); _DD_HVAR_GAIN=HP.get("dd_hvar_gain",1.0,low=0.0,high=20.0); _DD_HVAR_SCORE_GAIN=HP.get("dd_hvar_score_gain",0.10,low=0.0,high=4.0); _DD_PRIOR_REL_GAIN=HP.get("dd_prior_rel_gain",0.5,low=0.0,high=5.0)
_FINAL_AFFINE_ENABLE=HP.get("final_affine_enable",1,choices=[0,1]); _PATCH_RIDGE0=HP.get("final_affine_ridge_center",0.025,low=0.0,high=0.5); _PATCH_RIDGES=HP.get("final_affine_ridge_slope",0.16,low=0.0,high=2.0); _PATCH_W_FLOOR=HP.get("final_affine_weight_floor",0.02,low=0.0,high=0.5); _PATCH_CONF_POWER=HP.get("final_affine_conf_power",0.85,low=0.0,high=4.0)
_PATCH_DEN_FLOOR=HP.get("final_affine_den_floor",1e-5,low=1e-8,high=1e-2,log=True); _PATCH_SMOOTH_BLEND=HP.get("final_affine_smooth_blend",0.25,low=0.0,high=1.0); _PATCH_RES_GAIN=HP.get("final_affine_residual_gain",0.45,low=0.0,high=5.0); _PATCH_UNC_GAIN=HP.get("final_affine_unc_gain",0.45,low=0.0,high=5.0)
_FINAL_AFFINE_TUPLE_GAIN=HP.get("final_affine_ratio_gate_gain",0.85,low=0.0,high=4.0); _FINAL_AFFINE_DD_GAIN=HP.get("final_affine_dd_gate_gain",0.55,low=0.0,high=4.0); _FINAL_AFFINE_GATE_BIAS=HP.get("final_affine_gate_bias",0.10,low=0.0,high=1.0); _FINAL_AFFINE_GATE_TEMP=HP.get("final_affine_gate_temp",0.18,low=0.01,high=2.0,log=True)
_FINAL_AFFINE_VAR_GAIN=HP.get("final_affine_var_gain",0.70,low=0.0,high=5.0); _FINAL_AFFINE_SLOPE_HVAR_GAIN=HP.get("final_affine_slope_hvar_gain",0.20,low=0.0,high=8.0)
_FINAL_SYM_TOPL=int(HP.get("final_compact_symbol_topl",8,choices=[6,8,10,12,16])); _FINAL_TAIL_LOGW=HP.get("final_compact_tail_log_weight",0.0,low=-5.0,high=1.0); _FINAL_TAIL_TEMP=HP.get("final_compact_tail_temp",1.0,low=0.5,high=2.0,log=True)

_DT_CPU=torch.tensor([-2.,-2.,-2.,-1.,-1.,-1.,0.,0.,0.,1.,1.,1.,2.,2.,2.]); _DF_CPU=torch.tensor([-1.,0.,1.,-1.,0.,1.,-1.,0.,1.,-1.,0.,1.,-1.,0.,1.])
_DMAT_CPU=torch.stack([torch.ones_like(_DT_CPU),_DT_CPU,_DF_CPU],dim=-1).to(torch.complex64).contiguous(); _DHD_CPU=(_DMAT_CPU.conj()[:,:,None]*_DMAT_CPU[:,None,:]).contiguous(); _EYE3_CPU=torch.eye(3,dtype=torch.complex64)

def _odd(k:int)->int:
    k=int(k); return k if k%2 else k+1

def _gauss(kt,kf,st,sf):
    kt=_odd(kt); kf=_odd(kf); at=torch.arange(kt,dtype=torch.float32)-kt//2; af=torch.arange(kf,dtype=torch.float32)-kf//2
    yy,xx=torch.meshgrid(at,af,indexing="ij"); ker=torch.exp(-yy.square()/(2*st*st)-xx.square()/(2*sf*sf)); ker=ker/ker.sum(); return ker.view(1,1,kt,kf).contiguous()
_GAUSS_CPU=_gauss(_DD_KT,_DD_KF,_DD_SIGMA_T,_DD_SIGMA_F)

def _mask_den(active,kt,kf):
    kt=_odd(kt); kf=_odd(kf)
    return F.avg_pool2d(active[None,None],(kt,kf),stride=1,padding=(kt//2,kf//2),count_include_pad=False)[0,0].clamp_min(_MASK_EPS)
_KEYS=[]
def _add(kt,kf):
    key=(_odd(kt),_odd(kf))
    if key not in _KEYS: _KEYS.append(key)
for a,b in ((_SCALE_TF_T,_SCALE_TF_F),(_SCALE_BAL_T,_SCALE_BAL_F),(_SCALE_FT_T,_SCALE_FT_F),(_SCALE_TF_T+_SCORE_T-1,_SCALE_TF_F+_SCORE_F-1),(_SCALE_BAL_T+_SCORE_T-1,_SCALE_BAL_F+_SCORE_F-1),(_SCALE_FT_T+_SCORE_T-1,_SCALE_FT_F+_SCORE_F-1),(_EM_KT,_EM_KF),(_WEIGHT_KT,_WEIGHT_KF),(_CRF_KT,_CRF_KF),(_HVAR_SMOOTH_T,_HVAR_SMOOTH_F),(3,3)):
    _add(a,b)
_DEN_CPU={k:_mask_den(_ACTIVE_CPU,k[0],k[1]).contiguous() for k in _KEYS}

if torch.cuda.is_available():
    _CR_CUDA=_CR_CPU.cuda(); _CI_CUDA=_CI_CPU.cuda(); _BITS0_CUDA=_BITS0_CPU.cuda(); _BITS1_CUDA=_BITS1_CPU.cuda(); _BITSF_CUDA=_BITSF_CPU.cuda(); _ACTIVE_CUDA=_ACTIVE_CPU.cuda(); _DEN_CUDA={k:v.cuda() for k,v in _DEN_CPU.items()}; _GAUSS_CUDA=_GAUSS_CPU.cuda(); _MOMPOW_CUDA=_MOMPOW_CPU.cuda()
    _M2R_CUDA=_M2R_CPU.cuda(); _M2I_CUDA=_M2I_CPU.cuda(); _M3R_CUDA=_M3R_CPU.cuda(); _M3I_CUDA=_M3I_CPU.cuda(); _M4R_CUDA=_M4R_CPU.cuda(); _M4I_CUDA=_M4I_CPU.cuda(); _M6R_CUDA=_M6R_CPU.cuda(); _M6I_CUDA=_M6I_CPU.cuda(); _DMAT_CUDA=_DMAT_CPU.cuda(); _DHD_CUDA=_DHD_CPU.cuda(); _EYE3_CUDA=_EYE3_CPU.cuda()
else:
    _CR_CUDA=_CI_CUDA=_BITS0_CUDA=_BITS1_CUDA=_BITSF_CUDA=_ACTIVE_CUDA=_DEN_CUDA=_GAUSS_CUDA=_MOMPOW_CUDA=None; _M2R_CUDA=_M2I_CUDA=_M3R_CUDA=_M3I_CUDA=_M4R_CUDA=_M4I_CUDA=_M6R_CUDA=_M6I_CUDA=None; _DMAT_CUDA=_DHD_CUDA=_EYE3_CUDA=None

def _select(y):
    if y.is_cuda and _CR_CUDA is not None:
        return _CR_CUDA,_CI_CUDA,_BITS0_CUDA,_BITS1_CUDA,_BITSF_CUDA,_ACTIVE_CUDA,_DEN_CUDA,_GAUSS_CUDA,_MOMPOW_CUDA,_M2R_CUDA,_M2I_CUDA,_M3R_CUDA,_M3I_CUDA,_M4R_CUDA,_M4I_CUDA,_M6R_CUDA,_M6I_CUDA,_DMAT_CUDA,_DHD_CUDA,_EYE3_CUDA
    return _CR_CPU,_CI_CPU,_BITS0_CPU,_BITS1_CPU,_BITSF_CPU,_ACTIVE_CPU,_DEN_CPU,_GAUSS_CPU,_MOMPOW_CPU,_M2R_CPU,_M2I_CPU,_M3R_CPU,_M3I_CPU,_M4R_CPU,_M4I_CPU,_M6R_CPU,_M6I_CPU,_DMAT_CPU,_DHD_CPU,_EYE3_CPU

def _avg(x,mask,dens,kt,kf):
    kt=_odd(kt); kf=_odd(kf); z=F.avg_pool2d((x*mask[None])[:,None],(kt,kf),stride=1,padding=(kt//2,kf//2),count_include_pad=False)[:,0]
    return z/dens[(kt,kf)][None]
def _avg_pair(x0,x1,mask,dens,kt,kf):
    b,t,f=x0.shape; z=torch.stack((x0,x1),1).reshape(b*2,t,f); z=_avg(z,mask,dens,kt,kf).reshape(b,2,t,f); return z[:,0],z[:,1]
def _avg_triple(x0,x1,x2,mask,dens,kt,kf):
    b,t,f=x0.shape; z=torch.stack((x0,x1,x2),1).reshape(b*3,t,f); z=_avg(z,mask,dens,kt,kf).reshape(b,3,t,f); return z[:,0],z[:,1],z[:,2]
def _avg_k(x,mask,dens,kt,kf):
    k,b,t,f=x.shape; z=x.permute(1,0,2,3).reshape(b*k,t,f); z=_avg(z,mask,dens,kt,kf).reshape(b,k,t,f).permute(1,0,2,3); return z
def _avg_k_pair(x0,x1,mask,dens,kt,kf):
    k,b,t,f=x0.shape; z=torch.stack((x0,x1),2).permute(1,0,2,3,4).reshape(b*k*2,t,f); z=_avg(z,mask,dens,kt,kf).reshape(b,k,2,t,f).permute(1,0,2,3,4); return z[:,:,0],z[:,:,1]
def _avg_k_triple(x0,x1,x2,mask,dens,kt,kf):
    k,b,t,f=x0.shape; z=torch.stack((x0,x1,x2),2).permute(1,0,2,3,4).reshape(b*k*3,t,f); z=_avg(z,mask,dens,kt,kf).reshape(b,k,3,t,f).permute(1,0,2,3,4); return z[:,:,0],z[:,:,1],z[:,:,2]
def _cmul(ar,ai,br,bi): return ar*br-ai*bi, ar*bi+ai*br

def _ypows_fused(yr, yi, kt, kf, mask, dens):
    rr=torch.ones_like(yr); ri=torch.zeros_like(yi); outs=[]
    for o in range(1,7):
        rr,ri=_cmul(rr,ri,yr,yi)
        if o in (2,3,4,6):
            outs.append(rr); outs.append(ri)
    b,t,f=yr.shape
    z=torch.stack(outs,1).reshape(b*8,t,f)
    z=_avg(z,mask,dens,kt,kf).reshape(b,8,t,f)
    return (z[:,0],z[:,1],z[:,2],z[:,3],z[:,4],z[:,5],z[:,6],z[:,7])
def _roots(myr,myi,mcr,mci,o,amp):
    rr=myr*mcr+myi*mci; ri=myi*mcr-myr*mci; base=torch.atan2(ri,rr+1e-12)/float(o); out=[]
    for r in range(o):
        th=base+(2.0*torch.pi*float(r)/float(o)); out.append((amp*torch.cos(th),amp*torch.sin(th)))
    return out
def _cloud(yr,yi,hr,hi,crv,civ,den):
    sr=hr[...,None]*crv-hi[...,None]*civ; si=hr[...,None]*civ+hi[...,None]*crv
    return -((yr[...,None]-sr).square()+(yi[...,None]-si).square())/den[...,None]

def _scale_cand(yr,yi,no,crv,civ,mom,m2r,m2i,m3r,m3i,m4r,m4i,m6r,m6i,kt,kf,mask,dens):
    py=_avg(yr.square()+yi.square(),mask,dens,kt,kf); amp=torch.sqrt(torch.clamp((py-no[:,None,None])/(mom+1e-12),min=_H_FLOOR*_H_FLOOR))
    p2r,p2i,p3r,p3i,p4r,p4i,p6r,p6i=_ypows_fused(yr,yi,kt,kf,mask,dens)
    roots=[]; roots+=_roots(p2r,p2i,m2r,m2i,2,amp); roots+=_roots(p3r,p3i,m3r,m3i,3,amp); roots+=_roots(p4r,p4i,m4r,m4i,4,amp); roots+=_roots(p6r,p6i,m6r,m6i,6,amp)
    den=no[:,None,None]*_TEMP_GAIN+_TEMP_FLOOR; scores=[]; hrs=[]; his=[]
    for hr,hi in roots:
        scores.append(torch.logsumexp(_cloud(yr,yi,hr,hi,crv,civ,den),-1)); hrs.append(hr); his.append(hi)
    ss=torch.stack(scores,0); ss=_avg_k(ss,mask,dens,kt+_SCORE_T-1,kf+_SCORE_F-1); hra=torch.stack(hrs,0); hia=torch.stack(his,0); ts,ti=torch.topk(ss,k=_TOP_PER_SCALE,dim=0)
    hr=torch.gather(hra,0,ti); hi=torch.gather(hia,0,ti); hr,hi=_avg_k_pair(hr,hi,mask,dens,kt,kf)
    lse=torch.logsumexp(_cloud(yr.unsqueeze(0),yi.unsqueeze(0),hr,hi,crv,civ,den.view(1,den.shape[0],den.shape[1],den.shape[2])),-1)
    return hr,hi,_avg_k(lse,mask,dens,kt+_SCORE_T-1,kf+_SCORE_F-1)
def _blend(a,b,c,d,e,f,g,h,i):
    hs_r=torch.stack((a,d,g),0); hs_i=torch.stack((b,e,h),0); ss=torch.stack((c,f,i),0); best=torch.max(ss,0,keepdim=True).values; close=(ss>=best-_BLEND_GAP).to(ss.dtype); wt=torch.softmax((ss-best)/_BLEND_TEMP,0)*close; wt=wt/wt.sum(0,keepdim=True).clamp_min(1e-8)
    br=(wt*hs_r).sum(0); bi=(wt*hs_i).sum(0); bs=torch.logsumexp(ss/_BLEND_TEMP+torch.log(close.clamp_min(1e-8)),0)*_BLEND_TEMP
    bid=torch.argmax(ss,0,keepdim=True); br=(1-_BLEND_WITH_BEST)*br+_BLEND_WITH_BEST*torch.gather(hs_r,0,bid)[0]; bi=(1-_BLEND_WITH_BEST)*bi+_BLEND_WITH_BEST*torch.gather(hs_i,0,bid)[0]
    return torch.cat((br,a,d,g),0), torch.cat((bi,b,e,h),0), torch.cat((bs,c,f,i),0)
def _crf(hr,hi,unary,mask,dens,sweeps,lam,temp,kt,kf):
    q=torch.softmax((_CRF_UNARY_SCALE*unary)/temp,0); h2=hr.square()+hi.square()
    for _ in range(sweeps):
        mr=(q*hr).sum(0); mi=(q*hi).sum(0); m2=(q*h2).sum(0); nr,ni,n2=_avg_triple(mr,mi,m2,mask,dens,kt,kf); cost=h2-2*hr*nr.unsqueeze(0)-2*hi*ni.unsqueeze(0)+n2.unsqueeze(0); q=torch.softmax((_CRF_UNARY_SCALE*unary-lam*cost)/temp,0)
    return q, torch.log(q.clamp_min(1e-12))

def _ratio_phi(q, ratio, tol):
    err2=torch.abs(q[...,None,None]-ratio).square().to(torch.float32)
    scale=tol[...,None,None]*_RATIO_TEMP+1e-7
    phi=-_RATIO_CAUCHY_GAIN*torch.log1p(err2/(scale*(_RATIO_CAUCHY_C*_RATIO_CAUCHY_C)+1e-7))
    return phi.clamp(-_RATIO_MSG_CLIP,0.0)

def _ratio_prior(yr,yi,top_idx,single_logp,no,cr,ci,active,hspread):
    B,T,Fsz=yr.shape; K=top_idx.shape[-1]; logb=torch.zeros((B,T,Fsz,K),dtype=yr.dtype,device=yr.device); no_b=no.view(B,1); am=active[:T,:Fsz].view(1,T,Fsz,1)
    sp=torch.exp(single_logp).clamp_min(1e-12); ent=-(sp*single_logp).sum(-1)/float(math.log(float(K)))
    ent_rel=(1.0-ent).clamp(0.0,1.0)**_RATIO_REL_ENT_POWER; yp=(yr.square()+yi.square()).clamp_min(0.0); snr=yp/(no[:,None,None]+1e-8); snr_rel=(snr/(snr+_RATIO_REL_SNR_K)).clamp(0.0,1.0); ch_rel=(1.0/(1.0+_RATIO_REL_HVAR_GAIN*hspread.clamp_min(0.0))).clamp(0.0,1.0)
    node_rel=(_RATIO_REL_FLOOR+(1.0-_RATIO_REL_FLOOR)*ent_rel*snr_rel*ch_rel)*active[:T,:Fsz].view(1,T,Fsz)
    for _ in range(_RATIO_BP_ITERS):
        acc=torch.zeros_like(logb); deg=torch.zeros((B,T,Fsz,1),dtype=yr.dtype,device=yr.device)
        ew0=(active[:T,:Fsz-1]*active[:T,1:Fsz]).view(1,T,Fsz-1,1); q=torch.complex(yr[:,:,:-1],yi[:,:,:-1])/(torch.complex(yr[:,:,1:],yi[:,:,1:])+(1e-7+0j)); aq=torch.abs(q).to(torch.float32); q=q*torch.minimum(torch.ones_like(aq),_RATIO_QMAX/(aq+1e-7)).to(torch.complex64)
        ia=top_idx[:,:,:-1,:]; ib=top_idx[:,:,1:,:]; ratio=torch.complex(cr[ia].unsqueeze(-1),ci[ia].unsqueeze(-1))/(torch.complex(cr[ib].unsqueeze(-2),ci[ib].unsqueeze(-2))+(1e-7+0j)); den=(yr[:,:,1:].square()+yi[:,:,1:].square()).clamp_min(0.0)+_RATIO_DEN_FLOOR; tol=_RATIO_NOISE*no_b[:,None,:]/den*(1+torch.abs(q).square().to(torch.float32))+_RATIO_HCHVAR
        den_rel=(den/(den+_RATIO_REL_DEN_NOISE*no_b[:,None,:]+_RATIO_DEN_FLOOR)).clamp(0.0,1.0); tol=tol*(1.0+_RATIO_FADE_TOL_GAIN*(1.0-den_rel)); erel=torch.sqrt((node_rel[:,:,:-1]*node_rel[:,:,1:]).clamp_min(0.0))*(den_rel.clamp_min(0.0)**_RATIO_DEN_REL_POWER); ew=ew0*erel.unsqueeze(-1)
        phi=_ratio_phi(q,ratio,tol); la=_RATIO_BP_W*logb[:,:,:-1,:]; lb=_RATIO_BP_W*logb[:,:,1:,:]
        ma=torch.logsumexp(phi+lb[...,None,:],-1); mb=torch.logsumexp(phi+la[..., :,None],-2); ma=(ma-torch.logsumexp(ma,-1,keepdim=True))*ew; mb=(mb-torch.logsumexp(mb,-1,keepdim=True))*ew; acc=acc+F.pad(ma,(0,0,0,1))+F.pad(mb,(0,0,1,0)); deg=deg+F.pad(ew,(0,0,0,1))+F.pad(ew,(0,0,1,0))
        ew0=(active[:T-1,:Fsz]*active[1:T,:Fsz]).view(1,T-1,Fsz,1); q=torch.complex(yr[:,:-1,:],yi[:,:-1,:])/(torch.complex(yr[:,1:,:],yi[:,1:,:])+(1e-7+0j)); aq=torch.abs(q).to(torch.float32); q=q*torch.minimum(torch.ones_like(aq),_RATIO_QMAX/(aq+1e-7)).to(torch.complex64)
        ia=top_idx[:,:-1,:,:]; ib=top_idx[:,1:,:,:]; ratio=torch.complex(cr[ia].unsqueeze(-1),ci[ia].unsqueeze(-1))/(torch.complex(cr[ib].unsqueeze(-2),ci[ib].unsqueeze(-2))+(1e-7+0j)); den=(yr[:,1:,:].square()+yi[:,1:,:].square()).clamp_min(0.0)+_RATIO_DEN_FLOOR; tol=_RATIO_NOISE*no_b[:,:,None]/den*(1+torch.abs(q).square().to(torch.float32))+_RATIO_VCHVAR
        den_rel=(den/(den+_RATIO_REL_DEN_NOISE*no_b[:,:,None]+_RATIO_DEN_FLOOR)).clamp(0.0,1.0); tol=tol*(1.0+_RATIO_FADE_TOL_GAIN*(1.0-den_rel)); erel=torch.sqrt((node_rel[:,:-1,:]*node_rel[:,1:,:]).clamp_min(0.0))*(den_rel.clamp_min(0.0)**_RATIO_DEN_REL_POWER); ew=ew0*erel.unsqueeze(-1)
        phi=_ratio_phi(q,ratio,tol); la=_RATIO_BP_W*logb[:,:-1,:,:]; lb=_RATIO_BP_W*logb[:,1:,:,:]
        ma=torch.logsumexp(phi+lb[...,None,:],-1); mb=torch.logsumexp(phi+la[..., :,None],-2); ma=(ma-torch.logsumexp(ma,-1,keepdim=True))*ew; mb=(mb-torch.logsumexp(mb,-1,keepdim=True))*ew; acc=acc+F.pad(ma,(0,0,0,0,0,1))+F.pad(mb,(0,0,0,0,1,0)); deg=deg+F.pad(ew,(0,0,0,0,0,1))+F.pad(ew,(0,0,0,0,1,0))
        src=_RATIO_SOURCE_W*single_logp; src=src-src.mean(-1,keepdim=True); src=src*node_rel.unsqueeze(-1); logb=(acc/deg.clamp_min(1.0)+src); logb=(logb-logb.mean(-1,keepdim=True)).clamp(-_RATIO_PRIOR_CLIP,_RATIO_PRIOR_CLIP)*am
    prior=torch.zeros((B,T,Fsz,_M),dtype=yr.dtype,device=yr.device); prior.scatter_add_(3,top_idx,logb); prior=prior-prior.mean(-1,keepdim=True); return prior.clamp(-_RATIO_PRIOR_CLIP,_RATIO_PRIOR_CLIP)*am, node_rel

def _bit_llr_prior(prior,bits0,bits1):
    met=prior[...,None]; big=-1e9; m1=bits1.view(1,1,1,_M,NUM_BITS_PER_SYMBOL); m0=bits0.view(1,1,1,_M,NUM_BITS_PER_SYMBOL); return torch.logsumexp(torch.where(m1,met,big),-2)-torch.logsumexp(torch.where(m0,met,big),-2)

def _gauss_masked(x,rel,active,ker):
    pt=int(ker.shape[-2]//2); pf=int(ker.shape[-1]//2); m=rel.clamp_min(0.0)*active[None]; num=F.conv2d((x*m)[:,None],ker,padding=(pt,pf))[:,0]; den=F.conv2d(m[:,None],ker,padding=(pt,pf))[:,0]; return num/den.clamp_min(_MASK_EPS),den

def _dd(yr,yi,no,sym_post,bitsf,cr,ci,active,gker,hpr,hpi):
    crv=cr.view(1,1,1,_M); civ=ci.view(1,1,1,_M); pm=sym_post.clamp_min(1e-12); ent=-(pm*torch.log(pm)).sum(-1)/_LOGM; p1=(sym_post[...,None]*bitsf.view(1,1,1,_M,NUM_BITS_PER_SYMBOL)).sum(-2).clamp(1e-6,1-1e-6); hb=-(p1*torch.log(p1)+(1-p1)*torch.log(1-p1))/_LN2; mi=(1-hb.mean(-1)).clamp(0,1)
    ecr=(sym_post*crv).sum(-1); eci=(sym_post*civ).sum(-1); ec2=(sym_post*(crv.square()+civ.square())).sum(-1).clamp_min(_DD_SOFT_DEN_EPS); emu2=ecr.square()+eci.square(); varc=(ec2-emu2).clamp_min(0); hpow=(hpr.square()+hpi.square()).clamp_min(_H_FLOOR*_H_FLOOR)
    phr=(yr*ecr+yi*eci)/ec2; phi=(yi*ecr-yr*eci)/ec2; rel=(mi.clamp(0,1)**_DD_MI_POWER)*((emu2/ec2).clamp(0,1)**_DD_COH_POWER)*((1-ent).clamp(0,1)**_DD_SOFT_POWER); obs=(no[:,None,None]+hpow*varc)/ec2; inv=(hpow/(hpow+obs+1e-9)).clamp(0,1); rel=(rel*inv).clamp_min(_DD_MASK_FLOOR)
    dhr,dden=_gauss_masked(phr,rel,active,gker); dhi,_=_gauss_masked(phi,rel,active,gker); svar,_=_gauss_masked(obs,rel,active,gker); hv=(_DD_HVAR_GAIN*svar/dden.clamp_min(_MASK_EPS)).clamp(0,_HVAR_MAX); prior=(_DD_WEIGHT_MASK_GAIN*torch.log(dden.clamp_min(_MASK_EPS))-_DD_HVAR_SCORE_GAIN*torch.log1p(hv)+_DD_PRIOR_REL_GAIN*torch.log(rel.clamp_min(1e-6)))/_DD_WEIGHT_TEMP
    return dhr,dhi,prior.to(torch.float32),hv.to(torch.float32)

def _unfold_c(z):
    return torch.complex(F.unfold(F.pad(z.real[:,None],(1,1,2,2),mode="replicate"),(5,3)),F.unfold(F.pad(z.imag[:,None],(1,1,2,2),mode="replicate"),(5,3))).transpose(1,2)
def _unfold_r(x): return F.unfold(F.pad(x[:,None],(1,1,2,2),mode="replicate"),(5,3)).transpose(1,2)

def _affine_candidate(z,xmean,x2,xvar,conf,hprior,base_hvar,mask,dens,dmat,dhd,eye,mom):
    B,T,Fsz=z.shape; N=T*Fsz; yp=_unfold_c(z); mup=_unfold_c(xmean); x2p=_unfold_r(x2).clamp_min(_PATCH_DEN_FLOOR); xvp=_unfold_r(xvar).clamp_min(0); wp=(_unfold_r(conf).clamp_min(_PATCH_W_FLOOR)**_PATCH_CONF_POWER)*_unfold_r(mask[None].expand(B,T,Fsz)); hp=hprior.reshape(B,N)
    regd=torch.tensor([_PATCH_RIDGE0,_PATCH_RIDGES,_PATCH_RIDGES],dtype=torch.float32,device=z.device).to(torch.complex64); reg=torch.diag(regd)+1e-6*eye
    G=((wp*x2p).to(torch.complex64)[..., :,None,None]*dhd[None,None]).sum(2)+reg[None,None]
    rhs=((wp.to(torch.complex64)*mup.conj()*yp)[..., :,None]*dmat.conj()[None,None]).sum(2); rhs0=torch.zeros((B,N,3),dtype=torch.complex64,device=z.device); rhs0[...,0]=_PATCH_RIDGE0*hp
    par=torch.linalg.solve(G,(rhs+rhs0).unsqueeze(-1))[...,0]; h=par[...,0].reshape(B,T,Fsz); hsr=_avg(h.real.to(torch.float32),mask,dens,3,3); hsi=_avg(h.imag.to(torch.float32),mask,dens,3,3); h=(1-_PATCH_SMOOTH_BLEND)*h+_PATCH_SMOOTH_BLEND*torch.complex(hsr,hsi)
    hloc=(dmat[None,None]*par.unsqueeze(2)).sum(-1); res=torch.abs(yp-hloc*mup).square()+torch.abs(hloc).square()*xvp; wsum=wp.sum(2).clamp_min(1.0); rm=(wp*res).sum(2)/wsum; unc=(wp*(torch.abs(hloc).square()*xvp)).sum(2)/wsum; bv=base_hvar.reshape(B,N); var=(_PATCH_RES_GAIN*(rm-bv*mom).clamp_min(0)/(mom+1e-12)+_PATCH_UNC_GAIN*unc/(mom+1e-12)+bv).to(torch.float32).reshape(B,T,Fsz).clamp(0,_HVAR_MAX)
    return h.real.to(torch.float32),h.imag.to(torch.float32),var

def _final_affine(z,all_hr,all_hi,all_hvar,evid,logw,prior_view,cr,ci,active,dens,dmat,dhd,eye,mom):
    K,B,T,Fsz=all_hr.shape; metric=evid+logw[...,None]+prior_view; sp=torch.softmax(torch.logsumexp(metric,0),-1); xmr=(sp*cr.view(1,1,1,_M)).sum(-1); xmi=(sp*ci.view(1,1,1,_M)).sum(-1); xm=torch.complex(xmr,xmi); x2=(sp*(cr.square()+ci.square()).view(1,1,1,_M)).sum(-1); xv=(x2-xmr.square()-xmi.square()).clamp_min(0); conf=sp.max(-1).values.clamp_min(1.0/float(_M))
    cp=torch.softmax(logw,0); hpr=(cp*all_hr).sum(0); hpi=(cp*all_hi).sum(0); hp=torch.complex(hpr,hpi); bhv=(cp*all_hvar).sum(0); phr,phi,phv=_affine_candidate(z,xm,x2,xv,conf,hp,bhv,active,dens,dmat,dhd,eye,mom)
    top2=torch.topk(cp,k=2,dim=0).values; gap=(top2[0]-top2[1]).clamp(0,1); prrel=(prior_view[0].abs().mean(-1)/(_RATIO_DEMAP_W+1e-6)).clamp(0,1); ddrel=cp[-1].clamp(0,1) if K>1 else torch.zeros_like(gap); gate=torch.sigmoid((gap+_FINAL_AFFINE_TUPLE_GAIN*prrel+_FINAL_AFFINE_DD_GAIN*ddrel-_FINAL_AFFINE_GATE_BIAS)/_FINAL_AFFINE_GATE_TEMP)*active.view(1,T,Fsz)
    best=torch.argmax(cp,0,keepdim=True); ids=torch.arange(K,dtype=torch.long,device=z.device).view(K,1,1,1); gk=(ids==best).to(all_hr.dtype)*gate.unsqueeze(0); slope=(phr-hpr).square()+(phi-hpi).square(); add=(_FINAL_AFFINE_VAR_GAIN*phv+_FINAL_AFFINE_SLOPE_HVAR_GAIN*slope).clamp(0,_HVAR_MAX)
    return (1-gk)*all_hr+gk*phr.unsqueeze(0),(1-gk)*all_hi+gk*phi.unsqueeze(0),(all_hvar+gk*add.unsqueeze(0)).clamp(0,_HVAR_MAX),logw

def receiver(y,no):
    cr,ci,bits0,bits1,bitsf,active,dens,gker,mom,m2r,m2i,m3r,m3i,m4r,m4i,m6r,m6i,dmat,dhd,eye=_select(y)
    yr=y[:,0,0].real.to(torch.float32); yi=y[:,0,0].imag.to(torch.float32); zc=torch.complex(yr,yi); no=no.to(torch.float32); B,T,Fsz=yr.shape
    crv3=cr.view(1,1,1,_M); civ3=ci.view(1,1,1,_M)
    h1r,h1i,s1=_scale_cand(yr,yi,no,crv3,civ3,mom,m2r,m2i,m3r,m3i,m4r,m4i,m6r,m6i,_SCALE_TF_T,_SCALE_TF_F,active,dens); h2r,h2i,s2=_scale_cand(yr,yi,no,crv3,civ3,mom,m2r,m2i,m3r,m3i,m4r,m4i,m6r,m6i,_SCALE_BAL_T,_SCALE_BAL_F,active,dens); h3r,h3i,s3=_scale_cand(yr,yi,no,crv3,civ3,mom,m2r,m2i,m3r,m3i,m4r,m4i,m6r,m6i,_SCALE_FT_T,_SCALE_FT_F,active,dens)
    hr,hi,init=_blend(h1r,h1i,s1,h2r,h2i,s2,h3r,h3i,s3); K=hr.shape[0]; q,logq=_crf(hr,hi,init,active,dens,_CRF_SWEEPS,_CRF_LAMBDA,_CRF_TEMP,_CRF_KT,_CRF_KF); emw=q.clamp_min(1e-4)**_EM_CAND_POWER
    crv=cr.view(1,1,1,1,_M); civ=ci.view(1,1,1,1,_M); cabs2=crv.square()+civ.square(); yrk=yr.unsqueeze(0); yik=yi.unsqueeze(0); den_em=no.view(1,B,1,1,1)*_TEMP_GAIN+_TEMP_FLOOR
    for _ in range(_EM_ITERS):
        oldr,oldi=hr,hi; sr=hr[...,None]*crv-hi[...,None]*civ; si=hr[...,None]*civ+hi[...,None]*crv; dist=(yrk[...,None]-sr).square()+(yik[...,None]-si).square(); p=torch.softmax(-dist/den_em,-1); ecr=(p*crv).sum(-1); eci=(p*civ).sum(-1); ec2=(p*cabs2).sum(-1); conf=p.max(-1).values.clamp_min(1.0/float(_M))**_CONF_POWER*emw; ad,anr,ani=_avg_k_triple(ec2*conf,(yrk*ecr+yik*eci)*conf,(yik*ecr-yrk*eci)*conf,active,dens,_EM_KT,_EM_KF); nr=anr/ad.clamp_min(_H_FLOOR*_H_FLOOR); ni=ani/ad.clamp_min(_H_FLOOR*_H_FLOOR); hr=(1-_H_PRIOR_BLEND)*nr+_H_PRIOR_BLEND*oldr; hi=(1-_H_PRIOR_BLEND)*ni+_H_PRIOR_BLEND*oldi
    sr=hr[...,None]*crv-hi[...,None]*civ; si=hr[...,None]*civ+hi[...,None]*crv; dist=(yrk[...,None]-sr).square()+(yik[...,None]-si).square(); post=-dist/den_em; mix=torch.logsumexp(post,-1); cand=_avg_k(mix,active,dens,_WEIGHT_KT,_WEIGHT_KF)/_CAND_WEIGHT_TEMP; fq,flogq=_crf(hr,hi,cand,active,dens,_FINAL_CRF_SWEEPS,_FINAL_CRF_LAMBDA,_CRF_TEMP,_CRF_KT,_CRF_KF); logw=(1-_FINAL_PRIOR_BLEND)*torch.log_softmax(cand,0)+_FINAL_PRIOR_BLEND*flogq; logw=logw-torch.logsumexp(logw,0,keepdim=True); w=torch.softmax(logw,0)
    p=torch.softmax(post,-1); ec2=(p*cabs2).sum(-1).clamp_min(1e-5); res=(p*dist).sum(-1); conf=p.max(-1).values.clamp_min(1.0/float(_M)); avn,avd,_=_avg_k_triple((res-no.view(1,B,1,1)).clamp_min(0)*conf,ec2*conf,conf,active,dens,_EM_KT,_EM_KF); hvr=(avn/avd.clamp_min(_H_FLOOR*_H_FLOOR)); mhr=(w*hr).sum(0,keepdim=True); mhi=(w*hi).sum(0,keepdim=True); hspread=(w*((hr-mhr).square()+(hi-mhi).square())).sum(0); hvar=_avg((_HVAR_SPREAD_GAIN*hspread+_HVAR_RES_GAIN*(w*hvr).sum(0)).clamp(0,_HVAR_MAX),active,dens,_HVAR_SMOOTH_T,_HVAR_SMOOTH_F).clamp(0,_HVAR_MAX)
    base_log=torch.logsumexp(post+logw[...,None],0); base_log=base_log-torch.logsumexp(base_log,-1,keepdim=True)
    rtlog,rtidx=torch.topk(base_log,k=_RATIO_TOPK,dim=-1); rtlog=rtlog-torch.logsumexp(rtlog,-1,keepdim=True); ratio_prior,ratio_rel=_ratio_prior(yr,yi,rtidx,rtlog,no,cr,ci,active,hspread)
    prior_view=(_RATIO_DEMAP_W*ratio_prior).view(1,B,T,Fsz,_M); score_view=(_RATIO_SCORE_W*ratio_prior).view(1,B,T,Fsz,_M); ratio_llr=_bit_llr_prior(ratio_prior,bits0,bits1)
    if _RATIO_EM_BLEND!=0.0:
        oldr,oldi=hr,hi; sr=hr[...,None]*crv-hi[...,None]*civ; si=hr[...,None]*civ+hi[...,None]*crv; dist=(yrk[...,None]-sr).square()+(yik[...,None]-si).square(); pp=torch.softmax(-dist/den_em+_RATIO_EM_PRIOR_WEIGHT*ratio_prior.view(1,B,T,Fsz,_M),-1)
        ecr=(pp*crv).sum(-1); eci=(pp*civ).sum(-1); ec2=(pp*cabs2).sum(-1); conf=(pp.max(-1).values.clamp_min(1.0/float(_M))**_RATIO_EM_CONF_POWER)*emw; ad,anr,ani=_avg_k_triple(ec2*conf,(yrk*ecr+yik*eci)*conf,(yik*ecr-yrk*eci)*conf,active,dens,_EM_KT,_EM_KF)
        nr=anr/ad.clamp_min(_H_FLOOR*_H_FLOOR); ni=ani/ad.clamp_min(_H_FLOOR*_H_FLOOR); hr=(1-_RATIO_EM_BLEND)*oldr+_RATIO_EM_BLEND*nr; hi=(1-_RATIO_EM_BLEND)*oldi+_RATIO_EM_BLEND*ni
        sr=hr[...,None]*crv-hi[...,None]*civ; si=hr[...,None]*civ+hi[...,None]*crv; dist=(yrk[...,None]-sr).square()+(yik[...,None]-si).square(); post=-dist/den_em; mix=torch.logsumexp(post,-1); cand=_avg_k(mix,active,dens,_WEIGHT_KT,_WEIGHT_KF)/_CAND_WEIGHT_TEMP; fq,flogq=_crf(hr,hi,cand,active,dens,_FINAL_CRF_SWEEPS,_FINAL_CRF_LAMBDA,_CRF_TEMP,_CRF_KT,_CRF_KF); logw=(1-_FINAL_PRIOR_BLEND)*torch.log_softmax(cand,0)+_FINAL_PRIOR_BLEND*flogq; logw=logw-torch.logsumexp(logw,0,keepdim=True); w=torch.softmax(logw,0); mhr=(w*hr).sum(0,keepdim=True); mhi=(w*hi).sum(0,keepdim=True); base_log=torch.logsumexp(post+logw[...,None],0); base_log=base_log-torch.logsumexp(base_log,-1,keepdim=True)
    sym_post=torch.softmax(base_log,-1); all_hr=hr; all_hi=hi; all_prior=logw; all_hvar=hvar.unsqueeze(0).expand(K,B,T,Fsz)
    if _DD_ENABLE==1:
        dhr,dhi,dlog,dhv=_dd(yr,yi,no,sym_post,bitsf,cr,ci,active,gker,mhr[0],mhi[0]); all_hr=torch.cat((all_hr,dhr.unsqueeze(0)),0); all_hi=torch.cat((all_hi,dhi.unsqueeze(0)),0); all_prior=torch.cat((all_prior,dlog.unsqueeze(0)),0); all_hvar=torch.cat((all_hvar,dhv.unsqueeze(0)),0)
    sr=all_hr[...,None]*crv-all_hi[...,None]*civ; si=all_hr[...,None]*civ+all_hi[...,None]*crv; dist=(yrk[...,None]-sr).square()+(yik[...,None]-si).square(); den=no.view(1,B,1,1,1)+_DEMAP_FLOOR+cabs2*all_hvar[...,None]; evid=-dist/den-torch.log(den.clamp_min(1e-9)); score=_avg_k(torch.logsumexp(evid+score_view,-1),active,dens,_WEIGHT_KT,_WEIGHT_KF)/_ALL_CAND_TEMP; unary=(1-_ALL_PRIOR_BLEND)*score+_ALL_PRIOR_BLEND*all_prior; aq,alogq=_crf(all_hr,all_hi,unary,active,dens,_ALL_CRF_SWEEPS,_ALL_CRF_LAMBDA,_CRF_TEMP,_CRF_KT,_CRF_KF); logw_all=(1-_ALL_FINAL_PRIOR_BLEND)*torch.log_softmax(unary,0)+_ALL_FINAL_PRIOR_BLEND*alogq; logw_all=logw_all-torch.logsumexp(logw_all,0,keepdim=True)
    if _FINAL_AFFINE_ENABLE==1:
        all_hr,all_hi,all_hvar,logw_all=_final_affine(zc,all_hr,all_hi,all_hvar,evid,logw_all,prior_view,cr,ci,active,dens,dmat,dhd,eye,mom); sr=all_hr[...,None]*crv-all_hi[...,None]*civ; si=all_hr[...,None]*civ+all_hi[...,None]*crv; dist=(yrk[...,None]-sr).square()+(yik[...,None]-si).square(); den=no.view(1,B,1,1,1)+_DEMAP_FLOOR+cabs2*all_hvar[...,None]; evid=-dist/den-torch.log(den.clamp_min(1e-9))
    metric=evid+logw_all[...,None]+prior_view
    # Final compact symbol marginalization: keep all channel hypotheses, but first
    # marginalize them per symbol, then use a ratio-aware top-L + per-bit guard set.
    # Unselected symbols enter as a tail correction (exact for default temp=1, logw=0).
    big=-1e9
    sym_metric=torch.logsumexp(metric,0)  # [B,T,F,M], all channel candidates retained
    sel_score=base_log+prior_view[0]
    ksym=_FINAL_SYM_TOPL if _FINAL_SYM_TOPL < _M else _M
    _,top_sym=torch.topk(sel_score,k=ksym,dim=-1)
    scb=sel_score[...,None].expand(B,T,Fsz,_M,NUM_BITS_PER_SYMBOL)
    b1v=bits1.view(1,1,1,_M,NUM_BITS_PER_SYMBOL); b0v=bits0.view(1,1,1,_M,NUM_BITS_PER_SYMBOL)
    guard1=torch.argmax(torch.where(b1v,scb,big),dim=-2)
    guard0=torch.argmax(torch.where(b0v,scb,big),dim=-2)
    sel_idx=torch.cat((top_sym,guard1,guard0),dim=-1)
    selected=torch.zeros((B,T,Fsz,_M),dtype=torch.bool,device=yr.device)
    selected.scatter_(3,sel_idx,True)
    sm=sym_metric[...,None]
    sel_log=torch.where(selected[...,None],sm,big)
    tail_log=torch.where((~selected)[...,None],sm/_FINAL_TAIL_TEMP+_FINAL_TAIL_LOGW,big)
    m1=bits1.view(1,1,1,_M,NUM_BITS_PER_SYMBOL); m0=bits0.view(1,1,1,_M,NUM_BITS_PER_SYMBOL)
    llr1=torch.logaddexp(torch.logsumexp(torch.where(m1,sel_log,big),-2),torch.logsumexp(torch.where(m1,tail_log,big),-2))
    llr0=torch.logaddexp(torch.logsumexp(torch.where(m0,sel_log,big),-2),torch.logsumexp(torch.where(m0,tail_log,big),-2))
    llr_raw=llr1-llr0
    sub_damp=(_RATIO_LLR_SUB_DAMP*(1.0+_RATIO_LLR_SUB_UNREL_GAIN*(1.0-ratio_rel))).clamp(0.0,_RATIO_LLR_SUB_MAX); llr=torch.clamp(llr_raw-sub_damp[...,None]*_RATIO_DEMAP_W*ratio_llr,min=-_LLR_CLIP,max=_LLR_CLIP).to(torch.float32)
    return llr[:,None,None]