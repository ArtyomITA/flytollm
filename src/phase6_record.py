"""Record finished control runs into SUITE_TEST_FASE_6B.md: for every known run whose results JSON exists, replace the
'Risultato: in coda.' / 'Risultato: in corso.' line of its test with the DEV curve every 500 updates, the final CE and
accuracy, the delta against the reference and the pre-registered verdict, plus the greedy generation.
Idempotent: a test already recorded (marker 'Risultato (run ') is left alone unless --force."""
import argparse, json, re
from bench_runtime import ROOT

REAL_2000 = 5.087
BASE7_2000 = 4.580  # phase7_baseline_4000 at 2000 updates (standard configuration)
REAL_8000 = 3.959
RUNS = {
    # name: (heading marker in the .md, reference CE, reference label)
    'phase6c_permweights_2000': ('### B3 =', REAL_2000, 'reale 5,087'),
    'phase6c_permsigns_2000': ('### B4 =', REAL_2000, 'reale 5,087'),
    'phase6c_erdosrenyi_2000': ('### B5 =', 4.533, 'configuration model 4,533'),
    'phase6c_withinsuperclass_2000': ('### B6 =', 4.931, 'gradi conservati 4,931'),
    'phase6c_ports_random_2000': ('### B7 =', REAL_2000, 'reale 5,087'),
    'phase6c_ports_olfactory_2000': ('### B7 =', REAL_2000, 'reale 5,087'),
    'phase6c_readout_anatomical_2000': ('### D3 =', REAL_2000, 'reale 5,087'),
    'phase6c_core_apl_2000': ('### D4 =', REAL_2000, 'reale 5,087'),
    'phase6c_core_graded_ol_2000': ('### D5 =', REAL_2000, 'reale 5,087'),
    'phase6c_core_tau_type_2000': ('### D6 =', REAL_2000, 'reale 5,087'),
    'phase6c_core_reversal_2000': ('### D7 =', REAL_2000, 'reale 5,087'),
    'phase6c_relthr_2000': ('### D8 =', REAL_2000, 'reale 5,087'),
    'phase6c_softsign_2000': ('### D9 =', REAL_2000, 'reale 5,087'),
    'phase6d_ports_visual_2000': ('### F1 ', REAL_2000, 'reale 5,087'),
    'phase6d_readout_fru_2000': ('### F2 ', REAL_2000, 'reale 5,087'),
    'phase6d_visual_fru_2000': ('### F3 ', REAL_2000, 'reale 5,087'),
    'phase6d_readout_hub_2000': ('### F4 ', REAL_2000, 'reale 5,087'),
    'phase6d_readout_cx_2000': ('### F5 ', REAL_2000, 'reale 5,087'),
    'phase6d_ports_auditory_2000': ('### F6 ', REAL_2000, 'reale 5,087'),
    'phase6d_visual_hub_2000': ('### F7 ', REAL_2000, 'reale 5,087'),
    'phase6s_S8_fused_reorder_2000': ('S8 (2000 update, kernel fuso + riordino', REAL_2000, 'pilota 5,087'),
    'phase6s_S0c_reference_2000': ('S8 (2000 update, kernel fuso + riordino', REAL_2000, 'pilota 5,087'),
    'phase6s_S8b_fused_2000': ('S8 (2000 update, kernel fuso + riordino', REAL_2000, 'pilota 5,087'),
    # phase 6f promotions: fresh 8000-update runs, compared with the real fly at 8000
    'phase6b_configmodel_s41_8000': ('### B2 =', REAL_8000, 'reale 8000 3,959'),
    'phase6c_erdosrenyi_8000': ('### B5 =', REAL_8000, 'reale 8000 3,959'),
    'phase6c_permsigns_8000': ('### B4 =', REAL_8000, 'reale 8000 3,959'),
    'phase6c_ports_random_8000': ('### B7 =', REAL_8000, 'reale 8000 3,959'),
    'phase6c_readout_anatomical_8000': ('### D3 =', REAL_8000, 'reale 8000 3,959'),
    'phase6c_relthr_8000': ('### D8 =', REAL_8000, 'reale 8000 3,959'),
    'phase6c_softsign_8000': ('### D9 =', REAL_8000, 'reale 8000 3,959'),
    # phase 7: standard baseline and 'tengo' levers (SUITE section G); tengo runs compared with the baseline at 2000 once known
    'phase7_baseline_4000': ('### G0 Baseline', 4.070, 'porte random 4000 4,070'),
    'phase7_tengo_T8_2000': ('### G1 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_tengo_shortpath_2000': ('### G2 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_tengo_relthr5_2000': ('### G3 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_tengo_all_2000': ('### G4 ', BASE7_2000, 'baseline standard 2000 4,580'),
    # phase 7c levers (SUITE section H), compared with the standard baseline at 2000
    'phase7_lever_bias_2000': ('### H1 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_bias_lr1e4_2000': ('### H1 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_bias_lr3e4_2000': ('### H1 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_bias_scaled_2000': ('### H1 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_tau_2000': ('### H2 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_reversal_2000': ('### H3 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_excit_2000': ('### H4 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_mono_2000': ('### H5 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_scale05_2000': ('### H6 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_scale2_2000': ('### H6 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_scale025_2000': ('### H6 ', BASE7_2000, 'baseline standard 2000 4,580'),
    # phase 7e: Muon on the dense matrices (SUITE section I)
    'phase7_lever_muon_2000': ('### I1 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_muon3e4_2000': ('### I2 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_T8muon_2000': ('### I3 ', 4.399, '8+8 sottopassi (G1) 4,399'),
    'phase7_lever_muonhead_2000': ('### I4 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_T8rev_2000': ('### G5 ', 4.399, '8+8 sottopassi (G1) 4,399'),
    # phase 7f (SUITE section J): advised tests and promotions
    'phase7_lever_tau3e3_2000': ('### J1 ', BASE7_2000, 'baseline standard 2000 4,580'),
    'phase7_lever_revtau_2000': ('### J2 ', 4.459, 'conduttanza (H3) 4,459'),
    'phase7_promote_T8_8000': ('### J4 ', 3.689, 'porte random 8000 3,689'),
    'phase7_promote_reversal_8000': ('### J4 ', 3.689, 'porte random 8000 3,689'),
    'phase7_promote_T8rev_8000': ('### J4 ', 3.689, 'porte random 8000 3,689'),
    # phase 7g (SUITE section K): follow-ups of the winners; K6 is re-referenced to the standard baseline at 8000 once K0 exists
    'phase7_lever_muon1e3_2000': ('### K1 ', 4.411, 'Muon 3e-4 (I2) 4,411'),
    'phase7_lever_tau1e2_2000': ('### K2 ', 4.462, 'tau lr 3e-3 (J1) 4,462'),
    'phase7_lever_T8muon3e4_2000': ('### K3 ', 4.399, '8+8 sottopassi (G1) 4,399'),
    'phase7_lever_revtau3e3_2000': ('### K4 ', 4.419, 'conduttanza + tau 1e-3 (J2) 4,419'),
    'phase7_lever_best_2000': ('### K5 ', 4.350, '8+8 + conduttanza (G5) 4,350'),
    'phase7_baseline_8000': ('### K0 ', 3.689, 'porte random 8000 3,689'),
    'phase7_promote_muon3e4_8000': ('### K6 ', 3.689, 'porte random 8000 3,689'),
    'phase7_lever_muon3e3_2000': ('### K7 ', 4.276, 'Muon 1e-3 (K1) 4,276'),
    'phase7_promote_muon_8000': ('### K8 ', 3.689, 'porte random 8000 3,689'),
    'phase7_promote_muon1e3_8000': ('### K8 ', 3.689, 'porte random 8000 3,689'),
    # phase 8a (SUITE section L): short tests on the safe default candidate (standard + Muon 1e-3, 4,275 at 2000)
    'phase8_L4_T8_2000': ('### L4 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_F9_shortpath_T8_2000': ('### F9 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_L5_T12_2000': ('### L5 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_L3_best_seed23_2000': ('### L3 ', 4.152, 'K5 seed 17 4,152'),
    'phase8_L7_best_scale05_2000': ('### L7 ', 4.152, 'K5 4,152'),
    # phase 8b (SUITE section M): short tests with new control code, on the safe default candidate
    'phase8_C14_core_lr1e3_2000': ('### C14 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_C14_core_lr1e2_2000': ('### C14 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_C16_soft_gw_2000': ('### C16 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_C16_soft_gw_corelr1e3_2000': ('### C16 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_C16_soft_gw_corelr1e2_2000': ('### C16 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_N6d_depth_shock_2000': ('### N6d ', 4.162, '12+12 costante (L5) 4,162'),
    'phase8_micro_default_300': ('### M0 ', 6.0, 'riferimento provvisorio: il default a 300 update viene letto dal suo JSON'),
    'phase8_micro_corelr1e3_300': ('### M0 ', 6.0, 'riferimento provvisorio: il default a 300 update viene letto dal suo JSON'),
    'phase8_micro_corelr3e3_300': ('### M0 ', 6.0, 'riferimento provvisorio: il default a 300 update viene letto dal suo JSON'),
    'phase8_micro_corelr1e2_300': ('### M0 ', 6.0, 'riferimento provvisorio: il default a 300 update viene letto dal suo JSON'),
    'phase8_micro_corelr3e2_300': ('### M0 ', 6.0, 'riferimento provvisorio: il default a 300 update viene letto dal suo JSON'),
    'phase8_micro_soft_corelr1e3_300': ('### M0 ', 6.0, 'riferimento provvisorio: il default a 300 update viene letto dal suo JSON'),
    'phase8_micro_soft_corelr3e3_300': ('### M0 ', 6.0, 'riferimento provvisorio: il default a 300 update viene letto dal suo JSON'),
    'phase8_micro_soft_corelr1e2_300': ('### M0 ', 6.0, 'riferimento provvisorio: il default a 300 update viene letto dal suo JSON'),
    'phase8_micro_soft_corelr3e2_300': ('### M0 ', 6.0, 'riferimento provvisorio: il default a 300 update viene letto dal suo JSON'),
    'phase8_H1_homeo_2000': ('### H1o ', 4.275, 'default candidato 2000 4,275'),
    'phase8_H2_arousal_pulse_2000': ('### H2a ', 4.275, 'default candidato 2000 4,275'),
    'phase8_H3_arousal_smooth_2000': ('### H2a ', 4.275, 'default candidato 2000 4,275'),
    'phase8_H4_homeo_T12_2000': ('### H1o ', 4.162, '12+12 costante (L5) 4,162'),
    'phase8_N1_reads2_2000': ('### N1 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_N2_every_2000': ('### N2 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_N3_every_firstkv_2000': ('### N3 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_N4_reads2_noid_2000': ('### N4 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_N5_inject_first_2000': ('### N5 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_M4_cond_ports_2000': ('### M4 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_A272_gain_group_2000': ('### A2.7.2 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_B253a_identity_2000': ('### B2.5.3a ', 4.275, 'default candidato 2000 4,275'),
    'phase8_B10_reservoir_2000': ('### B10 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_B8_ports_seed23_2000': ('### B8 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_B9a_ports2639_2000': ('### B9 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_B9b_ports1733_2000': ('### B9 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_B252a_real_on_2000': ('### B2.5.2 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_B252b_real_off_2000': ('### B2.5.2 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_B252c_rewired_on_2000': ('### B2.5.2 ', 4.275, 'default candidato 2000 4,275'),
    'phase8_B252d_rewired_off_2000': ('### B2.5.2 ', 4.275, 'default candidato 2000 4,275'),
}


def fmt(x):
    return f'{x:.3f}'.replace('.', ',')


def block(name, data, ref, ref_label):
    r = data['result']
    curve = ' / '.join(f"{e['update']}: {fmt(e['dev']['ce'])} ({e['dev']['accuracy']:.3f})" for e in r.get('curve', []))
    final = r['curve'][-1]['dev']['ce'] if r.get('curve') else float('nan')
    delta = ref - final
    if abs(delta) < 0.10:
        verdict = f'entro 0,10 dal riferimento ({ref_label}): nessun effetto misurabile'
    elif delta > 0:
        verdict = (f'migliore del riferimento ({ref_label}) di {fmt(delta)} nat: effetto positivo, vantaggio confermato a 8000' if ref == REAL_8000                   else f'migliore del riferimento ({ref_label}) di {fmt(delta)} nat: effetto positivo, candidata alla promozione a 8000')
    else:
        verdict = f'peggiore del riferimento ({ref_label}) di {fmt(-delta)} nat: effetto negativo'
    gen = next((g['text'] for g in r.get('generation', []) if g.get('temperature') == 0 and g.get('prompt') == 'Once upon a time'), '')
    ctrl = r.get('control', {})
    extra = []
    for key in ('ports', 'readout', 'core'):
        v = ctrl.get('stats', {}).get(key)
        if isinstance(v, dict):
            extra.append(f"{key}: " + ', '.join(f"{k}={val}" for k, val in v.items() if k != 'labels'))
    return (f"Risultato (run `{name}`, {r['updates']} update, {r['targets']} target, {r['mean_step_s']*1000:.0f} ms/update, core {ctrl.get('fast_mode', 'off')}): "
            f"curva CE DEV (accuracy) ogni 500: {curve}. Finale {fmt(final)}; Δ = {fmt(delta)} rispetto a {ref_label}. Lettura pre-registrata: {verdict}. "
            f"Generazione greedy: \"{' '.join(gen.split())}\". " + (' '.join(extra) if extra else ''))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--force', action='store_true'); a = p.parse_args()
    micro = ROOT / 'results/phase8_micro_default_300.json'
    if micro.exists():
        d = json.loads(micro.read_text())
        if d.get('ok') and d.get('result', {}).get('curve'):
            ce = d['result']['curve'][-1]['dev']['ce']
            for name in list(RUNS):
                if name.startswith('phase8_micro_') and name != 'phase8_micro_default_300':
                    RUNS[name] = ('### M0 ', ce, f'default a 300 update {fmt(ce)}')
            RUNS['phase8_micro_default_300'] = ('### M0 ', ce, f'se stesso {fmt(ce)}')
    base8000 = ROOT / 'results/phase7_baseline_8000.json'
    if base8000.exists():
        d = json.loads(base8000.read_text())
        if d.get('ok') and d.get('result', {}).get('curve'):
            ce = d['result']['curve'][-1]['dev']['ce']
            RUNS['phase7_promote_muon3e4_8000'] = ('### K6 ', ce, f'baseline standard 8000 {fmt(ce)}')
            RUNS['phase7_promote_muon_8000'] = ('### K8 ', ce, f'baseline standard 8000 {fmt(ce)}')
            RUNS['phase7_promote_muon1e3_8000'] = ('### K8 ', ce, f'baseline standard 8000 {fmt(ce)}')
    path = ROOT / 'SUITE_TEST_FASE_6B.md'; text = path.read_text(encoding='utf8'); changed = []
    for name, (marker, ref, ref_label) in RUNS.items():
        jp = ROOT / 'results' / f'{name}.json'
        if not jp.exists():
            continue
        data = json.loads(jp.read_text())
        if not data.get('ok') or not isinstance(data.get('result'), dict):
            continue
        start = text.find(marker)
        if start < 0:
            print('marker not found', name, marker); continue
        ends = [e for e in (text.find('\n### ', start + 1), text.find('\n## ', start + 1), text.find('\n---', start + 1)) if e >= 0]
        section_end = min(ends) if ends else len(text)
        section = text[start:section_end]
        if f'run `{name}`' in section and not a.force:
            continue
        new_block = block(name, data, ref, ref_label)
        if re.search(r'Risultato: in (coda|corso)\.', section):
            section_new = re.sub(r'Risultato: in (coda|corso)\.', new_block, section, count=1)
        elif f'run `{name}`' in section:
            section_new = re.sub(r'Risultato \(run `' + re.escape(name) + r'`.*?(?=\n|$)', new_block, section, count=1)
        else:
            section_new = section.rstrip('\n') + '\n' + new_block + '\n'
        text = text[:start] + section_new + text[section_end:]
        changed.append(name)
    path.write_text(text, encoding='utf8')
    print('recorded:', changed)


if __name__ == '__main__':
    main()
