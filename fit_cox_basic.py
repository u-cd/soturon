import pandas as pd
from pathlib import Path
from lifelines import CoxPHFitter

def prepare(df: pd.DataFrame):
    df = df.copy()
    # Drop missing stage
    df['figo_stage'] = df['figo_stage'].replace('', pd.NA)
    df = df.dropna(subset=['figo_stage'])
    # Grouping: EarlyMid (all except IIIC, IV), IIIC, IV
    def stage_group(s):
        if s == 'Stage IIIC':
            return 'IIIC'
        if s == 'Stage IV':
            return 'IV'
        return 'EarlyMid'
    df['stage_group'] = df['figo_stage'].apply(stage_group)
    # Encode categorical using pandas get_dummies (reference: IIIC)
    dummies = pd.get_dummies(df['stage_group'])
    # Drop reference column IIIC
    if 'IIIC' in dummies.columns:
        dummies = dummies.drop(columns=['IIIC'])
    df = pd.concat([df, dummies], axis=1)
    # Rename columns for clarity
    rename_map = {}
    if 'EarlyMid' in dummies.columns:
        rename_map['EarlyMid'] = 'stage_EarlyMid_vs_IIIC'
    if 'IV' in dummies.columns:
        rename_map['IV'] = 'stage_IV_vs_IIIC'
    df = df.rename(columns=rename_map)
    # Survival time and event
    df['survival_days'] = pd.to_numeric(df['survival_days'], errors='coerce')
    df['event'] = pd.to_numeric(df['event'], errors='coerce')
    df = df.dropna(subset=['survival_days', 'event'])
    return df

def fit_basic(df: pd.DataFrame):
    cph = CoxPHFitter()
    # Select columns: survival_days, event + covariates
    covars = [c for c in df.columns if c.startswith('stage_') and c != 'stage_group']
    # Force numeric type
    for c in covars:
        df[c] = pd.to_numeric(df[c], errors='raise')
    use = df[['survival_days', 'event'] + covars].copy()
    cph.fit(use, duration_col='survival_days', event_col='event')
    return cph, covars

def main():
    base = Path(__file__).parent
    csv_path = base / 'figo_stage_survival.csv'
    out_path = base / 'cox_basic_results.txt'
    df = pd.read_csv(csv_path, encoding='utf-8')
    df_prep = prepare(df)
    cph, covars = fit_basic(df_prep)
    summary = cph.summary.loc[covars]
    lines = []
    lines.append('基本Cox比例ハザードモデル（単変量: 病期グループ）')
    lines.append('参照カテゴリ: Stage IIIC')
    lines.append('グループ化: EarlyMid = (I/II/IIIA/IIIB/IIC など), IIIC, IV')
    lines.append('')
    lines.append('係数推定 (β), HR=exp(β), 95%CI, p値:')
    for idx, row in summary.iterrows():
        hr = row['exp(coef)']
        beta = row['coef']
        ci_lower = row['exp(coef) lower 95%']
        ci_upper = row['exp(coef) upper 95%']
        pval = row['p']
        lines.append(f'  {idx}: beta={beta:.4f} HR={hr:.3f} 95%CI=({ci_lower:.3f}, {ci_upper:.3f}) p={pval:.3g}')
    lines.append('')
    # Concordance index
    lines.append(f'C-index: {cph.concordance_index_:.3f}')
    # Global log-likelihood & AIC
    lines.append(f'Log-likelihood: {cph.log_likelihood_:.3f}')
    lines.append(f'AIC (approx): {-2 * cph.log_likelihood_ + 2 * len(covars):.3f}')
    lines.append('')
    # Minimal proportional hazards check using Schoenfeld残差ベース検定
    # 使用データフレームはモデルに投入した列（survival_days, event, 共変量）のみに限定し、非数値列を除く
    try:
        from lifelines.statistics import proportional_hazard_test
        # df_prep にはダミー共変量が含まれているのでこちらを使用する
        covars_df = df_prep[['survival_days', 'event'] + covars].copy()
        test = proportional_hazard_test(cph, covars_df, time_transform='rank')
        lines.append('PH仮定検定（Schoenfeld残差, time_transform=rank）:')
        for cov in covars:
            p_ph = test.summary.loc[cov, 'p']
            stat = test.summary.loc[cov].get('chi2', test.summary.loc[cov].get('test', float('nan')))
            lines.append(f'  {cov}: stat={stat:.3f} p={p_ph:.3g}')
        if 'global' in test.summary.index:
            g_p = test.summary.loc['global', 'p']
            g_stat = test.summary.loc['global'].get('chi2', test.summary.loc['global'].get('test', float('nan')))
            lines.append(f'  global: stat={g_stat:.3f} p={g_p:.3g}')
        lines.append('PH検定 summary table:')
        lines.append(str(test.summary))
    except Exception as e:
        lines.append(f'PH検定実行失敗: {e}')
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'結果を書き出しました: {out_path}')

if __name__ == '__main__':
    main()
