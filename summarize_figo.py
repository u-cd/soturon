import pandas as pd
from pathlib import Path

def main():
    base = Path(__file__).parent
    csv_path = base / 'figo_stage_survival.csv'
    df = pd.read_csv(csv_path, encoding='utf-8')
    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    # Clean stage: empty strings -> NaN
    df['figo_stage'] = df['figo_stage'].replace('', pd.NA)
    total = len(df)
    missing_stage = df['figo_stage'].isna().sum()
    event_col = 'event'
    # Ensure event numeric
    df[event_col] = pd.to_numeric(df[event_col], errors='coerce')
    events = int(df[event_col].sum())
    censored = total - events
    # Stage counts
    stage_counts = df['figo_stage'].value_counts(dropna=False).to_dict()
    # Event rate by stage
    stage_event_rate = (
        df.groupby('figo_stage')[event_col]
        .agg(['count','sum'])
        .assign(event_rate=lambda x: x['sum']/x['count'])
    )
    # Median survival days overall and by stage
    median_overall = int(df['survival_days'].median())
    survival_by_stage = (
        df.groupby('figo_stage')['survival_days']
        .agg(['count','median','min','max'])
        .sort_values('median', ascending=False)
    )
    # Longest and shortest observed survival (overall)
    min_surv = int(df['survival_days'].min())
    max_surv = int(df['survival_days'].max())

    lines = []
    lines.append('データ概要: figo_stage_survival.csv')
    lines.append(f'総観測数: {total}')
    lines.append(f'イベント(=1)数: {events}')
    lines.append(f'打ち切り(=0)数: {censored}')
    lines.append(f'ステージ欠損数: {missing_stage}')
    lines.append('')
    lines.append('ステージ別件数:')
    for stage, cnt in stage_counts.items():
        label = stage if stage is not pd.NA else 'MISSING'
        lines.append(f'  {label}: {cnt}')
    lines.append('')
    lines.append('ステージ別イベント率: (count, events, event_rate)')
    for stage, row in stage_event_rate.iterrows():
        lines.append(f'  {stage}: {int(row["count"])} {int(row["sum"])} {row["event_rate"]:.3f}')
    lines.append('')
    lines.append(f'全体 生存日数 中央値: {median_overall} (最小 {min_surv}, 最大 {max_surv})')
    lines.append('')
    lines.append('ステージ別 生存日数要約: (count, median, min, max)')
    for stage, row in survival_by_stage.iterrows():
        lines.append(f'  {stage}: {int(row["count"])} {int(row["median"])} {int(row["min"])} {int(row["max"])}')
    lines.append('')
    lines.append('変数定義案:')
    lines.append('  case_id: 患者/症例識別子 (非解析目的、レポートで直接利用しない)')
    lines.append('  figo_stage: FIGO病期分類 (カテゴリ変数)。欠損は除外または「不明」カテゴリ。')
    lines.append('  survival_days: 観測開始からイベントまたは打ち切りまでの日数。')
    lines.append('  event: 1=事象発生 (例: 死亡), 0=右打ち切り。')
    lines.append('解析方針案:')
    lines.append('  1. 欠損ステージの除外または不明カテゴリ化。')
    lines.append('  2. FIGO病期を基準カテゴリ (例: Stage IA/IB) としたハザード比推定。')
    lines.append('  3. 生存曲線: stage別Kaplan-Meierを付録、本文はハザード比解釈中心。')
    lines.append('  4. 比例性診断: Schoenfeld残差でステージ係数の時間依存性確認。')
    lines.append('  5. 必要なら病期を順序尺度化し線形トレンド検定。')
    lines.append('  6. 外れ値: survival_days極小/極大は計測異常か確認。')
    lines.append('  7. 長期生存 (最大値付近) の影響評価。')

    out_path = base / 'データ概要.txt'
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'概要を書き出しました: {out_path}')

if __name__ == '__main__':
    main()
