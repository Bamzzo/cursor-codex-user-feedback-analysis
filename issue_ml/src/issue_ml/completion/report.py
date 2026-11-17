import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve
from .data import TIME, STRUCTURE, EARLY, PROXY

DESCRIPTIONS = {
 'hour_sin':'创建小时的周期编码（UTC）','hour_cos':'创建小时的周期编码（UTC）',
 'weekday_sin':'创建星期的周期编码（UTC）','weekday_cos':'创建星期的周期编码（UTC）','weekend':'创建日是否周末（UTC）',
 'title_length':'快照标题字符数','body_length':'快照正文字符数','word_count':'快照正文空白分词数',
 'has_code':'快照正文出现代码围栏或 code 标签','has_log':'快照正文包含 traceback/stack trace/exception/error 模式',
 'has_version':'快照正文包含点分版本号','url_count':'快照正文 URL 数量','body_missing':'快照正文为空',
 'comment_count':'h小时内所有评论数','participant_count':'h小时内不同评论账号数',
 'author_comment_count':'h小时内作者本人评论数','other_comment_count':'h小时内非作者评论数（含机器人）',
 'other_participant_count':'h小时内非作者不同账号数','has_other':'h小时内是否有非作者评论',
 'first_other_hours':'首条非作者评论等待小时；无回复填h并结合has_other',
 'author_after_other':'首条非作者评论之后且h小时内作者是否再次评论',
 'external_nonbot_count':'h小时内非作者且非机器人代理评论数','external_nonbot_participants':'对应不同账号数',
 'has_external_nonbot':'h小时内是否有外部非机器人代理评论',
 'first_external_nonbot_hours':'首条外部非机器人等待小时；无回复填h并结合缺失标记',
 'author_after_external_nonbot':'外部非机器人回复之后且h小时内作者是否再评论',
 'bot_proxy_count':'账号后缀或App标记判为机器人代理的早期评论数',
 'maintainer_proxy_count':'早期外部非机器人且身份为MEMBER/COLLABORATOR/OWNER的评论数',
}

def table(frame,cols):
    lines=['|'+'|'.join(cols)+'|','|'+'|'.join(['---']*len(cols))+'|']
    for _,row in frame.iterrows():
        lines.append('|'+'|'.join(f'{row[c]:.4f}' if isinstance(row[c],(float,np.floating)) else str(row[c]) for c in cols)+'|')
    return '\n'.join(lines)

def write_report(output,selection,audit,flow,results,predictions):
    output=Path(output);h,d,variant=selection['h'],selection['d'],selection['variant']
    picked=results[(results.h==h)&(results.d==d)&(results.variant==variant)]
    model=picked[picked.model.eq('random_forest')].iloc[0];base=picked[picked.model.eq('train_prior')].iloc[0]
    pred=predictions[(predictions.h==h)&(predictions.d==d)&(predictions.variant==variant)]
    compare=results[(results.h==h)&(results.d==d)&results.model.eq('random_forest')]
    params=json.loads((output/'config.json').read_text(encoding='utf-8'))
    validation=pd.read_csv(output/'window_validation.csv')
    split=pd.read_csv(output/'split_counts.csv')
    selected_split=split[(split.h==h)&(split.d==d)&split.variant.eq(variant)]
    ci=json.loads((output/'paired_uncertainty.json').read_text())
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    precision,recall,_=precision_recall_curve(pred.y,pred.p_completed)
    fig,ax=plt.subplots(figsize=(7,5));ax.plot(recall,precision,label=f'随机森林 AP={model.average_precision:.3f}')
    ax.axhline(model.positive_rate,color='gray',linestyle='--',label=f'无排序信息 AP={model.positive_rate:.3f}')
    ax.set(xlabel='召回率',ylabel='精确率',title=f'{h}小时观察 / 创建后{d}天内 completed');ax.legend();fig.tight_layout();fig.savefig(output/'precision_recall.png',dpi=160);plt.close(fig)
    actual,mean=calibration_curve(pred.y,pred.p_completed,n_bins=8,strategy='quantile')
    pd.DataFrame({'mean_prediction':mean,'observed_rate':actual}).to_csv(output/'calibration.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,5));ax.plot(mean,actual,'o-',label='随机森林');ax.plot([0,1],[0,1],'--',color='gray')
    upper=min(1,max(.2,float(max(mean.max(),actual.max()))*1.2));ax.set(xlim=(0,upper),ylim=(0,upper),xlabel='预测完成概率均值',ylabel='实际完成率',title='测试集概率校准');ax.legend();fig.tight_layout();fig.savefig(output/'calibration.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,['roc_auc','average_precision']):
        ax.bar(compare.variant,compare[metric]);ax.set_title(metric);ax.tick_params(axis='x',rotation=25)
    fig.suptitle(f'同一队列的随机森林特征比较：h={h}, d={d}');fig.tight_layout();fig.savefig(output/'feature_comparison.png',dpi=160);plt.close(fig)
    window=validation[validation.variant.eq('structure_early')].copy()
    window['ap_lift']=window.average_precision/window.positive_rate
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    by_h=window[window.d.eq(30)].sort_values('h')
    by_d=window[window.h.eq(h)].sort_values('d')
    axes[0].plot(by_h.h,by_h.ap_lift,'o-');axes[0].set(xlabel='观察小时 h',ylabel='验证 AP / 验证完成率',title='固定创建后30天期限')
    axes[1].plot(by_d.d,by_d.ap_lift,'o-');axes[1].set(xlabel='创建后完成期限 d（天）',ylabel='验证 AP / 验证完成率',title=f'固定观察{h}小时')
    fig.tight_layout();fig.savefig(output/'window_comparison.png',dpi=160);plt.close(fig)
    field_rows=[]
    for f in TIME+STRUCTURE+EARLY+PROXY:
        group='创建时间' if f in TIME else '快照描述结构' if f in STRUCTURE else '早期互动计数' if f in EARLY else '身份/机器人代理'
        field_rows.append({'feature':f,'group':group,'definition':DESCRIPTIONS[f],'availability':'snapshot text limitation' if f in STRUCTURE else 'snapshot identity limitation' if f in PROXY else 'created timestamp and account-based counts'})
    field_rows.append({'feature':'text_svd_0..63','group':'快照文本','definition':'训练集拟合单/双词TF-IDF后截断SVD到64维，随机森林使用稠密表示','availability':'snapshot text limitation'})
    pd.DataFrame(field_rows).to_csv(output/'feature_dictionary.csv',index=False,encoding='utf-8-sig')
    lift=model.average_precision/model.positive_rate
    lines=[
      '# Issue completed 预测：随机森林结果', '',
      f'验证阶段选定：观察 **{h} 小时**，预测创建后 **{d} 天**内的 completed；特征 **{variant}**。',
      f'测试 ROC-AUC **{model.roc_auc:.3f}**，AP **{model.average_precision:.3f}**，测试完成率 **{model.positive_rate:.2%}**，AP/完成率 **{lift:.2f}**。',
      f'Brier={model.brier:.4f}，训练先验基线={base.brier:.4f}；阈值={model.threshold:.4f}，Precision={model.precision:.2%}，Recall={model.recall:.2%}，F1={model.f1:.3f}。',
      f'预测最不易完成的前20%共 {int(model.top20_count)} 条，实际未完成率 **{model.top20_incomplete_rate:.2%}**；总体未完成率 **{model.overall_incomplete_rate:.2%}**，提升倍数 **{model.top20_lift:.3f}**。',
      '', '## 数据与代理标签',
      f'分析截点 {audit["analysis_cutoff"]}，为保守分析截点，尚无独立采集结束时间。Issue最大更新时间为 {audit["issue_time_maxima"]["updated_at"]}。',
      '排除非法时间、截点后创建、当前已知 reopened；按创建时间保留最早的完全相同文本、规范化相同标题及高相似词面近重复。',
      '近重复规则：至少30词，三词shingle的64位SimHash；四段LSH检索，Hamming≤3且Jaccard≥0.85。该规则处理词面近重复，不能消除全部语义近重复。',
      '保留创建满d天且快照未显示h小时内关闭的记录。y=1当且仅当快照state=closed、state_reason=completed且h<关闭等待小时≤24d；其余可判定记录y=0。',
      '关闭时间超过分析截点的记录在截点视作尚未观测关闭。duplicate/not_planned均为负例。',
      '由于没有完整事件历史，当前open记录可能曾关闭，当前closed记录也可能多次关闭。队列与标签是快照代理，不是首次完成事件或精确历史存续状态。',
      '', table(flow,['h','d','after_deduplication','mature_d','closed_by_h_removed','eligible','positives','rate','train_n','validation_n','test_n','unused_n']),
      '', '## 时间划分和调参',
      f'训练创建时间 < {params["train_end"]}；验证 [{params["validation_start"]}, {params["validation_end"]})；测试 [{params["test_start"]}, {params["test_end"]})。所有时间UTC。',
      '训练和验证后分别留出至少30天标签成熟期，各期限用同一套日期。成熟但位于时间间隔或测试窗口外的记录不纳入拟合或评分。',
      '每个窗口/特征方案尝试6组随机森林参数。TF-IDF与SVD只拟合训练集；没有逻辑回归。先在d=30比较h，再固定h比较d；窗口选择规则见selection.json。',
      '参数按验证AP选择，阈值按验证F1选择。h选及时性调整AP提升达到最佳95%的最早时点；d按同一效用最大值选择。最终测试前锁定参数、窗口、特征与阈值，不根据测试成绩改选。',
      '所有模型均只使用训练期拟合，没有把验证集并入重训，以保持特征比较与阈值对应的模型一致。',
      '',table(selected_split,['split','n','positive','rate']),
      '', '## 窗口验证结果',table(validation,['h','d','variant','candidate','n','positives','positive_rate','roc_auc','average_precision','brier','utility']),
      '', '## 最终测试结果（测试后不再选窗口）',table(results,['h','d','variant','model','n','positives','positive_rate','roc_auc','average_precision','brier','precision','recall','f1','top20_incomplete_rate']),
      '', '## 特征与结果解读',
      'early_only只使用创建星期/小时和早期评论账号计数、回复时延，排除快照正文、描述结构、作者关联身份及机器人/App代理。其他方案的身份代理也可能不是当时身份。',
      'text_structure使用快照文本、长度/代码等结构及创建时间；structure_early加入早期互动与身份代理；combined再加入文本。只在相同h/d和划分内讨论特征增益。',
      '文本通过TF-IDF→64维TruncatedSVD输入森林，避免用数万维稀疏文本直接训练大量决策树。这是本实验的文本表示选择，不把不同特征效果差异归因为算法差异。',
      f'同队列combined−text_structure配对bootstrap 95%区间：AUC增量 {ci["delta_auc_ci95"]}；AP增量 {ci["delta_ap_ci95"]}。该区间只反映当前测试集抽样不确定性。',
      'AP为Average Precision，按召回增量加权的精确率，不是准确率或简单梯形PR曲线面积；常数概率AP等于测试完成率。Brier越低越好。',
      '未完成通常占绝大多数，前20%未完成比例即使很高，也必须对照总体未完成率；不能仅凭百分比宣称跟进排序有效。',
      '训练先验基线会受时间完成率变化影响。class_weight会影响概率校准，已同时报告Brier与校准曲线；未使用测试标签校准。',
      '模型重要性只是关联；不能推出“回复促进完成”。正文与身份快照误差、未识别的重开和评论缺失均限制历史预测解释。',
      '', '## 产物',
      'audit.json、sample_flow.csv、feature_dictionary.csv：审计/样本/字段；tuning_validation.csv：全部调参记录；selection.json：测试前选择；test_metrics.csv：测试指标；selected_predictions.csv：最终逐条预测；selected_errors.csv：误判；selected_model.joblib：可复用模型包。',
      'sample_membership.csv包含用于核查标签的状态字段，仅作为审计输出，从未作为模型输入。',
      'comment_completeness_sensitivity.json是在父表评论数与实际匹配数一致的测试子集上的敏感性检查；一致也不等于历史完整。',
    ]
    (output/'结论报告.md').write_text('\n\n'.join(lines),encoding='utf-8')
