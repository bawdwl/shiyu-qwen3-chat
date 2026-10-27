"""生成仅供评估的20题；拒绝覆盖，引用事实全部写在题内。"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def cases():
    rows = []

    def add(category, prompts, reference, rubric):
        rows.append(dict(id=f'q3_{len(rows)+1:03}', category=category,
                         prompts=[prompts] if isinstance(prompts, str) else prompts,
                         content_reference=reference, content_rubric=rubric,
                         format_check=None, review_required=True,
                         source='assistant-authored-synthetic-quality-holdout-v1'))

    add('材料与证据', '记录写明：任务甲收到请求后生成失败，未执行保存；任务乙生成成功并完成保存。有人据此说两个任务都保存了。请只依据记录，在60字内纠正。',
        '只有任务乙完成保存，任务甲生成失败且未保存；两个都保存的说法与记录矛盾。', ['明确甲未保存、乙已保存', '不编造失败原因'])
    add('材料与证据', '测试报告：只测了单用户连续请求，全部成功。负责人声称已经证明100人同时使用也稳定。只依据这段材料，用两句话评价证据是否足够。',
        '证据不足。单用户连续请求的结果不能证明100人同时使用，需要对应并发测试。', ['区分顺序与并发证据', '不把缺乏证明说成已经证明会失败'])
    add('材料与证据', '记录：版本甲回答用了80个token、8秒；版本乙用了20个token、4秒。只按输出token/总秒数计算，两者各是多少？能否仅凭总时长说乙生成速度更高？限80字。',
        '甲10 token/s，乙5 token/s；乙总时长短但该口径速率更低，不能仅凭总时长说更快。', ['计算两个速率正确', '区分答案长度、总耗时与速率'])
    add('材料与证据', '评估规定：24题中只有9题可自动检查；这9题通过6题，其余15题尚未人工评分。请在60字内说明可以报告什么，不能报告什么。',
        '可报告自动规则6/9；其余15题未评分，不能把6/24称为整体正确率或默认其余都错。', ['分母是有规则的9题', '未评审不是错误或通过'])
    add('材料与证据', '实验只报告验证loss由4.8降到3.9，没有新问题回答或评分。请用两句话说明这支持什么结论，还缺少什么证据。',
        '支持该验证目标上的预测损失降低；尚缺独立回答质量评估，不能推出整体问答能力提高。', ['限定loss含义', '不宣称必然提高或必然退化'])

    add('给定规则推理', '本题规定：只有“生成成功且保存成功”才算完成。记录：生成成功，保存失败。此任务算完成吗？请给出结论和原因，40字内。',
        '不算完成，因为保存失败，不满足两个条件同时成立。', ['按合取条件判断', '不因生成成功忽略保存失败'])
    add('给定规则推理', '约定：取消早于提交则不保存；提交完成后取消不撤销已保存内容。时间线：10:00:01提交完成，10:00:02取消。请用40字内说明最终是否保存。',
        '已保存，取消在提交完成之后，不撤销已保存内容。', ['按时间先后使用规则', '不把取消等同回滚已提交内容'])
    add('给定规则推理', '会话规则：切换模式创建新会话并保留旧会话；清空只删除当前会话消息。甲会话有两条消息，切到乙后清空乙。甲的两条消息还在吗？40字内说明。',
        '还在。切换保留甲，清空乙只影响乙的消息。', ['区别会话身份', '不将清空当前扩展到所有会话'])
    add('给定规则推理', '本题规则：额度至少18才可执行，每执行一次扣6。当前额度18，依次请求两次，第一次成功扣费后重新检查。两次结果如何，最终额度多少？60字内。',
        '第一次可执行，额度降为12；第二次不满足至少18，不执行不扣费，最终12。', ['第二次用更新后额度检查', '不把执行费用当准入门槛'])
    add('给定规则推理', '输入限制分别是消息最多90字符、模板后总输入最多120 token。某消息70字符，但历史加模板共125 token。按规则应接受还是拒绝？40字内解释。',
        '拒绝。字符限制通过，但总输入125 token超过120 token。', ['两个限制分别检查', '不把字符与token混为一谈'])

    add('多轮更正与作用域', ['项目代号是 river_482，负责人暂定周岚。只需确认。', '更正：代号改为 pine_915，负责人不变。请只回答当前代号和负责人。'],
        'pine_915，周岚。', ['使用最新代号', '未修改负责人'])
    add('多轮更正与作用域', ['有甲、乙两个会话。甲用Base，乙用LoRA。先记下。', '只将乙改成Base；甲保持原样。现在两个会话分别用什么？不超过40字。'],
        '甲Base，乙Base。', ['局部更正乙', '甲仍为Base'])
    add('多轮更正与作用域', ['仓库有红、蓝两个箱子，钥匙在红箱；蓝箱放卡片。请确认。', '只交换两个箱子的标签，不移动物品。现在钥匙所在箱子贴什么标签？一句话。'],
        '钥匙所在的原红箱现在贴蓝标签。', ['跟随标签交换', '不误认为物品移动'])
    add('多轮更正与作用域', ['规则：成绩至少72算通过。林禾成绩71，许宁成绩72。请确认。', '只把林禾成绩更正为73。请分别给出两人的最终通过状态，40字内。'],
        '林禾通过，许宁通过。', ['林禾使用更正成绩', '至少包含等于边界'])
    add('多轮更正与作用域', ['计划记录：周一检查数据，周二训练，周三评估。先记下。', '只取消周二训练，不移动其他任务。周三做什么？一句话，勿补充新安排。'],
        '周三评估。', ['取消仅作用周二', '不编造顺延规则'])

    add('信息不足与不编造', '仅有记录“服务器昨天启动成功”，没有今天的日志和探测。请问服务器此刻可用吗？两句话以内回答，并指出需要的证据。',
        '无法从昨天记录确定此刻可用，需要当前健康探测或日志。', ['不保证当前状态', '建议当前而非过去或未来的证据'])
    add('信息不足与不编造', '只知道训练集有240条样本，没有批次大小、累积次数或轮数。能确定总参数更新次数吗？60字内说明缺少哪些信息，不猜数字。',
        '不能。需要批次大小、梯度累积次数、轮数及末尾不足批次的处理等信息。', ['不从样本数直接断言更新数', '指出关键缺失配置'])
    add('信息不足与不编造', '材料只给出“文件已保存”，没有哈希、文件内容或加载结果。有人问Adapter是否完整且能加载。请在60字内回答，并给出下一步检查。',
        '不能确认；应校验文件/哈希及配置，并实际重新加载验证，存在文件不等于可用。', ['不从保存推断完整可用', '提出能直接支持结论的检查'])
    add('信息不足与不编造', '候选甲回答80字，候选乙回答25字。没有问题原文、答案内容或评分。能判断谁更正确吗？请在40字内回答，不替两者评分。',
        '不能，长度不足以判断正确性，需要题目、具体答案和评分依据。', ['不以长度代替正确性', '不编造质量分'])
    add('信息不足与不编造', '合同片段只写“可内部使用”，没有再分发、公开或商业授权条款。仅依据这段文字，能断言允许公开发布完整材料吗？两句话内回答，不提供法律结论。',
        '不能据此断言可以公开发布；需核对完整授权范围或向权利人确认。', ['结论局限于给定文本证据', '不把内部使用扩展为公开许可，也不断言必然违法'])
    return rows


def main():
    root = Path('data/quality-holdout-v1')
    rows = cases()
    assert len(rows) == 20 and len({r['id'] for r in rows}) == 20
    # Literal overlap check is deliberately weaker than semantic independence.
    known = []
    protected = {}
    for path in [Path('data/v2/train.jsonl'), Path('data/v2/validation.jsonl'), Path('data/v2/test.jsonl')]:
        protected[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            known.extend(m['content'] for m in item.get('messages', []) if m['role'] == 'user')
            known.extend(item.get('prompts', []))
    assert not any(p in known for row in rows for p in row['prompts'])
    root.mkdir(parents=True, exist_ok=False)
    raw = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode('utf-8')
    (root / 'test.jsonl').write_bytes(raw)
    manifest = dict(version='quality-holdout-v1', role='frozen-quality-holdout-v1',
                    frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                    test=dict(count=len(rows), sha256=hashlib.sha256(raw).hexdigest()),
                    protected_v2_files=protected,
                    notes='Frozen before these prompts are run. Assistant-authored, not independent human-authored. Same broad task families; no semantic independence guarantee. Never used for training or checkpoint selection.')
    (root / 'freeze-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
