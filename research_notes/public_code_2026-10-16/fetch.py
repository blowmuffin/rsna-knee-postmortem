"""Fetch every topic + all comments of the competition forum into forum/<id>.md (plain text) and forum/index.tsv."""
import os, re, html, json, time
from kagglesdk import KaggleClient
from kagglesdk.discussions.types.discussions_api_service import ApiGetTopicRequest, ApiListCommentsRequest, ApiListTopicsRequest
tok = open(os.path.expanduser('~/.kaggle/access_token')).read().strip()
d = KaggleClient(api_token=tok).discussions.discussion_api_client
OUT = 'forum'

def txt(h):
    h = re.sub(r'<a [^>]*href="([^"]*)"[^>]*>(.*?)</a>', lambda m: m.group(2) if m.group(1) in m.group(2) else f'{m.group(2)} <{m.group(1)}>', h or '', flags=re.S)
    h = re.sub(r'<img [^>]*>', '[img]', h)
    h = re.sub(r'<(br|/p|/li|/tr|/h\d)\s*/?>', '\n', h)
    h = re.sub(r'<li>', '- ', h)
    h = re.sub(r'</t[dh]>', ' | ', h)
    h = re.sub(r'<[^>]+>', '', h)
    return re.sub(r'\n{3,}', '\n\n', html.unescape(h)).strip()

def call(f, r):
    for i in range(5):
        try: return f(r)
        except Exception as e:
            print('retry', e); time.sleep(3 * (i + 1))
    raise RuntimeError('failed')

ids = [int(x) for x in open('ids.txt').read().split()]
print('topics', len(ids))

def dump(cs, depth, out):
    for c in cs or []:
        out.append(f"{'  ' * depth}- [{c.get('authorName')} | {c.get('postDate', '')[:16]} | v{c.get('votes', 0)}] " + txt(c.get('content')).replace('\n', '\n' + '  ' * (depth + 1)))
        dump(c.get('replies'), depth + 1, out)

rows = []
for tid in ids:
    t = call(d.get_topic, ApiGetTopicRequest(id=tid) if False else (lambda r: (setattr(r, 'id', tid), r)[1])(ApiGetTopicRequest())).to_dict()['topic']
    out = [f"# {t['title']}", f"{tid} | {t.get('authorName')} | {t.get('postDate', '')[:16]} | votes {t.get('votes', 0)}", '', txt(t.get('content')), '', '## Comments']
    pt, n = '', 0
    while True:
        r = ApiListCommentsRequest(); r.topic_id = tid; r.page_size = 100
        if pt: r.page_token = pt
        res = call(d.list_comments, r).to_dict()
        dump(res.get('comments'), 0, out); n += len(res.get('comments', []))
        pt = res.get('nextPageToken', '')
        if not pt: break
    open(f'{OUT}/{tid}.md', 'w', encoding='utf-8').write('\n'.join(out))
    rows.append(f"{tid}\t{t.get('postDate', '')[:10]}\t{t.get('votes', 0)}\t{n}\t{len(chr(10).join(out))}\t{t['title']}")
open(f'{OUT}/index.tsv', 'w', encoding='utf-8').write('\n'.join(rows))
print('done', len(rows))
