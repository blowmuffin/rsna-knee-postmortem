"""Save competition solution write-ups (topic ids in argv) as markdown, with their comments."""
import os, sys, json, time
from markdownify import markdownify
from kagglesdk import KaggleClient
from kagglesdk.discussions.types.discussions_api_service import ApiGetTopicRequest, ApiListCommentsRequest
c = KaggleClient(api_token=open(os.path.expanduser('~/.kaggle/access_token')).read().strip())
d = c.discussions.discussion_api_client
out = os.path.dirname(os.path.abspath(__file__))
for tid in map(int, sys.argv[1:]):
    r = ApiGetTopicRequest(); r.id = tid
    t = d.get_topic(r).to_dict()['topic']
    rc = ApiListCommentsRequest(); rc.topic_id = tid
    try: cm = d.list_comments(rc).to_dict().get('comments', [])
    except Exception as e: cm = []; print('comments failed', tid, e)
    def flat(cs, depth=0):
        for x in cs:
            yield depth, x
            yield from flat(x.get('replies', []), depth + 1)
    md = f"# {t.get('title')}\n\nTopic {tid}, author {t.get('authorName')} ({t.get('authorUrl')}), posted {t.get('postDate')}, votes {t.get('votes')}\n\n"
    md += markdownify(t.get('content') or '', heading_style='ATX')
    md += "\n\n## Comments\n\n" + "\n\n".join(f"{'  ' * dp}- **{x.get('authorName')}**: {markdownify(x.get('content') or '').strip()}" for dp, x in flat(cm))
    open(os.path.join(out, f"{tid}.md"), 'w', encoding='utf-8').write(md)
    print(tid, t.get('title'), len(md), 'chars,', len(cm), 'comments')
    time.sleep(2)
