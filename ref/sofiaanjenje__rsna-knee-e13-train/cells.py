
# ======== CELL 0 [markdown] ========
# V41: DINOv3 + E10 alpha 0.60

This version changes only the precommitted E10 configuration from uniform 0.35 to uniform 0.60. Grouped nested selection chose 0.60 in all five public outer folds; an independent OOF source chose 0.70 in all five. No visible-test tuning or identifier is used.


# ======== CELL 1 [markdown] ========
# V40: DINOv3 cross-series + E10 RadImageNet hybrid

This hidden-test-safe candidate retains Mattia Angeli's cross-series DINOv3 member, removes its older Rad15 stage, and applies fishface's hash-pinned E10 uniform-0.35 RadImageNet correction to the resulting dynamic DINOv2+DINOv3 parent. No visible test identifier or prediction is embedded, and no test-time weight selection is performed.


# ======== CELL 2 [markdown] ========
# Bend the Knee Ensemble

A community pipeline, with one more model added.

## Thank you

This is built on work other people did first and shared:

- **pilkwang** — twenty trained models that are part of this ensemble, and a
  set of labels read from the reports.
- **stevenleehans** and **lixin73** — two more sets of labels read from the
  reports, so one reading could be checked against another.
- **tonylica** — four more trained models that join the ensemble.
- **marwanmath** — the official RadImageNet ResNet-50 weights.
- **prvsiyan** — the notebook this was forked from (Apache 2.0). Most of its
  later stages have been removed here.
- **cf696666** — for leaving two findings out of the RadImageNet blend, which
  is done here too.

# ======== CELL 3 [markdown] ========
### The added member

The submission this forks from is a rank mean of twenty **DINOv2** models. Added
here is one **DINOv3 ViT-S/16** — self-supervised and pretrained without labels,
then fine-tuned on this competition's knee MRI — rank-blended into that
family.

Above the encoder, each series carries a learned type embedding — plane crossed
with fat suppression — added to all of its tokens. The tokens of every series in
a study are then concatenated into one key/value sequence, and twelve learned
queries, one per finding, cross-attend over it with multi-head attention. A
finding therefore draws evidence from any series in the study at once, rather
than from per-series summaries combined afterwards. Each query's output is
concatenated with the mean and the max of the per-series CLS embeddings before
the classifier. Series a study does not contain are removed by the key-padding
mask, so nothing is imputed for them.

**RadImageNet** joins the vote at the end — a ResNet-50 pretrained on
radiology images rather than natural ones. Its heads were retrained here
rather than used as published.

Its five folds are combined on ranks rather than by averaging probabilities, matching
how the other two families are already combined — the metric is macro ROC-AUC, so only
ordering matters. That correction is due to **romantamrazov** (*RSNA Knee | DINOsaur V2*).

# ======== CELL 4 [code] ========
from __future__ import annotations
import re
import unicodedata
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
_PRE = str.maketrans({'ı': 'i', 'İ': 'i', 'I': 'i', 'ß': 'ss', 'đ': 'd', 'Đ': 'd', 'ø': 'o', 'Ø': 'o', 'æ': 'ae', 'Æ': 'ae'})

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.translate(_PRE).lower()
    text = unicodedata.normalize('NFKD', text)
    text = ''.join((ch for ch in text if not unicodedata.combining(ch)))
    text = text.replace('\xad', '')
    text = re.sub('[_\\-/\\\\]+', ' ', text)
    text = re.sub('[ \\t]+', ' ', text)
    return text
_SENT_SPLIT = re.compile('(?<=[.;!?])\\s+|\\n+')

def unwrap(text: str) -> str:
    if not isinstance(text, str):
        return ''
    out = []
    for line in text.split('\n'):
        s = line.strip()
        if out and out[-1] and (not re.search('[.;:!?>*•]$', out[-1])) and (len(out[-1].split()) >= 4) and s and (not s[:1].isupper()):
            out[-1] = out[-1] + ' ' + s
        else:
            out.append(s)
    return '\n'.join(out)

def clauses(text: str):
    norm = normalize(unwrap(text) if FEATURES['unwrap'] else text)
    raw = [c.strip() for c in _SENT_SPLIT.split(norm) if c and c.strip()]
    merged = []
    for i, c in enumerate(raw):
        if c.endswith(':') and len(c.split()) <= 14 and (i + 1 < len(raw)):
            merged.append(c + ' ' + raw[i + 1])
        merged.append(c)
    out = []
    for c in merged:
        out.append(c)
        if len(c.split()) > 25:
            out.extend((p.strip() for p in c.split(',') if len(p.split()) > 2))
    return out
FEATURES = {'unwrap': True, 'directional_negation': True, 'oa_inherit': True, 'graded_pathology': True, 'synovitis_backoff': True}

def _rx(*alts: str) -> re.Pattern:
    return re.compile('|'.join(alts))
PRE_NEG = _rx('\\bno\\b', '\\bnot\\b', '\\bwithout\\b', '\\bnegative for\\b', '\\babsence\\b', '\\bno evidence\\b', '\\bfree of\\b', '\\bnone\\b', '\\bneither\\b', '\\bnor\\b', '\\bsin\\b', '\\bno hay\\b', '\\bausencia\\b', '\\bausentes?\\b', '\\bno se\\b', '\\bpas de\\b', '\\bsans\\b', '\\baucune?\\b', '\\bgeen\\b', '\\bzonder\\b', '\\bniet\\b', '\\bkeine?[nmrs]?\\b', '\\bohne\\b', '\\bnicht\\b', '\\bkein\\b', '\\bnema\\b', '\\bbez\\b', '\\bnisu\\b', '\\bnije\\b', '\\bδεν\\b', '\\bχωρις\\b', 'ουδεν', '\\bουτε\\b', '\\bбез\\b', '\\bне\\b', 'липсва', '\\bняма\\b')
POST_NEG = _rx('\\byok\\b', '\\byoktur\\b', 'izlenmemekte', 'saptanmadi', '\\bdegil\\b', 'gozlenmemekte', 'mevcut degil', 'eslik etmiyor', '\\bizlenmedi\\b', 'izlenmemistir', 'saptanmamistir', 'gorulmemistir', '\\bnema znakova\\b', 'bez znakova')
NEGATION = _rx(PRE_NEG.pattern, POST_NEG.pattern, '\\bunremarkable\\b')
NEG_WINDOW = 90

def _negated(clause: str, start: int, end: int) -> bool:
    for m in PRE_NEG.finditer(clause):
        if m.end() <= start and start - m.end() <= NEG_WINDOW:
            if not re.search('\\b(but|however|ancak|fakat|pero|maar|aber|no i|ali|ωστοσο|αλλα|но)\\b', clause[m.end():start]):
                return True
    for m in POST_NEG.finditer(clause):
        if m.start() >= end and m.start() - end <= NEG_WINDOW:
            return True
    return False
NORMALITY = _rx('\\bnormal', '\\bintact\\b', '\\bpreserved\\b', '\\bwithin normal limits\\b', 'limites normales', '\\bconservad', '\\bintegr', '\\bnormales\\b', '\\bdoga(l|ll)\\b', 'korunmus', '\\bnormaldir\\b', 'olagan', '\\buredn', '\\bocuvan', '\\bodrzan', '\\bintakt', '\\bprimjeren', '\\bodrzanog kontinuiteta', '\\bodržan', 'φυσιολογικ', 'ακεραι', 'δεν παρατηρουνται', 'δεν σημειωνονται', 'unauffallig', 'regelrecht', '\\bo\\.?b\\.?\\b', 'нормал', 'запазен', 'съхранен', '\\bбез особености\\b', 'интактн', '\\bgaaf\\b', '\\bnormaal\\b')
NORMAL_PHRASE = _rx('\\bsin alteracion', '\\bsin cambios\\b', '\\bsin particularidad', '\\bsin hallazgos\\b', '\\bsin lesion', '\\bsin signos de (rotura|lesion)', '\\bcontinu[oa]s?\\b', '\\bcontinuidad conservada\\b', '\\bno abnormalit', '\\bno significant abnormalit', '\\bunremarkable\\b', '\\bno evidence of (tear|injury|abnormalit)', '\\bohne auffalligkeit', '\\bkein nachweis\\b', '\\bohne befund\\b', '\\bgeen afwijking', '\\bzonder afwijking', '\\bsans anomalie', "\\bpas d[e']anomalie", '\\bbez osobitosti\\b', '\\bbez znakova (rupture|lezije)\\b', '\\bbez patoloskih\\b', 'χωρις αλλοιωσ', 'χωρις παθολογ', 'δεν παρατηρουνται (αξιολογα|παθολογ)', '\\bбез особености\\b', '\\bбез патологич', '\\bбез данни за\\b', '\\bozel bir ozellik yok', '\\bpatolojik bulgu (yok|izlenmemis)')
UNCERTAIN = _rx('\\bpossible\\b', '\\bprobable\\b', '\\bsuspicious\\b', '\\bsuspected?\\b', 'cannot (be )?exclude', '\\bmay\\b', '\\bquestionable\\b', '\\bequivocal\\b', '\\br/o\\b', '\\bdd\\b', '\\blikely\\b', '\\bsuggest', '\\bcompatible with\\b', '\\bposible\\b', 'sin criterios categoricos', '\\bdudos', '\\bsugier', '\\bmuhtemel\\b', '\\bolasi\\b', '\\bsupheli\\b', '\\bizlenim', '\\bdusundur', '\\bmoguce\\b', '\\bvjerojatno\\b', '\\bsumnja\\b', '\\bmoze odgovarati\\b', 'πιθαν', 'υποπτ', '\\bmoglich', '\\bverdachtig', '\\bfraglich', '\\bv\\.?a\\.?\\b', '\\bwohl\\b', '\\bвъзможно\\b', '\\bвероятно\\b', 'суспект', '\\bmogelijk\\b', '\\bverdacht\\b')

# ======== CELL 5 [code] ========
TEAR = _rx('\\btear', '\\btorn\\b', '\\brupture', '\\bdisruption\\b', 'discontinuit', '\\bavuls', '\\bmacerat', '\\bbuckethandle\\b', 'bucket handle', '\\brotura\\b', '\\broturas\\b', '\\bruptura', '\\bdesgarro', '\\broto\\b', '\\bdechirure', '\\bdechire', '\\bscheur', '\\bruptuur', 'gescheurd', '\\briss\\b', 'einriss', '\\bruptur', 'zerreiss', '\\blasion', '\\bausriss', '\\byirtik', '\\byirtig', '\\bkopma\\b', 'butunluk kaybi', '\\brupturu\\b', 'devamsizlik', '\\brupture\\b', '\\bdevamliligi secilememis', '\\bpuknuce', '\\bprekid\\b', '\\bpukotin', '\\bruptur', 'ρηξη', 'ρηξις', 'ρηγμα', 'ασυνεχεια', 'руптура', 'разкъсв', 'разрив', 'скъсв', '\\bлезия\\b')
DEGEN = _rx('degenerat', '\\bmucoid\\b', '\\bmyxoid\\b', '\\bfray', '\\bfissur', 'dejeneratif', '\\bmukoid\\b', 'degenerativn', 'εκφυλ', 'дегенерат', '\\bμυξοειδ', '\\bμυξωδ', '\\bmeniskopat', '\\bmeniscopath', '\\bmuco ?ide\\b', 'aufgefasert', '\\bdejenerasyon\\b')
INJURY = _rx('\\binjur', '\\bsprain', '\\blesion', '\\blasion', '\\bedema\\b', '\\boedema\\b', '\\bodem\\b', '\\bedem\\b', '\\bοιδημα', '\\bодем', '\\bедем', '\\bstrain\\b', '\\bhigh signal\\b', '\\bsignal alteration\\b', '\\bhiperintens', '\\bhyperintens', 'aumento de senal', 'alteracion de senal', 'cambio de senal', '\\bsignalanhebung', '\\bsignalalteration', 'verhoogd signaal', 'sinyal artis', 'αυξημενο σημα', 'повишен сигнал', '\\besguince\\b', '\\bthicken', '\\bzadebljanje\\b', '\\bverdikking\\b', '\\bdistenzij', '\\blaksite\\b', '\\blaxity\\b', '\\bpartial\\b', '\\bparcijaln', '\\bparcial', '\\bpartiel', '\\bpartiell')
_GRADE_RX = re.compile('(?:grade|grad|grado|grau|derece|stupnja|stupanj|βαθμ|степен|icrs|outerbridge)[\\s:]*(?:grade\\s*)?([1-4]|iv|iii|ii|i)\\b')
_ROMAN = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4}

def _grade_of(clause: str):
    best = None
    for m in _GRADE_RX.finditer(clause):
        v = m.group(1)
        n = _ROMAN.get(v, None) if not v.isdigit() else int(v)
        if n is not None and (best is None or n > best):
            best = n
    return best
ANAT = {'ACL': _rx('anterior cruciate', '\\bacl\\b', 'cruzado anterior', '\\blca\\b', 'croise anterieur', 'voorste kruisband', '\\bvkb\\b', 'vorderes kreuzband', 'vorderen kreuzband', 'vordere kreuzband', 'on capraz', '\\bocb\\b', 'anterior capraz', 'prednji krizni', 'prednjeg krizn', 'προσθι[οα][^ ]* χιαστ', 'προσθιου χιαστου', 'χιαστο[^ ]* συνδεσμ', '\\bχιαστ\\w*', 'предна кръстна', 'предната кръстна', 'предна кръста', 'cruciate ligaments', 'ligamentos cruzados', 'ligaments croises', 'kruisbanden', 'kreuzbander', 'capraz baglar', 'krizn[a-z]* ligament[a-z]*', 'χιαστοι συνδεσμ', 'χιαστων συνδεσμ', 'кръстните връзки', 'кръстни връзки'), 'MCL': _rx('medial collateral', '\\bmcl\\b', 'tibial collateral', 'colateral medial', 'colateral interno', '\\blcm\\b', 'collateral medial', 'collateral interne', 'mediale collaterale', 'binnenband', '\\b(mediale|laterale) banden\\b', '\\bcollaterale banden\\b', 'innenband', 'mediales? kollateral', '\\bic yan bag', 'medial kollateral', '\\biyb\\b', 'medyal kollateral', 'medijalni kolateraln', 'medijalnog kolateraln', 'εσω πλαγι', 'εσωτερικο πλαγι', '\\bπλαγι\\w* συνδεσμ', '\\bπλαγιοι\\b', 'медиален колатерал', 'вътрешна странична', '\\bколатерал\\w*', '\\bcolaterales\\b', '\\bcollateraux\\b', '\\bcollateralen\\b', '\\bkolateralni\\b', 'collateral ligaments', 'ligamentos colaterales', 'ligaments collateraux', 'collaterale banden', 'kollateralbander', 'seitenbander', 'yan baglar', 'kolateraln[a-z]* ligament[a-z]*', 'πλαγιοι συνδεσμ', 'πλαγιων συνδεσμ', 'колатерални връзки', 'страничните връзки'), 'Medial Meniscus': _rx('medial meniscus', '\\bmm\\b(?= tear)', 'medial menisc', 'menisco medial', 'menisco interno', 'menisque medial', 'menisque interne', 'mediale meniscus', 'binnenmeniscus', 'innenmeniskus', 'medialen? meniskus', 'innenmeniskushinterhorn', 'medyal menisk', '\\bic menisk', 'medijalni meniskus', 'medijalnog meniskusa', 'medijalnom meniskusu', 'medijaln\\w* menisk\\w*', '\\bmedijalnog meniska\\b', 'medijalni menisk', 'εσω μηνισκ', 'μηνισκ[^ ]* του εσω', 'εσω διαμερισμα[^.]{0,40}μηνισκ', 'медиалния менискус', 'медиален менискус', 'вътрешния менискус', 'oba meniska', 'both menisci', 'ambos meniscos', 'beide menisci', 'her iki menisku', 'amfoteroi\\w* mhnisk', 'αμφοτερ\\w* μηνισκ', 'двата менискуса', 'medial (and|&) lateral menisc'), 'Lateral Meniscus': _rx('lateral meniscus', 'lateral menisc', 'menisco lateral', 'menisco externo', 'menisque lateral', 'menisque externe', 'laterale meniscus', 'buitenmeniscus', 'aussenmeniskus', 'lateralen? meniskus', 'aussenmeniskushinterhorn', 'lateral menisk', '\\bdis menisk', 'lateralni meniskus', 'lateralnog meniskusa', 'lateralnom meniskusu', 'lateraln\\w* menisk\\w*', '\\blateralnog meniska\\b', 'εξω μηνισκ', 'μηνισκ[^ ]* του εξω', 'εξω διαμερισμα[^.]{0,40}μηνισκ', 'латералния менискус', 'латерален менискус', 'външния менискус', 'oba meniska', 'both menisci', 'ambos meniscos', 'beide menisci', 'her iki menisku', 'αμφοτερ\\w* μηνισκ', 'двата менискуса', 'medial (and|&) lateral menisc')}
OA_EVIDENCE = _rx('osteoarthrit', '\\barthros', '\\bgonarthros', '\\bosteoarthros', 'chondropath', 'chondromalac', 'condropat', 'condromalac', '\\bchondros', '\\bchondrosis\\b', 'chondral (loss|defect|ulcer|thinning|injury|fissur|wear)', 'cartilage (loss|thinning|defect|fissur|wear|damage|heterogeneity|irregularit)', '(loss|thinning|fissur|defect|ulcer|erosion|denudation) of[^.]{0,20}cartilage', 'articular cartilage[^.]{0,30}(loss|thin|fissur|defect|erosion|wear|irregular)', 'osteophyt', 'osteofit', 'osteofyt', 'osteofito', 'osteophyten', 'spurring', 'joint space narrowing', 'pinzamiento articular', 'reduced joint space', 'kikirdak kayb', 'kikirdak incelme', 'kondropati', 'kondral', 'kikirdak dejener', 'eklem aralig\\w* daral', 'eklem mesafesi daral', 'kikirdak kalinlig\\w* azal', 'kraakbeen', 'gonartrose', 'artrose', '\\bknorpel', 'arthrose', 'gonarthrose', 'hrskavic', 'hondromalac', 'artroz', 'osteoartrit', 'artrotsk', 'artrotick', '\\boa promjen', '\\boa\\b', 'degenerativne promjene hrskav', 'χονδρ[^ ]*παθ', 'αρθριτ', 'αρθρωσ', 'οστεοφυτ', 'χονδρομαλακ', 'αρθρικου χονδρου', 'εξαλειψη του αρθρικου χονδρου', 'διαβρωση του αρθρικου χονδρ', 'λεπτυνση[^.]{0,30}χονδρ', 'φθορα[^.]{0,20}χονδρ', 'артроз', 'хондропат', 'остеофит', 'хрущял[^.]{0,40}(изтън|увред|дефект|липс)', 'изтъняване[^.]{0,30}хрущял', 'хондромалац', 'ulcera[s]? condral', 'cartilago[^.]{0,25}(perdida|adelgaz)', 'icrs grade', 'icrs\\b', 'outerbridge', '\\bdenudation\\b', 'denudacij', 'erozivne promjene', '\\berosion of[^.]{0,20}cartilage', 'kraakbeenlijden', 'kraakbeenverlies')
TF_SITE = _rx('compartment', 'compartimento', 'compartiment', 'kompartman', 'kompartiment', 'kompartment', 'odjelj', 'διαμερισμα', 'компартм', '\\bотдел', 'femorotibial', 'tibiofemoral', 'femoro tibial', 'femorotibiaal', 'femorotibijaln', 'феморотибиал', '\\bft zglob', 'tibiofemoraln', 'condyle', 'condilo', 'kondyl', 'kondil', 'condyl', 'κονδυλ', 'кондил', '\\bplateau', '\\bplato\\b', 'platillo', 'meseta', 'плато', 'tibiaplateau', 'tibijaln\\w* plato', 'tibyal plato', 'tibia plato', 'κνημιαι', 'μηριαι', 'weightbearing', 'weightbaring', 'zona de carga', 'dragende deel', 'agirlik tasiyan', '\\bfemur\\b', '\\btibia\\b', '\\bfemoral\\b', '\\btibial\\b', '\\bfemura\\b', '\\btibije\\b', '\\bmesarthrio\\b', 'μεσαρθριο')
PF_SITE = _rx('patellofemoral', 'femoropatellar', 'femoropatelar', 'patelofemoral', 'retropatellar', 'retrorotulian', 'trochlea', 'troclea', 'troklea', 'trochlear', 'trohlej', 'τροχιλ', '\\bpatella', '\\bpatellar', 'rotulian', '\\brotula\\b', '\\bpatele\\b', 'patellofemoraal', 'femoropatellair', 'επιγονατιδ', 'μηροεπιγονατιδ', 'пател', 'феморопател', 'anterior compartment', 'compartimento anterior', 'prednj\\w* odjeljk', '\\bfp zglob', '\\bpf zglob', '\\bfaset', '\\bfacet', 'patellofemoraln')
SIDE_MEDIAL = _rx('\\bmedial\\w*', '\\bmedyal\\w*', '\\bmedijaln\\w*', '\\bmediaal\\w*', '\\bmediale\\w*', '\\binterno\\b', '\\binterna\\b', '\\binternos\\b', '\\binterne\\b', '\\binnen\\w*', '\\bic\\b', '\\bunutarnj\\w*', '\\bεσω\\w*', '\\bεσωτερικ\\w*', '\\bмедиал\\w*', '\\bвътреш\\w*', '\\bbinnen\\w*', '\\bmediaal\\b', '\\bmediales?\\b')
SIDE_LATERAL = _rx('\\blateral\\w*', '\\bexterno\\b', '\\bexterna\\b', '\\bexternos\\b', '\\bexterne\\b', '\\bdis\\b', '\\blateraln\\w*', '\\baussen\\w*', '\\bbuiten\\w*', '\\bεξω\\w*', '\\bεξωτερικ\\w*', '\\bлатерал\\w*', '\\bвъншн\\w*', '\\bvanjsk\\w*')
SIDE_ANTERIOR = _rx('\\banterior\\w*', '\\bant\\b', '\\bon\\b', '\\bprednj\\w*', '\\bvorder\\w*', '\\bvoorste\\b', '\\bπροσθι\\w*', '\\bпредн\\w*', '\\banteriyor\\w*', '\\bavant\\b', '\\banterieur\\w*')
GLOBAL_OA = _rx('tri ?compartment', 'all three compartment', 'global(ised)? (oa|osteoarthrit)', '\\bgonarthros', '\\bgonartros', '\\bgonarthrose', '\\bgonartrose', 'gonartro', 'goanrtrot', 'gonartrot', 'osteoarthritis of the knee', 'artrosis (de |)(la )?rodilla', 'knee osteoarthrit', '\\bdiz osteoartrit', '\\bgonartroz', 'artroza koljena', 'οστεοαρθριτιδα', 'αρθριτιδα του γονατος', 'εκφυλιστικη οστεοαρθριτ', 'артроза на колянната', 'гонартроз', 'degenerative joint disease', '\\bdjd\\b', 'three compartments', 'compartmens', 'compartments')
DIRECT = {'Effusion': _rx('\\beffusion', 'joint fluid', 'intra ?articular fluid', '\\bhydrops\\b', '\\bhemarthros', '\\bhaemarthros', 'derrame articular', '\\bderrame\\b', 'liquido articular', 'hemartrosis', 'epanchement', 'gewrichtsvocht', '\\bvocht\\b', 'gewrichtseffusie', 'opzetting van suprapatell', 'gelenkerguss', '\\berguss\\b', 'gelenksergu', 'gelenksflussigkeit', 'eklem\\w* ic\\w* sivi', 'efuzyon', 'eklem sivisi', 'eklem mesafesinde sivi', 'sivi (miktari|artisi|birikimi)', 'sivi artis', '\\bsivi\\b[^.]{0,25}artmis', '\\bizljev', '\\bizliv', 'zglobn[^ ]* tekucin', '\\bhidrops\\b', 'αρθρικ[^ ]* υγρ', 'υγρου ενδαρθρικα', 'ενδαρθρικ[^ ]* υγρ', 'ποσοτητα υγρου', 'ενδαρθρικ', 'αρθρικη συλλογη', 'υγρο στην αρθρωση', 'υγρου στην αρθρωση', 'συλλογη υγρου', 'ενθαρθρικ', 'ставен излив', 'излив', 'ставна течност', 'синовиална течност'), 'Synovitis': _rx('synovit', 'sinovit', 'synovial (thickening|proliferation|hypertroph)', 'thicken\\w* synovial', 'hypertroph\\w* of the synovium', 'synoviale? (verdikking|proliferatie)', 'verdikkingen van (het )?synovium', 'synovialitis', 'synovialis(verdickung|proliferation)', 'reizsynovial', 'sinovijalitis', 'sinovitis', 'zadebljanje sinovij', 'proliferacij\\w* sinovij', 'sinovijaln\\w* proliferacij', 'υμενιτιδα', 'συνοβιτιδα', 'υμενικ[^ ]* υπερτροφ', 'αρθρικου υμεν', 'παχυνση[^.]{0,20}υμεν', 'υμενα', 'синовит', 'синовиал[^ ]* (задебел|пролифер)', '\\bpannus\\b', '\\bhoffit', 'sinovyal\\w* (kalinlas|proliferas)', 'sinovyal hipertrof', '\\bartrit\\b', '\\barthritis\\b'), "Baker's": _rx('baker', 'popliteal cyst', 'quiste popliteo', 'quistes popliteos', 'kyste poplite', 'popliteale? cyst', 'poplitealzyste', 'bakerzyste', 'popliteal kist', '\\bbakerova\\b', 'poplitealn[^ ]* cist', 'popliteal\\w* cist', 'κυστη baker', 'πολυχωρη συνοβιακη κυστη', 'κυστη του baker', 'συνοβιακη κυστη', 'κυστη τυπου baker', 'киста на бейкър', 'бейкърова киста', 'поплитеална киста', 'бекеров', 'gastrocnemio ?semimembranos', 'gastrocnemius semimembranosus burs'), 'Contusion': _rx('\\bcontusion', 'bone bruise', 'bone marrow (o?edema|contusion)', 'marrow o?edema', '\\bkontuz', 'medular bone o?edema', 'osseous contusion', 'contusion osea', 'edema oseo', 'edema de medula osea', 'contusiones oseas', 'oedeme osseux', 'contusion osseuse', 'botcontusie', 'botoedeem', 'beenmergoedeem', 'botmergoedeem', 'knochenmarkodem', 'knochenodem', 'knochenmarksodem', 'kontusion', 'kemik kontuzyonu', 'kemik iligi odemi', 'kemik odemi', 'kemik iliginde odem', 'kontuzyonel kemik', 'kemik iligi odemleri', 'kostani edem', 'edem kosti', 'kontuzij', 'kostane srzi[^.]{0,20}edem', 'οστεομυελικ[^ ]* οιδημα', 'οστικο οιδημα', 'μυελικο οιδημα', 'οστικο μωλωπ', 'костномозъчен едем', 'костен едем', 'контузионен', 'костно мозъчен едем'), 'Fracture': _rx('\\bfractur', '\\bfract\\b', '\\bfractura', '\\bfracturas\\b', '\\bfractuur', '\\bbreuk\\b', '\\bfraktur', '\\bbruch\\b', '\\bkirik\\b', '\\bkirigi\\b', '\\bkiri[kg]\\w*', '\\bprijelom', 'impresijsk[^ ]* fraktur', 'impaktcij', 'καταγμα', 'καταγματ', 'фрактур', 'счупван', 'фисур', 'insufficiency fracture', 'stress fracture', 'avulsion fracture', 'subchondral fracture', 'subkondral kiri', 'impaction (fracture|injury)', 'osteochondral (fracture|impaction)', '\\bsegond\\b', 'impactiefractuur', 'subchondrale impression', 'subchondraler? impress')}
DECOY = {'Fracture': _rx('microfractur', '\\bfracture (risk|prophyla)'), "Baker's": _rx('meniscal cyst', 'quiste meniscal', 'parameniscal')}
PAIRED = {'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus'}
OA_TARGETS = ['Medial OA', 'Lateral OA', 'PF OA']
PLURAL_MENISCI = _rx('\\bmenisci\\b', '\\bmeniscos\\b', '\\bmenisques\\b', '\\bmenisken\\b', '\\bmeniskusi\\b', '\\bmenisk\\w*ler\\b', '\\bμηνισκοι\\b', '\\bμηνισκων\\b', '\\bменискуси\\b', '\\bменискусите\\b', '\\bmenisci\\w*\\b')
ANY_SIDE = _rx(SIDE_MEDIAL.pattern, SIDE_LATERAL.pattern)
STEM_MENISCUS = _rx('menisc\\w*', 'menisk\\w*', 'μηνισκ\\w*', 'мениск\\w*')
STEM_CRUCIATE = _rx('cruciate', 'cruzado', 'croise', 'kruisband', 'kreuzband', 'capraz bag\\w*', 'krizn\\w*', 'χιαστ\\w*', 'кръстн\\w*', '\\bacl\\b', '\\blca\\b', '\\bvkb\\b', '\\bocb\\b', '\\bacb\\b')
STEM_COLLATERAL = _rx('collateral\\w*', 'colateral\\w*', 'kollateral\\w*', 'collaterale\\w*', 'kolateraln\\w*', 'yan bag\\w*', 'πλαγι\\w*', 'колатерал\\w*', 'странич\\w*', 'innenband\\w*', 'binnenband\\w*', '\\bmcl\\b', '\\blcm\\b', '\\biyb\\b')
STEM_FRACTURE = _rx('fractur\\w*', 'fraktur\\w*', 'fractuur\\w*', '\\bfract\\b', 'kiri[kgğ]\\w*', 'prijelom\\w*', 'lom kosti', '\\bbreuk\\w*', '\\bbruch\\w*', 'καταγμα\\w*', 'καταγματ\\w*', 'фрактур\\w*', 'счупван\\w*', 'fisur\\w* (osea|oseas|kost)', 'fissur\\w* kost')
POSTERIOR_ONLY = _rx('\\bpcl\\b', '\\blcp\\b', '\\bhkb\\b', '\\bacb\\b', 'posterior cruciate', 'cruzado posterior', 'croise posterieur', 'achterste kruisband', 'hinteres kreuzband', 'arka capraz', 'straznji krizn', 'οπισθι[οα]\\w* χιαστ', 'задна кръстн', 'задната кръстн')
LATERAL_COLL_ONLY = _rx('\\blcl\\b', '\\bfcl\\b', 'lateral collateral', 'fibular collateral', 'colateral lateral', 'colateral externo', 'buitenband', 'aussenband', 'dis yan bag', 'lateralni kolateraln', 'εξω πλαγι', 'латерален колатерал')

# ======== CELL 6 [code] ========
def _near(clause: str, stem_rx: re.Pattern, qual_rx: re.Pattern, window: int=55):
    for m in stem_rx.finditer(clause):
        lo = max(0, m.start() - window)
        hi = min(len(clause), m.end() + window)
        if qual_rx.search(clause[lo:hi]):
            return True
    return False
STEM_RULES = {'ACL': (STEM_CRUCIATE, SIDE_ANTERIOR), 'MCL': (STEM_COLLATERAL, SIDE_MEDIAL), 'Medial Meniscus': (STEM_MENISCUS, SIDE_MEDIAL), 'Lateral Meniscus': (STEM_MENISCUS, SIDE_LATERAL)}

class _Matcher:

    def __init__(self, phrase_rx, stem=None, side=None, window=55):
        self.phrase_rx = phrase_rx
        self.stem = stem
        self.side = side
        self.window = window

    def search(self, clause):
        m = self.phrase_rx.search(clause)
        if m is not None:
            return m
        if self.stem is not None and _near(clause, self.stem, self.side, self.window):
            return self.stem.search(clause)
        return None
ANAT_MATCH = {t: _Matcher(ANAT[t], *STEM_RULES[t]) for t in PAIRED}
DIRECT_MATCH = {t: _Matcher(_rx(rx.pattern, STEM_FRACTURE.pattern) if t == 'Fracture' else rx) for t, rx in DIRECT.items()}
SEV_LOW = _rx('\\bsmall\\b', '\\bminimal\\b', '\\btrace\\b', '\\bmild\\b', '\\bslight\\b', '\\btiny\\b', '\\bscant\\b', '\\bdiscrete\\b', '\\blow ?grade\\b', '\\bincipient\\b', '\\bleve\\b', '\\bminim', '\\bpeque', '\\bfina\\b', '\\bfino\\b', '\\bligero\\b', '\\bescaso\\b', '\\bdiscreto\\b', '\\bhafif\\b', '\\baz miktarda\\b', '\\bsilik\\b', '\\bmanj\\w*', '\\bblago\\b', '\\bdiskretn', '\\bmalo\\b', '\\bpocetn', '\\bgering', '\\bdiskret', '\\bkleine?r?\\b', '\\bwenig\\b', '\\bzarte?\\b', '\\bbeperkte?\\b', '\\bgeringe\\b', '\\bweinig\\b', '\\blichte?\\b', '\\blicht\\b', '\\bηπι', '\\bμικρ', '\\bελαχιστ', '\\bαρχομεν', '\\bминимал', '\\bлек', '\\bмалк', '\\bнеголям')
SEV_HIGH = _rx('\\blarge\\b', '\\bmarked\\b', '\\bmassive\\b', '\\bsevere\\b', '\\bextensive\\b', '\\bmoderate\\b', '\\bgross\\b', '\\bsignificant\\b', '\\babundant\\b', '\\btense\\b', '\\bcomplete\\b', '\\bfull ?thickness\\b', '\\bhigh ?grade\\b', '\\badvanced\\b', '\\bmoderad', '\\bimportante\\b', '\\bsevera?\\b', '\\bmarcad', '\\bcuantios', '\\bespesor total\\b', '\\bcompleta?\\b', '\\bbelirgin\\b', '\\byaygin\\b', '\\bileri\\b', '\\bciddi\\b', '\\bbol\\b', '\\bkomplet', '\\bopsezan\\b', '\\bveliki\\b', '\\bizrazit', '\\bznacajn', '\\bumjeren', '\\buznapredoval', '\\bpotpun', '\\bkompleksn', '\\bausgepragt', '\\bdeutlich', '\\bmassiv', '\\bmassig', '\\bgross', '\\buitgebreid', '\\bgevorderd', '\\bveel\\b', '\\bmatige?\\b', '\\bvolledig', '\\bμετρι', '\\bμεγαλ', '\\bεκτεταμεν', '\\bευμεγεθ', '\\bσοβαρ', '\\bπληρη', '\\bголям', '\\bизразен', '\\bзначим', '\\bумерен', '\\bобилен', '\\bпълн')
GRADE_HIGH = re.compile('grade?[ao]?\\s*(3|4|iii|iv)\\b|icrs grade (iii|iv|3|4)|stupnja iv|stupnja iii|\\bgrado (3|4)\\b|\\bgrad (3|4)\\b|\\bgrade (3|4)\\b')
DEGENERATIVE_MARROW = _rx('subchondral', 'subcondral', 'subkondral', 'supkondraln', 'subchondraln', 'υποχονδρι', 'υπαρθρικ', 'субхондрал', 'subchondrale?', 'subartikuler', '\\bcyst', '\\bquist', '\\bzyste\\b', '\\bcistic', 'reactive', 'reactivo', 'degenerative', 'degenerativ', 'reaktiv', '\\bcisti\\b')
TRAUMA = _rx('\\bbruise\\b', '\\bcontusion', '\\bkontuz', '\\btrauma', '\\bimpaction\\b', '\\bpivot shift\\b', '\\bkissing\\b', '\\bacute\\b', '\\bagudo\\b', '\\bakut', '\\bpivot kaymasi\\b', '\\bcontusion osseuse\\b', '\\bbone bruise\\b', '\\bbotcontusie\\b', '\\bконтузион', '\\bμωλωπ', '\\bkontuzij', '\\bimpaktcij', '\\bimpakcij', '\\bfall\\b', '\\binjury\\b', '\\bimpression\\b')
SYNOVIAL_PROXY = _rx('bursit', 'burzit', '\\bbursa\\b[^.]{0,30}(fluid|distend|sivi|tekucin|opzetting)', 'suprapatellar (bursitis|effusion|recess)', 'suprapatellar bursa', 'suprapatellar bursada', 'suprapatelarno', 'suprapatellaire recessus', 'hoffa', 'hoffit', 'plica', 'plika', 'πλικα', 'fat pad[^.]{0,20}(edema|oedema)', 'kapsul', 'capsul', 'καψ', 'капсул', '\\bpannus\\b', '\\bsinov', '\\bsynov')

def _polarity(clause: str, span=None) -> str:
    if UNCERTAIN.search(clause):
        return 'uncertain'
    if span is None or not FEATURES['directional_negation']:
        if NEGATION.search(clause):
            return 'negative'
    elif _negated(clause, span[0], span[1]):
        return 'negative'
    if NORMALITY.search(clause):
        if TEAR.search(clause) or GRADE_HIGH.search(clause):
            return 'positive'
        return 'negative'
    return 'positive'

def _severity(clause: str) -> float:
    high = SEV_HIGH.search(clause) is not None
    low = SEV_LOW.search(clause) is not None
    if high and (not low):
        return 1.0
    if low and (not high):
        return 0.45
    if high and low:
        return 0.8
    return 0.75

def _grade(n_pos, n_neg, n_unc, best):
    if n_pos or n_unc:
        score = min(0.97, 0.5 + 0.45 * best + 0.015 * min(n_pos, 3))
        conf = min(1.0, 0.55 + 0.15 * n_pos)
    elif n_neg:
        score = max(0.04, 0.2 - 0.04 * n_neg)
        conf = min(0.9, 0.45 + 0.12 * n_neg)
    else:
        score, conf = (0.28, 0.05)
    return (score, conf)

def _paired_weight(clause: str, meniscus: bool) -> float:
    g = _grade_of(clause) if FEATURES['graded_pathology'] else None
    tear = TEAR.search(clause) is not None
    if meniscus:
        if tear:
            base = 1.0
        elif g is not None:
            base = 0.95 if g >= 3 else 0.3
        elif DEGEN.search(clause):
            base = 0.35
        else:
            base = 0.45
    elif tear:
        base = 1.0
    elif g is not None:
        base = 0.85 if g >= 2 else 0.3
    elif DEGEN.search(clause):
        base = 0.4
    else:
        base = 0.55
    if SEV_HIGH.search(clause) and (not SEV_LOW.search(clause)):
        base = min(1.0, base * 1.2)
    elif SEV_LOW.search(clause) and (not SEV_HIGH.search(clause)):
        base *= 0.7
    return base

def _score_paired(cls, tgt):
    anat_rx = ANAT_MATCH[tgt]
    path_rx = _rx(TEAR.pattern, DEGEN.pattern, INJURY.pattern)
    meniscus = 'Meniscus' in tgt
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        hit = anat_rx.search(c)
        if hit is None and meniscus and PLURAL_MENISCI.search(c) and (not ANY_SIDE.search(c)):
            hit = PLURAL_MENISCI.search(c)
        if hit is None:
            continue
        pm = path_rx.search(c)
        if pm is None and _grade_of(c) is None:
            if NORMAL_PHRASE.search(c) or (NORMALITY.search(c) and (not NEGATION.search(c))):
                n_neg += 1
            continue
        span = (pm.start(), pm.end()) if pm is not None else None
        pol = _polarity(c, span)
        if pol == 'positive':
            n_pos += 1
            best = max(best, _paired_weight(c, meniscus))
        elif pol == 'negative':
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.45 * _paired_weight(c, meniscus))
    s, cf = _grade(n_pos, n_neg, n_unc, best)
    return (s, cf, n_pos, n_neg)

def _score_clauses(cls, anat_rx, path_rx=None, decoy_rx=None, context_penalty=None, context_bonus=None):
    n_pos = n_neg = n_unc = 0
    best = 0.0
    for c in cls:
        m = anat_rx.search(c)
        if not m:
            continue
        if decoy_rx is not None and decoy_rx.search(c):
            continue
        if path_rx is not None and (not path_rx.search(c)):
            if NORMAL_PHRASE.search(c) or (NORMALITY.search(c) and (not NEGATION.search(c))):
                n_neg += 1
            continue
        pol = _polarity(c, (m.start(), m.end()))
        if pol == 'positive':
            n_pos += 1
            w = _severity(c)
            if context_penalty is not None and context_penalty.search(c):
                w *= 0.45
            if context_bonus is not None and context_bonus.search(c):
                w = min(1.0, w * 1.35)
            best = max(best, w)
        elif pol == 'negative':
            n_neg += 1
        else:
            n_unc += 1
            best = max(best, 0.3)
    s, c = _grade(n_pos, n_neg, n_unc, best)
    return (s, c, n_pos, n_neg)

def _score_oa(cls):
    acc = {t: {'pos': 0, 'neg': 0, 'unc': 0, 'best': 0.0} for t in OA_TARGETS}
    g_pos, g_neg, g_best = (0, 0, 0.0)
    for c in cls:
        m = OA_EVIDENCE.search(c)
        if not m:
            continue
        pol = _polarity(c, (m.start(), m.end()))
        sev = _severity(c)
        tf_med = _near(c, TF_SITE, SIDE_MEDIAL, 45)
        tf_lat = _near(c, TF_SITE, SIDE_LATERAL, 45)
        pf = PF_SITE.search(c) is not None
        hits = []
        if tf_med:
            hits.append('Medial OA')
        if tf_lat:
            hits.append('Lateral OA')
        if pf:
            hits.append('PF OA')
        if not hits:
            if pol == 'positive':
                g_pos += 1
                g_best = max(g_best, sev if GLOBAL_OA.search(c) else sev * 0.7)
            elif pol == 'negative':
                g_neg += 1
            continue
        for t in hits:
            if pol == 'positive':
                acc[t]['pos'] += 1
                acc[t]['best'] = max(acc[t]['best'], sev)
            elif pol == 'negative':
                acc[t]['neg'] += 1
            else:
                acc[t]['unc'] += 1
                acc[t]['best'] = max(acc[t]['best'], 0.3)
    out = {}
    for t in OA_TARGETS:
        a = acc[t]
        pos, neg, unc, best = (a['pos'], a['neg'], a['unc'], a['best'])
        if not (pos or unc) and g_pos and FEATURES['oa_inherit']:
            if neg:
                score, conf = _grade(0, neg, 0, 0.0)
                score = max(score, 0.35)
                conf *= 0.7
            else:
                score, conf = _grade(g_pos, 0, 0, g_best * 0.92)
                conf *= 0.75
        else:
            score, conf = _grade(pos, neg + g_neg, unc, best)
        out[t] = (score, conf, pos, neg)
    return out

def extract(report: str) -> dict:
    cls = clauses(report)
    out = {}
    for tgt in PAIRED:
        s, c, npos, nneg = _score_paired(cls, tgt)
        out[tgt] = s
        out[tgt + '__conf'] = c
        out[tgt + '__npos'] = npos
        out[tgt + '__nneg'] = nneg
    for tgt, (s, c, npos, nneg) in _score_oa(cls).items():
        out[tgt] = s
        out[tgt + '__conf'] = c
        out[tgt + '__npos'] = npos
        out[tgt + '__nneg'] = nneg
    for tgt in ('Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture'):
        if tgt == 'Contusion':
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt), context_penalty=DEGENERATIVE_MARROW, context_bonus=TRAUMA)
        else:
            s, c, npos, nneg = _score_clauses(cls, DIRECT_MATCH[tgt], None, DECOY.get(tgt))
        out[tgt] = s
        out[tgt + '__conf'] = c
        out[tgt + '__npos'] = npos
        out[tgt + '__nneg'] = nneg
    if FEATURES['synovitis_backoff'] and out['Synovitis__npos'] == 0 and (out['Synovitis__nneg'] == 0):
        proxy = sum((1 for c in cls if SYNOVIAL_PROXY.search(c) and _polarity(c) == 'positive'))
        eff = out['Effusion']
        prior = 0.3 + 0.3 * max(0.0, (eff - 0.5) / 0.45) + 0.06 * min(proxy, 3)
        out['Synovitis'] = min(0.72, prior)
        out['Synovitis__conf'] = 0.18
    return out

# ======== CELL 7 [code] ========
import os
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
T0 = time.time()

def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)

def find_root():
    for c in [Path('/kaggle/input/rsna-knee-abnormality-detection'), Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('data'), Path('.')]:
        if (c / 'test.csv').is_file() and (c / 'test_series').is_dir():
            return c
    base = Path('/kaggle/input')
    if base.is_dir():
        for d1 in sorted((p for p in base.iterdir() if p.is_dir())):
            for cand in [d1] + sorted((p for p in d1.iterdir() if p.is_dir())):
                if (cand / 'test.csv').is_file():
                    return cand
    raise FileNotFoundError('competition mount not found')
ROOT = find_root()
log(f'input root: {ROOT}')
_test_df = pd.read_csv(ROOT / 'test.csv')
_bench = _test_df[['StudyInstanceUID']].copy()
for _c in TARGETS:
    _bench[_c] = 0.5
_bench.to_csv('submission.csv', index=False)
log(f'benchmark submission.csv written ({len(_bench)} rows)')
STAGE_OK = {}

def stage(name):

    def deco(fn):

        def run(*a, **k):
            t = time.time()
            try:
                out = fn(*a, **k)
                STAGE_OK[name] = True
                log(f"stage '{name}' ok in {time.time() - t:.1f}s")
                return out
            except Exception:
                import traceback
                traceback.print_exc()
                STAGE_OK[name] = False
                log(f"stage '{name}' FAILED after {time.time() - t:.1f}s")
                return None
        return run
    return deco
train_df = pd.read_csv(ROOT / 'train.csv')
log(f'train {train_df.shape}  test {_test_df.shape}')
t = time.time()
LAB = pd.DataFrame([extract(r) for r in train_df['Report'].fillna('')])
LAB['StudyInstanceUID'] = train_df['StudyInstanceUID'].values
LAB = LAB.set_index('StudyInstanceUID')
log(f'read {len(LAB)} reports in {time.time() - t:.1f}s')
GOLD = train_df.dropna(subset=TARGETS).set_index('StudyInstanceUID')[TARGETS]
log(f'{len(GOLD)} studies carry the twelve annotations')
pos = (LAB[TARGETS] > 0.5).mean()
sil = pd.Series({t_: float(((LAB[t_ + '__npos'] == 0) & (LAB[t_ + '__nneg'] == 0)).mean()) for t_ in TARGETS})
print(pd.DataFrame({'derived positive rate': pos.round(3), 'silence rate': sil.round(3), 'annotated positive rate': GOLD.mean().round(3)}).to_string())

# ======== CELL 8 [code] ========
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
plt.rcParams.update({'figure.dpi': 120, 'font.size': 8, 'axes.grid': True, 'grid.alpha': 0.25, 'axes.spines.top': False, 'axes.spines.right': False})
INK, ACC, WARN = ('#22303f', '#2b7a9b', '#c25a3d')

def agreement(lab, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = lab.loc[GOLD.index]
    rows = []
    for t_ in TARGETS:
        y = GOLD[t_].values.astype(int)
        p = g[t_].values
        if len(set(y)) < 2:
            rows.append((t_, np.nan, np.nan, np.nan, int(y.sum()), int((1 - y).sum())))
            continue
        a = roc_auc_score(y, p)
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) > 1:
                bs.append(roc_auc_score(y[i], p[i]))
        rows.append((t_, a, np.percentile(bs, 2.5), np.percentile(bs, 97.5), int(y.sum()), int((1 - y).sum())))
    return pd.DataFrame(rows, columns=['target', 'auc', 'lo', 'hi', 'npos', 'nneg'])
AGREE = agreement(LAB).dropna(subset=['auc'])
print(AGREE.round(3).to_string(index=False))
print(f'\nmacro agreement AUC: {AGREE.auc.mean():.4f}   mean silence rate: {sil.mean() * 100:.1f}%')
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={'width_ratios': [1.25, 1]})
o = AGREE.sort_values('auc')
y = np.arange(len(o))
ax[0].hlines(y, o.lo, o.hi, color=ACC, lw=3, alpha=0.35)
ax[0].plot(o.auc, y, 'o', color=ACC, ms=5)
ax[0].axvline(0.5, color=INK, lw=0.8, ls=':')
ax[0].axvline(o.auc.mean(), color=WARN, lw=1, ls='--')
ax[0].text(o.auc.mean(), -0.9, f' macro {o.auc.mean():.3f}', color=WARN, fontsize=7)
ax[0].set_ylim(-1.4, len(o) - 0.4)
ax[0].set_yticks(y)
ax[0].set_yticklabels([f'{t_}  ({p}+/{n}-)' for t_, p, n in zip(o.target, o.npos, o.nneg)])
ax[0].set_xlim(0.35, 1.02)
ax[0].set_xlabel('AUC of the derived score against the annotation')
ax[0].set_title('gauge one: agreement, n = 58\nbars are 95% bootstrap intervals — they are this wide on purpose', loc='left', fontsize=8)
o2 = sil.sort_values()
ax[1].barh(np.arange(len(o2)), o2.values * 100, color=INK, alpha=0.8, height=0.65)
ax[1].set_yticks(np.arange(len(o2)))
ax[1].set_yticklabels(o2.index)
ax[1].set_xlabel('% of studies where no rule fired at all')
ax[1].set_title('gauge two: coverage, n = 4 407\nno labels needed, so it runs on the whole corpus', loc='left', fontsize=8)
for i, v in enumerate(o2.values * 100):
    ax[1].text(v + 1, i, f'{v:.0f}', va='center', fontsize=6.5, color=INK)
fig.tight_layout()
plt.show()

# ======== CELL 9 [code] ========
_SCRIPT = {'el': re.compile('[Ͱ-Ͽ]'), 'bg/ru': re.compile('[Ѐ-ӿ]')}
_STOP = {'en': '\\b(the|and|is|with|there is|normal)\\b', 'es': '\\b(del|los|las|con|sin|senal|rodilla|hallazgos|tecnica|resultados|impresion|menisco|rotura)\\b', 'fr': '\\b(des|les|avec|sans|genou|aucune)\\b', 'nl': '\\b(van|het|een|geen|met|voorste|knie)\\b', 'de': '\\b(der|die|und|mit|ohne|kein|keine|nachweis)\\b', 'tr': '\\b(ve|ile|izlenmistir|mevcut|normaldir|diz|bulgular)\\b', 'hr': '\\b(se|te|uz|bez|prikaz|uredan|koljena|meniska)\\b'}
_STOP = {k: re.compile(v) for k, v in _STOP.items()}

def guess_language(report):
    n = normalize(report)
    for tag, rx in _SCRIPT.items():
        if rx.search(n):
            return tag
    score = {k: len(rx.findall(n)) for k, rx in _STOP.items()}
    best = max(score, key=score.get)
    return best if score[best] >= 2 else '?'
LANG = pd.Series([guess_language(r) for r in train_df['Report'].fillna('')], index=train_df['StudyInstanceUID'])
print(LANG.value_counts().to_string())
SIL = pd.DataFrame({t: ((LAB[t + '__npos'] == 0) & (LAB[t + '__nneg'] == 0)).values for t in TARGETS}, index=LAB.index)
by_lang = SIL.groupby(LANG.reindex(SIL.index).values).mean() * 100
by_lang = by_lang.loc[LANG.value_counts().index.intersection(by_lang.index)]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={'width_ratios': [1.7, 1]})
im = ax[0].imshow(by_lang.values, cmap='RdYlBu_r', vmin=0, vmax=100, aspect='auto')
ax[0].set_xticks(range(len(TARGETS)))
ax[0].set_xticklabels(TARGETS, rotation=55, ha='right', fontsize=6.5)
ax[0].set_yticks(range(len(by_lang)))
ax[0].set_yticklabels([f'{l}  (n={int((LANG == l).sum())})' for l in by_lang.index], fontsize=7)
ax[0].grid(False)
for i in range(by_lang.shape[0]):
    for j in range(by_lang.shape[1]):
        v = by_lang.values[i, j]
        ax[0].text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=5.5, color='white' if v > 62 or v < 12 else INK)
ax[0].set_title('gauge two, broken down: % of studies where no rule fired\na common finding silent in one language and not another is a lexicon gap — or a reporting style', loc='left', fontsize=8)
fig.colorbar(im, ax=ax[0], fraction=0.02, pad=0.01)
CLAUSES = {u: clauses(r) for u, r in zip(train_df['StudyInstanceUID'], train_df['Report'].fillna(''))}

def names_it(uid, target):
    cs = CLAUSES[uid]
    if any((ANAT_MATCH[target].search(c) for c in cs)):
        return True
    if 'Meniscus' in target:
        return any((PLURAL_MENISCI.search(c) and (not ANY_SIDE.search(c)) for c in cs))
    return False
rows = []
for t_ in ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus']:
    sel = SIL[t_].values
    if not sel.sum():
        continue
    named = np.array([names_it(u, t_) for u in SIL.index[sel]])
    rows.append((t_, int(sel.sum()), 100 * named.mean()))
D = pd.DataFrame(rows, columns=['target', 'silent', 'names it anyway'])
y = np.arange(len(D))
ax[1].barh(y, 100 - D['names it anyway'], color='#8a97a3', label='never mentioned')
ax[1].barh(y, D['names it anyway'], left=100 - D['names it anyway'], color=WARN, label='mentioned, missed')
ax[1].set_yticks(y)
ax[1].set_yticklabels([f'{t_}\n({n} silent)' for t_, n in zip(D.target, D.silent)], fontsize=6.5)
ax[1].set_xlabel('% of the silent studies')
ax[1].legend(fontsize=6.5, frameon=False, loc='lower right')
ax[1].set_title('why it was silent\nonly the orange half is a lexicon gap', loc='left', fontsize=8)
fig.tight_layout()
plt.show()
print(D.round(1).to_string(index=False))

# ======== CELL 10 [code] ========
from __future__ import annotations
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
import gc
import hashlib
import json
import re
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

def _cuda_execution_probe(index):
    dev = torch.device(f'cuda:{index}')
    try:
        major, minor = torch.cuda.get_device_capability(index)
        probe = nn.Conv2d(3, 4, kernel_size=3, padding=1).eval().to(dev)
        with torch.inference_mode():
            out = probe(torch.zeros((1, 3, 16, 16), device=dev))
            if tuple(out.shape) != (1, 4, 16, 16):
                raise RuntimeError(f'unexpected CUDA probe shape {tuple(out.shape)}')
        torch.cuda.synchronize(index)
        print(f'cuda:{index} probe PASS (compute {major}.{minor})')
        del probe, out
        torch.cuda.empty_cache()
        return True
    except Exception as exc:
        print(f'cuda:{index} probe FAIL ({type(exc).__name__}: {exc}); using CPU fallback')
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False
DEVS = []
if torch.cuda.is_available():
    DEVS = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count()) if _cuda_execution_probe(i)]
if not DEVS:
    DEVS = [torch.device('cpu')]
print(f'devices: {[str(d) for d in DEVS]}')
T0 = time.time()
SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
CROP_MM = 130.0
CACHE_IMG = 336
GROUP = 3
N_GROUP_MAX = 1
CACHE_FRACTION = 0.45
CACHE_BUDGET_MAX_GB = 24.0
CACHE_BUDGET_GB = 12.0
TEST_SHARE = 0.3
HDR_THREADS = 16
PIX_THREADS = 12
ORDER_THREADS = 32
ORDER_BUDGET_S = 5400
RUNS = [{'name': 'r224', 'img': 224}, {'name': 'r336', 'img': 336}]
EPOCHS = 10
BATCH_STUDIES = 8
AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.1
LAT_MIN_OFFSET_MM = 20.0
SLICE_BAND = (0.2, 0.8)
RULES_NATIVE = {'order': 'normal', 'lat': 'centre', 'slot_fallback': False, 'decode_fill': 'nearest'}
RULES_LEGACY = {'order': 'dominant_axis', 'lat': 'corner_x', 'slot_fallback': True, 'decode_fill': 'zero'}
RULES = dict(RULES_NATIVE)
LEGACY_LAT_OFFSET_MM = 5.0
LR_HEAD = 0.001
LR_BACKBONE = 8e-06
UNFREEZE_LAST = 6
WEIGHT_DECAY = 0.02
EVAL_BATCH = 8
TIME_BUDGET = 8.0 * 3600
SLOTS_RECOVERED = [('SAG_FLUID_FS', 'Sagittal', True, True), ('COR_FLUID_FS', 'Coronal', True, True), ('AX_FLUID_FS', 'Axial', True, True), ('SAG_FLUID_NOFS', 'Sagittal', True, False), ('COR_T1', 'Coronal', False, False), ('SAG_T1', 'Sagittal', False, False)]
SLOTS_PUBLIC = [('SAG_FLUID', 'Sagittal', None, True), ('COR_FLUID', 'Coronal', None, True), ('AX_FLUID', 'Axial', None, True), ('SAG_STRUCT', 'Sagittal', None, False), ('COR_STRUCT', 'Coronal', None, False), ('AX_STRUCT', 'Axial', None, False)]
SLOT_SCHEME = os.environ.get('SLOT_SCHEME', 'recovered')
SLOTS = SLOTS_PUBLIC if SLOT_SCHEME == 'public' else SLOTS_RECOVERED
N_SLOT = len(SLOTS)
POOL_PARTS = {'cls_mean': 2, 'cls_mean_focal': 3}
SLOT_PRIOR_TABLE = {'ACL': (0, 3, 5), 'MCL': (1, 4), 'Medial Meniscus': (0, 1, 3, 4), 'Lateral Meniscus': (0, 1, 3, 4), 'Medial OA': (1, 4, 5), 'Lateral OA': (1, 4, 5), 'PF OA': (0, 2, 5), 'Effusion': (0, 2), 'Synovitis': (0, 2), "Baker's": (0,), 'Contusion': (0, 1, 2), 'Fracture': (0, 1, 2, 4, 5)}
SLOT_PRIOR_STRENGTH = 0.55
FATSAT_OPTS = {'FS', 'FATSAT', 'FAT_SAT', 'FSAT'}
_SEP = re.compile('[_\\-.]')
_FATSAT_RX = re.compile('\\bfs\\b|fatsat|fat sat|\\bstir\\b|\\bspair\\b|\\bspir\\b|\\bwe\\b|water excit|\\btirm\\b|\\bsting\\b|\\bfatsup\\b')
_T1_RX = re.compile('\\bt1\\b|\\bt1w\\b')
_T2_RX = re.compile('\\bt2\\b|\\bt2w\\b')
_PD_RX = re.compile('\\bpd\\b|\\bpdw\\b|proton|\\bdp\\b|dens')

# ======== CELL 11 [code] ========
def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)

def find_root():
    for c in [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'), Path('/kaggle/input/rsna-knee-abnormality-detection'), Path('data'), Path('.')]:
        if (c / 'test.csv').is_file() and (c / 'test_series').is_dir():
            return c
    base = Path('/kaggle/input')
    if base.is_dir():
        for depth1 in sorted((p for p in base.iterdir() if p.is_dir())):
            for cand in [depth1] + sorted((p for p in depth1.iterdir() if p.is_dir())):
                if (cand / 'test.csv').is_file():
                    return cand
    raise FileNotFoundError(f'competition mount not found (cwd {Path.cwd()}); expected a directory holding test.csv and test_series/')

def find_dinov2(variant='small'):
    base = Path('/kaggle/input')
    if not base.is_dir():
        return None
    hits = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        if 'config.json' in files and 'dinov2' in root.lower():
            hits.append(Path(root))
    for h in hits:
        if variant in str(h).lower():
            return h
    return hits[0] if hits else None
LABEL_COLS = TARGETS + [t + '__conf' for t in TARGETS]

class LabelSourceError(RuntimeError):
    pass

def find_label_table():
    base = Path('/kaggle/input')
    cands = []
    if base.is_dir():
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
            cands += [Path(root) / f for f in files if f.startswith('report_labels') and f.endswith('.csv')]
    cands += [p for p in (Path('data/derived/report_labels_v2.csv'),) if p.is_file()]
    for c in cands:
        try:
            head = pd.read_csv(c, nrows=1)
        except Exception:
            continue
        if 'StudyInstanceUID' in head.columns and all((t in head.columns for t in TARGETS)):
            return c
    return None

def label_mount_attached():
    base = Path('/kaggle/input')
    if not base.is_dir():
        return False
    return any(('label' in p.name.lower() for p in base.iterdir() if p.is_dir()))

def read_labels(train_df):
    n = len(train_df)
    lab = pd.DataFrame([extract(r) for r in train_df['Report'].fillna('')])
    lab['StudyInstanceUID'] = train_df['StudyInstanceUID'].values
    lab = lab.set_index('StudyInstanceUID')
    src = find_label_table()
    if src is None:
        if label_mount_attached():
            raise LabelSourceError('LABEL SOURCE: a label dataset is mounted but no usable table was found in it. Falling back to the lexicon here would train on the weaker labels and say so only in a log line, so the run stops instead.')
        log(f'LABEL SOURCE: lexicon, {n} studies (no table mounted)')
        return lab
    tab = pd.read_csv(src).set_index('StudyInstanceUID')
    missing = [c for c in LABEL_COLS if c not in tab.columns]
    if missing:
        raise LabelSourceError(f'LABEL SOURCE: {src} is missing {len(missing)} expected columns (first: {missing[0]!r}). Refusing to fall back silently.')
    hit = lab.index.intersection(tab.index)
    if not len(hit):
        raise LabelSourceError(f'LABEL SOURCE: {src} shares no StudyInstanceUID with train.csv.')
    log(f'LABEL SOURCE: {src.name} covers {len(hit)} of {n} studies, lexicon for the remaining {n - len(hit)}')
    lab.loc[hit, LABEL_COLS] = tab.loc[hit, LABEL_COLS].values
    return lab
ROOT = find_root()
log(f'input root: {ROOT}')
IMG = CACHE_IMG

def available_gb():
    try:
        with open('/proc/meminfo') as fh:
            info = {k.strip(): v for k, v in (l.split(':', 1) for l in fh if ':' in l)}
        return int(info['MemAvailable'].split()[0]) / 1024 ** 2
    except Exception:
        return CACHE_BUDGET_GB / CACHE_FRACTION

def plan_cache(n_study, n_test=0):
    avail = available_gb()
    budget = min(avail * CACHE_FRACTION, CACHE_BUDGET_MAX_GB)
    n_total = n_study + max(n_test, int(TEST_SHARE * n_study))
    per_slice = n_total * N_SLOT * IMG * IMG
    afford = int(budget * 1024 ** 3 // max(per_slice, 1))
    groups = max(1, min(N_GROUP_MAX, afford // GROUP))
    log(f'memory: {avail:.1f} GB available, {budget:.1f} GB to the cache; sizing for {n_study} train + {n_total - n_study} test studies -> {groups} group(s) of {GROUP} = {groups * GROUP} slices per slot' + (f' (wanted {N_GROUP_MAX})' if groups < N_GROUP_MAX else ''))
    return groups
N_GROUP = plan_cache(len(pd.read_csv(ROOT / 'train.csv')), len(pd.read_csv(ROOT / 'test.csv')))
CACHE_SLICES = GROUP * N_GROUP
log(f'cache layout: {N_GROUP} groups x {GROUP} slices = {CACHE_SLICES} per slot')

# ======== CELL 12 [code] ========
HDR_TAGS = ['SeriesDescription', 'SequenceName', 'ScanOptions', 'ScanningSequence', 'RepetitionTime', 'EchoTime', 'Laterality', 'PixelSpacing', 'Rows', 'Columns', 'RescaleSlope', 'RescaleIntercept', 'ImagePositionPatient', 'ImageOrientationPatient']

def _hdr_vec(s, n):
    if not isinstance(s, str):
        return None
    try:
        v = [float(x) for x in s.split('|')]
    except ValueError:
        return None
    return np.array(v) if len(v) >= n else None

def side_from_geometry(h):
    cx = {}
    for r in h.itertuples(index=False):
        ipp = _hdr_vec(getattr(r, 'ImagePositionPatient', None), 3)
        iop = _hdr_vec(getattr(r, 'ImageOrientationPatient', None), 6)
        ps = _hdr_vec(getattr(r, 'PixelSpacing', None), 2)
        rows, cols = (getattr(r, 'Rows', None), getattr(r, 'Columns', None))
        if ipp is None or iop is None or ps is None or (not rows) or (not cols):
            continue
        try:
            c = ipp[:3] + iop[:3] * ps[1] * float(cols) / 2 + iop[3:6] * ps[0] * float(rows) / 2
        except (TypeError, ValueError):
            continue
        cx.setdefault(r.StudyInstanceUID, []).append(float(c[0]))
    out = {}
    for st, xs in cx.items():
        m = float(np.median(xs))
        out[st] = None if abs(m) < LAT_MIN_OFFSET_MM else 'R' if m < 0 else 'L'
    return out

def side_from_corner_x(h):
    out = {}
    for st, g in h.groupby('StudyInstanceUID'):
        xs = []
        for r in g.itertuples(index=False):
            ipp = _hdr_vec(getattr(r, 'ImagePositionPatient', None), 3)
            if ipp is not None and np.isfinite(ipp).all():
                xs.append(float(ipp[0]))
        if not xs:
            out[st] = None
            continue
        x = float(np.median(xs))
        out[st] = None if abs(x) < LEGACY_LAT_OFFSET_MM else 'R' if x < 0 else 'L'
    return out

def lat_of(h, tag=''):
    geo = side_from_corner_x(h) if RULES['lat'] == 'corner_x' else side_from_geometry(h)
    d, n_tag, n_geo, n_none, n_disagree = ({}, 0, 0, 0, 0)
    for st, g in h.groupby('StudyInstanceUID'):
        v = [str(x).strip().upper() for x in g['Laterality'].dropna()]
        if RULES['lat'] == 'corner_x' and 'ImageLaterality' in g.columns:
            v += [str(x).strip().upper() for x in g['ImageLaterality'].dropna()]
        v = [x[0] for x in v if x and x[0] in ('L', 'R')]
        side = v[0] if v else None
        if side is not None:
            n_tag += 1
            if geo.get(st) is not None and geo[st] != side:
                n_disagree += 1
        else:
            side = geo.get(st)
            n_geo += side is not None
            n_none += side is None
        d[st] = side
    log(f'{tag}laterality: {n_tag} from the tag, {n_geo} from geometry, {n_none} unresolved; tag and geometry disagree on {n_disagree} ({n_disagree / max(n_tag, 1):.1%} of the tagged)')
    return d

def probe(item):
    split, study, series, path = item
    row = {'split': split, 'StudyInstanceUID': study, 'SeriesInstanceUID': series, 'dir': path}
    try:
        files = sorted((e.name for e in os.scandir(path) if e.name.endswith('.dcm')))
        row['files'] = files
        row['n_slices'] = len(files)
        if not files:
            return row
        ds = pydicom.dcmread(os.path.join(path, files[len(files) // 2]), stop_before_pixels=True, force=True)
        for t in HDR_TAGS:
            v = getattr(ds, t, None)
            if v is None:
                row[t] = None
            elif isinstance(v, (list, tuple)) or type(v).__name__ == 'MultiValue':
                row[t] = '|'.join((str(x) for x in v))
            else:
                row[t] = str(v)
    except Exception as exc:
        row['err'] = str(exc)[:120]
    return row

def walk(split):
    base = ROOT / split
    items = []
    if not base.is_dir():
        return pd.DataFrame(columns=['split', 'StudyInstanceUID', 'SeriesInstanceUID', 'dir', 'files', 'n_slices'] + HDR_TAGS)
    for study in os.scandir(base):
        if study.is_dir():
            for series in os.scandir(study.path):
                if series.is_dir():
                    items.append((split, study.name, series.name, series.path))
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as pool:
        rows = list(pool.map(probe, items))
    return pd.DataFrame(rows)

def annotate(df):
    desc = df['SeriesDescription'].fillna('') + ' ' + df['SequenceName'].fillna('')
    desc = desc.str.lower().str.replace(_SEP, ' ', regex=True)
    opts = df['ScanOptions'].fillna('').str.upper().str.split('|')
    opts_fs = opts.apply(lambda ts: any((t.strip() in FATSAT_OPTS for t in ts)))
    df['fatsat'] = desc.str.contains(_FATSAT_RX) | opts_fs
    tr = pd.to_numeric(df['RepetitionTime'], errors='coerce')
    te = pd.to_numeric(df['EchoTime'], errors='coerce')
    gre = df['ScanningSequence'].fillna('').str.upper().str.contains('GR')
    t1, t2, pdw = (desc.str.contains(_T1_RX), desc.str.contains(_T2_RX), desc.str.contains(_PD_RX))
    df['weight'] = np.where(t1 & ~t2 & ~pdw, 'T1', np.where(t2 & ~pdw, 'T2', np.where(pdw, 'PD', np.where(gre, 'GRE', np.where(tr < 800, 'T1', np.where(te > 60, 'T2', np.where(tr >= 800, 'PD', 'UNK')))))))
    df['fluid'] = np.isin(df['weight'], ['PD', 'T2'])
    df['px'] = pd.to_numeric(df['PixelSpacing'].fillna('').str.split('|').str[0].replace('', np.nan), errors='coerce')
    return df

# ======== CELL 13 [code] ========
def pick_slots(series_df, plane_map):
    series_df = series_df.copy()
    series_df['plane'] = series_df['SeriesInstanceUID'].map(plane_map)
    out = {}
    for study, g in series_df.groupby('StudyInstanceUID'):
        chosen = {}
        for name, plane, fluid, fs in SLOTS:
            sel = (g['plane'] == plane) & (g['fatsat'] == fs)
            if fluid is not None:
                sel &= g['fluid'] == fluid
            cand = g[sel]
            if len(cand) == 0 and RULES['slot_fallback'] and (fluid is False):
                cand = g[(g['plane'] == plane) & ~g['fatsat']]
            if len(cand):
                chosen[name] = cand.sort_values('n_slices', ascending=False).iloc[0]
        out[study] = chosen
    return out

# ======== CELL 14 [code] ========
ORDER_TAGS = [(32, 50), (32, 55), (32, 19)]
DECODE_FAILED = []

def cache_tag(rules=None):
    r = dict(RULES if rules is None else rules)
    t = f'{CACHE_IMG}px_{CACHE_SLICES}sl_{int(CROP_MM)}mm_{SLICE_BAND[0]:.2f}-{SLICE_BAND[1]:.2f}'
    if {k: r.get(k, v) for k, v in RULES_NATIVE.items()} != RULES_NATIVE:
        t += '_' + hashlib.md5(json.dumps(r, sort_keys=True).encode()).hexdigest()[:6]
    return t

def _natural_key(name):
    return tuple((int(x) if x.isdigit() else x.lower() for x in re.split('(\\d+)', str(name))))

def _order_dominant_axis(rec):
    files, d = (rec['files'], rec['dir'])
    rows = []
    for pos, f in enumerate(files):
        ipp = inst = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True, specific_tags=['ImagePositionPatient', 'InstanceNumber'])
            raw = getattr(ds, 'ImagePositionPatient', None)
            if raw is not None and len(raw) >= 3:
                c = np.asarray(raw[:3], dtype=np.float64)
                if np.isfinite(c).all():
                    ipp = c
            n = getattr(ds, 'InstanceNumber', None)
            if n is not None:
                inst = float(n)
        except Exception:
            pass
        rows.append((f, ipp, inst, pos))
    placed = [r for r in rows if r[1] is not None]
    need = max(2, int(0.8 * len(rows)))
    if len(placed) >= need:
        xyz = np.stack([r[1] for r in placed])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        rows.sort(key=lambda r: (float(r[1][axis]) if r[1] is not None else spare, r[2] if r[2] is not None else float('inf'), r[3]))
    elif sum((r[2] is not None for r in rows)) >= need:
        rows.sort(key=lambda r: (r[2] if r[2] is not None else float('inf'), r[3]))
    else:
        rows.sort(key=lambda r: _natural_key(r[0]))
    return ([r[0] for r in rows], True)

def order_slices(rec):
    if RULES['order'] == 'dominant_axis':
        return _order_dominant_axis(rec)
    files, d = (rec['files'], rec['dir'])
    keyed = []
    for f in files:
        k = None
        try:
            ds = pydicom.dcmread(os.path.join(d, f), force=True, stop_before_pixels=True, specific_tags=ORDER_TAGS)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            k = float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
        except Exception:
            try:
                k = float(ds.InstanceNumber)
            except Exception:
                k = None
        keyed.append((k, f))
    if any((k is None for k, _ in keyed)):
        return (files, False)
    return ([f for _, f in sorted(keyed, key=lambda t: t[0])], True)

def read_slot(rec, n_slice=None, out_size=None):
    n_slice = GROUP if n_slice is None else n_slice
    out_size = IMG if out_size is None else out_size
    files, d, px = (rec.get('ordered') or rec['files'], rec['dir'], rec['px'])
    n = len(files)
    if n == 0:
        return None
    lo, hi = (int(SLICE_BAND[0] * (n - 1)), int(SLICE_BAND[1] * (n - 1)))
    idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])
    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, 'RescaleSlope', 1) or 1)
            ic = float(getattr(ds, 'RescaleIntercept', 0) or 0)
            a = a * sl + ic
        except Exception:
            a = None
        planes.append(a)
    got = [k for k, p in enumerate(planes) if p is not None]
    if RULES['decode_fill'] == 'zero':
        if not got:
            DECODE_FAILED.append(rec.get('SeriesInstanceUID', d))
        planes = [np.zeros((out_size, out_size), np.float32) if p is None else p for p in planes]
        got = list(range(len(planes)))
    if not got:
        DECODE_FAILED.append(rec.get('SeriesInstanceUID', d))
        return None
    if len(got) < len(planes):
        DECODE_FAILED.append(rec.get('SeriesInstanceUID', d))
        for k, p in enumerate(planes):
            if p is None:
                planes[k] = planes[min(got, key=lambda j: abs(j - k))]
    shp = planes[0].shape
    planes = [p if p.shape == shp else np.zeros(shp, np.float32) for p in planes]
    vol = np.stack(planes)
    if px and np.isfinite(px) and (px > 0):
        want = int(round(CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = (h // 2, w // 2)
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]
    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-06), 0, 1)
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode='bilinear', align_corners=False)
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)

# ======== CELL 15 [code] ========
def normalise_laterality(img, plane, lat):
    if lat != 'R':
        return img
    if plane in ('Coronal', 'Axial'):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])

# ======== CELL 16 [code] ========
ORDER_CACHE = os.environ.get('RSNA_ORDER_CACHE') or None

def build_cache(slot_map, plane_map, lat_map, tag):
    studies = sorted(slot_map)
    sidx = {s: i for i, s in enumerate(studies)}
    cache = np.zeros((len(studies), N_SLOT, CACHE_SLICES, IMG, IMG), np.uint8)
    mask = np.zeros((len(studies), N_SLOT), np.float32)
    log(f'{tag}: cache {cache.shape} = {cache.nbytes / 1024 ** 3:.1f} GB')
    jobs = [(st, k, plane, slot_map[st][name]) for st in studies for k, (name, plane, _, _) in enumerate(SLOTS) if name in slot_map[st]]
    n_job = len(jobs)
    t_ord = time.time()
    n_slice_total = sum((len(j[3]['files']) for j in jobs))
    log(f'{tag}: ordering {len(jobs)} slot-series ({n_slice_total} slice headers)')
    ok = done = 0
    CHUNK_O = 1024
    seen = {}
    if ORDER_CACHE and Path(ORDER_CACHE).is_file():
        try:
            import json as _json
            seen = _json.loads(Path(ORDER_CACHE).read_text())
        except (OSError, ValueError):
            seen = {}
        hit = 0
        for _, _, _, rec in jobs:
            e = seen.get(rec['SeriesInstanceUID'])
            if e and len(e['files']) == len(rec['files']):
                rec['ordered'] = e['files']
                ok += int(e['good'])
                hit += 1
        jobs = [j for j in jobs if 'ordered' not in j[3]]
        log(f'{tag}: {hit} slot-series ordered from {ORDER_CACHE}, {len(jobs)} to read')
    with ThreadPoolExecutor(max_workers=ORDER_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK_O):
            block = jobs[c0:c0 + CHUNK_O]
            for (_, _, _, rec), (files, good) in zip(block, pool.map(lambda j: order_slices(j[3]), block)):
                rec['ordered'] = files
                ok += int(good)
                done += 1
                if ORDER_CACHE:
                    seen[rec['SeriesInstanceUID']] = {'files': files, 'good': bool(good)}
            budget = min(ORDER_BUDGET_S, max(60.0, (TIME_BUDGET - (time.time() - T0)) * 0.35))
            if time.time() - t_ord > budget:
                log(f'{tag}: ordering budget spent at {done}/{len(jobs)}; the rest keep file order')
                break
    if ORDER_CACHE and done:
        import json as _json
        _t = Path(ORDER_CACHE).with_suffix('.tmp')
        _t.write_text(_json.dumps(seen))
        _t.replace(Path(ORDER_CACHE))
    log(f'{tag}: ordered {ok}/{n_job} by geometry ({n_job - ok} kept arbitrary) in {time.time() - t_ord:.0f}s')
    jobs = [(st, k, plane, slot_map[st][name]) for st in studies for k, (name, plane, _, _) in enumerate(SLOTS) if name in slot_map[st]]
    log(f'{tag}: decoding {len(jobs)} slot-series')
    n_failed_before = len(DECODE_FAILED)
    CHUNK = 512
    done = 0
    with ThreadPoolExecutor(max_workers=PIX_THREADS) as pool:
        for c0 in range(0, len(jobs), CHUNK):
            block = jobs[c0:c0 + CHUNK]
            for (st, k, plane, _), img in zip(block, pool.map(lambda j: read_slot(j[3], CACHE_SLICES, IMG), block)):
                done += 1
                if img is None:
                    continue
                cache[sidx[st], k] = normalise_laterality(img, plane, lat_map.get(st)).numpy()
                mask[sidx[st], k] = 1.0
            if done % 4096 < CHUNK:
                log(f'  {tag} {done}/{len(jobs)}')
            if time.time() - T0 > TIME_BUDGET:
                log(f'  {tag}: time budget reached during decode')
                break
    n_failed = len(DECODE_FAILED) - n_failed_before
    log(f'{tag}: {int(mask.sum())}/{len(jobs)} slots filled' + (f'; {n_failed} series had a slice that would not decode' if n_failed else ''))
    gc.collect()
    return (studies, cache, mask)

# ======== CELL 17 [code] ========
class SlotHead(nn.Module):

    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        p_ = torch.zeros(n_out, n_slot)
        if prior and n_slot == len(SLOTS) and (n_out == len(TARGETS)):
            for t, slots in SLOT_PRIOR_TABLE.items():
                if t in TARGETS:
                    p_[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
        self.prior = prior
        if prior:
            self.register_buffer('slot_prior', p_)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum('bsh,oh->bos', h, self.query) / self.hidden ** 0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -10000.0).softmax(-1)
        ctx = self.drop(torch.einsum('bos,bsh->boh', att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias

# ======== CELL 18 [code] ========
class Model(nn.Module):

    def __init__(self, backbone, dim, pool='cls_mean', prior=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, imgs, mask, img_size=None):
        B, S = imgs.shape[:2]
        x = imgs.reshape(B * S, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == 'cls_mean_focal':
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        feat = torch.cat(parts, dim=1).reshape(B, S, -1)
        return self.head(feat, mask)

# ======== CELL 19 [code] ========
def build_model(unfreeze_last, source=None, variant='small', pool='cls_mean', prior=False):
    from transformers import AutoModel
    p = source if source is not None else find_dinov2(variant)
    if p is None:
        raise FileNotFoundError('DINOv2 weights not attached')
    bb = AutoModel.from_pretrained(str(p))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size
    trainable = sum((p.numel() for p in bb.parameters() if p.requires_grad))
    log(f'backbone: {n_layer} blocks, last {unfreeze_last} trainable ({trainable / 1000000.0:.1f}M params), feature dim {dim * POOL_PARTS[pool]}')
    return Model(bb, dim, pool=pool, prior=prior)

# ======== CELL 20 [code] ========
FINGERPRINT_TOL = 0.002

def fingerprint(model, dev, img_size, n_slot=None, group=None, seed=None):
    n_slot = N_SLOT if n_slot is None else n_slot
    group = GROUP if group is None else group
    seed = SEED if seed is None else seed
    g = torch.Generator().manual_seed(seed)
    imgs = torch.randint(0, 256, (2, n_slot, group, img_size, img_size), generator=g, dtype=torch.uint8).to(dev)
    mask = torch.ones(2, n_slot, device=dev)
    mask[1, -1] = 0.0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(imgs, mask, img_size).float().cpu().numpy()
    if was_training:
        model.train()
    return out

def check_fingerprint(model, dev, img_size, expected, tol=FINGERPRINT_TOL, tag=''):
    got = fingerprint(model, dev, img_size)
    exp = np.asarray(expected, np.float32)
    if got.shape != exp.shape:
        raise WeightsError(f'{tag}fingerprint shape {got.shape} != stored {exp.shape}: the architecture is not the one these weights were fitted to')
    d = float(np.abs(got - exp).max())
    if d > tol:
        raise WeightsError(f'{tag}fingerprint differs by {d:.4g} (tolerance {tol:g}). The weights load but do not compute what they computed when fitted - preprocessing, resolution or architecture has moved between the two runs.')
    log(f'{tag}fingerprint matches within {d:.2g}')
    return d

class WeightsError(RuntimeError):
    pass

def find_weights(name='manifest.json'):
    import json
    base = Path('/kaggle/input')
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        if name not in files:
            continue
        try:
            man = json.loads((Path(root) / name).read_text())
        except (OSError, ValueError):
            continue
        if isinstance(man.get('members'), list) and man['members']:
            missing = [m['file'] for m in man['members'] if not (Path(root) / m['file']).is_file()]
            if missing:
                raise WeightsError(f"{root} holds a manifest listing {len(man['members'])} members but {len(missing)} of their files are absent (first {missing[0]!r})")
            return Path(root)
    return None
TTA_OVERLAP = True
TTA_POOL = 'prob'
PUBLIC_FRONTIER_TARGET_POOL = {'Fracture': 'max', 'Contusion': 'max', 'Medial Meniscus': 'max', 'Lateral Meniscus': 'max', 'ACL': 'top2', 'MCL': 'top2', "Baker's": 'max'}
TTA_TARGET_POOL = {**PUBLIC_FRONTIER_TARGET_POOL, 'Synovitis': 'original_mean'}
LEGACY_MEMBER_WEIGHT_BY_TARGET = {'Lateral Meniscus': 15.0, 'Medial OA': 2.5, 'Lateral OA': 15.0, 'Contusion': 5.0}

def window_starts(n_slice, group, overlap=None):
    overlap = TTA_OVERLAP if overlap is None else overlap
    if overlap and n_slice >= group:
        return list(range(n_slice - group + 1))
    return [g * group for g in range(max(n_slice // group, 1))]

def apply_target_window_pool(values, probs, logits, original_probs, mapping, target_idx):
    for target, mode in mapping.items():
        j = target_idx[target]
        if mode == 'max':
            values[:, j] = probs[:, :, j].max(0).values
        elif mode == 'mean':
            values[:, j] = probs[:, :, j].mean(0)
        elif mode == 'logit_mean':
            values[:, j] = torch.sigmoid(logits[:, :, j].mean(0))
        elif mode == 'original_mean':
            values[:, j] = original_probs[:, :, j].mean(0)
        elif mode in ('top2', 'top3'):
            k = min(int(mode[3:]), probs.shape[0])
            values[:, j] = probs[:, :, j].topk(k, dim=0).values.mean(0)
        else:
            raise ValueError(f'unknown TTA pooling mode for {target}: {mode}')
    return values

@torch.no_grad()
def predict_member(model, cache, mask, idx, dev, img_size, group=None, pool=None, starts=None, jitter=False, jitter_seed=SEED, return_public_frontier=False):
    group = GROUP if group is None else group
    pool = TTA_POOL if pool is None else pool
    starts = window_starts(cache.shape[2], group) if starts is None else list(starts)
    if not starts:
        raise ValueError('predict_member was given no windows to average over')
    target_idx = {t: j for j, t in enumerate(TARGETS)}
    unknown = (set(TTA_TARGET_POOL) | set(PUBLIC_FRONTIER_TARGET_POOL)) - set(target_idx)
    if unknown:
        raise ValueError(f'unknown target(s) in TTA_TARGET_POOL: {unknown}')
    jitter_gen = torch.Generator(device=dev)
    jitter_gen.manual_seed(int(jitter_seed) % (2 ** 63 - 1))
    model.eval()
    out, public_frontier_out = ([], [])
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[sel]).to(dev)
        win_probs, win_logits, win_original_probs = ([], [], [])
        for st in starts:
            rows = torch.from_numpy(np.ascontiguousarray(cache[sel, :, st:st + group])).to(dev)
            views = [rows] + ([augment(rows, generator=jitter_gen)] if jitter else [])
            view_probs, view_logits = ([], [])
            for view in views:
                with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                    z = model(view, m, img_size).float()
                view_logits.append(z)
                view_probs.append(torch.sigmoid(z))
            win_logits.append(torch.stack(view_logits).mean(0))
            win_probs.append(torch.stack(view_probs).mean(0))
            win_original_probs.append(view_probs[0])
        probs = torch.stack(win_probs)
        logits = torch.stack(win_logits)
        original_probs = torch.stack(win_original_probs)
        v = torch.sigmoid(logits.mean(0)) if pool == 'logit' else probs.mean(0)
        v = apply_target_window_pool(v, probs, logits, original_probs, TTA_TARGET_POOL, target_idx)
        out.append(v.cpu().numpy())
        if return_public_frontier:
            public_v = apply_target_window_pool(original_probs.mean(0), original_probs, logits, original_probs, PUBLIC_FRONTIER_TARGET_POOL, target_idx)
            public_frontier_out.append(public_v.cpu().numpy())
    primary = np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)
    if not return_public_frontier:
        return primary
    public_frontier = np.concatenate(public_frontier_out) if public_frontier_out else np.zeros((0, len(TARGETS)), np.float32)
    return (primary, public_frontier)
BUILD_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
LEGACY_BUNDLE_FILE = 'rsna_20260807_v1.pt'
LEGACY_WEIGHT = 0.5

def find_legacy_bundle():
    base = Path('/kaggle/input')
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ('train_series', 'test_series')]
        if LEGACY_BUNDLE_FILE in files:
            return Path(root) / LEGACY_BUNDLE_FILE
    return None

def legacy_group_members():
    p = find_legacy_bundle()
    if p is None:
        log('no legacy bundle attached; blending skipped')
        return {}
    try:
        b = torch.load(p, map_location='cpu', weights_only=False)
        folds = b.get('fold_states') or []
        b_slots = [tuple(s)[0] for s in b.get('slots', SLOTS)]
        if list(b.get('targets', TARGETS)) != TARGETS or b_slots != [s[0] for s in SLOTS]:
            log(f'legacy bundle {p.name}: target/slot contract differs; blending skipped')
            return {}
        gr, n_gr = (int(b.get('group', 3)), int(b.get('n_group', 3)))
        variant = str(b.get('model_variant', 'dinov2-small')).split('-')[-1]
        key = json.dumps({'img': int(b.get('img', 224)), 'group': gr, 'slices': gr * n_gr, 'crop_mm': 160.0, 'band': [0.2, 0.8], 'rules': RULES_LEGACY, 'slots': [s[0] for s in SLOTS]}, sort_keys=True)
        ms = [{'id': f"legacy-f{f.get('fold', k)}", 'fold': f.get('fold', k), 'state': f['state_dict'], 'holdout': None, 'weight': LEGACY_WEIGHT, 'target_weight': [LEGACY_MEMBER_WEIGHT_BY_TARGET.get(t, 0.0) for t in TARGETS], 'pixel_group': key, 'config': {'unfreeze_last': 6, 'variant': 'base' if variant == 'base' else 'small', 'pool': 'cls_mean_focal', 'prior': True}} for k, f in enumerate(folds)]
        if ms:
            active = sorted(set(LEGACY_MEMBER_WEIGHT_BY_TARGET.values()))
            log(f'legacy bundle {p.name}: {len(ms)} fold(s) join with target-specific per-member weights {active}')
        return {key: ms} if ms else {}
    except Exception as exc:
        log(f'legacy bundle unusable ({type(exc).__name__}: {exc}); blending skipped')
        return {}

def _run_member(path, m, dev, Cte, Mte, idx, starts, jitter):
    t0 = time.time()
    with BUILD_LOCK:
        if 'state' in m:
            state, fp = (m['state'], None)
        else:
            ck = torch.load(Path(path) / m['file'], map_location='cpu', weights_only=False)
            state, fp = (ck['model'], ck.get('fingerprint'))
        model = build_model(int(m['config']['unfreeze_last']), variant=m['config']['variant'], pool=m['config'].get('pool', 'cls_mean'), prior=bool(m['config'].get('prior', False))).to(dev)
        model.load_state_dict(state)
        if fp is not None:
            check_fingerprint(model, dev, IMG, fp, tag=f"{m['id']}: ")
        else:
            log(f"  {m['id']}: no stored fingerprint (legacy bundle) -- accepted at reduced weight")
    t_ready = time.time()
    jitter_seed = SEED + int(hashlib.sha256(str(m['id']).encode()).hexdigest()[:8], 16)
    public_member = 'state' not in m
    predicted = predict_member(model, Cte, Mte, idx, dev, IMG, starts=starts, jitter=jitter, jitter_seed=jitter_seed, return_public_frontier=public_member)
    if public_member:
        p, public_p = predicted
    else:
        p, public_p = (predicted, None)
    t_done = time.time()
    del model, state
    gc.collect()
    if dev.type == 'cuda':
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    passes = len(starts) * (2 if jitter else 1)
    return (p, public_p, (t_ready - t0, (t_done - t_ready) / max(passes, 1)))

def _combine(per_member):
    all_ids = sorted({s for m in per_member for s in m['ids']})
    pos = {s: i for i, s in enumerate(all_ids)}
    acc = np.zeros((len(all_ids), len(TARGETS)), np.float64)
    tot = np.zeros(len(TARGETS), np.float64)
    for m in per_member:
        target_weight = m.get('target_weight')
        w = np.asarray(target_weight if target_weight is not None else [float(m.get('weight', 1.0))] * len(TARGETS), dtype=np.float64)
        if w.shape != (len(TARGETS),) or np.any(w < 0):
            raise ValueError(f"invalid target weights for {m.get('id')}: {w}")
        r = pd.DataFrame(m['pred']).rank(pct=True).to_numpy()
        acc[[pos[s] for s in m['ids']]] += r * w[None, :]
        tot += w
    if np.any(tot <= 0):
        raise ValueError(f'at least one target has no ensemble vote: {tot}')
    return (all_ids, acc / tot[None, :])

def infer_from_package(path, dev=None):
    man = json.loads((Path(path) / 'manifest.json').read_text())
    members = man['members']
    log(f'weights package: {len(members)} member(s) from {path}; {len(DEVS)} device(s)')
    test_df = pd.read_csv(ROOT / 'test.csv')
    test_series = pd.read_csv(ROOT / 'test_series.csv')
    plane_map = dict(zip(test_series['SeriesInstanceUID'], test_series['Anatomical_Plane']))
    hte = annotate(walk('test_series'))
    log(f'test header pass: {len(hte)} series')
    groups = {}
    for m in members:
        groups.setdefault(m['pixel_group'], []).append(m)
    groups.update(legacy_group_members())
    per_member, public_frontier_members = ([], [])
    est = {'fixed': None, 'win': None}

    def bank(m, ids, pred, starts, jitter, public_pred=None):
        if float(np.std(pred)) < 1e-09:
            log(f"  {m['id']}: degenerate predictions; not banked")
            return
        with STATE_LOCK:
            per_member.append({'id': m['id'], 'ids': ids, 'pred': pred, 'weight': m.get('weight', 1.0), 'target_weight': m.get('target_weight'), 'holdout': m.get('holdout')})
            if public_pred is not None and len(starts) == len(starts_full):
                if float(np.std(public_pred)) < 1e-09:
                    raise WeightsError(f"{m['id']}: degenerate public-frontier prediction")
                public_frontier_members.append({'id': m['id'], 'ids': ids, 'pred': public_pred})
            elif public_pred is not None:
                log(f"  {m['id']}: public-frontier vote omitted because only {len(starts)} / {len(starts_full)} windows completed")
            all_ids, acc = _combine(per_member)
            write_submission(acc, all_ids, test_df, 'submission.csv')
            log(f"  banked {m['id']} fold {m.get('fold', '?')} ({len(starts)} window(s){(', jitter' if jitter else '')}); submission.csv = weighted rank mean of {len(per_member)} member(s)")
    for gi, (key, gm) in enumerate(groups.items(), 1):
        cfg = json.loads(key)
        adopt_config_globals(cfg)
        log(f"decode group {gi}/{len(groups)}: {cfg['img']}px x {cfg['slices']} slices, crop {cfg['crop_mm']} mm -> {len(gm)} member(s)")
        st_te, Cte, Mte = build_cache(pick_slots(hte, plane_map), plane_map, lat_of(hte, 'test '), f'test g{gi}')
        idx = np.arange(len(st_te))
        starts_full = window_starts(Cte.shape[2], GROUP)
        pending = sorted(gm, key=lambda m: -(m.get('holdout') or 0))
        left_after = sum((len(g) for j, (_, g) in enumerate(groups.items(), 1) if j > gi))

        def pop_next():
            with STATE_LOCK:
                if not pending:
                    return (None, None, False)
                left = TIME_BUDGET - (time.time() - T0)
                remaining = len(pending) + left_after
                slots_left = -(-remaining // len(DEVS))
                starts, jit = (starts_full, False)
                if est['fixed'] is not None and est['win'] is not None:
                    afford = max(left * 0.9, 0.0)
                    room = afford / max(slots_left, 1)
                    if est['fixed'] + est['win'] > room:
                        log(f'  {left / 60:.0f} min left: surrendering {len(pending)} member(s); not one more fits')
                        pending.clear()
                        return (None, None, False)
                    jit = est['fixed'] + 2 * len(starts_full) * est['win'] <= room * 0.6
                    per_win = est['win'] * (2 if jit else 1)
                    n_win = int((room - est['fixed']) / per_win) if per_win > 0 else len(starts_full)
                    n_win = max(1, min(len(starts_full), n_win))
                    if n_win < len(starts_full):
                        mid = (len(starts_full) - n_win) // 2
                        starts = starts_full[mid:mid + n_win]
                return (pending.pop(0), starts, jit)

        def worker(dev):
            others = [d for d in DEVS if d is not dev]
            while True:
                m, starts, jit = pop_next()
                if m is None:
                    return
                for attempt, d in enumerate([dev] + others[:1]):
                    try:
                        p, public_p, (fs, ws) = _run_member(path, m, d, Cte, Mte, idx, starts, jit)
                        with STATE_LOCK:
                            est['fixed'], est['win'] = (fs, ws)
                        bank(m, st_te, p, starts, jit, public_p)
                        break
                    except Exception as exc:
                        log(f"  MEMBER {m['id']} failed on {d} ({type(exc).__name__}: {exc}); " + ('retrying on peer device' if attempt == 0 and others else 'dropped -- costs one vote, not the run'))
                        if d.type == 'cuda':
                            with torch.cuda.device(d):
                                torch.cuda.empty_cache()
        threads = [threading.Thread(target=worker, args=(d,)) for d in DEVS]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        del Cte, Mte
        gc.collect()
    if not per_member:
        raise WeightsError('no member produced predictions; submission stays at 0.5')
    all_ids, acc = _combine(per_member)
    sub = write_submission(acc, all_ids, test_df, 'submission.csv')
    log(f'final submission.csv = weighted rank mean of {len(per_member)} member(s); {sub.shape}; nulls {int(sub[TARGETS].isna().sum().sum())}')
    if len(public_frontier_members) == len(members):
        frontier_ids, frontier_acc = _combine(public_frontier_members)
        frontier_sub = write_submission(frontier_acc, frontier_ids, test_df, 'submission_public_0899.csv')
        log(f'submission_public_0899.csv = exact no-jitter public-frontier rank mean of {len(public_frontier_members)} member(s); {frontier_sub.shape}; nulls {int(frontier_sub[TARGETS].isna().sum().sum())}')
    else:
        log(f'public-frontier fallback not emitted: {len(public_frontier_members)} / {len(members)} required public members completed')
    return sub

def adopt_config_globals(cfg):
    global IMG, CACHE_IMG, GROUP, CACHE_SLICES, N_GROUP, CROP_MM, SLICE_BAND, RULES
    CACHE_IMG = IMG = int(cfg['img'])
    GROUP = int(cfg['group'])
    CACHE_SLICES = int(cfg['slices'])
    N_GROUP = max(CACHE_SLICES // GROUP, 1)
    CROP_MM = float(cfg['crop_mm'])
    SLICE_BAND = tuple((float(x) for x in cfg['band']))
    rules = cfg.get('rules') or RULES_NATIVE
    unknown = {k: v for k, v in rules.items() if k not in RULES_NATIVE or v not in (RULES_NATIVE[k], RULES_LEGACY[k])}
    if unknown:
        raise WeightsError(f'the members record pixel rules this pipeline cannot reproduce: {unknown}')
    RULES = {**RULES_NATIVE, **rules}
    if [s[0] for s in SLOTS] != list(cfg['slots']):
        raise WeightsError(f"the members were fitted on slots {cfg['slots']} and this pipeline defines {[s[0] for s in SLOTS]}; a weight would be read against the wrong slot")

# ======== CELL 21 [code] ========
def take_group(cache_rows, g):
    return cache_rows[:, :, g * GROUP:(g + 1) * GROUP]

def augment(imgs, generator=None):
    lead = imgs.shape[:-3]
    x = imgs.reshape(-1, *imgs.shape[-3:]).float()
    n, dev = (x.shape[0], x.device)
    rot = (torch.rand(n, device=dev, generator=generator) - 0.5) * 2 * (AUG_ROT_DEG * np.pi / 180)
    sc = 1.0 + torch.rand(n, device=dev, generator=generator) * AUG_SCALE
    tx = (torch.rand(n, device=dev, generator=generator) - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(n, device=dev, generator=generator) - 0.5) * 2 * AUG_SHIFT
    cos, sin = (torch.cos(rot) / sc, torch.sin(rot) / sc)
    theta = torch.zeros(n, 2, 3, device=dev, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = (cos, -sin, tx)
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = (sin, cos, ty)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
    scale = 1.0 + (torch.rand(n, 1, 1, 1, device=dev, generator=generator) - 0.5) * 2 * AUG_INTENSITY
    x = (x * scale).clamp(0, 255)
    return x.reshape(*lead, *x.shape[-3:]).to(imgs.dtype)

@torch.no_grad()
def predict(model, cache, mask, idx, dev, img_size=None):
    model.eval()
    out = []
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        m = torch.from_numpy(mask[sel]).to(dev)
        acc = None
        for g in range(N_GROUP):
            rows = torch.from_numpy(np.ascontiguousarray(cache[sel, :, g * GROUP:(g + 1) * GROUP])).to(dev)
            with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                z = model(rows, m, img_size).float()
            acc = z if acc is None else acc + z
        out.append(torch.sigmoid(acc / N_GROUP).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)

def macro_auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(np.nanmean([roc_auc_score(y[:, j], p[:, j]) if len(set(y[:, j])) > 1 else np.nan for j in range(y.shape[1])]))

# ======== CELL 22 [code] ========
import math
import cv2
TARGET_FAMILIES = ['acl', 'mcl', 'medial_meniscus', 'lateral_meniscus', 'medial_oa', 'lateral_oa', 'pf_oa', 'effusion', 'synovitis', 'baker', 'contusion', 'fracture']
GROUP_NAMES = ['ligament', 'meniscus', 'oa', 'inflammation', 'bone', 'other']

def target_group_id(family):
    if family in {'acl', 'mcl'}:
        return 0
    if family in {'medial_meniscus', 'lateral_meniscus'}:
        return 1
    if family in {'medial_oa', 'lateral_oa', 'pf_oa'}:
        return 2
    if family in {'effusion', 'synovitis', 'baker'}:
        return 3
    if family in {'contusion', 'fracture'}:
        return 4
    return 5
TARGET_GROUP_IDS = torch.tensor([target_group_id(f) for f in TARGET_FAMILIES], dtype=torch.long)

class RTAHMIL(nn.Module):

    def __init__(self, in_dim, hidden_dim, n_targets, n_slots, n_slices, dropout, series_dropout):
        super().__init__()
        self.n_targets = n_targets
        self.n_slots = n_slots
        self.n_slices = n_slices
        self.series_dropout = float(series_dropout)
        self.input_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.slice_pos_emb = nn.Parameter(torch.randn(n_slices, hidden_dim) / math.sqrt(hidden_dim))
        slice_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.slice_encoder = nn.TransformerEncoder(slice_layer, num_layers=1)
        self.series_query = nn.Parameter(torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim))
        self.series_pool = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.slot_emb = nn.Embedding(n_slots, hidden_dim)
        self.plane_emb = nn.Embedding(3, hidden_dim)
        self.sequence_emb = nn.Embedding(2, hidden_dim)
        study_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.study_encoder = nn.TransformerEncoder(study_layer, num_layers=2)
        self.target_queries = nn.Parameter(torch.randn(n_targets, hidden_dim) / math.sqrt(hidden_dim))
        self.group_emb = nn.Embedding(len(GROUP_NAMES), hidden_dim)
        self.target_cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.target_fuse = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.target_heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1)) for _ in range(n_targets)])
        slot_plane = [0, 0, 1, 1, 2, 2]
        slot_sequence = [0, 1, 0, 1, 0, 1]
        self.register_buffer('slot_plane_ids', torch.tensor(slot_plane, dtype=torch.long), persistent=False)
        self.register_buffer('slot_sequence_ids', torch.tensor(slot_sequence, dtype=torch.long), persistent=False)
        self.register_buffer('target_group_ids', TARGET_GROUP_IDS, persistent=False)

    def stochastic_slot_mask(self, mask):
        if not self.training or self.series_dropout <= 0:
            return mask
        keep = torch.rand(mask.shape, device=mask.device) > self.series_dropout
        new_mask = mask & keep
        all_missing = (~new_mask).all(dim=1)
        if all_missing.any():
            for row in torch.where(all_missing)[0]:
                valid = torch.where(mask[row])[0]
                if len(valid) > 0:
                    new_mask[row, valid[0]] = True
        return new_mask

    def forward(self, x, slot_mask):
        B, S, K, _ = x.shape
        z = self.input_proj(x)
        z = z + self.slice_pos_emb[None, None, :K, :]
        z = z.reshape(B * S, K, -1)
        z = self.slice_encoder(z)
        q = self.series_query.expand(B * S, -1, -1)
        series_token, _ = self.series_pool(q, z, z, need_weights=False)
        series_token = series_token[:, 0].reshape(B, S, -1)
        slot_ids = torch.arange(S, device=x.device)
        plane_ids = self.slot_plane_ids[:S]
        seq_ids = self.slot_sequence_ids[:S]
        series_token = series_token + self.slot_emb(slot_ids)[None, :, :] + 0.35 * self.plane_emb(plane_ids)[None, :, :] + 0.35 * self.sequence_emb(seq_ids)[None, :, :]
        effective_mask = self.stochastic_slot_mask(slot_mask)
        no_valid_slot = (~effective_mask).all(dim=1)
        if no_valid_slot.any():
            effective_mask = effective_mask.clone()
            series_token = series_token.clone()
            effective_mask[no_valid_slot, 0] = True
            series_token[no_valid_slot, 0] = 0.0
        series_token = self.study_encoder(series_token, src_key_padding_mask=~effective_mask)
        denom = effective_mask.sum(dim=1, keepdim=True).clamp_min(1).to(series_token.dtype)
        study_global = (series_token * effective_mask.unsqueeze(-1)).sum(dim=1) / denom
        target_q = self.target_queries + 0.25 * self.group_emb(self.target_group_ids)
        target_q = target_q.unsqueeze(0).expand(B, -1, -1)
        target_context, _ = self.target_cross_attn(target_q, series_token, series_token, key_padding_mask=~effective_mask, need_weights=False)
        global_expand = study_global.unsqueeze(1).expand(-1, self.n_targets, -1)
        fused = self.target_fuse(torch.cat([target_context, global_expand], dim=-1))
        logits = []
        for j, head in enumerate(self.target_heads):
            logits.append(head(fused[:, j]))
        return torch.cat(logits, dim=1)
'Runtime helpers embedded into the V26 Kaggle notebook.\n\nThe exact RTAHMIL class from the public report-teacher notebook is prepended by the\ncandidate builder. This file contains only hidden-test feature extraction, checkpoint\ninference, and the fail-safe Synovitis blend.\n'
RT_START_CUTOFF_S = 5.9 * 3600
RT_DEADLINE_S = 7.1 * 3600
RT_IMG_SIZE = 336
RT_TARGET_SPACING = 0.42
RT_SLICES = 7
RT_SYN_WEIGHT = 0.75
RT_SEEDS = (2026, 3407)

def _rt_find_checkpoint_dir():
    root = Path('/kaggle/input')
    required = [f'rta_final_seed{seed}_fold{fold}.pth' for seed in RT_SEEDS for fold in range(4)]
    for first in required[:1]:
        for hit in root.glob(f'*/{first}'):
            parent = hit.parent
            if all(((parent / name).is_file() for name in required)):
                return parent
    raise FileNotFoundError('the complete eight-checkpoint report-teacher package is absent')

def _rt_find_dino_base():
    direct = [Path('/kaggle/input/dinov2/pytorch/base/1'), Path('/kaggle/input/models/metaresearch/dinov2/pytorch/base/1')]
    for path in direct:
        if (path / 'config.json').is_file():
            return path
    for top in Path('/kaggle/input').iterdir():
        if not top.is_dir() or 'dino' not in top.name.lower():
            continue
        for config in top.glob('**/config.json'):
            try:
                if 'dinov2' in config.read_text(errors='ignore').lower():
                    model_type = json.loads(config.read_text()).get('model_type', '')
                    if model_type == 'dinov2' and 'base' in str(config.parent).lower():
                        return config.parent
            except Exception:
                continue
    raise FileNotFoundError('offline DINOv2-base model is absent')

def _rt_binary_flag(value):
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        return int(value.strip().lower() in {'1', 'true', 'yes', 'y'})
    try:
        return int(float(value) > 0)
    except Exception:
        return 0

def _rt_plane_id(value):
    text = str(value).lower()
    if 'sag' in text:
        return 0
    if 'cor' in text:
        return 1
    if 'axi' in text or 'trans' in text or 'tra' == text.strip():
        return 2
    return 3

def _rt_assign_slots(series_df):
    x = series_df.copy()
    x['StudyInstanceUID'] = x['StudyInstanceUID'].astype(str)
    x['SeriesInstanceUID'] = x['SeriesInstanceUID'].astype(str)
    x['_plane_id'] = x['Anatomical_Plane'].map(_rt_plane_id)
    fluid = x['Fluid_Sensitive'].map(_rt_binary_flag)
    fat = x['Fat_Suppression'].map(_rt_binary_flag)
    x['_fluid_like'] = np.maximum(fluid.astype(int), fat.astype(int))
    slot_defs = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1))
    lookup = {}
    for study_uid, group in x.groupby('StudyInstanceUID', sort=False):
        group = group.sort_values(['SeriesInstanceUID']).copy()
        used = set()
        for slot_id, (plane, fluid_like) in enumerate(slot_defs):
            desired = group[(group['_plane_id'] == plane) & (group['_fluid_like'] == fluid_like) & ~group['SeriesInstanceUID'].isin(used)]
            if len(desired) == 0:
                desired = group[(group['_plane_id'] == plane) & ~group['SeriesInstanceUID'].isin(used)]
            if len(desired) == 0:
                continue
            series_uid = str(desired.iloc[0]['SeriesInstanceUID'])
            used.add(series_uid)
            lookup[str(study_uid), slot_id] = series_uid
    return lookup

def _rt_locate_series_dir(study_uid, series_uid):
    canonical = ROOT / 'test_series'
    candidates = (canonical / str(series_uid), canonical / str(study_uid) / str(series_uid), ROOT / 'test' / str(study_uid) / str(series_uid), ROOT / 'test_images' / str(study_uid) / str(series_uid), ROOT / 'test_dicom' / str(study_uid) / str(series_uid), ROOT / 'test_dicoms' / str(study_uid) / str(series_uid), ROOT / 'images' / 'test' / str(study_uid) / str(series_uid))
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f'report-teacher series missing: study={study_uid}, series={series_uid}')

def _rt_sorted_dicom_files(series_dir):
    files = list(Path(series_dir).glob('*.dcm'))
    if not files:
        files = [path for path in Path(series_dir).iterdir() if path.is_file()]
    if not files:
        raise FileNotFoundError(f'no DICOM files in {series_dir}')
    simple_numeric = [path.stem.isdigit() and len(path.stem) <= 8 for path in files]
    if np.mean(simple_numeric) >= 0.9:
        number_re = re.compile('(\\d+)')

        def key(path):
            matches = number_re.findall(path.stem)
            return int(matches[-1]) if matches else 10 ** 12
        return sorted(files, key=key)
    keyed = []
    for index, path in enumerate(files):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            if hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) >= 3:
                key = float(ds.ImagePositionPatient[2])
            else:
                key = float(getattr(ds, 'InstanceNumber', index))
        except Exception:
            key = float(index)
        keyed.append((key, path))
    return [path for _, path in sorted(keyed, key=lambda pair: pair[0])]

def _rt_robust_uint8(array):
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.uint8)
    values = array[finite]
    lo, hi = np.percentile(values, [1.0, 99.0])
    if hi <= lo:
        lo, hi = (float(values.min()), float(values.max()) + 1e-06)
    array = np.clip(array, lo, hi)
    array = (array - lo) / max(hi - lo, 1e-06)
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)

def _rt_center_crop_or_pad(image):
    height, width = image.shape[:2]
    pad_y, pad_x = (max(0, RT_IMG_SIZE - height), max(0, RT_IMG_SIZE - width))
    if pad_y or pad_x:
        top, left = (pad_y // 2, pad_x // 2)
        image = cv2.copyMakeBorder(image, top, pad_y - top, left, pad_x - left, borderType=cv2.BORDER_CONSTANT, value=0)
    height, width = image.shape[:2]
    y0, x0 = (max(0, (height - RT_IMG_SIZE) // 2), max(0, (width - RT_IMG_SIZE) // 2))
    return image[y0:y0 + RT_IMG_SIZE, x0:x0 + RT_IMG_SIZE]

def _rt_read_dicom(path):
    ds = pydicom.dcmread(str(path), force=True)
    array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0) or 1.0)
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0) or 0.0)
    image = _rt_robust_uint8(array * slope + intercept)
    if str(getattr(ds, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1':
        image = 255 - image
    spacing = getattr(ds, 'PixelSpacing', None)
    if spacing is not None and len(spacing) >= 2:
        try:
            scale_y = np.clip(float(spacing[0]) / RT_TARGET_SPACING, 0.4, 3.0)
            scale_x = np.clip(float(spacing[1]) / RT_TARGET_SPACING, 0.4, 3.0)
            new_h = max(32, int(round(image.shape[0] * scale_y)))
            new_w = max(32, int(round(image.shape[1] * scale_x)))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return _rt_center_crop_or_pad(image)
        except Exception:
            pass
    return cv2.resize(image, (RT_IMG_SIZE, RT_IMG_SIZE), interpolation=cv2.INTER_AREA)

def _rt_load_series_25d(study_uid, series_uid):
    files = _rt_sorted_dicom_files(_rt_locate_series_dir(study_uid, series_uid))
    quantiles = np.array([0.08, 0.23, 0.38, 0.5, 0.62, 0.77, 0.92], np.float32)
    centers = np.zeros(RT_SLICES, np.int64) if len(files) <= 1 else np.round(quantiles * (len(files) - 1)).astype(np.int64)
    centers = np.clip(centers, 0, len(files) - 1)
    views = []
    for center in centers:
        channels = []
        for index in (max(0, center - 1), center, min(len(files) - 1, center + 1)):
            try:
                channels.append(_rt_read_dicom(files[index]))
            except Exception:
                channels.append(np.zeros((RT_IMG_SIZE, RT_IMG_SIZE), dtype=np.uint8))
        views.append(np.stack(channels, axis=-1))
    return np.stack(views, axis=0)

def _rt_try_attached_visible_features(checkpoint_dir, expected_uids):
    uid_path = checkpoint_dir / 'rta_final_test_uids.txt'
    feature_path = checkpoint_dir / 'rta_final_test_features.npy'
    mask_path = checkpoint_dir / 'rta_final_test_slot_mask.npy'
    if not (uid_path.is_file() and feature_path.is_file() and mask_path.is_file()):
        return None
    if uid_path.read_text().splitlines() != list(expected_uids):
        return None
    features = np.load(feature_path, mmap_mode='r')
    mask = np.load(mask_path, mmap_mode='r')
    if features.shape[:3] != (len(expected_uids), 6, 7) or mask.shape != (len(expected_uids), 6):
        return None
    log('report-teacher: exact attached visible-test features reused')
    return (features, mask)

def _rt_extract_features(test_df, series_df, checkpoint_dir, dev):
    from transformers import AutoModel
    expected_uids = test_df['StudyInstanceUID'].astype(str).tolist()
    attached = _rt_try_attached_visible_features(checkpoint_dir, expected_uids)
    if attached is not None:
        return attached
    if time.time() - T0 > RT_START_CUTOFF_S:
        raise TimeoutError('insufficient runtime reserve for report-teacher feature extraction')
    dino_dir = _rt_find_dino_base()
    log(f'report-teacher: DINOv2 base from {dino_dir}')
    dino = AutoModel.from_pretrained(str(dino_dir), local_files_only=True).eval().to(dev)
    for parameter in dino.parameters():
        parameter.requires_grad_(False)
    dino_dim = int(dino.config.hidden_size)
    if dino_dim != 768:
        raise AssertionError(f'expected DINOv2-base hidden size 768, got {dino_dim}')
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    slot_lookup = _rt_assign_slots(series_df)
    features = np.zeros((len(expected_uids), 6, RT_SLICES, dino_dim * 2), np.float16)
    slot_mask = np.zeros((len(expected_uids), 6), bool)

    @torch.inference_mode()
    def encode(images):
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        tensor = (tensor - mean) / std
        parts = []
        for start in range(0, len(tensor), 8):
            batch = tensor[start:start + 8].to(dev, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                output = dino(pixel_values=batch, interpolate_pos_encoding=True)
                tokens = output.last_hidden_state
                part = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=-1)
            parts.append(part.float().cpu())
        return torch.cat(parts, dim=0).numpy().astype(np.float32)
    for row_index, study_uid in enumerate(expected_uids):
        if time.time() - T0 > RT_DEADLINE_S:
            raise TimeoutError('report-teacher deadline reached before submission overwrite')
        jobs = [(slot_id, slot_lookup[study_uid, slot_id]) for slot_id in range(6) if (study_uid, slot_id) in slot_lookup]

        def load_job(job):
            slot_id, series_uid = job
            return (slot_id, _rt_load_series_25d(study_uid, series_uid))
        if jobs:
            with ThreadPoolExecutor(max_workers=4) as executor:
                loaded = list(executor.map(load_job, jobs))
            encoded = encode(np.concatenate([views for _, views in loaded], axis=0))
            cursor = 0
            for slot_id, views in loaded:
                count = len(views)
                features[row_index, slot_id] = encoded[cursor:cursor + count].astype(np.float16)
                slot_mask[row_index, slot_id] = True
                cursor += count
        if row_index == 0 or (row_index + 1) % 100 == 0 or row_index + 1 == len(expected_uids):
            log(f'report-teacher features {row_index + 1}/{len(expected_uids)}')
        if dev.type == 'cuda' and (row_index + 1) % 100 == 0:
            torch.cuda.empty_cache()
    del dino
    gc.collect()
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return (features, slot_mask)

@torch.inference_mode()
def _rt_predict_checkpoints(features, slot_mask, checkpoint_dir, dev):
    syn_index = TARGETS.index('Synovitis')
    seed_predictions = []
    for seed in RT_SEEDS:
        seed_prediction = np.zeros(len(features), np.float32)
        for fold in range(4):
            if time.time() - T0 > RT_DEADLINE_S:
                raise TimeoutError('report-teacher deadline reached during checkpoint ensemble')
            path = checkpoint_dir / f'rta_final_seed{seed}_fold{fold}.pth'
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint.get('targets') != TARGETS:
                raise AssertionError(f'target order mismatch in {path.name}')
            cfg = checkpoint['cfg']
            model = RTAHMIL(in_dim=int(checkpoint['slice_feat_dim']), hidden_dim=int(cfg['hidden_dim']), n_targets=len(TARGETS), n_slots=int(cfg['n_slots']), n_slices=int(cfg['slices_per_series']), dropout=float(cfg['dropout']), series_dropout=float(cfg['series_dropout']))
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            model.eval().to(dev)
            fold_prediction = []
            for start in range(0, len(features), 48):
                x = torch.from_numpy(np.asarray(features[start:start + 48])).float().to(dev)
                mask = torch.from_numpy(np.asarray(slot_mask[start:start + 48])).bool().to(dev)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                    logits = model(x, mask)
                fold_prediction.append(torch.sigmoid(logits[:, syn_index]).float().cpu().numpy())
            seed_prediction += np.concatenate(fold_prediction) / 4.0
            del model, checkpoint
            gc.collect()
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
        seed_predictions.append(seed_prediction)
    return np.mean(np.stack(seed_predictions, axis=0), axis=0)

def _rt_blend_synovitis(primary, teacher_synovitis, teacher_uids):
    result = primary.copy()
    primary_uids = result['StudyInstanceUID'].astype(str)
    teacher = pd.Series(np.asarray(teacher_synovitis, dtype=np.float64), index=pd.Index([str(uid) for uid in teacher_uids], name='StudyInstanceUID'))
    if teacher.index.has_duplicates or set(primary_uids) != set(teacher.index):
        raise AssertionError('report-teacher and primary StudyInstanceUID sets differ')
    teacher = teacher.reindex(primary_uids.values)
    if not np.isfinite(teacher.values).all():
        raise AssertionError('non-finite report-teacher prediction')
    base_rank = result['Synovitis'].rank(pct=True).to_numpy(np.float64)
    teacher_rank = teacher.rank(pct=True).to_numpy(np.float64)
    result['Synovitis'] = (1.0 - RT_SYN_WEIGHT) * base_rank + RT_SYN_WEIGHT * teacher_rank
    return result

def run_report_teacher_synovitis_specialist():
    if time.time() - T0 > RT_START_CUTOFF_S:
        log('report-teacher skipped: the primary ensemble used its runtime reserve')
        return False
    checkpoint_dir = _rt_find_checkpoint_dir()
    primary_path = Path('submission.csv')
    primary = pd.read_csv(primary_path, dtype={'StudyInstanceUID': str})
    if primary.columns.tolist() != ['StudyInstanceUID'] + TARGETS:
        raise AssertionError('primary submission schema mismatch')
    test_df = pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str})
    series_df = pd.read_csv(ROOT / 'test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    expected_uids = test_df['StudyInstanceUID'].astype(str).tolist()
    dev = DEVS[0]
    features, slot_mask = _rt_extract_features(test_df, series_df, checkpoint_dir, dev)
    teacher_synovitis = _rt_predict_checkpoints(features, slot_mask, checkpoint_dir, dev)
    result = _rt_blend_synovitis(primary, teacher_synovitis, expected_uids)
    untouched = [target for target in TARGETS if target != 'Synovitis']
    if not result[untouched].equals(primary[untouched]):
        raise AssertionError('report-teacher changed a non-Synovitis target')
    if result.shape != primary.shape or not np.isfinite(result[TARGETS].to_numpy()).all():
        raise AssertionError('invalid report-teacher blend')
    temp_path = Path('submission_v26_synovitis.tmp.csv')
    result.to_csv(temp_path, index=False)
    reread = pd.read_csv(temp_path)
    if reread.shape != primary.shape or not np.isfinite(reread[TARGETS].to_numpy()).all():
        raise AssertionError('serialized report-teacher blend is invalid')
    temp_path.replace(primary_path)
    log('report-teacher complete: 0.75 Synovitis rank blend; all other targets preserved')
    return True

# ======== CELL 23 [code] ========
import base64
import gc
import hashlib
import io
import json
import math
import os
import random
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
import cv2
import joblib
import numpy as np
import pandas as pd
import pydicom
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
HYB_PREFIX = 'v8_hybrid_dino224_6slot_5pos_radiomics'
HYB_EXPECTED_TRAIN_ID_SHA256 = '21c1944bd15c3397290f0816de614ad4153f62e84c4bfb0e4d6147ac72084af8'
HYB_TEACHER_PAYLOAD = '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'
HYB_START_CUTOFF_S = 7.2 * 3600
HYB_DEADLINE_S = 8.72 * 3600
HYB_IMG_SIZE = 224
HYB_N_POSITIONS = 5
HYB_N_SLOTS = 6
HYB_SLOT_META_DIM = 8
HYB_STUDY_META_DIM = 13
HYB_RAD_DIM = 14
HYB_RAD_AGG_DIM = 56
HYB_SEEDS = (20260809, 20260810, 20260811, 20260814, 20260815, 20260816, 20260817, 20260818)
HYB_LM_FAMILY_WEIGHTS = (0.5227272727272728, 0.27272727272727276, 0.20454545454545456, 0.0)
HYB_FAMILIES = ('lr', 'et', 'hgb', 'exact_lr')
HYB_FAMILY_WEIGHTS = (0.46, 0.24, 0.18, 0.12)
HYB_LR_CS = (0.015, 0.05, 0.16, 0.5)
HYB_EXACT_LR_CS = (0.01, 0.03, 0.1, 0.3)
HYB_TARGET = 'Lateral Meniscus'
HYB_OA_TARGET = 'Lateral OA'
HYB_OA_SEEDS = (20260809, 20260810, 20260811, 20260812, 20260813, 20260814, 20260815, 20260816, 20260817, 20260818)
HYB_OA_FAMILY_WEIGHTS = (0.5227272727272728, 0.27272727272727276, 0.20454545454545456, 0.0)
HYB_PLANES = ('Sagittal', 'Coronal', 'Axial')
HYB_CONTRASTS = ('Fluid', 'Structural')
HYB_SLOT_NAMES = tuple((f'{plane}_{contrast}' for plane in HYB_PLANES for contrast in HYB_CONTRASTS))

def _hyb_required_cache_names():
    names = []
    for split in ('train', 'test'):
        stem = f'{split}_{HYB_PREFIX}_'
        names.extend((stem + suffix for suffix in ('slot_features.npy', 'slot_mask.npy', 'slot_meta.npy', 'radiomics.npy', 'study_meta.npy', 'sex.npy', 'ids.npy')))
    names.append(f'{HYB_PREFIX}_ipca_192.joblib')
    return names

def _hyb_find_cache_dir():
    local = globals().get('HYB_LOCAL_CACHE_DIR')
    candidates = []
    if local:
        candidates.append(Path(local))
    root = Path('/kaggle/input')
    if root.is_dir():
        marker = f'train_{HYB_PREFIX}_slot_features.npy'
        candidates.extend((hit.parent for hit in root.glob(f'*/{marker}')))
        candidates.extend((hit.parent for hit in root.glob(f'*/*/{marker}')))
    required = _hyb_required_cache_names()
    for path in candidates:
        if all(((path / name).is_file() for name in required)):
            return path
    raise FileNotFoundError('the complete public hybrid feature/PCA cache is absent')

def _hyb_find_dino_small():
    preferred = (Path('/kaggle/input/models/metaresearch/dinov2/pytorch/small/1'), Path('/kaggle/input/dinov2/pytorch/small/1'), Path('/kaggle/input/dinov2-small/pytorch/small/1'))
    for path in preferred:
        if (path / 'config.json').is_file():
            return path
    for top in Path('/kaggle/input').glob('*'):
        if not top.is_dir() or 'dino' not in top.name.lower():
            continue
        for config_path in top.glob('**/config.json'):
            try:
                cfg = json.loads(config_path.read_text())
                if cfg.get('model_type') == 'dinov2' and int(cfg.get('hidden_size', -1)) == 384:
                    return config_path.parent
            except Exception:
                continue
    raise FileNotFoundError('offline DINOv2-small model is absent')

def _hyb_numeric(value, default=0.0):
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except Exception:
        return default

def _hyb_sex_to_id(row):
    value = str(row.get('PatientSex', '')).strip().lower()
    if value.startswith('m'):
        return 1
    if value.startswith('f'):
        return 2
    return 0

def _hyb_choose_series(part, contrast, used_ids):
    if len(part) == 0:
        return None
    fluid = part['Fluid_Sensitive'].fillna(0).astype(float)
    fat = part['Fat_Suppression'].fillna(0).astype(float)
    if contrast == 'Fluid':
        score = 4.0 * fluid + 2.0 * fat
    else:
        score = 3.5 * (1.0 - fluid) + 1.5 * (1.0 - fat)
    ordered = part.assign(_slot_score=score).sort_values('_slot_score', ascending=False)
    for _, row in ordered.iterrows():
        series_id = str(row['SeriesInstanceUID'])
        if series_id not in used_ids:
            return row
    return ordered.iloc[0]

def _hyb_build_slots(series_df):
    slots, study_meta = ({}, {})
    for study_id, rows in series_df.groupby('StudyInstanceUID', sort=False):
        study_id = str(study_id)
        selected, meta = ({}, [])
        plane_lower = rows['Anatomical_Plane'].astype(str).str.lower()
        for plane in HYB_PLANES:
            part = rows[plane_lower == plane.lower()]
            count = len(part)
            fluid_mean = part['Fluid_Sensitive'].fillna(0).astype(float).mean() if count else 0.0
            fat_mean = part['Fat_Suppression'].fillna(0).astype(float).mean() if count else 0.0
            meta.extend([np.log1p(count) / 3.0, fluid_mean, fat_mean])
            used = set()
            for contrast in HYB_CONTRASTS:
                row = _hyb_choose_series(part, contrast, used)
                if row is None:
                    continue
                sid = str(row['SeriesInstanceUID'])
                used.add(sid)
                selected[f'{plane}_{contrast}'] = {'series_id': sid, 'contrast': contrast, 'fluid': _hyb_numeric(row.get('Fluid_Sensitive', 0)), 'fat': _hyb_numeric(row.get('Fat_Suppression', 0))}
        total = len(rows)
        meta.extend([np.log1p(total) / 4.0, rows['Fluid_Sensitive'].fillna(0).astype(float).mean() if total else 0.0, rows['Fat_Suppression'].fillna(0).astype(float).mean() if total else 0.0, rows['SeriesInstanceUID'].nunique() / 12.0 if total else 0.0])
        slots[study_id] = selected
        study_meta[study_id] = np.asarray(meta, dtype=np.float32)
    return (slots, study_meta)

def _hyb_locate_series_dir(study_uid, series_uid):
    candidates = (ROOT / 'test_series' / str(study_uid) / str(series_uid), ROOT / 'test_series' / str(series_uid), ROOT / 'test' / str(study_uid) / str(series_uid), ROOT / 'test_images' / str(study_uid) / str(series_uid), ROOT / 'test_dicom' / str(study_uid) / str(series_uid), ROOT / 'test_dicoms' / str(study_uid) / str(series_uid), ROOT / 'images' / 'test' / str(study_uid) / str(series_uid))
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f'hybrid series missing: study={study_uid}, series={series_uid}')

def _hyb_read_header(path):
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None

def _hyb_header_position(ds):
    if ds is None:
        return None
    try:
        ipp = np.asarray([float(x) for x in ds.ImagePositionPatient], dtype=np.float64)
        iop = np.asarray([float(x) for x in ds.ImageOrientationPatient], dtype=np.float64)
        return float(np.dot(ipp, np.cross(iop[:3], iop[3:])))
    except Exception:
        pass
    for name in ('SliceLocation', 'InstanceNumber'):
        try:
            return float(getattr(ds, name))
        except Exception:
            continue
    return None

@lru_cache(maxsize=8192)
def _hyb_ordered_files(folder_str):
    files = sorted(Path(folder_str).glob('*.dcm'))
    if not files:
        files = sorted((path for path in Path(folder_str).iterdir() if path.is_file()))
    keyed, ok = ([], 0)
    for fallback, path in enumerate(files):
        key = _hyb_header_position(_hyb_read_header(path))
        if key is None:
            key = fallback
        else:
            ok += 1
        keyed.append((key, str(path)))
    if ok >= max(3, len(files) // 3):
        keyed.sort(key=lambda pair: pair[0])
    return tuple((path for _, path in keyed))

def _hyb_spacing(ds):
    spacing_x = spacing_y = thickness = 0.0
    try:
        ps = [float(x) for x in ds.PixelSpacing]
        spacing_y, spacing_x = (ps[0], ps[1])
    except Exception:
        pass
    for name in ('SliceThickness', 'SpacingBetweenSlices'):
        try:
            thickness = float(getattr(ds, name))
            break
        except Exception:
            continue
    return (spacing_x, spacing_y, thickness)

def _hyb_read_pixel(path):
    ds = pydicom.dcmread(str(path), force=True)
    array = ds.pixel_array.astype(np.float32)
    array = array * _hyb_numeric(getattr(ds, 'RescaleSlope', 1.0), 1.0)
    array += _hyb_numeric(getattr(ds, 'RescaleIntercept', 0.0), 0.0)
    if str(getattr(ds, 'PhotometricInterpretation', '')).upper() == 'MONOCHROME1':
        array = array.max() - array
    return (array, ds)

def _hyb_robust_uint8(stack):
    stack = np.asarray(stack, dtype=np.float32)
    finite = stack[np.isfinite(stack)]
    if finite.size == 0:
        return np.zeros(stack.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.4])
    if high <= low:
        low, high = (float(finite.min()), float(finite.max()))
    if high <= low:
        return np.zeros(stack.shape, dtype=np.uint8)
    return (255.0 * np.clip((stack - low) / (high - low), 0.0, 1.0)).astype(np.uint8)

def _hyb_crop_foreground(image):
    gray = image.max(axis=2)
    mask = gray > max(8, np.percentile(gray, 55) * 0.18)
    if mask.sum() < 64:
        return image
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    pad_y, pad_x = (int(0.08 * (y1 - y0 + 1)), int(0.08 * (x1 - x0 + 1)))
    y0, y1 = (max(0, y0 - pad_y), min(image.shape[0], y1 + pad_y))
    x0, x1 = (max(0, x0 - pad_x), min(image.shape[1], x1 + pad_x))
    if y1 - y0 < 32 or x1 - x0 < 32:
        return image
    return image[y0:y1, x0:x1]

def _hyb_resize(image):
    return cv2.resize(image, (HYB_IMG_SIZE, HYB_IMG_SIZE), interpolation=cv2.INTER_AREA)

def _hyb_view_radiomics(image):
    gray = image.astype(np.float32).mean(axis=2) / 255.0
    height, width = gray.shape
    q = np.percentile(gray, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    center = gray[height // 4:3 * height // 4, width // 4:3 * width // 4]
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy)
    foreground = gray > 0.08
    return np.asarray([gray.mean(), gray.std(), q[0], q[2], q[4], q[6], q[8], center.mean(), center.std(), grad.mean(), grad.std(), foreground.mean(), gray[foreground].mean() if foreground.any() else 0.0, gray[foreground].std() if foreground.any() else 0.0], dtype=np.float32)

def _hyb_make_view(paths):
    arrays, first_ds = ([], None)
    for path in paths:
        try:
            array, ds = _hyb_read_pixel(path)
            first_ds = ds if first_ds is None else first_ds
            arrays.append(array)
        except Exception:
            return (None, np.zeros(HYB_RAD_DIM, np.float32), (0.0, 0.0, 0.0))
    image = np.transpose(_hyb_robust_uint8(np.stack(arrays)), (1, 2, 0))
    image = _hyb_resize(_hyb_crop_foreground(image))
    return (np.transpose(image, (2, 0, 1)), _hyb_view_radiomics(image), _hyb_spacing(first_ds) if first_ds is not None else (0.0, 0.0, 0.0))

def _hyb_sampled_triplets(study_uid, series_uid):
    files = list(_hyb_ordered_files(str(_hyb_locate_series_dir(study_uid, series_uid))))
    if not files:
        return ([], 0)
    centers = np.round(np.linspace(0.08 * (len(files) - 1), 0.92 * (len(files) - 1), HYB_N_POSITIONS)).astype(int)
    centers = np.clip(centers, 0, len(files) - 1)
    return ([[files[max(0, center - 1)], files[center], files[min(len(files) - 1, center + 1)]] for center in centers], len(files))

def _hyb_load_study(row, slots, study_meta):
    study_uid = str(row['StudyInstanceUID'])
    images = np.zeros((6, 5, 3, 224, 224), np.uint8)
    view_mask = np.zeros((6, 5), bool)
    slot_meta = np.zeros((6, 8), np.float32)
    radiomics = np.zeros((6, 56), np.float32)
    selected = slots.get(study_uid, {})
    for slot_index, slot_name in enumerate(HYB_SLOT_NAMES):
        info = selected.get(slot_name)
        if info is None:
            continue
        triplets, n_files = _hyb_sampled_triplets(study_uid, info['series_id'])
        rad_values, spacings = ([], [])
        for pos_index, paths in enumerate(triplets[:5]):
            image, rad, spacing = _hyb_make_view(paths)
            if image is None:
                continue
            images[slot_index, pos_index] = image
            view_mask[slot_index, pos_index] = True
            rad_values.append(rad)
            spacings.append(spacing)
        if rad_values:
            rad_array = np.stack(rad_values).astype(np.float32)
            radiomics[slot_index] = np.concatenate([rad_array.mean(0), rad_array.std(0), rad_array.min(0), rad_array.max(0)])
            spacing_mean = np.asarray(spacings, np.float32).mean(0)
        else:
            spacing_mean = np.zeros(3, np.float32)
        slot_meta[slot_index] = np.asarray([info['fluid'], info['fat'], float(info['contrast'] == 'Structural'), np.log1p(n_files) / 6.0, float(view_mask[slot_index].mean()), spacing_mean[0] / 2.5 if spacing_mean[0] else 0.0, spacing_mean[1] / 2.5 if spacing_mean[1] else 0.0, spacing_mean[2] / 8.0 if spacing_mean[2] else 0.0], np.float32)
    return (images, view_mask, slot_meta, radiomics, study_meta.get(study_uid, np.zeros(13, np.float32)), _hyb_sex_to_id(row), study_uid)

class _HybridDinoEncoder(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values):
        tokens = self.backbone(pixel_values=pixel_values).last_hidden_state
        patches = tokens[:, 1:]
        return torch.cat([F.normalize(tokens[:, 0], dim=1), F.normalize(patches.mean(dim=1), dim=1), F.normalize(patches.amax(dim=1), dim=1)], dim=1)

def _hyb_load_cached_split(cache_dir, split):
    stem = f'{split}_{HYB_PREFIX}_'
    return tuple((np.load(cache_dir / (stem + suffix), mmap_mode=None if suffix == 'ids.npy' else 'r', allow_pickle=suffix == 'ids.npy') for suffix in ('slot_features.npy', 'slot_mask.npy', 'slot_meta.npy', 'radiomics.npy', 'study_meta.npy', 'sex.npy', 'ids.npy')))

def _hyb_extract_test_bundle(test_df, series_df, cache_dir, dev):
    expected_uids = test_df['StudyInstanceUID'].astype(str).to_numpy()
    cached = _hyb_load_cached_split(cache_dir, 'test')
    if np.array_equal(np.asarray(cached[-1]).astype(str), expected_uids):
        log('hybrid: exact attached visible-test features reused')
        return cached
    if time.time() - T0 > HYB_START_CUTOFF_S:
        raise TimeoutError('insufficient runtime reserve for hybrid feature extraction')
    slots, study_meta = _hyb_build_slots(series_df)
    backbone = AutoModel.from_pretrained(str(_hyb_find_dino_small()), local_files_only=True, trust_remote_code=False)
    if int(backbone.config.hidden_size) != 384:
        raise AssertionError('the hybrid specialist requires DINOv2-small hidden size 384')
    model = _HybridDinoEncoder(backbone).eval().to(dev)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    @torch.inference_mode()
    def encode(images):
        outputs = []
        for start in range(0, len(images), 64):
            batch = images[start:start + 64].to(dev, non_blocking=True).float().div_(255.0)
            batch = (batch - mean) / std
            with torch.autocast('cuda', dtype=torch.float16, enabled=dev.type == 'cuda'):
                outputs.append(model(batch).float().cpu())
        return torch.cat(outputs, dim=0)
    n = len(test_df)
    features = np.zeros((n, 6, 3456), np.float16)
    masks = np.zeros((n, 6), bool)
    slot_meta_array = np.zeros((n, 6, 8), np.float16)
    radiomics_array = np.zeros((n, 6, 56), np.float16)
    study_meta_array = np.zeros((n, 13), np.float16)
    sexes = np.zeros(n, np.int8)
    ids = expected_uids.astype(object)

    def safe_load(index):
        try:
            return _hyb_load_study(test_df.iloc[index], slots, study_meta)
        except Exception as exc:
            uid = str(test_df.iloc[index]['StudyInstanceUID'])
            log(f'hybrid study decode failed safely: {uid}: {exc}')
            return (np.zeros((6, 5, 3, 224, 224), np.uint8), np.zeros((6, 5), bool), np.zeros((6, 8), np.float32), np.zeros((6, 56), np.float32), np.zeros(13, np.float32), 0, uid)
    workers = max(1, min(8, os.cpu_count() or 8))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, n, 6):
            if time.time() - T0 > HYB_DEADLINE_S:
                raise TimeoutError('hybrid deadline reached during feature extraction')
            stop = min(start + 6, n)
            items = list(executor.map(safe_load, range(start, stop)))
            image_batch = torch.from_numpy(np.stack([item[0] for item in items]))
            view_mask = torch.from_numpy(np.stack([item[1] for item in items]))
            valid = view_mask.reshape(-1)
            view_features = torch.zeros(len(items) * 30, 1152, dtype=torch.float32)
            if valid.any():
                flat_images = image_batch.reshape(-1, 3, 224, 224)
                view_features[valid] = encode(flat_images[valid])
            view_features = view_features.reshape(len(items), 6, 5, 1152)
            aggregate = torch.zeros(len(items), 6, 3456, dtype=torch.float32)
            slot_mask = view_mask.any(dim=2)
            for batch_index in range(len(items)):
                for slot_index in range(6):
                    present = view_mask[batch_index, slot_index]
                    if present.any():
                        values = view_features[batch_index, slot_index, present]
                        aggregate[batch_index, slot_index] = torch.cat([values.mean(0), values.amax(0), values.std(0, unbiased=False)])
            features[start:stop] = aggregate.numpy().astype(np.float16)
            masks[start:stop] = slot_mask.numpy()
            slot_meta_array[start:stop] = np.stack([item[2] for item in items]).astype(np.float16)
            radiomics_array[start:stop] = np.stack([item[3] for item in items]).astype(np.float16)
            study_meta_array[start:stop] = np.stack([item[4] for item in items]).astype(np.float16)
            sexes[start:stop] = np.asarray([item[5] for item in items], np.int8)
            if start == 0 or stop % 100 == 0 or stop == n:
                log(f'hybrid features {stop}/{n}')
    del model, backbone
    gc.collect()
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return (features, masks, slot_meta_array, radiomics_array, study_meta_array, sexes, ids)

def _hyb_align_bundle(bundle, expected_uids):
    ids = np.asarray(bundle[-1]).astype(str)
    expected_uids = np.asarray(expected_uids).astype(str)
    if np.array_equal(ids, expected_uids):
        return bundle
    if len(set(ids)) != len(ids) or set(ids) != set(expected_uids):
        raise AssertionError('hybrid cache StudyInstanceUID set mismatch')
    positions = {uid: index for index, uid in enumerate(ids)}
    order = np.asarray([positions[uid] for uid in expected_uids], dtype=int)
    return tuple((np.asarray(array)[order] for array in bundle[:-1])) + (expected_uids,)

def _hyb_transform(bundle, pca):
    features, masks, slot_meta, radiomics, study_meta, sex, _ = bundle
    n = len(features)
    result = np.zeros((n, 6, 192), np.float32)
    for start in range(0, n, 96):
        stop = min(start + 96, n)
        block = np.asarray(features[start:stop], np.float32)
        block_mask = np.asarray(masks[start:stop]).reshape(-1).astype(bool)
        flat = block.reshape(-1, block.shape[-1])
        transformed = np.zeros((len(flat), 192), np.float32)
        if block_mask.any():
            transformed[block_mask] = pca.transform(flat[block_mask]).astype(np.float32)
        result[start:stop] = transformed.reshape(stop - start, 6, 192)
    sex_onehot = np.eye(3, dtype=np.float32)[np.asarray(sex, dtype=int).clip(0, 2)]
    matrix = np.concatenate([result.reshape(n, -1), np.asarray(masks, np.float32), np.asarray(slot_meta, np.float32).reshape(n, -1), np.asarray(radiomics, np.float32).reshape(n, -1), np.asarray(study_meta, np.float32), sex_onehot], axis=1)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.shape != (n, 1558):
        raise AssertionError(f'unexpected hybrid matrix shape {matrix.shape}')
    return matrix.astype(np.float32)

def _hyb_rank(values, denominator_offset=0.0):
    values = np.asarray(values, np.float64)
    if len(values) <= 1 or np.ptp(values) < 1e-12:
        return np.full(len(values), 0.5, np.float64)
    return rankdata(values, method='average') / (len(values) + denominator_offset)

def _hyb_payload():
    raw = zlib.decompress(base64.b64decode(HYB_TEACHER_PAYLOAD.encode('ascii')))
    with np.load(io.BytesIO(raw), allow_pickle=False) as payload:
        return {name: payload[name].astype(np.float32) for name in payload.files}

def _hyb_select_top(indices, scores, labels, class_value, cap=2200):
    keep = indices[labels == class_value]
    if len(keep) <= cap:
        return keep
    keep_scores = scores[labels == class_value]
    return keep[np.argsort(-keep_scores)[:cap]]

def _hyb_training_arrays(pseudo_y, pseudo_conf, exact_mask, exact_y):
    exact_idx = np.flatnonzero(exact_mask)
    pseudo_pool = ~exact_mask
    confidence = np.clip(pseudo_conf, 0.0, 1.0)
    candidates = np.flatnonzero(pseudo_pool & (confidence >= 0.2))
    candidate_labels = (pseudo_y[candidates] >= 0.5).astype(int)
    pos = _hyb_select_top(candidates, confidence[candidates], candidate_labels, 1)
    neg = _hyb_select_top(candidates, confidence[candidates], candidate_labels, 0)
    pseudo_idx = np.concatenate([pos, neg]).astype(int)
    pseudo_labels = (pseudo_y[pseudo_idx] >= 0.5).astype(int)
    pseudo_weight = 0.18 + 1.15 * np.power(np.clip(confidence[pseudo_idx], 0, 1), 1.4)
    fit_idx = np.concatenate([exact_idx, pseudo_idx]).astype(int)
    fit_y = np.concatenate([exact_y[exact_idx].astype(int), pseudo_labels]).astype(int)
    fit_weight = np.concatenate([np.full(len(exact_idx), 7.0, np.float32), pseudo_weight.astype(np.float32)])
    return (fit_idx, fit_y, fit_weight, exact_idx, exact_y[exact_idx].astype(int))

def _hyb_fit_lr(x_fit, y_fit, x_test, weights, cs, seed):
    predictions = []
    for index, c_value in enumerate(cs):
        model = LogisticRegression(C=c_value, solver='liblinear', class_weight='balanced', max_iter=3000, random_state=seed + 31 * index)
        model.fit(x_fit, y_fit, sample_weight=weights)
        predictions.append(model.predict_proba(x_test)[:, 1])
    return np.mean(predictions, axis=0).astype(np.float32)

def _hyb_fit_family(family, x_train, fit_idx, fit_y, fit_weight, x_test, seed):
    if time.time() - T0 > HYB_DEADLINE_S:
        raise TimeoutError('hybrid deadline reached during model fitting')
    if family == 'lr':
        return _hyb_fit_lr(x_train[fit_idx], fit_y, x_test, fit_weight, HYB_LR_CS, seed)
    if family == 'exact_lr':
        return _hyb_fit_lr(x_train[fit_idx], fit_y, x_test, fit_weight, HYB_EXACT_LR_CS, seed)
    if family == 'et':
        model = ExtraTreesClassifier(n_estimators=420, max_features='sqrt', min_samples_leaf=4, min_samples_split=8, bootstrap=False, class_weight='balanced', random_state=seed, n_jobs=-1)
    elif family == 'hgb':
        model = HistGradientBoostingClassifier(learning_rate=0.035, max_iter=180, max_leaf_nodes=15, min_samples_leaf=18, l2_regularization=0.25, early_stopping=True, validation_fraction=0.15, random_state=seed)
    else:
        raise ValueError(f'unknown hybrid family {family}')
    model.fit(x_train[fit_idx], fit_y, sample_weight=fit_weight)
    return model.predict_proba(x_test)[:, 1].astype(np.float32)

def _hyb_teacher_arm(x_train, x_test, pseudo_y, pseudo_conf, exact_mask, exact_y, exact_lr):
    fit_idx, fit_y, fit_weight, _, _ = _hyb_training_arrays(pseudo_y, pseudo_conf, exact_mask, exact_y)
    predictions = {}
    for family in ('lr', 'et', 'hgb'):
        seed_predictions = []
        for seed in HYB_SEEDS:
            model_seed = seed + 101 * TARGETS.index(HYB_TARGET) + len(family)
            seed_predictions.append(_hyb_fit_family(family, x_train, fit_idx, fit_y, fit_weight, x_test, model_seed))
        predictions[family] = np.mean(np.stack(seed_predictions), axis=0)
    predictions['exact_lr'] = exact_lr
    weighted_rank = np.zeros(len(x_test), np.float64)
    weighted_prob = np.zeros(len(x_test), np.float64)
    for weight, family in zip(HYB_FAMILY_WEIGHTS, HYB_FAMILIES):
        pred = np.clip(predictions[family], 1e-05, 1.0 - 1e-05)
        weighted_rank += weight * _hyb_rank(pred, denominator_offset=1.0)
        weighted_prob += weight * pred
    return 0.9 * weighted_rank + 0.1 * weighted_prob

def run_hybrid_lm_and_lateral_oa_specialists():
    if time.time() - T0 > HYB_START_CUTOFF_S:
        log('hybrid skipped: insufficient runtime reserve')
        return False
    cache_dir = _hyb_find_cache_dir()
    primary_path = Path('submission.csv')
    primary = pd.read_csv(primary_path, dtype={'StudyInstanceUID': str})
    train_df = pd.read_csv(ROOT / 'train.csv', dtype={'StudyInstanceUID': str})
    test_df = pd.read_csv(ROOT / 'test.csv', dtype={'StudyInstanceUID': str})
    series_df = pd.read_csv(ROOT / 'test_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
    if primary.columns.tolist() != ['StudyInstanceUID'] + TARGETS:
        raise AssertionError('primary submission schema mismatch')
    train_uids = train_df['StudyInstanceUID'].astype(str).to_numpy()
    uid_hash = hashlib.sha256('\n'.join(train_uids).encode()).hexdigest()
    if uid_hash != HYB_EXPECTED_TRAIN_ID_SHA256:
        raise AssertionError('competition train StudyInstanceUID order drifted')
    test_uids = test_df['StudyInstanceUID'].astype(str).to_numpy()
    dev = DEVS[0]
    train_bundle = _hyb_align_bundle(_hyb_load_cached_split(cache_dir, 'train'), train_uids)
    test_bundle = _hyb_align_bundle(_hyb_extract_test_bundle(test_df, series_df, cache_dir, dev), test_uids)
    pca = joblib.load(cache_dir / f'{HYB_PREFIX}_ipca_192.joblib')
    x_train_raw = _hyb_transform(train_bundle, pca)
    x_test_raw = _hyb_transform(test_bundle, pca)
    joint = np.concatenate([x_train_raw, x_test_raw], axis=0)
    mean = joint.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = joint.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-05] = 1.0
    x_train = ((x_train_raw - mean) / scale).astype(np.float32)
    x_test = ((x_test_raw - mean) / scale).astype(np.float32)
    payload = _hyb_payload()
    if any((len(payload[name]) != len(train_df) for name in payload)):
        raise AssertionError('embedded hybrid teacher length mismatch')

    def fit_target(target_name, pseudo_y, pseudo_conf, seeds):
        exact_mask = train_df[target_name].notna().to_numpy()
        exact_y = np.nan_to_num(train_df[target_name].to_numpy(np.float32), nan=0.0)
        exact_idx = np.flatnonzero(exact_mask)
        exact_labels = exact_y[exact_idx].astype(int)
        exact_predictions = []
        for seed in seeds:
            model_seed = seed + 101 * TARGETS.index(target_name) + len('exact_lr')
            exact_predictions.append(_hyb_fit_family('exact_lr', x_train, exact_idx, exact_labels, np.full(len(exact_idx), 3.0, np.float32), x_test, model_seed))
        exact_lr = np.mean(np.stack(exact_predictions), axis=0)
        fit_idx, fit_y, fit_weight, _, _ = _hyb_training_arrays(pseudo_y, pseudo_conf, exact_mask, exact_y)
        predictions = {}
        for family in ('lr', 'et', 'hgb'):
            seed_predictions = []
            for seed in seeds:
                model_seed = seed + 101 * TARGETS.index(target_name) + len(family)
                seed_predictions.append(_hyb_fit_family(family, x_train, fit_idx, fit_y, fit_weight, x_test, model_seed))
            predictions[family] = np.mean(np.stack(seed_predictions), axis=0)
        predictions['exact_lr'] = exact_lr
        weighted_rank = np.zeros(len(x_test), np.float64)
        weighted_prob = np.zeros(len(x_test), np.float64)
        family_weights = HYB_LM_FAMILY_WEIGHTS if target_name == HYB_TARGET else HYB_OA_FAMILY_WEIGHTS
        for weight, family in zip(family_weights, HYB_FAMILIES):
            pred = np.clip(predictions[family], 1e-05, 1.0 - 1e-05)
            weighted_rank += weight * _hyb_rank(pred, denominator_offset=1.0)
            weighted_prob += weight * pred
        return 0.9 * weighted_rank + 0.1 * weighted_prob
    lm_consensus = fit_target(HYB_TARGET, payload['lm_consensus_y'], payload['lm_consensus_conf'], HYB_SEEDS)
    lm_pilkwang = fit_target(HYB_TARGET, payload['lm_pilkwang_y'], payload['lm_pilkwang_conf'], HYB_SEEDS)
    lm_teacher_rank = 1.0 * _hyb_rank(lm_consensus) + 0.0 * _hyb_rank(lm_pilkwang)
    oa_pilkwang = fit_target(HYB_OA_TARGET, payload['oa_pilkwang_y'], payload['oa_pilkwang_conf'], HYB_OA_SEEDS)
    oa_teacher_rank = _hyb_rank(oa_pilkwang)
    train_mean = x_train_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    train_scale = x_train_raw.std(axis=0, dtype=np.float64).astype(np.float32)
    train_scale[train_scale < 1e-05] = 1.0
    x_train_pca = ((x_train_raw - train_mean) / train_scale).astype(np.float32)
    x_test_pca = ((x_test_raw - train_mean) / train_scale).astype(np.float32)
    pca_specs = {HYB_TARGET: (128, 0.1), 'Synovitis': (32, 1.0)}
    pca_predictions = {target: [] for target in pca_specs}
    for pca_seed in (20260809, 20260819, 20260829, 20260839):
        decomposition = PCA(n_components=128, whiten=True, svd_solver='randomized', n_oversamples=20, random_state=pca_seed)
        train_embedding = decomposition.fit_transform(x_train_pca)
        test_embedding = decomposition.transform(x_test_pca)
        for target_name, (dimensions, c_value) in pca_specs.items():
            exact_mask = train_df[target_name].notna().to_numpy()
            exact_labels = train_df.loc[exact_mask, target_name].to_numpy(int)
            if int(exact_mask.sum()) != 58 or set(np.unique(exact_labels)) != {0, 1}:
                raise AssertionError(f'unexpected exact-label support for {target_name}')
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, solver='liblinear', class_weight='balanced', max_iter=5000, random_state=pca_seed))
            model.fit(train_embedding[exact_mask, :dimensions], exact_labels)
            pca_predictions[target_name].append(model.predict_proba(test_embedding[:, :dimensions])[:, 1])
    pca_predictions = {target: np.mean(np.stack(predictions), axis=0) for target, predictions in pca_predictions.items()}
    result = primary.copy()
    primary_uids = result['StudyInstanceUID'].astype(str)
    if set(primary_uids) != set(test_uids):
        raise AssertionError('hybrid and primary StudyInstanceUID sets differ')
    lm_by_uid = pd.Series(lm_teacher_rank, index=test_uids).reindex(primary_uids.values)
    oa_by_uid = pd.Series(oa_teacher_rank, index=test_uids).reindex(primary_uids.values)
    lm_pca_by_uid = pd.Series(_hyb_rank(pca_predictions[HYB_TARGET]), index=test_uids).reindex(primary_uids.values)
    syn_pca_by_uid = pd.Series(_hyb_rank(pca_predictions['Synovitis']), index=test_uids).reindex(primary_uids.values)
    lm_base_rank = result[HYB_TARGET].rank(pct=True).to_numpy(np.float64)
    oa_base_rank = result[HYB_OA_TARGET].rank(pct=True).to_numpy(np.float64)
    syn_base_rank = result['Synovitis'].rank(pct=True).to_numpy(np.float64)
    result[HYB_TARGET] = 0.125 * lm_base_rank + 0.5 * lm_by_uid.to_numpy(np.float64) + 0.375 * lm_pca_by_uid.to_numpy(np.float64)
    result[HYB_OA_TARGET] = 0.125 * oa_base_rank + 0.875 * oa_by_uid.to_numpy(np.float64)
    result['Synovitis'] = 0.75 * syn_base_rank + 0.25 * syn_pca_by_uid.to_numpy(np.float64)
    changed = {HYB_TARGET, HYB_OA_TARGET, 'Synovitis'}
    untouched = [target for target in TARGETS if target not in changed]
    if not result[untouched].equals(primary[untouched]):
        raise AssertionError('hybrid changed an unevaluated target')
    if result.shape != primary.shape or not np.isfinite(result[TARGETS].to_numpy()).all():
        raise AssertionError('invalid hybrid blend')
    temp_path = Path('submission_v34_hybrid.tmp.csv')
    result.to_csv(temp_path, index=False)
    reread = pd.read_csv(temp_path)
    if reread.shape != primary.shape or not np.isfinite(reread[TARGETS].to_numpy()).all():
        raise AssertionError('serialized hybrid blend is invalid')
    temp_path.replace(primary_path)
    log('hybrid complete: LM 0.125 base / 0.500 consensus / 0.375 PCA; Lateral OA 0.125 base / 0.875 Pilkwang; Synovitis 0.75 base / 0.25 PCA; nine other targets preserved')
    return True

# ======== CELL 24 [code] ========
def write_submission(pred, studies, test_df, path):
    sub = pd.DataFrame(pd.DataFrame(pred).rank(pct=True).values, columns=TARGETS)
    sub.insert(0, 'StudyInstanceUID', studies)
    sub = test_df[['StudyInstanceUID']].merge(sub, on='StudyInstanceUID', how='left')
    sub[TARGETS] = sub[TARGETS].fillna(0.5)
    sub.to_csv(path, index=False)
    return sub

def write_benchmark_submission():
    t = pd.read_csv(ROOT / 'test.csv')
    for c in TARGETS:
        t[c] = 0.5
    t.to_csv('submission.csv', index=False)

def _v37_validate_submission(path, test_df, tag):
    path = Path(path)
    frame = pd.read_csv(path)
    expected = ['StudyInstanceUID'] + TARGETS
    if list(frame.columns) != expected:
        raise ValueError(f'{tag}: columns differ from the competition contract')
    if len(frame) != len(test_df) or not frame['StudyInstanceUID'].is_unique:
        raise ValueError(f'{tag}: row count or StudyInstanceUID uniqueness failed')
    if set(frame['StudyInstanceUID'].astype(str)) != set(test_df['StudyInstanceUID'].astype(str)):
        raise ValueError(f'{tag}: StudyInstanceUID set differs from test.csv')
    values = frame[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f'{tag}: non-finite prediction')
    return test_df[['StudyInstanceUID']].merge(frame, on='StudyInstanceUID', how='left')

def _v37_find_yash_submission():
    candidates = []
    local = globals().get('YASH_LOCAL_SOURCE_DIR')
    if local:
        candidates.append(Path(local) / 'submission.csv')
    root = Path('/kaggle/input')
    candidates.append(root / 'rsna-knee-infer-v1' / 'submission.csv')
    if root.is_dir():
        candidates.extend((meta.parent / 'submission.csv' for meta in root.glob('**/infer_meta.json')))
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        meta_path = path.with_name('infer_meta.json')
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            if int(meta.get('errors', -1)) != 0:
                raise ValueError(f"Yash source reports {meta.get('errors')} inference errors")
        return path
    raise FileNotFoundError('the attached yashbishnoi98/rsna-knee-infer-v1 output is absent')

def run_yash_public_ensemble():
    import shutil
    test_df = pd.read_csv(ROOT / 'test.csv')
    native_path = Path('submission.csv')
    public_path = Path('submission_public_0899.csv')
    native = _v37_validate_submission(native_path, test_df, 'native V36')
    public = _v37_validate_submission(public_path, test_df, 'public DINO family')
    yash_path = _v37_find_yash_submission()
    yash = _v37_validate_submission(yash_path, test_df, 'Yash public image family')
    meta_path = yash_path.with_name('infer_meta.json')
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        if int(meta.get('studies', -1)) != len(test_df):
            raise ValueError('Yash source study count differs from test.csv')
    shutil.copyfile(native_path, 'submission_native_v36.csv')
    shutil.copyfile(yash_path, 'submission_yash_reference.csv')
    yr = yash[TARGETS].rank(pct=True).to_numpy(np.float64)
    dr = public[TARGETS].rank(pct=True).to_numpy(np.float64)
    blend = 0.55 * yr + 0.45 * dr
    result = test_df[['StudyInstanceUID']].copy()
    result[TARGETS] = blend
    if result.shape != yash.shape or not np.isfinite(result[TARGETS].to_numpy()).all():
        raise AssertionError('invalid Yash/DINO rank blend')
    candidate_path = Path('submission_yash_dino_rankblend.csv')
    result.to_csv(candidate_path, index=False)
    reread = _v37_validate_submission(candidate_path, test_df, 'Yash/DINO rank blend')
    changed = sum((tuple(reread[target].rank(method='first')) != tuple(yash[target].rank(method='first')) for target in TARGETS))
    if changed == 0:
        raise AssertionError('Yash/DINO blend is rank-identical to its Yash parent')
    temp_path = Path('submission_v37_yash_dino.tmp.csv')
    reread.to_csv(temp_path, index=False)
    temp_path.replace(native_path)
    log(f'Yash public family banked; V37 primary = 0.55 Yash / 0.45 public DINO rank blend ({changed} target orderings differ from Yash); exact Yash and native V36 outputs retained')
    return True

def main():
    write_benchmark_submission()
    pkg = find_weights()
    if pkg is not None:
        dev = DEVS[0]
        infer_from_package(pkg, dev)
        try:
            test_df = pd.read_csv(ROOT / 'test.csv')
            native_path = Path('submission.csv')
            public_path = Path('submission_public_0899.csv')
            native = _v37_validate_submission(native_path, test_df, 'native 24-member')
            public = _v37_validate_submission(public_path, test_df, 'public DINO frontier')
            native.to_csv('submission_native_v38.csv', index=False)
            public.to_csv(native_path, index=False)
            promoted = _v37_validate_submission(native_path, test_df, 'V40 primary')
            if not promoted.equals(public):
                raise AssertionError('V40 serialization differs from validated public frontier')
            log('V40 primary = exact no-jitter public-frontier target pooling; native 24-member output retained')
        except Exception as public_frontier_error:
            log(f'public-frontier promotion skipped safely: {public_frontier_error}')
            traceback.print_exc()
        log('done')
        return
    read_labels(pd.read_csv(ROOT / 'train.csv', usecols=['StudyInstanceUID', 'Report']))
    test_df = pd.read_csv(ROOT / 'test.csv')
    test_series = pd.read_csv(ROOT / 'test_series.csv')
    train_df = pd.read_csv(ROOT / 'train.csv')
    train_series = pd.read_csv(ROOT / 'train_series.csv')
    log(f'train {train_df.shape} test {test_df.shape}')
    both = pd.concat([train_series, test_series])
    plane_map = dict(zip(both['SeriesInstanceUID'], both['Anatomical_Plane']))
    log('header pass: test')
    hte = annotate(walk('test_series'))
    log(f'  {len(hte)} test series')
    log('header pass: train')
    htr = annotate(walk('train_series'))
    log(f'  {len(htr)} train series')
    slots_te, slots_tr = (pick_slots(hte, plane_map), pick_slots(htr, plane_map))
    cov = pd.Series([len(v) for v in slots_tr.values()]).describe()
    log(f"train slots per study: mean {cov['mean']:.2f} min {cov['min']:.0f} max {cov['max']:.0f}")
    st_tr, Ctr, Mtr = build_cache(slots_tr, plane_map, lat_of(htr, 'train '), 'train')
    st_te, Cte, Mte = build_cache(slots_te, plane_map, lat_of(hte, 'test '), 'test')
    t_lab = time.time()
    lab = read_labels(train_df)
    log(f'derived labels for {len(lab)} studies in {time.time() - t_lab:.1f}s')
    gold = train_df.set_index('StudyInstanceUID')[TARGETS]
    gold = gold[gold.notna().all(axis=1)]
    Y = np.zeros((len(st_tr), len(TARGETS)), np.float32)
    W = np.zeros_like(Y)
    for i, st in enumerate(st_tr):
        if st in gold.index:
            Y[i], W[i] = (gold.loc[st].values, 3.0)
        elif st in lab.index:
            r = lab.loc[st]
            Y[i] = r[TARGETS].values
            W[i] = 0.25 + 0.75 * r[[t + '__conf' for t in TARGETS]].values
    keep = np.where(W.sum(1) > 0)[0]
    log(f'supervised {len(keep)} of {len(st_tr)} studies (annotated {len(gold)})')
    import hashlib
    rep = train_df.set_index('StudyInstanceUID')['Report'].fillna('')
    grp = np.array([int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % 5 for s in st_tr])
    va = np.array([i for i in keep if grp[i] == 0])
    tr = np.array([i for i in keep if grp[i] != 0])
    if len(va) == 0 or len(tr) < BATCH_STUDIES:
        cut = max(1, len(keep) // 5)
        va, tr = (keep[:cut], keep[cut:])
    log(f'train {len(tr)} / holdout {len(va)} studies')
    gpos = {s: i for i, s in enumerate(st_tr)}
    va_set = set(va.tolist())
    gi = np.array([gpos[s] for s in gold.index if s in gpos and gpos[s] in va_set])
    gold_y = gold.loc[[st_tr[i] for i in gi]].values.astype(int) if len(gi) else None
    yv = (Y[va] > 0.5).astype(int)
    log(f'annotation check: {len(gi)} of {len(gold)} annotated studies are in the holdout')
    dev = DEVS[0]
    results, test_preds = ({}, {})
    for cfg in RUNS:
        pitch = CROP_MM / cfg['img']
        log(f"=== {cfg['name']}: {cfg['img']} px, {pitch:.3f} mm/pixel, {pitch * 14:.2f} mm per patch token ===")
        torch.manual_seed(SEED)
        model = build_model(UNFREEZE_LAST).to(dev)
        opt = torch.optim.AdamW([{'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': LR_BACKBONE}, {'params': model.head.parameters(), 'lr': LR_HEAD}], weight_decay=WEIGHT_DECAY)
        steps = max(EPOCHS * (len(tr) // BATCH_STUDIES), 1)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_BACKBONE, LR_HEAD], total_steps=steps, pct_start=0.15)
        scaler = torch.amp.GradScaler('cuda', enabled=dev.type == 'cuda')
        best, best_state, best_annot = (-1.0, None, float('nan'))
        for ep in range(EPOCHS):
            model.train()
            perm = np.random.permutation(tr)
            tot, nstep = (0.0, 0)
            for b in range(0, len(perm) - BATCH_STUDIES + 1, BATCH_STUDIES):
                sel = perm[b:b + BATCH_STUDIES]
                rows = torch.from_numpy(Ctr[sel]).to(dev)
                g = int(torch.randint(N_GROUP, (1,)).item())
                imgs = augment(take_group(rows, g))
                m = torch.from_numpy(Mtr[sel]).to(dev)
                y = torch.from_numpy(Y[sel]).to(dev)
                w = torch.from_numpy(W[sel]).to(dev)
                with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                    loss = (F.binary_cross_entropy_with_logits(model(imgs, m, cfg['img']), y, reduction='none') * w).mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item()
                nstep += 1
            pv = predict(model, Ctr, Mtr, va, dev, cfg['img'])
            d = macro_auc(yv, pv)
            g_auc = float('nan')
            if gold_y is not None and len(gi):
                g_auc = macro_auc(gold_y, predict(model, Ctr, Mtr, gi, dev, cfg['img']))
            log(f'  epoch {ep + 1}/{EPOCHS}  loss {tot / max(nstep, 1):.4f}  holdout {d:.4f}  annot(n={len(gi)}) {g_auc:.4f}')
            if d > best:
                best, best_annot = (d, g_auc)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if time.time() - T0 > TIME_BUDGET:
                log('  time budget reached')
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        results[cfg['name']] = (best, best_annot)
        test_preds[cfg['name']] = predict(model, Cte, Mte, np.arange(len(st_te)), dev, cfg['img'])
        log(f"  {cfg['name']}: best holdout {best:.4f} (annot {best_annot:.4f})")
        del model, opt, sched, scaler, best_state
        gc.collect()
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
    log('---- summary ----')
    for n, (d, g_auc) in results.items():
        log(f'  {n:12s} holdout {d:.4f}   annot {g_auc:.4f}')
    pick = max(results, key=lambda k: results[k][0])
    log(f'best on the holdout: {pick} ({results[pick][0]:.4f})')
    for name, pred in test_preds.items():
        sub = write_submission(pred, st_te, test_df, f'submission_{name}.csv')
        log(f'  submission_{name}.csv {sub.shape}; nulls {int(sub[TARGETS].isna().sum().sum())}')
    ens = np.mean([pd.DataFrame(p).rank(pct=True).values for p in test_preds.values()], axis=0)
    write_submission(ens, st_te, test_df, 'submission_rankmean.csv')
    log(f'  submission_rankmean.csv (rank mean of {len(test_preds)})')
    sub = write_submission(test_preds[pick], st_te, test_df, 'submission.csv')
    log(f'submission.csv = {pick}; {sub.shape}; nulls {int(sub[TARGETS].isna().sum().sum())}')
    print(sub.head().to_string())

# ======== CELL 25 [code] ========
try:
    main()
except LabelSourceError:
    traceback.print_exc()
    raise
except Exception:
    traceback.print_exc()
    t = pd.read_csv(find_root() / 'test.csv')
    for c in TARGETS:
        t[c] = 0.5
    t.to_csv('submission.csv', index=False)
    print('wrote fallback submission.csv')
log('done')

# ======== CELL 26 [code] ========
_A5_SAVED = dict(globals())
import gc, os, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings('ignore')
cv2.setNumThreads(1)
CROP_MM = 130.0
SIZE = 336
SLICE_BAND = (0.12, 0.88)
N_SLICE = 16
INTENSITY = 'slice'
SLOTS = [('Sagittal', 1), ('Sagittal', 0), ('Coronal', 1), ('Coronal', 0), ('Axial', 1), ('Axial', 0)]
N_SLOT = len(SLOTS)
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']

def _find_dir(*names):
    root = Path('/kaggle/input')
    cand = []
    for n in names:
        cand += [root / n, root / 'competitions' / n, root / 'datasets' / n]
        for parent in (root / 'datasets', root / 'competitions', root):
            if parent.is_dir():
                try:
                    cand += [d / n for d in parent.iterdir() if d.is_dir()]
                except OSError:
                    pass
    for p in cand:
        if p.is_dir():
            return p
    return None
COMP = _find_dir('rsna-knee-abnormality-detection')
CKPT = _find_dir('knee-mri-fold-weights')
assert COMP is not None, 'competition data not attached'
assert CKPT is not None, 'fold weights not attached'
assert (COMP / 'sample_submission.csv').exists(), f'no competition data at {COMP}'
assert list(CKPT.glob('*_f*.pt')), f'no checkpoints at {CKPT}'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'competition : {COMP}')
print(f'checkpoints : {CKPT}')
print(f'device      : {DEV}')
for i in range(torch.cuda.device_count() if DEV == 'cuda' else 0):
    cc = torch.cuda.get_device_capability(i)
    print(f'  gpu{i}       : {torch.cuda.get_device_name(i)} sm_{cc[0]}{cc[1]}, {torch.cuda.get_device_properties(i).total_memory / 2 ** 30:.0f} GiB, native bf16={cc >= (8, 0)}')

# ======== CELL 27 [code] ========
SERIES_ROOT = COMP / 'test_series'
if not SERIES_ROOT.exists():
    SERIES_ROOT = COMP / 'train_series'
print('series root:', SERIES_ROOT)

def ordered_files(sdir, cap=64):
    keyed = []
    for f in sdir.glob('*.dcm'):
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            keyed.append((int(ds.InstanceNumber), str(f)))
        except Exception:
            continue
        if len(keyed) >= cap * 4:
            break
    return [f for _, f in sorted(keyed)]

def series_side(path):
    try:
        return float(pydicom.dcmread(path, stop_before_pixels=True).ImagePositionPatient[0])
    except Exception:
        return 0.0

def read_crop(path):
    try:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return None
    try:
        ps = float(ds.PixelSpacing[0])
    except Exception:
        ps = CROP_MM / max(arr.shape)
    half = int(round(CROP_MM / ps / 2))
    cy, cx = (arr.shape[0] // 2, arr.shape[1] // 2)
    y0, y1 = (max(0, cy - half), min(arr.shape[0], cy + half))
    x0, x1 = (max(0, cx - half), min(arr.shape[1], cx + half))
    crop = arr[y0:y1, x0:x1]
    return None if crop.size == 0 else crop

def window(crop, lo, hi, flip):
    c = np.clip((crop - lo) / max(hi - lo, 1e-06), 0, 1)
    img = cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return img[:, ::-1].copy() if flip else img

def render(path, flip):
    crop = read_crop(path)
    if crop is None:
        return None
    lo, hi = np.percentile(crop[::4, ::4], [1, 99])
    return window(crop, lo, hi, flip)

def build_study(args):
    idx, study, recs = args
    out = np.zeros((N_SLOT, N_SLICE, SIZE, SIZE), np.uint8)
    mask = np.zeros(N_SLOT, np.uint8)
    rows = pd.DataFrame(recs)
    if len(rows):
        for s_i, (plane, fs) in enumerate(SLOTS):
            sub = rows[(rows.Anatomical_Plane == plane) & (rows.Fat_Suppression == fs)]
            if sub.empty:
                continue
            files = ordered_files(SERIES_ROOT / study / sub.iloc[0].SeriesInstanceUID)
            if not files:
                continue
            flip = plane != 'Sagittal' and series_side(files[0]) < 0
            lo, hi = SLICE_BAND
            i0 = int(round(lo * (len(files) - 1)))
            i1 = int(round(hi * (len(files) - 1)))
            avail = list(range(i0, i1 + 1))
            if len(avail) >= N_SLICE:
                picks = [avail[int(round(t))] for t in np.linspace(0, len(avail) - 1, N_SLICE)]
                off = 0
            else:
                picks, off = (avail, (N_SLICE - len(avail)) // 2)
            if INTENSITY == 'series':
                crops = [read_crop(files[p]) for p in picks]
                got = [x for x in crops if x is not None]
                if got:
                    samp = np.concatenate([x[::4, ::4].ravel() for x in got])
                    lo_, hi_ = np.percentile(samp, [1, 99])
                    for c, x in enumerate(crops):
                        if x is None:
                            x = read_crop(files[min(len(files) - 1, picks[c] + 1)])
                        if x is not None:
                            out[s_i, off + c] = (window(x, lo_, hi_, flip) * 255).astype(np.uint8)
            else:
                for c, p in enumerate(picks):
                    img = render(files[p], flip)
                    if img is None:
                        img = render(files[min(len(files) - 1, p + 1)], flip)
                    if img is not None:
                        out[s_i, off + c] = (img * 255).astype(np.uint8)
            mask[s_i] = len(picks)
    return (idx, out, mask)
sub_df = pd.read_csv(COMP / 'sample_submission.csv')
ser_csv = pd.read_csv(COMP / 'test_series.csv')
if not (COMP / 'test_series').exists():
    ser_csv = pd.read_csv(COMP / 'train_series.csv')
ser_csv = ser_csv.loc[:, ~ser_csv.columns.duplicated()]
studies = sub_df.StudyInstanceUID.tolist()
by = {s: g.to_dict('records') for s, g in ser_csv[ser_csv.StudyInstanceUID.isin(set(studies))].groupby('StudyInstanceUID')}
print(f'{len(studies):,} test studies, {len(by):,} with series metadata')

# ======== CELL 28 [code] ========
N_SLOT_TYPES, MASK_IDX = (6, 0)

def segment_softmax(scores, sidx, B):
    T, K = scores.shape
    idx = sidx.unsqueeze(1).expand(-1, K)
    m = torch.full((B, K), float('-inf'), device=scores.device, dtype=scores.dtype)
    m = m.scatter_reduce(0, idx, scores, reduce='amax', include_self=True)
    e = (scores - m[sidx]).exp()
    s = torch.zeros(B, K, device=scores.device, dtype=scores.dtype).index_add_(0, sidx, e)
    return e / s[sidx].clamp(min=1e-06)

class MeanMaxPool(nn.Module):

    def forward(self, f, sidx, B, slot=None, return_attn=False):
        D = f.shape[1]
        cnt = torch.zeros(B, device=f.device, dtype=f.dtype).index_add_(0, sidx, torch.ones(f.shape[0], device=f.device, dtype=f.dtype))
        mean = torch.zeros(B, D, device=f.device, dtype=f.dtype).index_add_(0, sidx, f)
        mean = mean / cnt.clamp(min=1).unsqueeze(1)
        mx = torch.full((B, D), -10000.0, device=f.device, dtype=f.dtype)
        mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), f, reduce='amax', include_self=True)
        return (torch.cat([mean, mx], 1), None)

class LabelAttentionPool(nn.Module):

    def __init__(self, d, n_labels=12, n_heads=4, slot_bias=True):
        super().__init__()
        self.d, self.k, self.h = (d, n_labels, n_heads)
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.key, self.val = (nn.Linear(d, d), nn.Linear(d, d))
        self.slot_bias = nn.Parameter(torch.zeros(n_labels, N_SLOT_TYPES + 1)) if slot_bias else None

    def forward(self, f, sidx, B, slot=None, return_attn=False):
        scores = self.key(f) @ self.q.t() / self.d ** 0.5
        if self.slot_bias is not None and slot is not None:
            scores = scores + self.slot_bias.t()[slot]
        a = segment_softmax(scores, sidx, B)
        out = torch.zeros(B, self.k, self.d, device=f.device, dtype=f.dtype)
        out = out.index_add_(0, sidx, a.unsqueeze(-1) * self.val(f).unsqueeze(1))
        return (out, a)

class TokenXAttnPool(nn.Module):

    def __init__(self, d, n_labels=12, n_heads=6, dropout=0.2):
        super().__init__()
        self.d, self.k = (d, n_labels)
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.slot_emb = nn.Embedding(N_SLOT_TYPES + 1, d, padding_idx=0)
        self.kv_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

    def forward(self, tok, sidx, B, slot=None, return_attn=False):
        T, N, D = tok.shape
        cnt = torch.bincount(sidx, minlength=B)
        S = int(cnt.max().item())
        starts = torch.cumsum(cnt, 0) - cnt
        pos = torch.arange(T, device=tok.device) - starts[sidx]
        kv = tok + self.slot_emb(slot).unsqueeze(1)
        pad = tok.new_zeros(B, S, N, D)
        pad[sidx, pos] = kv
        keep = torch.zeros(B, S, dtype=torch.bool, device=tok.device)
        keep[sidx, pos] = True
        kpm = ~keep.repeat_interleave(N, dim=1)
        pad = self.kv_norm(pad.reshape(B, S * N, D))
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        att, w = self.attn(q, pad, pad, key_padding_mask=kpm, need_weights=return_attn, average_attn_weights=True)
        cls = tok[:, 0]
        mean = torch.zeros(B, D, device=tok.device, dtype=tok.dtype).index_add_(0, sidx, cls) / cnt.clamp(min=1).unsqueeze(1)
        mx = torch.full((B, D), -10000.0, device=tok.device, dtype=tok.dtype)
        mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), cls, reduce='amax', include_self=True)
        base = torch.cat([mean, mx], 1).unsqueeze(1).expand(-1, self.k, -1)
        return (torch.cat([att, base], -1), w)

class ViTSlotToken(nn.Module):

    def __init__(self, vit, n_cat, dim=None):
        super().__init__()
        self.vit = vit
        d = dim or vit.embed_dim
        self.tok = nn.Embedding(n_cat + 1, d, padding_idx=MASK_IDX)
        self.num_features = vit.num_features
        self._orig_prefix = getattr(vit, 'num_prefix_tokens', 1)
        vit.num_prefix_tokens = self._orig_prefix + 1
        for blk in vit.blocks:
            a = getattr(blk, 'attn', None)
            if a is not None and hasattr(a, 'num_prefix_tokens'):
                a.num_prefix_tokens = a.num_prefix_tokens + 1

    @staticmethod
    def _maybe(mod, x):
        return x if mod is None else mod(x)

    def forward_features(self, x, cat):
        v = self.vit
        x = v.patch_embed(x)
        pos = v._pos_embed(x)
        rope = None
        if isinstance(pos, tuple):
            x, rope = pos
        else:
            x = pos
        x = self._maybe(getattr(v, 'patch_drop', None), x)
        x = self._maybe(getattr(v, 'norm_pre', None), x)
        npt = self._orig_prefix
        tok = self.tok(cat).unsqueeze(1)
        x = torch.cat([x[:, :npt], tok, x[:, npt:]], dim=1)
        if rope is not None:
            if getattr(v, 'rope_mixed', False):
                for i, blk in enumerate(v.blocks):
                    x = blk(x, rope=rope[i])
            else:
                for blk in v.blocks:
                    x = blk(x, rope=rope)
        else:
            x = v.blocks(x)
        return v.norm(x)

    def forward_head(self, x, pre_logits=True):
        return self.vit.forward_head(x, pre_logits=pre_logits)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class _GatedDepthBlock(nn.Module):

    def __init__(self, n_slice, dropout=0.0, ls_init=0.1):
        super().__init__()
        self.norm = nn.GroupNorm(1, n_slice)
        self.v = nn.Conv2d(n_slice, n_slice, 1)
        self.g = nn.Conv2d(n_slice, n_slice, 1)
        self.out = nn.Conv2d(n_slice, n_slice, 1)
        self.gamma = nn.Parameter(torch.full((n_slice, 1, 1), ls_init))
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, x):
        z = self.norm(x)
        return x + self.gamma * self.drop(self.out(self.v(z) * F.silu(self.g(z))))

class DepthCompress(nn.Module):

    def __init__(self, n_slice=16, out_ch=3, depth=1, dropout=0.0, ls_init=0.1, imagenet=True, proj_noise=0.25):
        super().__init__()
        self.imagenet = imagenet
        self.blocks = nn.ModuleList([_GatedDepthBlock(n_slice, dropout, ls_init) for _ in range(depth)])
        self.proj = nn.Conv2d(n_slice, out_ch, 1, bias=True)
        if imagenet:
            self.register_buffer('mu', torch.tensor(IMAGENET_MEAN).view(1, -1, 1, 1))
            self.register_buffer('sd', torch.tensor(IMAGENET_STD).view(1, -1, 1, 1))

    def forward(self, x):
        keep = (x.amax(dim=1, keepdim=True) > 0).to(x.dtype)
        z = x
        for b in self.blocks:
            z = b(z)
        z = self.proj(z)
        if self.imagenet:
            z = (z - self.mu.to(z.dtype)) / self.sd.to(z.dtype)
        return z * keep
N_PLANE, N_CONTRAST = (3, 2)
_PLANE_OF = lambda s: torch.clamp(s - 1, 0, 5) // 2
_CONTRAST_OF = lambda s: torch.clamp(s - 1, 0, 5) % 2

class SlotDepthMixer(nn.Module):

    def __init__(self, n_slice=16, ksize=5, alpha_max=0.25):
        super().__init__()
        self.n_slice, self.ksize, self.r = (n_slice, ksize, ksize // 2)
        self.alpha_max = alpha_max
        b = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        self.register_buffer('base', b.log()[self.r:])
        n_u = self.r + 1
        self.shared = nn.Parameter(torch.zeros(n_u))
        self.plane_k = nn.Parameter(torch.zeros(N_PLANE, n_u))
        self.contrast_k = nn.Parameter(torch.zeros(N_CONTRAST, n_u))
        self.g0 = nn.Parameter(torch.zeros(()))
        self.gate_p = nn.Parameter(torch.zeros(N_PLANE))
        self.gate_c = nn.Parameter(torch.zeros(N_CONTRAST))
        idx = torch.arange(n_slice)
        self.register_buffer('off', idx[None, :] - idx[:, None])

    def kernel(self, slot):
        p, c = (_PLANE_OF(slot), _CONTRAST_OF(slot))
        half = self.base + self.shared + self.plane_k[p] + self.contrast_k[c]
        full = torch.cat([half.flip(-1)[..., :self.r], half], dim=-1)
        return F.softmax(full, dim=-1)

    def alpha(self, slot):
        p, c = (_PLANE_OF(slot), _CONTRAST_OF(slot))
        return self.alpha_max * torch.tanh(self.g0 + self.gate_p[p] + self.gate_c[c])

    def forward(self, x, slot, vmask):
        T, S, H, W = x.shape
        if vmask is None:
            raise ValueError('stem=mixer requires the padding mask')
        k = self.kernel(slot)
        v = vmask.to(k.dtype)
        d = self.off + self.r
        inb = (d >= 0) & (d < self.ksize)
        kk = k[:, d.clamp(0, self.ksize - 1)] * inb
        M = kk * v[:, None, :]
        den = M.sum(-1, keepdim=True)
        eye = torch.eye(S, device=x.device, dtype=M.dtype).expand(T, S, S)
        ok = (den > 1e-06) & v[:, :, None].bool()
        M = torch.where(ok, M / den.clamp(min=1e-06), eye)
        a = self.alpha(slot)[:, None, None]
        Aop = ((1.0 - a) * eye + a * M).to(x.dtype)
        if x.is_contiguous(memory_format=torch.channels_last) and (not x.is_contiguous()):
            y = torch.bmm(x.permute(0, 2, 3, 1).reshape(T, H * W, S), Aop.transpose(1, 2))
            return y.reshape(T, H, W, S).permute(0, 3, 1, 2)
        return torch.bmm(Aop, x.reshape(T, S, H * W)).reshape(T, S, H, W)

def _seg_mean_max(v, sidx, B):
    D = v.shape[1]
    cnt = torch.zeros(B, device=v.device, dtype=v.dtype).index_add_(0, sidx, torch.ones(v.shape[0], device=v.device, dtype=v.dtype))
    mean = torch.zeros(B, D, device=v.device, dtype=v.dtype).index_add_(0, sidx, v)
    mean = mean / cnt.clamp(min=1).unsqueeze(1)
    mx = torch.full((B, D), -10000.0, device=v.device, dtype=v.dtype)
    mx = mx.scatter_reduce(0, sidx.unsqueeze(1).expand(-1, D), v, reduce='amax', include_self=True)
    return torch.cat([mean, mx], 1)

def _pad_kv(x, sidx, B, norm):
    T, P, D = x.shape
    cnt = torch.bincount(sidx, minlength=B)
    S = int(cnt.max().item())
    starts = torch.cumsum(cnt, 0) - cnt
    pos = torch.arange(T, device=x.device) - starts[sidx]
    pad = x.new_zeros(B, S, P, D)
    pad[sidx, pos] = x
    keep = torch.zeros(B, S, dtype=torch.bool, device=x.device)
    keep[sidx, pos] = True
    return (norm(pad.reshape(B, S * P, D)), ~keep.repeat_interleave(P, dim=1))

class _GatedDelta(nn.Module):

    def __init__(self, d, n_labels, n_heads, dropout):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.kv_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.d_norm = nn.LayerNorm(d)
        self.dw = nn.Parameter(torch.randn(n_labels, d) * (1.0 / d ** 0.5))
        self.db = nn.Parameter(torch.zeros(n_labels))
        self.gate = nn.Parameter(torch.zeros(n_labels))

    def delta(self, pat, sidx, B, return_attn):
        kv, kpm = _pad_kv(pat, sidx, B, self.kv_norm)
        q = self.q.unsqueeze(0).expand(B, -1, -1)
        att, w = self.attn(q, kv, kv, key_padding_mask=kpm, need_weights=return_attn, average_attn_weights=True)
        return ((self.d_norm(att) * self.dw).sum(-1) + self.db, w)

class TokenResidualPool(_GatedDelta):

    def __init__(self, d, n_labels=12, n_heads=6, pe=64, dropout=0.2):
        super().__init__(d, n_labels, n_heads, dropout)
        self.base = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(dropout), nn.Linear(2 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        base = self.base(torch.cat([_seg_mean_max(tok[:, 1:].mean(1), sidx, B), pres], 1))
        d_, w = self.delta(tok[:, 1:], sidx, B, return_attn)
        return (base + self.gate * d_, w)

class CodexResidualPool(_GatedDelta):

    def __init__(self, d, n_labels=12, n_heads=6, pe=64, dropout=0.2):
        super().__init__(d, n_labels, n_heads, dropout)
        self.base = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(dropout), nn.Linear(2 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        base = self.base(torch.cat([_seg_mean_max(tok[:, 0], sidx, B), pres], 1))
        d_, w = self.delta(tok[:, 1:], sidx, B, return_attn)
        return (base + self.gate * d_, w)

class ClsAddPool(nn.Module):

    def __init__(self, d, n_labels=12, pe=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(4 * d + pe), nn.Dropout(dropout), nn.Linear(4 * d + pe, n_labels))

    def forward(self, tok, slot, sidx, B, pres, return_attn=False):
        return (self.net(torch.cat([_seg_mean_max(tok[:, 1:].mean(1), sidx, B), _seg_mean_max(tok[:, 0], sidx, B), pres], 1)), None)

class Readout(nn.Module):

    def __init__(self, pool, d, n_labels=12, pe=64):
        super().__init__()
        self.pool_kind, self.k = (pool, n_labels)
        self.pres_emb = nn.Embedding(N_SLOT_TYPES + 1, pe, padding_idx=0)
        if pool in ('xres', 'clsadd', 'xcodex'):
            self.pool = {'xres': TokenResidualPool, 'clsadd': ClsAddPool, 'xcodex': CodexResidualPool}[pool](d, n_labels, pe=pe)
        elif pool in ('attn', 'xattn'):
            if pool == 'xattn':
                self.pool = TokenXAttnPool(d, n_labels)
                wd = 3 * d + pe
            else:
                self.pool = LabelAttentionPool(d, n_labels)
                wd = d + pe
            self.norm = nn.LayerNorm(wd)
            self.w = nn.Parameter(torch.randn(n_labels, wd) * (1.0 / wd ** 0.5))
            self.b = nn.Parameter(torch.zeros(n_labels))
        else:
            self.pool = MeanMaxPool()
            self.net = nn.Sequential(nn.LayerNorm(2 * d + pe), nn.Dropout(0.2), nn.Linear(2 * d + pe, n_labels))
        self.drop = nn.Dropout(0.2)

    def forward(self, f, slot, sidx, B, return_attn=False):
        pe = self.pres_emb(slot)
        pres = torch.zeros(B, pe.shape[1], device=f.device, dtype=f.dtype).index_add_(0, sidx, pe)
        if self.pool_kind in ('xres', 'clsadd', 'xcodex'):
            return self.pool(f, slot, sidx, B, pres)[0]
        pooled, attn = self.pool(f, sidx, B, slot=slot, return_attn=return_attn)
        if self.pool_kind in ('attn', 'xattn'):
            x = torch.cat([pooled, pres.unsqueeze(1).expand(-1, self.k, -1)], -1)
            x = self.drop(self.norm(x))
            return (x * self.w).sum(-1) + self.b
        return self.net(torch.cat([pooled, pres], 1))

class Net(nn.Module):

    def __init__(self, enc, cond, n_meta=0, pool='mean_max', stem='native', n_slice=16):
        super().__init__()
        self.enc, self.cond = (enc, cond)
        self.compress = DepthCompress(n_slice, 3) if stem == 'compress' else None
        self.mixer = SlotDepthMixer(n_slice) if stem == 'mixer' else None
        self.tokens = pool in ('xattn', 'xres', 'clsadd', 'xcodex')
        D = enc.num_features
        self.meta_mlp = nn.Sequential(nn.LayerNorm(n_meta), nn.Linear(n_meta, 128), nn.GELU(), nn.Linear(128, D)) if n_meta > 0 else None
        self.readout = Readout(pool, D)
        if cond == 'post':
            self.slot_emb = nn.Embedding(N_SLOT_TYPES + 1, D, padding_idx=MASK_IDX)

    def forward(self, im, slot, smeta, sidx, B, vm=None):
        if self.mixer is not None:
            im = self.mixer(im, slot, vm)
        if self.compress is not None:
            im = self.compress(im)
        f = self.enc.forward_features(im, slot) if self.cond == 'token' else self.enc.forward_features(im)
        if self.tokens:
            inner = getattr(self.enc, 'vit', self.enc)
            orig = getattr(self.enc, '_orig_prefix', getattr(inner, 'num_prefix_tokens', 1))
            f = torch.cat([f[:, :1], f[:, orig:]], 1)
        else:
            f = self.enc.forward_head(f, pre_logits=True)
            if f.dim() > 2:
                f = f.flatten(1)
        ex = (lambda v: v.unsqueeze(1)) if self.tokens else lambda v: v
        if self.cond == 'post':
            f = f + ex(self.slot_emb(slot))
        if self.meta_mlp is not None and smeta.shape[1] > 0:
            mt = self.meta_mlp(smeta)
            f = torch.cat([f, mt.unsqueeze(1)], 1) if self.tokens else f + mt
        return self.readout(f, slot, sidx, B)
models = []
for ckpt_path in sorted(CKPT.glob('*_f*.pt')):
    z = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = z['cfg']
    _stem = cfg.get('stem', 'native')
    _in = 3 if _stem == 'compress' else cfg.get('n_slice', 16)
    enc = timm.create_model(cfg['backbone'], pretrained=False, num_classes=0, in_chans=_in, **{'img_size': cfg['img']} if 'vit_' in cfg['backbone'] else {})
    if cfg['cond'] == 'token':
        enc = ViTSlotToken(enc, N_SLOT_TYPES)
    m = Net(enc, cfg['cond'], cfg.get('n_meta', 0), cfg['pool'], stem=_stem, n_slice=cfg.get('n_slice', 16))
    missing, unexpected = m.load_state_dict(z['state_dict'], strict=False)
    assert not [k for k in missing if not k.startswith('enc.')], f'missing {missing[:5]}'
    assert not unexpected, f'unexpected {unexpected[:5]}'
    models.append(m.eval())
    print(f"loaded {ckpt_path.name}  fold {z['fold']}  {cfg['backbone']} pool={cfg['pool']} meta={cfg['meta']}")
CFG = cfg
assert CFG.get('n_meta', 0) == 0, f"checkpoint expects {CFG['n_meta']} metadata features -- build slot_meta for the TEST studies and pass it to predict() before submitting"
print(f"\n{len(models)} fold models ready | input norm: {CFG.get('norm', 'none')}")

# ======== CELL 29 [code] ========
AMP_PREF = 'bf16'

def amp_for(dev):
    if not str(dev).startswith('cuda'):
        return (torch.float32, False)
    cc = torch.cuda.get_device_capability(dev)
    if AMP_PREF == 'bf16':
        return (torch.bfloat16, True)
    if AMP_PREF == 'fp16':
        return (torch.float16, True)
    if AMP_PREF == 'fp32':
        return (torch.float32, False)
    return (torch.bfloat16 if cc >= (8, 0) else torch.float16, True)
AMP_DT, AMP_ON = amp_for(DEV)
WORKERS = max(1, min(4, os.cpu_count() or 4))
CHUNK = 48
MICRO = 8
models = [m.to(DEV).eval() for m in models]
print(f"device {DEV} | amp {str(AMP_DT).split('.')[-1]} (on={AMP_ON}) | workers {WORKERS} | chunk {CHUNK} | micro {MICRO}")

def _norm_(im):
    k = CFG.get('norm', 'none')
    if k == 'zscore':
        m = (im > 0).float()
        n = m.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
        mu = (im * m).sum(dim=(1, 2, 3), keepdim=True) / n
        var = (((im - mu) * m) ** 2).sum(dim=(1, 2, 3), keepdim=True) / n
        return (im - mu) / (var.sqrt() + 1e-06) * m
    if k == 'imagenet':
        m = (im > 0).float()
        return (im - 0.485) / 0.229 * m
    return im

@torch.no_grad()
def _micro(images, masks):
    dev = DEV
    ims, slots, sidx, vms = ([], [], [], [])
    for b in range(len(masks)):
        present = np.nonzero(masks[b] > 0)[0]
        if len(present) == 0:
            continue
        blk = images[b][present]
        ims.append(torch.from_numpy(blk))
        vms.append(torch.from_numpy(blk.reshape(blk.shape[0], blk.shape[1], -1).max(2) > 0))
        slots.append(torch.from_numpy(present + 1).long())
        sidx.append(torch.full((len(present),), b, dtype=torch.long))
    out = np.full((len(models), len(masks), len(LABELS)), np.nan, np.float32)
    if not ims:
        return out
    im = _norm_(torch.cat(ims).to(dev, non_blocking=True).float().div_(255.0))
    sl = torch.cat(slots).to(dev)
    si = torch.cat(sidx).to(dev)
    vm = torch.cat(vms).to(dev)
    sm = torch.zeros(len(sl), CFG.get('n_meta', 0), device=dev)
    per = torch.zeros(len(models), len(masks), len(LABELS), device=dev, dtype=torch.float32)
    with torch.autocast('cuda' if str(dev).startswith('cuda') else 'cpu', dtype=AMP_DT, enabled=AMP_ON):
        for _mi, m in enumerate(models):
            per[_mi] = torch.sigmoid(m(im, sl, sm, si, len(masks), vm=vm).float())
    got = per.cpu().numpy()
    keep = np.array([(masks[b] > 0).any() for b in range(len(masks))])
    out[:, keep] = got[:, keep]
    return out

def predict(images, masks):
    out = np.full((len(models), len(masks), len(LABELS)), np.nan, np.float32)
    for a in range(0, len(masks), MICRO):
        b = min(a + MICRO, len(masks))
        out[:, a:b] = _micro(images[a:b], masks[a:b])
    return out
# Per-fold predictions are kept whole: the metric is macro ROC-AUC, so folds are
# combined on RANKS across the full test set, matching what the DINOv2 frontier
# and the RadImageNet stage already do. Averaging probabilities first lets a
# fold with a shifted output range dominate the mean. (Observation due to
# romantamrazov, RSNA Knee | DINOsaur V2.)
preds = np.full((len(models), len(studies), len(LABELS)), np.nan, np.float32)
t0, done = (time.time(), 0)
with ProcessPoolExecutor(max_workers=WORKERS) as ex:
    for c0 in range(0, len(studies), CHUNK):
        block = studies[c0:c0 + CHUNK]
        imgs = np.zeros((len(block), N_SLOT, N_SLICE, SIZE, SIZE), np.uint8)
        msks = np.zeros((len(block), N_SLOT), np.uint8)
        futs = [ex.submit(build_study, (i, s, by.get(s, []))) for i, s in enumerate(block)]
        for f in as_completed(futs):
            try:
                i, a, k = f.result()
                imgs[i], msks[i] = (a, k)
            except Exception as e:
                print(f'  study failed: {type(e).__name__}: {e}')
        preds[:, c0:c0 + len(block)] = predict(imgs, msks)
        done += len(block)
        el = time.time() - t0
        print(f'  {done:,}/{len(studies):,}  {el / 60:.1f}m  eta {el / done * (len(studies) - done) / 60:.1f}m', flush=True)
        del imgs, msks
        gc.collect()
print(f'\ninference done in {(time.time() - t0) / 60:.1f} min')
A5_W = 0.45
A5_LABELS = list(LABELS)
_a5_ok = np.isfinite(preds).all(axis=(0, 2))
_a5_rankavg = np.zeros((len(studies), len(LABELS)), np.float64)
for _f in range(preds.shape[0]):
    _blk = preds[_f][_a5_ok]
    _o = _blk.argsort(0).argsort(0).astype(np.float64)
    _a5_rankavg[_a5_ok] += _o / max(len(_blk) - 1, 1)
_a5_rankavg /= preds.shape[0]
_a5_rankavg[~_a5_ok] = np.nan
print(f'a5: rank-averaged {preds.shape[0]} folds over {int(_a5_ok.sum()):,} studies')
A5_PREDS = dict(zip(sub_df['StudyInstanceUID'].astype(str), _a5_rankavg.astype(np.float32)))
for _a5k, _a5v in _A5_SAVED.items():
    globals()[_a5k] = _a5v
del _A5_SAVED, _a5k, _a5v

# ======== CELL 30 [code] ========
_a5_sub = pd.read_csv('/kaggle/working/submission.csv',
                      dtype={'StudyInstanceUID': str})
assert _a5_sub.columns.tolist()[1:] == A5_LABELS, 'submission schema drift'
if A5_W > 0:
    _a5_ours = np.stack([A5_PREDS[_u]
                         for _u in _a5_sub['StudyInstanceUID'].astype(str)])
    _a5_base_rank = _a5_sub[A5_LABELS].rank(method='average', pct=True)
    _a5_ours_rank = pd.DataFrame(_a5_ours, columns=A5_LABELS,
                                 index=_a5_sub.index).rank(method='average', pct=True)
    _a5_sub[A5_LABELS] = (1.0 - A5_W) * _a5_base_rank + A5_W * _a5_ours_rank
    assert np.isfinite(_a5_sub[A5_LABELS].to_numpy()).all()
    _a5_sub.to_csv('/kaggle/working/submission.csv', index=False)

# ======== CELL 31 [code] ========
# E9: independent RadImageNet ResNet-50 arm for the verified E2 parent.
#
# Adapted 2026-08-11 from the V52 cell in the public Kaggle competition notebook
# prvsiyan/rsna-knee-read-the-report-then-the-knee (latest source SHA-256
# b54aa529f38dc6f594478e7975d86459ddbff898453a2c2633f4c3be4b909e61).
# Kaggle's public-code rule deems public Competition Code open-source; Meta Kaggle
# documents public notebooks under Apache-2.0. Changes here remove the unavailable B3
# arm, pin the public E2 OOF bundle, add per-target/no-regression gates, and preserve E2
# byte-for-byte on every failure. This module is appended as the final notebook cell;
# its imports and DICOM helpers are supplied by the parent notebook.
SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
]
N_SLOT = len(SLOTS)
CACHE_SLICES = 8
TIME_BUDGET = 8.72 * 3600
IMG = CACHE_IMG = 224
# Match the released RadImageNet model's full-frame pretraining. Setting a crop
# larger than every acquisition disables the optional physical crop in read_slot.
CROP_MM = 10_000.0
SLICE_BAND = (0.12, 0.88)
# The audited public OOF was produced after the notebook's legacy member group, which
# left this process-global pixel contract active. Pin it instead of inheriting whichever
# E2 group happened to run last. This uses public preprocessing code only; no legacy
# checkpoint or unknown-license asset is attached.
RULES = dict(RULES_LEGACY)
TOKEN_DIM = 2048             # official ResNet-50 global-average feature
HEAD_DIM = 512
PINNED_HEADS_SHA256 = "0f465649799ecfbccaac1767844639e7ced44e1bc9babde6e4bac7c5d9b89eaa"
PINNED_REMOTE_AUDIT_SHA256 = "267f948078710d3ca8a6f0de4ce0a5e75e850e1f08d36451549a137d878a6fe8"
PINNED_PUBLIC_DIAGNOSTIC_SHA256 = "0f2f82fb40f0570d6766f73b0d7f51489df6d0faa8fd6e4c0f45bcd6c4c7b283"
PINNED_E9B_CONTRACT_SHA256 = "6777c0a0ba7dd044752fac948752dc39e9ca35b3c280fe74ee6d27a5865d87e7"
# E10 repairs E9's censored search: its alpha grid stopped at 0.25 and four of five outer
# folds selected that ceiling, so the deployed 0.20 was a boundary artifact rather than an
# optimum. The ladder, the two-source per-target gains and every deployable weight map live
# in the hash-pinned contract; this cell recomputes the remote half in-kernel before use.
# The contract also carries the held-out form of E10's own weight choice: selecting the rung
# on four grouped folds and scoring the fifth picks 0.60 (public) and 0.70 (v15) in all five
# outer folds, and never picks 0.20. So every deployable rung at or below 0.35 is below what
# honest selection would choose, which is the answer to "you tuned on the 58 gold rows".
PINNED_E10_CONTRACT_SHA256 = "219c91f40905181c222e2966b3fed01a96570ddfb64862357d5fd6cad500cd45"
E10_CONFIG = "uniform_060"
E10_PRESERVED_TARGETS = ["Baker's", "Fracture"]

# E11 trains a third arm whose diversity is in the pixels rather than in the weights. The
# existing arm reads three fat-suppressed slots at full frame; every one of E2's twenty
# members reads one DINOv2 recipe. Nothing in the portfolio has yet looked at a
# non-fat-suppressed series, where meniscal and ligament morphology is conventionally read,
# and nothing has given RadImageNet a physically normalised field of view. E11 changes both:
# three non-suppressed slots plus one suppressed anchor, cropped to 130 mm, which the parent
# notebook establishes is below the acquired field of view of 99.6% of series while still
# containing the joint. It is a training mode only; it never touches the submission.
ARM_MODE = "e11"
E11_SLOTS = [
    ("SAG_FS", "Sagittal", None, True),
    ("COR_FS", "Coronal", None, True),
    ("AX_FS", "Axial", None, True),
    ("SAG_NOFS", "Sagittal", None, False),
]
E11_CROP_MM = 130.0
E11_CACHE_SLICES = 8
E11_IMG = 224
# Availability of non-suppressed series per plane is unmeasured, so a fill floor rather than
# the parent's 90% rule: below this the run has found something structurally wrong, above it
# an empty slot is simply masked out of the token set like any other absent series.
E11_MIN_FILL = 0.45


def _v52_as_bool(value):
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text in {"1", "TRUE", "T", "YES", "Y"}:
        return True
    if text in {"0", "FALSE", "F", "NO", "N"}:
        return False
    try:
        number = float(text)
        return True if number == 1 else False if number == 0 else None
    except Exception:
        return None


def audit_official_sequence_metadata(inferred, official):
    """Audit metadata agreement without changing the checkpoint pixel contract."""
    needed = {"SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression"}
    if inferred.empty or official.empty or not needed.issubset(official.columns):
        return
    inferred_flags = inferred[["SeriesInstanceUID", "fluid", "fatsat"]].copy()
    official_flags = official[
        ["SeriesInstanceUID", "Fluid_Sensitive", "Fat_Suppression"]
    ].copy()
    official_flags["official_fluid"] = official_flags["Fluid_Sensitive"].map(
        _v52_as_bool
    )
    official_flags["official_fatsat"] = official_flags["Fat_Suppression"].map(
        _v52_as_bool
    )
    merged = inferred_flags.merge(
        official_flags[["SeriesInstanceUID", "official_fluid", "official_fatsat"]],
        on="SeriesInstanceUID",
        how="inner",
    )
    for inferred_col, official_col, name in [
        ("fluid", "official_fluid", "Fluid_Sensitive"),
        ("fatsat", "official_fatsat", "Fat_Suppression"),
    ]:
        valid = merged[official_col].notna() & merged[inferred_col].notna()
        if valid.any():
            agreement = (
                merged.loc[valid, inferred_col].astype(bool).to_numpy()
                == merged.loc[valid, official_col].astype(bool).to_numpy()
            ).mean()
            log(
                f"V52 metadata audit {name}: {agreement:.1%} agreement "
                f"on {int(valid.sum())} series"
            )


def find_input_file(name):
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            return Path(root) / name
    raise FileNotFoundError(name)


def find_input_dir(name):
    for root, dirs, files in os.walk("/kaggle/input"):
        if Path(root).name == name:
            return Path(root)
    raise FileNotFoundError(name)


def make_targets(train):
    """Three independent public report teachers; image-read gold always wins."""
    uid = "StudyInstanceUID"
    sources = [
        pd.read_csv(find_input_file("report_labels_v2.csv")),
        pd.read_csv(find_input_file("llm_labels_v2.csv")),
        pd.read_csv(find_input_file("labels_llm_gpt56sol.csv")),
    ]
    cube = []
    for frame in sources:
        if frame[uid].duplicated().any():
            raise ValueError("duplicate study in report-label source")
        aligned = train[[uid]].merge(frame[[uid] + TARGETS], on=uid, how="left")
        cube.append(aligned[TARGETS].to_numpy(float))
    cube = np.stack(cube)
    available = np.isfinite(cube).sum(0)
    if np.any(available < 2):
        raise ValueError("fewer than two report teachers for a study/target")
    y = np.nanmean(cube, axis=0).astype(np.float32)
    disagreement = np.nanmean(np.abs(cube - y[None]), axis=0)
    agreement = np.clip(1.0 - 2.0 * disagreement, 0, 1)
    certainty = np.clip(2.0 * np.abs(y - .5), 0, 1)
    w = (.15 + .85 * (.65 * agreement + .35 * certainty)).astype(np.float32)
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    y[gold] = train.loc[gold, TARGETS].to_numpy(np.float32)
    w[gold] = 3.0
    return y, w, gold


def report_groups(train):
    report = (train.Report.fillna("").astype(str).str.lower()
              .str.replace(r"\s+", " ", regex=True).str.strip())
    return np.array([hashlib.sha256(x.encode()).hexdigest()[:24] for x in report])


def _v52_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_radimagenet(device):
    """Strictly load the official RadImageNet ResNet-50 PyTorch checkpoint."""
    from torchvision.models import resnet50

    checkpoint = find_input_file("ResNet50.pt")
    expected_checkpoint = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
    observed_checkpoint = _v52_sha256(checkpoint)
    if observed_checkpoint != expected_checkpoint:
        raise RuntimeError(f"RadImageNet checkpoint drift: {observed_checkpoint}")

    class RadImageNetEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                *list(resnet50(weights=None).children())[:-2]
            )

        def forward(self, image):
            return self.backbone(image).mean(dim=(2, 3))

    model = RadImageNetEncoder()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not state or not all(str(key).startswith("backbone.") for key in state):
        raise RuntimeError("unexpected RadImageNet state-dict namespace")
    model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 23_508_032:
        raise RuntimeError(f"unexpected RadImageNet parameter count {parameter_count}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if gpu_count > 1:
        model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
    log(
        f"RadImageNet strict load: {parameter_count:,} params; "
        f"inference GPUs={max(1, gpu_count)}"
    )
    return model


@torch.inference_mode()
def encode_radimagenet(cache, slot_mask, device):
    """Encode acquired slices with the official [-1, 1] RadImageNet contract."""
    n, slots, slices, h, w = cache.shape
    features = np.zeros((n, slots * slices, TOKEN_DIM), np.float16)
    token_mask = np.repeat(slot_mask[:, :, None], slices, axis=2).reshape(n, -1)
    valid = np.flatnonzero(token_mask.reshape(-1) > 0)
    flat = cache.reshape(-1, h, w)
    model = load_radimagenet(device)
    if device.type == "cuda":
        batch = 192 if torch.cuda.device_count() > 1 else 96
    else:
        batch = 8
    for b0 in range(0, len(valid), batch):
        ix = valid[b0:b0 + batch]
        x = torch.from_numpy(flat[ix]).to(device).float().div_(127.5).sub_(1.0)
        x = x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            feat = model(x)
        if feat.shape[1:] != (TOKEN_DIM,):
            raise RuntimeError(f"unexpected RadImageNet feature shape {tuple(feat.shape)}")
        features.reshape(-1, TOKEN_DIM)[ix] = (
            feat.float().cpu().numpy().astype(np.float16)
        )
        if b0 % (batch * 100) == 0:
            log(f"RadImageNet encoded {b0}/{len(valid)} acquired slices")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, token_mask.astype(np.float32)


class FoundationQueryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(TOKEN_DIM),
                                     nn.Linear(TOKEN_DIM, HEAD_DIM), nn.GELU())
        self.plane = nn.Parameter(torch.randn(N_SLOT, HEAD_DIM) * .01)
        self.position = nn.Parameter(torch.randn(CACHE_SLICES, HEAD_DIM) * .01)
        self.query = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * .02)
        self.attn = nn.MultiheadAttention(HEAD_DIM, 8, dropout=.10, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(HEAD_DIM * 4), nn.Linear(HEAD_DIM * 4, HEAD_DIM),
            nn.GELU(), nn.Dropout(.15),
        )
        self.weight = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * .02)
        self.bias = nn.Parameter(torch.zeros(len(TARGETS)))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), N_SLOT, CACHE_SLICES, HEAD_DIM)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        key_padding = mask <= 0
        # No study should be empty, but keep MHA numerically defined if one is.
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(query, token, token,
                                     key_padding_mask=key_padding,
                                     need_weights=False)[0]
        denom = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdims=True) / denom
        mean = mean.expand(-1, len(TARGETS), -1)
        fused = self.fuse(torch.cat(
            [attended, mean, torch.abs(attended - mean), attended * mean], -1))
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


def macro_auc(y, pred):
    from sklearn.metrics import roc_auc_score
    hard = (np.asarray(y) >= .5).astype(np.uint8)
    values = [roc_auc_score(hard[:, j], pred[:, j])
              for j in range(hard.shape[1]) if np.unique(hard[:, j]).size == 2]
    return float(np.mean(values))


def _v52_target_auc(y, pred):
    from sklearn.metrics import roc_auc_score
    hard = (np.asarray(y) >= .5).astype(np.uint8)
    return {
        target: float(roc_auc_score(hard[:, index], pred[:, index]))
        for index, target in enumerate(TARGETS)
    }


@torch.inference_mode()
def predict_head(model, features, masks, indices, device, batch=64):
    model.eval()
    pred = []
    for b0 in range(0, len(indices), batch):
        ix = indices[b0:b0 + batch]
        x = torch.from_numpy(features[ix]).to(device)
        m = torch.from_numpy(masks[ix]).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            pred.append(torch.sigmoid(model(x, m)).float().cpu())
    return torch.cat(pred).numpy()


def train_fold(features, masks, y, weights, train_idx, val_idx, fold, device):
    from torch.utils.data import DataLoader, Dataset
    class Rows(Dataset):
        def __init__(self, indices): self.indices = np.asarray(indices)
        def __len__(self): return len(self.indices)
        def __getitem__(self, k):
            i = self.indices[k]
            return features[i], masks[i], y[i], weights[i]
    model = FoundationQueryHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=3e-3)
    generator = torch.Generator().manual_seed(SEED + 100 + fold)
    loader = DataLoader(Rows(train_idx), batch_size=48, shuffle=True,
                        generator=generator, num_workers=2, pin_memory=True,
                        persistent_workers=True)
    best, best_auc, stale = None, -1.0, 0
    for epoch in range(24):
        model.train()
        for x, m, target, weight in loader:
            x, m = x.to(device), m.to(device)
            target, weight = target.to(device), weight.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x, m)
                raw = F.binary_cross_entropy_with_logits(logits, target,
                                                          reduction="none")
                loss = (raw * weight).sum() / weight.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        pred = predict_head(model, features, masks, val_idx, device)
        score = macro_auc(y[val_idx], pred)
        log(f"fold {fold} epoch {epoch}: grouped weak-val AUC {score:.5f}")
        if score > best_auc + 2e-4:
            best_auc, stale = score, 0
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 5: break
    return best, best_auc


def _v52_rank_columns(values):
    frame = pd.DataFrame(np.asarray(values, dtype=np.float64))
    return frame.rank(method="average", pct=True).to_numpy(np.float64)


def _v52_validate_submission(frame, expected_ids):
    expected_columns = ["StudyInstanceUID", *TARGETS]
    if frame.columns.tolist() != expected_columns:
        raise RuntimeError("V52 submission schema drift")
    ids = frame["StudyInstanceUID"].astype(str).tolist()
    if ids != list(map(str, expected_ids)) or len(ids) != len(set(ids)):
        raise RuntimeError("V52 submission study identity/order drift")
    values = frame[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise RuntimeError("V52 submission values are invalid")


def main_v52():
    import shutil
    from sklearn.model_selection import GroupKFold

    output = Path("/kaggle/working/rsna_rad_e9")
    output.mkdir(parents=True, exist_ok=True)
    primary = Path("/kaggle/working/submission.csv")
    preserved = Path("/kaggle/working/submission_e2_preserved.csv")
    audit_path = Path("/kaggle/working/rad_e9_audit.json")
    audit = {
        "status": "E2_PRESERVED",
        "evidence_boundary": (
            "All OOF values are local diagnostics on 58 official image labels; "
            "they are not Kaggle competition scores. E2 remains the primary unless "
            "strict artifact, OOF, inference, and submission gates all pass."
        ),
        "encoder": "RadImageNet ResNet-50 official PyTorch release",
        "encoder_license": "CC-BY-NC-SA-4.0 (Kaggle-hosted weight metadata)",
        "encoder_sha256": "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734",
        "encoder_source_commit": "0ce16f7375db4236e646829d1eca61cdb4282133",
        "base_oof_sha256": "62d47ba4c0c8347b5b24e7fd2aa517aae0fd6d4656fd5d829fd3a50f0159909c",
        "parent": "E2 captured 20-member DINOv2 rank ensemble",
        "blend_contract": "rank columns independently, then 80% E2 plus 20% RadImageNet",
        "pixel_rules": dict(RULES),
    }
    if not primary.is_file():
        raise FileNotFoundError("E2 parent submission is absent")
    shutil.copy2(primary, preserved)

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("V52 RadImageNet experiment requires CUDA")
        elapsed = max(0.0, time.time() - float(globals().get("T0", time.time())))
        available = 8.72 * 3600 - elapsed
        audit["elapsed_before_v52_seconds"] = elapsed
        audit["available_at_start_seconds"] = available
        if available < 2.0 * 3600:
            raise TimeoutError(f"only {available / 60:.1f} minutes remain")

        train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
        train_series = pd.read_csv(
            ROOT / "train_series.csv",
            dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        )
        if len(train) != 4407:
            raise RuntimeError(f"unexpected train study count {len(train)}")
        plane = dict(zip(train_series.SeriesInstanceUID, train_series.Anatomical_Plane))
        headers = annotate(walk("train_series"))
        audit_official_sequence_metadata(headers, train_series)
        studies, pixels, slot_mask = build_cache(
            pick_slots(headers, plane), plane, lat_of(headers, "train-v52 "), "train-v52"
        )
        by_uid = {str(uid): i for i, uid in enumerate(studies)}
        missing = [uid for uid in train.StudyInstanceUID if uid not in by_uid]
        if missing:
            raise RuntimeError(f"{len(missing)} train studies absent from cache")
        order = np.array([by_uid[uid] for uid in train.StudyInstanceUID], dtype=np.int64)
        pixels, slot_mask = pixels[order], slot_mask[order]
        train_token_count = int(np.repeat(slot_mask[:, :, None], CACHE_SLICES, 2).sum())
        if train_token_count < int(0.90 * len(train) * N_SLOT * CACHE_SLICES):
            raise RuntimeError(f"insufficient acquired train slices: {train_token_count}")
        features, token_mask = encode_radimagenet(pixels, slot_mask, device)
        del pixels, slot_mask, headers
        gc.collect()

        y, weights, gold = make_targets(train)
        if int(gold.sum()) != 58:
            raise RuntimeError(f"expected 58 fully gold studies, observed {int(gold.sum())}")
        groups = report_groups(train)
        if len(np.unique(groups)) < 4000:
            raise RuntimeError("unexpected report-group collapse")

        splits = list(GroupKFold(5).split(features, groups=groups))
        fold_id = np.full(len(train), -1, dtype=np.int8)
        folds = []
        oof = np.zeros_like(y, dtype=np.float32)
        for fold, (tr, va) in enumerate(splits):
            if set(groups[tr]).intersection(groups[va]):
                raise RuntimeError(f"report leakage in fold {fold}")
            fold_id[va] = fold
            state, score = train_fold(
                features, token_mask, y, weights, tr, va, fold, device
            )
            if state is None:
                raise RuntimeError(f"fold {fold} produced no checkpoint")
            head = FoundationQueryHead().to(device)
            head.load_state_dict(state, strict=True)
            oof[va] = predict_head(head, features, token_mask, va, device)
            folds.append({"fold": fold, "weak_auc": float(score), "state_dict": state})
            del head
            torch.cuda.empty_cache()
        if (fold_id < 0).any() or not np.isfinite(oof).all():
            raise RuntimeError("incomplete V52 OOF")

        weak_auc = macro_auc(y, oof)
        gold_auc = macro_auc(y[gold], oof[gold])
        log(f"V52 RadImageNet OOF weak macro AUC {weak_auc:.5f}")
        log(f"V52 RadImageNet OOF gold macro AUC {gold_auc:.5f} on 58 studies")
        torch.save(
            {
                "version": "v52-radimagenet-resnet50-official-1",
                "targets": TARGETS,
                "encoder_sha256": audit["encoder_sha256"],
                "encoder_source_commit": audit["encoder_source_commit"],
                "img": IMG,
                "slices_per_plane": CACHE_SLICES,
                "feature": "global_average_pool",
                "folds": folds,
                "weak_oof_auc": weak_auc,
                "gold_oof_auc": gold_auc,
            },
            output / "v52_radimagenet_heads.pt",
        )
        oof_frame = pd.DataFrame(oof, columns=TARGETS)
        oof_frame.insert(0, "StudyInstanceUID", train.StudyInstanceUID)
        oof_frame["fold"] = fold_id
        oof_frame["is_gold"] = gold.astype(np.uint8)
        oof_frame.to_csv(output / "v52_oof.csv", index=False)

        base_npz = find_input_file("oof.npz")
        observed_base_hash = _v52_sha256(base_npz)
        if observed_base_hash != audit["base_oof_sha256"]:
            raise RuntimeError(f"E2 OOF artifact drift: {observed_base_hash}")
        with np.load(base_npz, allow_pickle=False) as base_bundle:
            expected_members = {"ids", "pred", "y_derived", "gold_mask", "targets"}
            if set(base_bundle.files) != expected_members:
                raise RuntimeError(f"unexpected E2 OOF members: {base_bundle.files}")
            base_ids = base_bundle["ids"].astype(str)
            base_targets = base_bundle["targets"].astype(str).tolist()
            base_gold = base_bundle["gold_mask"].astype(bool)
            base_prediction = base_bundle["pred"].astype(np.float64)
        if base_targets != TARGETS:
            raise RuntimeError("E2 OOF target order drift")
        if not np.array_equal(base_ids, train.StudyInstanceUID.astype(str).to_numpy()):
            raise RuntimeError("E2 OOF study order drift")
        if not np.array_equal(base_gold, gold):
            raise RuntimeError("E2 OOF gold mask differs from official train.csv")
        train_rows = np.flatnonzero(gold)
        if len(train_rows) != 58:
            raise RuntimeError(f"expected 58 E2 gold rows, observed {len(train_rows)}")
        gold_y = train.loc[gold, TARGETS].to_numpy(np.float64)
        exact_public = base_prediction[gold]
        rad = oof[gold].astype(np.float64)
        if not all(np.isfinite(x).all() for x in (gold_y, exact_public, rad)):
            raise RuntimeError("non-finite aligned E2/RadImageNet OOF value")

        base_rank = _v52_rank_columns(exact_public)
        rad_rank = _v52_rank_columns(rad)
        base_score = macro_auc(gold_y, base_rank)
        rad_score = macro_auc(gold_y, rad_rank)
        alpha_grid = np.array([0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25])
        gold_folds = fold_id[train_rows]
        if sorted(np.unique(gold_folds).tolist()) != [0, 1, 2, 3, 4]:
            raise RuntimeError("gold rows do not cover all five grouped folds")
        nested = np.zeros_like(base_rank)
        choices = []
        outer_train_scores = []
        for outer in range(5):
            tr = gold_folds != outer
            va = ~tr
            scored = []
            for alpha in alpha_grid:
                blend = (1.0 - alpha) * base_rank[tr] + alpha * rad_rank[tr]
                score = macro_auc(gold_y[tr], blend) - 0.01 * float(alpha)
                scored.append(float(score))
            best = max(range(len(alpha_grid)), key=lambda i: (scored[i], -alpha_grid[i]))
            alpha = float(alpha_grid[best])
            choices.append(alpha)
            outer_train_scores.append(scored)
            nested[va] = (1.0 - alpha) * base_rank[va] + alpha * rad_rank[va]
        nested_score = macro_auc(gold_y, nested)
        # Deployment weight is fixed by the independently scored public 0.906 mechanism.
        # The 58 gold rows may veto it and measure fold stability, but do not tune it.
        final_alpha = 0.20
        final_oof = (1.0 - final_alpha) * base_rank + final_alpha * rad_rank
        final_score = macro_auc(gold_y, final_oof)
        grid_scores = {
            f"{alpha:.3f}": macro_auc(
                gold_y, (1.0 - alpha) * base_rank + alpha * rad_rank
            )
            for alpha in alpha_grid
        }
        base_target_scores = _v52_target_auc(gold_y, base_rank)
        rad_target_scores = _v52_target_auc(gold_y, rad_rank)
        final_target_scores = _v52_target_auc(gold_y, final_oof)
        target_deltas = {
            target: final_target_scores[target] - base_target_scores[target]
            for target in TARGETS
        }
        target_regressions = {
            target: delta for target, delta in target_deltas.items() if delta < -1e-12
        }
        positive_folds = int(sum(alpha > 0 for alpha in choices))
        supported = bool(
            final_alpha > 0
            and positive_folds >= 3
            and nested_score >= base_score + 0.001
            and final_score >= base_score + 0.001
            and not target_regressions
        )
        audit["oof"] = {
            "rows": 58,
            "weak_macro_auc": weak_auc,
            "rad_gold_macro_auc": rad_score,
            "e2_macro_auc": base_score,
            "outer_fold_choices": choices,
            "outer_fold_penalized_train_scores": outer_train_scores,
            "nested_blend_macro_auc": nested_score,
            "final_alpha": final_alpha,
            "final_descriptive_macro_auc": final_score,
            "full_grid_macro_auc": grid_scores,
            "positive_outer_folds": positive_folds,
            "per_target": {
                target: {
                    "e2_auc": base_target_scores[target],
                    "radimagenet_auc": rad_target_scores[target],
                    "blend_auc": final_target_scores[target],
                    "blend_delta": target_deltas[target],
                }
                for target in TARGETS
            },
            "target_regressions": target_regressions,
            "gold_fold_counts": {
                str(fold): int((gold_folds == fold).sum()) for fold in range(5)
            },
            "selection_supported": supported,
        }
        audit["train_available_slice_tokens"] = train_token_count
        audit["head_count"] = len(folds)
        if not supported:
            audit["status"] = "OOF_REJECTED_E2_PRESERVED"
            log(
                f"V52 rejected by nested OOF: base={base_score:.5f}, "
                f"nested={nested_score:.5f}, final={final_score:.5f}, choices={choices}"
            )
            return

        del features, token_mask
        gc.collect()
        test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
        test_series = pd.read_csv(
            ROOT / "test_series.csv",
            dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        )
        test_plane = dict(zip(test_series.SeriesInstanceUID, test_series.Anatomical_Plane))
        test_headers = annotate(walk("test_series"))
        audit_official_sequence_metadata(test_headers, test_series)
        test_studies, test_pixels, test_slot_mask = build_cache(
            pick_slots(test_headers, test_plane),
            test_plane,
            lat_of(test_headers, "test-v52 "),
            "test-v52",
        )
        test_by_uid = {str(uid): i for i, uid in enumerate(test_studies)}
        test_missing = [uid for uid in test.StudyInstanceUID if uid not in test_by_uid]
        if test_missing:
            raise RuntimeError(f"{len(test_missing)} test studies absent from cache")
        test_order = np.array([test_by_uid[uid] for uid in test.StudyInstanceUID])
        test_pixels = test_pixels[test_order]
        test_slot_mask = test_slot_mask[test_order]
        test_token_count = int(
            np.repeat(test_slot_mask[:, :, None], CACHE_SLICES, 2).sum()
        )
        if test_token_count < int(0.85 * len(test) * N_SLOT * CACHE_SLICES):
            raise RuntimeError(f"insufficient acquired test slices: {test_token_count}")
        test_features, test_token_mask = encode_radimagenet(
            test_pixels, test_slot_mask, device
        )
        del test_pixels, test_slot_mask, test_headers
        gc.collect()

        fold_predictions = []
        all_test = np.arange(len(test), dtype=np.int64)
        for record in folds:
            head = FoundationQueryHead().to(device)
            head.load_state_dict(record["state_dict"], strict=True)
            fold_predictions.append(
                predict_head(head, test_features, test_token_mask, all_test, device)
            )
            del head
            torch.cuda.empty_cache()
        if len(fold_predictions) != 5:
            raise RuntimeError("test inference did not use all five heads")
        rad_test = np.mean(np.stack(fold_predictions), axis=0)
        if not np.isfinite(rad_test).all():
            raise RuntimeError("non-finite RadImageNet test prediction")

        baseline = pd.read_csv(preserved, dtype={"StudyInstanceUID": str})
        _v52_validate_submission(baseline, test.StudyInstanceUID)
        rad_frame = pd.DataFrame(rad_test, columns=TARGETS)
        rad_frame.insert(0, "StudyInstanceUID", test.StudyInstanceUID)
        _v52_validate_submission(rad_frame, test.StudyInstanceUID)
        rad_frame.to_csv(output / "submission_rad_only.csv", index=False)
        baseline_rank = _v52_rank_columns(baseline[TARGETS].to_numpy())
        rad_test_rank = _v52_rank_columns(rad_test)
        selected_path = None
        for alpha in alpha_grid[1:]:
            candidate = baseline.copy()
            candidate[TARGETS] = (
                (1.0 - alpha) * baseline_rank + alpha * rad_test_rank
            )
            _v52_validate_submission(candidate, test.StudyInstanceUID)
            path = output / f"submission_e2_rad_{int(round(1000 * alpha)):03d}.csv"
            candidate.to_csv(path, index=False)
            if abs(float(alpha) - final_alpha) < 1e-12:
                selected_path = path
        if selected_path is None or not selected_path.is_file():
            raise RuntimeError("selected V52 blend artifact is absent")
        selected = pd.read_csv(selected_path, dtype={"StudyInstanceUID": str})
        _v52_validate_submission(selected, test.StudyInstanceUID)
        audit["test_studies"] = len(test)
        audit["test_available_slice_tokens"] = test_token_count
        audit["test_head_count"] = len(fold_predictions)
        audit["selected_path"] = str(selected_path)
        audit["selected_sha256"] = _v52_sha256(selected_path)
        audit["fallback_sha256"] = _v52_sha256(preserved)
        shutil.copy2(selected_path, primary)
        if _v52_sha256(primary) != audit["selected_sha256"]:
            raise RuntimeError("primary V52 copy hash mismatch")
        audit["status"] = "CANDIDATE_SELECTED"
        log(
            f"E9 selected alpha={final_alpha:.3f}; "
            f"nested={nested_score:.5f} vs E2 OOF={base_score:.5f}"
        )
    except Exception as error:
        audit["status"] = "ERROR_E2_PRESERVED"
        audit["error"] = f"{type(error).__name__}: {error}"
        audit["traceback"] = traceback.format_exc()
        log(f"E9 preserves E2: {audit['error']}")
    finally:
        if audit.get("status") != "CANDIDATE_SELECTED" and preserved.is_file():
            shutil.copy2(preserved, primary)
        audit["primary_sha256"] = _v52_sha256(primary) if primary.is_file() else None
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def _v52_load_pinned_e9b():
    """Load the public v15 heads and reconstruct the dual-OOF target gate."""
    heads_path = find_input_file("v52_radimagenet_heads.pt")
    remote_path = find_input_file("rad_e9_audit.json")
    public_path = find_input_file("public_oof_diagnostic.json")
    contract_path = find_input_file("e9b_contract.json")
    expected_hashes = {
        heads_path: PINNED_HEADS_SHA256,
        remote_path: PINNED_REMOTE_AUDIT_SHA256,
        public_path: PINNED_PUBLIC_DIAGNOSTIC_SHA256,
        contract_path: PINNED_E9B_CONTRACT_SHA256,
    }
    for path, expected in expected_hashes.items():
        observed = _v52_sha256(path)
        if observed != expected:
            raise RuntimeError(f"pinned E9b artifact drift for {path.name}: {observed}")

    remote = json.loads(remote_path.read_text())
    public = json.loads(public_path.read_text())
    contract = json.loads(contract_path.read_text())
    if remote.get("status") != "OOF_REJECTED_E2_PRESERVED":
        raise RuntimeError(f"unexpected v15 audit status {remote.get('status')}")
    if remote.get("primary_sha256") != (
        "f9fb57b7bac8489a5d5285b3984b06df57f142572be6417eac6341c43e96707a"
    ):
        raise RuntimeError("v15 did not preserve the exact E2 visible artifact")
    remote_oof = remote.get("oof", {})
    public_target = public.get("per_target", {})
    remote_target = remote_oof.get("per_target", {})
    if set(public_target) != set(TARGETS) or set(remote_target) != set(TARGETS):
        raise RuntimeError("E9b diagnostic target set drift")
    if int(remote.get("head_count", -1)) != 5:
        raise RuntimeError("v15 remote audit does not contain five heads")
    if int(remote_oof.get("positive_outer_folds", -1)) != 5:
        raise RuntimeError("v15 remote outer-fold support drift")
    if int(public.get("positive_outer_folds", -1)) != 5:
        raise RuntimeError("public outer-fold support drift")

    selected_targets = [
        target for target in TARGETS
        if float(public_target[target]["blend_delta"]) > 1e-12
        and float(remote_target[target]["blend_delta"]) > 1e-12
    ]
    preserved_targets = [target for target in TARGETS if target not in selected_targets]
    if selected_targets != contract.get("selected_targets"):
        raise RuntimeError(f"E9b selected-target contract drift: {selected_targets}")
    if preserved_targets != contract.get("preserved_targets"):
        raise RuntimeError(f"E9b preserved-target contract drift: {preserved_targets}")
    if len(selected_targets) != 10 or preserved_targets != ["Baker's", "Fracture"]:
        raise RuntimeError("E9b requires the ten-target dual-OOF intersection")
    alpha = float(contract.get("alpha", -1))
    if abs(alpha - 0.20) > 1e-12:
        raise RuntimeError(f"E9b alpha drift: {alpha}")

    def selective_macro(records, base_key, blend_key):
        return float(np.mean([
            float(records[target][blend_key] if target in selected_targets
                  else records[target][base_key])
            for target in TARGETS
        ]))

    public_base = float(public["base_gold_macro_auc"])
    remote_base = float(remote_oof["e2_macro_auc"])
    public_selective = selective_macro(public_target, "base_auc", "blend_auc")
    remote_selective = selective_macro(remote_target, "e2_auc", "blend_auc")
    if public_selective < public_base + 0.001:
        raise RuntimeError("E9b public selective gate no longer improves E2")
    if remote_selective < remote_base + 0.001:
        raise RuntimeError("E9b remote selective gate no longer improves E2")

    payload = torch.load(heads_path, map_location="cpu", weights_only=True)
    expected_payload = {
        "version": "v52-radimagenet-resnet50-official-1",
        "targets": TARGETS,
        "encoder_sha256": (
            "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
        ),
        "encoder_source_commit": "0ce16f7375db4236e646829d1eca61cdb4282133",
        "img": 224,
        "slices_per_plane": 8,
        "feature": "global_average_pool",
    }
    for key, expected in expected_payload.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"pinned E9b head contract drift for {key}")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError("pinned E9b payload requires five folds")
    if sorted(int(record.get("fold", -1)) for record in folds) != list(range(5)):
        raise RuntimeError("pinned E9b fold identity drift")
    if any(not isinstance(record.get("state_dict"), dict) for record in folds):
        raise RuntimeError("pinned E9b state dictionary is absent")
    if abs(float(payload.get("weak_oof_auc", -1)) - 0.8278261335825697) > 1e-12:
        raise RuntimeError("pinned E9b weak OOF drift")
    if abs(float(payload.get("gold_oof_auc", -1)) - 0.8543239133509962) > 1e-12:
        raise RuntimeError("pinned E9b gold OOF drift")
    return {
        "payload": payload,
        "folds": folds,
        "alpha": alpha,
        "selected_targets": selected_targets,
        "preserved_targets": preserved_targets,
        "public_base": public_base,
        "public_selective": public_selective,
        "remote_base": remote_base,
        "remote_selective": remote_selective,
        "remote_outer_fold_choices": remote_oof["outer_fold_choices"],
    }


def _v52_e10_remote_ladder(contract):
    """Recompute this account's half of the ladder from artifacts the kernel can read.

    The contract carries per-target gains for two independent RadImageNet OOF runs. Only the
    public run is unverifiable here, so its numbers stay data. The remote run is rebuilt from
    the attached OOF table, the pinned E2 OOF bundle and the official labels, and must match
    the contract exactly or E10 refuses to deploy.
    """
    train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    oof_path = find_input_file("v52_oof.csv")
    observed = _v52_sha256(oof_path)
    if observed != contract["remote_oof_sha256"]:
        raise RuntimeError(f"E10 remote OOF drift: {observed}")
    rad_frame = pd.read_csv(oof_path, dtype={"StudyInstanceUID": str})
    if rad_frame.columns.tolist() != ["StudyInstanceUID", *TARGETS, "fold", "is_gold"]:
        raise RuntimeError("E10 remote OOF schema drift")
    aligned = train[["StudyInstanceUID"]].merge(
        rad_frame, on="StudyInstanceUID", how="left", validate="one_to_one"
    )
    if aligned[TARGETS].isna().any().any():
        raise RuntimeError("E10 remote OOF does not cover every official train study")

    base_npz = find_input_file("oof.npz")
    with np.load(base_npz, allow_pickle=False) as bundle:
        if bundle["targets"].astype(str).tolist() != TARGETS:
            raise RuntimeError("E10 E2 OOF target order drift")
        if not np.array_equal(
            bundle["ids"].astype(str), train.StudyInstanceUID.astype(str).to_numpy()
        ):
            raise RuntimeError("E10 E2 OOF study order drift")
        if not np.array_equal(bundle["gold_mask"].astype(bool), gold):
            raise RuntimeError("E10 E2 gold mask differs from official train.csv")
        base_prediction = bundle["pred"].astype(np.float64)

    # Rank within the scored rows, matching both the E9b parent and test-time deployment
    # where the ranked population and the scored population are the same studies.
    base = _v52_rank_columns(base_prediction[gold])
    rad = _v52_rank_columns(aligned[TARGETS].to_numpy(np.float64)[gold])
    gold_y = train.loc[gold, TARGETS].to_numpy(np.float64)
    if len(gold_y) != 58 or not np.isfinite(base).all() or not np.isfinite(rad).all():
        raise RuntimeError("E10 gold alignment is incomplete or non-finite")
    reference = _v52_target_auc(gold_y, base)
    if abs(
        float(np.mean([reference[t] for t in TARGETS]))
        - float(contract["base_gold_macro_auc"])
    ) > 1e-9:
        raise RuntimeError("E10 base gold diagnostic drift")

    rebuilt = {}
    for key in contract["ladder"]:
        alpha = float(key)
        scores = _v52_target_auc(gold_y, (1.0 - alpha) * base + alpha * rad)
        rebuilt[key] = {t: scores[t] - reference[t] for t in TARGETS}
    pinned_remote = contract["per_target_ladder_delta"]["remote_v15"]
    if set(rebuilt) != set(pinned_remote):
        raise RuntimeError("E10 ladder key drift")
    for key, deltas in rebuilt.items():
        for target, delta in deltas.items():
            if abs(delta - float(pinned_remote[key][target])) > 1e-9:
                raise RuntimeError(
                    f"E10 recomputed remote gain disagrees at {key}/{target}: {delta}"
                )
    if any(abs(reference[t] - float(contract["per_target_base_auc"][t])) > 1e-9 for t in TARGETS):
        raise RuntimeError("E10 per-target base AUC drift")
    return rebuilt, reference


def _v52_load_e10():
    """Validate the E10 contract, then return the weight map the kernel will deploy."""
    heads_path = find_input_file("v52_radimagenet_heads.pt")
    remote_path = find_input_file("rad_e9_audit.json")
    contract_path = find_input_file("e10_contract.json")
    for path, expected in (
        (heads_path, PINNED_HEADS_SHA256),
        (remote_path, PINNED_REMOTE_AUDIT_SHA256),
        (contract_path, PINNED_E10_CONTRACT_SHA256),
    ):
        observed = _v52_sha256(path)
        if observed != expected:
            raise RuntimeError(f"pinned E10 artifact drift for {path.name}: {observed}")

    contract = json.loads(contract_path.read_text())
    if contract.get("version") != "e10-alpha-ladder-2":
        raise RuntimeError(f"unexpected E10 contract version {contract.get('version')}")
    if contract.get("targets") != TARGETS:
        raise RuntimeError("E10 contract target order drift")
    remote = json.loads(remote_path.read_text())
    if remote.get("status") != "OOF_REJECTED_E2_PRESERVED":
        raise RuntimeError(f"unexpected v15 audit status {remote.get('status')}")
    if remote.get("primary_sha256") != (
        "f9fb57b7bac8489a5d5285b3984b06df57f142572be6417eac6341c43e96707a"
    ):
        raise RuntimeError("v15 did not preserve the exact E2 visible artifact")
    if int(remote.get("head_count", -1)) != 5:
        raise RuntimeError("v15 remote audit does not contain five heads")

    rebuilt, base_auc = _v52_e10_remote_ladder(contract)
    public_ladder = contract["per_target_ladder_delta"]["public"]
    configuration = contract["configurations"].get(E10_CONFIG)
    if configuration is None:
        raise RuntimeError(f"E10 contract has no configuration {E10_CONFIG!r}")
    alpha_map = {t: float(configuration["alpha_map"][t]) for t in TARGETS}
    if any(alpha < 0.0 or alpha > 1.0 for alpha in alpha_map.values()):
        raise RuntimeError("E10 weight outside the unit interval")
    preserved = sorted(t for t, alpha in alpha_map.items() if alpha == 0.0)
    if preserved != sorted(E10_PRESERVED_TARGETS):
        raise RuntimeError(f"E10 preserved-target drift: {preserved}")

    # Two-tier gate. The scored objective is macro AUC, so the binding requirement is that
    # the deployed map raise the macro in BOTH independent runs -- the public numbers as
    # pinned data, this account's numbers as recomputed above. A per-target "never harm any
    # single label" rule is strictly stronger than that objective and would veto rungs that
    # trade a small loss on one label for a large gain on another, so it is enforced only for
    # configurations that actually claim it. Whichever claim the contract makes is verified;
    # a configuration cannot quietly assert dual-positivity it no longer has.
    claims_dual_positive = bool(configuration["all_dual_positive"])
    observed_dual_positive = True
    for target, alpha in alpha_map.items():
        if alpha == 0.0:
            continue
        key = f"{alpha:.2f}"
        if key not in rebuilt:
            raise RuntimeError(f"E10 weight {key} is outside the audited ladder")
        gains = (float(public_ladder[key][target]), float(rebuilt[key][target]))
        if not all(gain > 0 for gain in gains):
            observed_dual_positive = False
            if claims_dual_positive:
                raise RuntimeError(
                    f"E10 dual-source gate rejects {target} at {key}: {gains}"
                )
    if observed_dual_positive != claims_dual_positive:
        raise RuntimeError(
            f"E10 contract claims all_dual_positive={claims_dual_positive} for "
            f"{E10_CONFIG!r} but recomputation observes {observed_dual_positive}"
        )

    macro = {}
    for name, ladder in (("public", public_ladder), ("remote_v15", rebuilt)):
        total = 0.0
        for target, alpha in alpha_map.items():
            gain = 0.0 if alpha == 0.0 else float(ladder[f"{alpha:.2f}"][target])
            total += float(base_auc[target]) + gain
        macro[name] = total / len(TARGETS)
        if macro[name] <= float(contract["base_gold_macro_auc"]):
            raise RuntimeError(
                f"E10 macro gate rejects {E10_CONFIG!r}: {name} macro {macro[name]} "
                f"does not beat base {contract['base_gold_macro_auc']}"
            )
        if abs(macro[name] - float(configuration["descriptive_macro"][name])) > 1e-9:
            raise RuntimeError(
                f"E10 recomputed {name} macro {macro[name]} disagrees with the contract "
                f"value {configuration['descriptive_macro'][name]}"
            )

    payload = torch.load(heads_path, map_location="cpu", weights_only=True)
    expected_payload = {
        "version": "v52-radimagenet-resnet50-official-1",
        "targets": TARGETS,
        "encoder_sha256": (
            "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
        ),
        "encoder_source_commit": "0ce16f7375db4236e646829d1eca61cdb4282133",
        "img": 224,
        "slices_per_plane": 8,
        "feature": "global_average_pool",
    }
    for key, expected in expected_payload.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"pinned E10 head contract drift for {key}")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 5:
        raise RuntimeError("pinned E10 payload requires five folds")
    if sorted(int(record.get("fold", -1)) for record in folds) != list(range(5)):
        raise RuntimeError("pinned E10 fold identity drift")
    if any(not isinstance(record.get("state_dict"), dict) for record in folds):
        raise RuntimeError("pinned E10 state dictionary is absent")
    if abs(float(payload.get("gold_oof_auc", -1)) - 0.8543239133509962) > 1e-12:
        raise RuntimeError("pinned E10 gold OOF drift")
    return {
        "payload": payload,
        "folds": folds,
        "contract": contract,
        "configuration": E10_CONFIG,
        "alpha_map": alpha_map,
        "preserved_targets": sorted(E10_PRESERVED_TARGETS),
        "diagnostic_macro": configuration["diagnostic_macro"],
        "recomputed_macro": macro,
        "all_dual_positive": claims_dual_positive,
        "rationale": configuration["rationale"],
        "recomputed_remote_ladder": rebuilt,
    }


def main_v52_pinned_e9b():
    """Inference-only E9b from hash-pinned v15 heads; preserve E2 on any failure."""
    import shutil

    output = Path("/kaggle/working/rsna_rad_e9b")
    output.mkdir(parents=True, exist_ok=True)
    primary = Path("/kaggle/working/submission.csv")
    preserved = Path("/kaggle/working/submission_e2_preserved.csv")
    audit_path = Path("/kaggle/working/rad_e9b_audit.json")
    audit = {
        "status": "E2_PRESERVED",
        "mode": "pinned_v15_heads_inference_only",
        "evidence_boundary": (
            "OOF values are diagnostics, not Kaggle scores. The fixed public 20-percent "
            "vote is applied only to targets improving in two independent OOF runs."
        ),
        "encoder": "RadImageNet ResNet-50 official PyTorch release",
        "encoder_license": "CC-BY-NC-SA-4.0",
        "encoder_sha256": (
            "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
        ),
        "heads_sha256": PINNED_HEADS_SHA256,
        "parent": "E2 captured 20-member DINOv2 rank ensemble",
        "pixel_rules": dict(RULES),
    }
    if not primary.is_file():
        raise FileNotFoundError("E2 parent submission is absent")
    shutil.copy2(primary, preserved)

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("E9b RadImageNet inference requires CUDA")
        elapsed = max(0.0, time.time() - float(globals().get("T0", time.time())))
        available = 8.72 * 3600 - elapsed
        audit["elapsed_before_e9b_seconds"] = elapsed
        audit["available_at_start_seconds"] = available
        if available < 45 * 60:
            raise TimeoutError(f"only {available / 60:.1f} minutes remain")

        pinned = _v52_load_pinned_e9b()
        audit["oof_gate"] = {
            "alpha": pinned["alpha"],
            "selected_targets": pinned["selected_targets"],
            "preserved_targets": pinned["preserved_targets"],
            "public_e2_macro_auc": pinned["public_base"],
            "public_selective_macro_auc": pinned["public_selective"],
            "remote_e2_macro_auc": pinned["remote_base"],
            "remote_selective_macro_auc": pinned["remote_selective"],
            "remote_outer_fold_choices": pinned["remote_outer_fold_choices"],
            "selection_supported": True,
        }

        test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
        test_series = pd.read_csv(
            ROOT / "test_series.csv",
            dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        )
        plane = dict(zip(test_series.SeriesInstanceUID, test_series.Anatomical_Plane))
        headers = annotate(walk("test_series"))
        audit_official_sequence_metadata(headers, test_series)
        studies, pixels, slot_mask = build_cache(
            pick_slots(headers, plane), plane, lat_of(headers, "test-e9b "), "test-e9b"
        )
        by_uid = {str(uid): index for index, uid in enumerate(studies)}
        missing = [uid for uid in test.StudyInstanceUID if uid not in by_uid]
        if missing:
            raise RuntimeError(f"{len(missing)} test studies absent from E9b cache")
        order = np.asarray([by_uid[uid] for uid in test.StudyInstanceUID], dtype=np.int64)
        pixels, slot_mask = pixels[order], slot_mask[order]
        token_count = int(np.repeat(slot_mask[:, :, None], CACHE_SLICES, 2).sum())
        if token_count < int(0.85 * len(test) * N_SLOT * CACHE_SLICES):
            raise RuntimeError(f"insufficient acquired E9b test slices: {token_count}")
        features, token_mask = encode_radimagenet(pixels, slot_mask, device)
        del pixels, slot_mask, headers
        gc.collect()

        all_test = np.arange(len(test), dtype=np.int64)
        fold_predictions = []
        for record in pinned["folds"]:
            head = FoundationQueryHead().to(device)
            head.load_state_dict(record["state_dict"], strict=True)
            fold_predictions.append(
                predict_head(head, features, token_mask, all_test, device)
            )
            del head
            torch.cuda.empty_cache()
        if len(fold_predictions) != 5:
            raise RuntimeError("E9b test inference did not use all five heads")
        rad_test = np.mean(np.stack(fold_predictions), axis=0)
        if rad_test.shape != (len(test), len(TARGETS)) or not np.isfinite(rad_test).all():
            raise RuntimeError(f"invalid E9b prediction shape/value: {rad_test.shape}")

        baseline = pd.read_csv(preserved, dtype={"StudyInstanceUID": str})
        _v52_validate_submission(baseline, test.StudyInstanceUID)
        rad_frame = pd.DataFrame(rad_test, columns=TARGETS)
        rad_frame.insert(0, "StudyInstanceUID", test.StudyInstanceUID)
        _v52_validate_submission(rad_frame, test.StudyInstanceUID)
        rad_frame.to_csv(output / "submission_rad_only.csv", index=False)
        baseline_rank = _v52_rank_columns(baseline[TARGETS].to_numpy())
        rad_rank = _v52_rank_columns(rad_test)
        selected = baseline.copy()
        alpha = pinned["alpha"]
        for target in pinned["selected_targets"]:
            index = TARGETS.index(target)
            selected[target] = (
                (1.0 - alpha) * baseline_rank[:, index] + alpha * rad_rank[:, index]
            )
        for target in pinned["preserved_targets"]:
            if not np.array_equal(
                selected[target].to_numpy(), baseline[target].to_numpy()
            ):
                raise RuntimeError(f"E9b failed to preserve {target}")
        _v52_validate_submission(selected, test.StudyInstanceUID)
        selected_path = output / "submission_e2_rad_robust_200.csv"
        selected.to_csv(selected_path, index=False)
        selected = pd.read_csv(selected_path, dtype={"StudyInstanceUID": str})
        _v52_validate_submission(selected, test.StudyInstanceUID)

        audit.update({
            "test_studies": len(test),
            "test_available_slice_tokens": token_count,
            "test_head_count": len(fold_predictions),
            "selected_path": str(selected_path),
            "selected_sha256": _v52_sha256(selected_path),
            "fallback_sha256": _v52_sha256(preserved),
        })
        shutil.copy2(selected_path, primary)
        if _v52_sha256(primary) != audit["selected_sha256"]:
            raise RuntimeError("primary E9b copy hash mismatch")
        audit["status"] = "CANDIDATE_SELECTED"
        log(
            f"E9b selected alpha={alpha:.3f} on "
            f"{len(pinned['selected_targets'])} dual-OOF-stable targets; "
            "Baker's and Fracture preserve E2"
        )
    except Exception as error:
        audit["status"] = "ERROR_E2_PRESERVED"
        audit["error"] = f"{type(error).__name__}: {error}"
        audit["traceback"] = traceback.format_exc()
        log(f"E9b preserves E2: {audit['error']}")
    finally:
        if audit.get("status") != "CANDIDATE_SELECTED" and preserved.is_file():
            shutil.copy2(preserved, primary)
        audit["primary_sha256"] = _v52_sha256(primary) if primary.is_file() else None
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def _v52_rad_test_predictions(pinned, test, device, tag):
    """Five-head RadImageNet test prediction on the official test tree."""
    test_series = pd.read_csv(
        ROOT / "test_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    plane = dict(zip(test_series.SeriesInstanceUID, test_series.Anatomical_Plane))
    headers = annotate(walk("test_series"))
    audit_official_sequence_metadata(headers, test_series)
    studies, pixels, slot_mask = build_cache(
        pick_slots(headers, plane), plane, lat_of(headers, f"{tag} "), tag
    )
    by_uid = {str(uid): index for index, uid in enumerate(studies)}
    missing = [uid for uid in test.StudyInstanceUID if uid not in by_uid]
    if missing:
        raise RuntimeError(f"{len(missing)} test studies absent from {tag} cache")
    order = np.asarray([by_uid[uid] for uid in test.StudyInstanceUID], dtype=np.int64)
    pixels, slot_mask = pixels[order], slot_mask[order]
    token_count = int(np.repeat(slot_mask[:, :, None], CACHE_SLICES, 2).sum())
    if token_count < int(0.85 * len(test) * N_SLOT * CACHE_SLICES):
        raise RuntimeError(f"insufficient acquired {tag} test slices: {token_count}")
    features, token_mask = encode_radimagenet(pixels, slot_mask, device)
    del pixels, slot_mask, headers
    gc.collect()

    rows = np.arange(len(test), dtype=np.int64)
    predictions = []
    for record in pinned["folds"]:
        head = FoundationQueryHead().to(device)
        head.load_state_dict(record["state_dict"], strict=True)
        predictions.append(predict_head(head, features, token_mask, rows, device))
        del head
        torch.cuda.empty_cache()
    if len(predictions) != 5:
        raise RuntimeError(f"{tag} test inference did not use all five heads")
    rad_test = np.mean(np.stack(predictions), axis=0)
    if rad_test.shape != (len(test), len(TARGETS)) or not np.isfinite(rad_test).all():
        raise RuntimeError(f"invalid {tag} prediction shape/value: {rad_test.shape}")
    return rad_test, token_count, len(predictions)


def main_v52_e10():
    """Deploy the audited E10 weight map; preserve the E2 parent on any failure."""
    import shutil

    output = Path("/kaggle/working/rsna_rad_e10")
    output.mkdir(parents=True, exist_ok=True)
    primary = Path("/kaggle/working/submission.csv")
    preserved = Path("/kaggle/working/submission_e2_preserved.csv")
    audit_path = Path("/kaggle/working/rad_e10_audit.json")
    audit = {
        "status": "E2_PRESERVED",
        "mode": "pinned_v15_heads_inference_only",
        "experiment": "E10",
        "configuration": E10_CONFIG,
        "evidence_boundary": (
            "OOF values are 58-study diagnostics on official train labels, not Kaggle "
            "scores. E10 widens the weight ladder that E9 truncated at 0.25 and votes only "
            "where two independent OOF runs agree at the deployed weight."
        ),
        "encoder": "RadImageNet ResNet-50 official PyTorch release",
        "encoder_license": "CC-BY-NC-SA-4.0",
        "heads_sha256": PINNED_HEADS_SHA256,
        "contract_sha256": PINNED_E10_CONTRACT_SHA256,
        "parent": "E2 captured 20-member DINOv2 rank ensemble",
        "pixel_rules": dict(RULES),
    }
    if not primary.is_file():
        raise FileNotFoundError("E2 parent submission is absent")
    shutil.copy2(primary, preserved)

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("E10 RadImageNet inference requires CUDA")
        elapsed = max(0.0, time.time() - float(globals().get("T0", time.time())))
        available = 8.72 * 3600 - elapsed
        audit["elapsed_before_e10_seconds"] = elapsed
        audit["available_at_start_seconds"] = available
        if available < 45 * 60:
            raise TimeoutError(f"only {available / 60:.1f} minutes remain")

        pinned = _v52_load_e10()
        audit["weight_gate"] = {
            "configuration": pinned["configuration"],
            "alpha_map": pinned["alpha_map"],
            "preserved_targets": pinned["preserved_targets"],
            "diagnostic_macro": pinned["diagnostic_macro"],
            "recomputed_macro": pinned["recomputed_macro"],
            "all_dual_positive": pinned["all_dual_positive"],
            "base_gold_macro_auc": pinned["contract"]["base_gold_macro_auc"],
            "rationale": pinned["rationale"],
            "remote_ladder_recomputed_in_kernel": True,
        }

        test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
        rad_test, token_count, head_count = _v52_rad_test_predictions(
            pinned, test, device, "test-e10"
        )

        baseline = pd.read_csv(preserved, dtype={"StudyInstanceUID": str})
        _v52_validate_submission(baseline, test.StudyInstanceUID)
        rad_frame = pd.DataFrame(rad_test, columns=TARGETS)
        rad_frame.insert(0, "StudyInstanceUID", test.StudyInstanceUID)
        _v52_validate_submission(rad_frame, test.StudyInstanceUID)
        rad_frame.to_csv(output / "submission_rad_only.csv", index=False)
        baseline_rank = _v52_rank_columns(baseline[TARGETS].to_numpy())
        rad_rank = _v52_rank_columns(rad_test)

        # Materialise every audited rung so the ladder is inspectable from one run; only the
        # configured map is promoted to the visible submission.
        written = {}
        for name, configuration in sorted(pinned["contract"]["configurations"].items()):
            frame = baseline.copy()
            for target, alpha in configuration["alpha_map"].items():
                alpha = float(alpha)
                if alpha > 0:
                    index = TARGETS.index(target)
                    frame[target] = (
                        (1.0 - alpha) * baseline_rank[:, index] + alpha * rad_rank[:, index]
                    )
            for target, alpha in configuration["alpha_map"].items():
                if float(alpha) == 0.0 and not np.array_equal(
                    frame[target].to_numpy(), baseline[target].to_numpy()
                ):
                    raise RuntimeError(f"E10 failed to preserve {target} in {name}")
            _v52_validate_submission(frame, test.StudyInstanceUID)
            path = output / f"submission_e10_{name}.csv"
            frame.to_csv(path, index=False)
            written[name] = _v52_sha256(path)
        audit["ladder_sha256"] = written

        selected_path = output / f"submission_e10_{pinned['configuration']}.csv"
        selected = pd.read_csv(selected_path, dtype={"StudyInstanceUID": str})
        _v52_validate_submission(selected, test.StudyInstanceUID)
        for target in pinned["preserved_targets"]:
            if not np.array_equal(
                selected[target].to_numpy(), baseline[target].to_numpy()
            ):
                raise RuntimeError(f"E10 promoted file does not preserve {target}")
        audit.update({
            "test_studies": len(test),
            "test_available_slice_tokens": token_count,
            "test_head_count": head_count,
            "selected_path": str(selected_path),
            "selected_sha256": _v52_sha256(selected_path),
            "fallback_sha256": _v52_sha256(preserved),
        })
        shutil.copy2(selected_path, primary)
        if _v52_sha256(primary) != audit["selected_sha256"]:
            raise RuntimeError("primary E10 copy hash mismatch")
        audit["status"] = "CANDIDATE_SELECTED"
        voted = sorted(t for t, alpha in pinned["alpha_map"].items() if alpha > 0)
        log(
            f"E10 promoted {pinned['configuration']} over {len(voted)} dual-OOF targets; "
            f"{', '.join(pinned['preserved_targets'])} preserve E2"
        )
    except Exception as error:
        audit["status"] = "ERROR_E2_PRESERVED"
        audit["error"] = f"{type(error).__name__}: {error}"
        audit["traceback"] = traceback.format_exc()
        log(f"E10 preserves E2: {audit['error']}")
    finally:
        if audit.get("status") != "CANDIDATE_SELECTED" and preserved.is_file():
            shutil.copy2(preserved, primary)
        audit["primary_sha256"] = _v52_sha256(primary) if primary.is_file() else None
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def _v52_e11_availability(headers, plane_map):
    """Count studies offering each (plane, fat-suppression) pair before any slot is picked.

    The parent arm reads only fat-suppressed series and never had to ask how many studies
    carry a non-suppressed one. E11 depends on that answer, so it is measured and logged
    rather than assumed: a slot nobody can fill is a masked column, and a run that produced
    one silently would look like a weak arm instead of an absent input.
    """
    frame = headers[["StudyInstanceUID", "SeriesInstanceUID", "fatsat"]].copy()
    frame["plane"] = frame.SeriesInstanceUID.map(plane_map)
    total = frame.StudyInstanceUID.nunique()
    table = {}
    for plane in ("Sagittal", "Coronal", "Axial"):
        for fatsat in (True, False):
            selected = frame[(frame.plane == plane) & (frame.fatsat == bool(fatsat))]
            studies = selected.StudyInstanceUID.nunique()
            key = f"{plane}_{'FS' if fatsat else 'NOFS'}"
            table[key] = {
                "studies": int(studies),
                "fraction": float(studies / total) if total else 0.0,
                "series": int(len(selected)),
            }
            log(f"E11 availability {key}: {studies}/{total} studies, {len(selected)} series")
    return table


def main_v52_e11():
    """Train a third arm on a deliberately different pixel recipe. Never ships a candidate."""
    import shutil
    from sklearn.model_selection import GroupKFold

    output = Path("/kaggle/working/rsna_rad_e11")
    output.mkdir(parents=True, exist_ok=True)
    primary = Path("/kaggle/working/submission.csv")
    preserved = Path("/kaggle/working/submission_e2_preserved.csv")
    audit_path = Path("/kaggle/working/rad_e11_audit.json")
    audit = {
        "status": "E2_PRESERVED",
        "mode": "e11-diverse-recipe-training-only",
        "evidence_boundary": (
            "Every value here is a local out-of-fold diagnostic on 58 official image "
            "labels. It is not a Kaggle score, and this mode never replaces the parent "
            "submission under any outcome."
        ),
        "recipe": {
            "slots": [list(slot) for slot in E11_SLOTS],
            "crop_mm": E11_CROP_MM,
            "cache_slices": E11_CACHE_SLICES,
            "img": E11_IMG,
            "differs_from_parent_arm": (
                "parent reads 3 fat-suppressed slots at full frame; this reads 3 "
                "non-suppressed slots plus 1 suppressed anchor at a 130 mm physical crop"
            ),
        },
        "encoder": "RadImageNet ResNet-50 official PyTorch release",
        "encoder_license": "CC-BY-NC-SA-4.0 (Kaggle-hosted weight metadata)",
        "encoder_sha256": "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734",
    }
    if not primary.is_file():
        raise FileNotFoundError("E2 parent submission is absent")
    shutil.copy2(primary, preserved)

    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("E11 training requires CUDA")
        elapsed = max(0.0, time.time() - float(globals().get("T0", time.time())))
        available = TIME_BUDGET - elapsed
        audit["elapsed_before_e11_seconds"] = elapsed
        audit["available_at_start_seconds"] = available
        if available < 2.5 * 3600:
            raise TimeoutError(f"only {available / 60:.1f} minutes remain")

        # Point the shared pixel path at the E11 recipe. These are the same process globals
        # the parent notebook's readers consult, so the override has to happen before any
        # slot is picked or any pixel is decoded, and nothing after this point may assume
        # the parent arm's values.
        globals().update(
            SLOTS=list(E11_SLOTS),
            N_SLOT=len(E11_SLOTS),
            CACHE_SLICES=int(E11_CACHE_SLICES),
            IMG=int(E11_IMG),
            CACHE_IMG=int(E11_IMG),
            CROP_MM=float(E11_CROP_MM),
        )
        log(
            f"E11 recipe: {[s[0] for s in E11_SLOTS]} at {E11_CROP_MM:.0f} mm, "
            f"{E11_IMG} px, {E11_CACHE_SLICES} slices/slot"
        )

        train = pd.read_csv(ROOT / "train.csv", dtype={"StudyInstanceUID": str})
        train_series = pd.read_csv(
            ROOT / "train_series.csv",
            dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        )
        if len(train) != 4407:
            raise RuntimeError(f"unexpected train study count {len(train)}")
        plane = dict(zip(train_series.SeriesInstanceUID, train_series.Anatomical_Plane))
        headers = annotate(walk("train_series"))
        audit["availability"] = _v52_e11_availability(headers, plane)

        studies, pixels, slot_mask = build_cache(
            pick_slots(headers, plane), plane, lat_of(headers, "train-e11 "), "train-e11"
        )
        by_uid = {str(uid): i for i, uid in enumerate(studies)}
        missing = [uid for uid in train.StudyInstanceUID if uid not in by_uid]
        if missing:
            raise RuntimeError(f"{len(missing)} train studies absent from cache")
        order = np.array([by_uid[uid] for uid in train.StudyInstanceUID], dtype=np.int64)
        pixels, slot_mask = pixels[order], slot_mask[order]
        fill = float(slot_mask.mean())
        per_slot = {
            name: float(slot_mask[:, k].mean())
            for k, (name, _, _, _) in enumerate(E11_SLOTS)
        }
        audit["slot_fill"] = per_slot
        audit["overall_fill"] = fill
        for name, value in per_slot.items():
            log(f"E11 slot fill {name}: {value:.1%}")
        if fill < E11_MIN_FILL:
            raise RuntimeError(f"E11 slot fill {fill:.1%} below the {E11_MIN_FILL:.0%} floor")

        features, token_mask = encode_radimagenet(pixels, slot_mask, device)
        del pixels, slot_mask, headers
        gc.collect()

        y, weights, gold = make_targets(train)
        if int(gold.sum()) != 58:
            raise RuntimeError(f"expected 58 fully gold studies, observed {int(gold.sum())}")
        groups = report_groups(train)
        if len(np.unique(groups)) < 4000:
            raise RuntimeError("unexpected report-group collapse")

        splits = list(GroupKFold(5).split(features, groups=groups))
        fold_id = np.full(len(train), -1, dtype=np.int8)
        folds = []
        oof = np.zeros_like(y, dtype=np.float32)
        for fold, (tr, va) in enumerate(splits):
            if set(groups[tr]).intersection(groups[va]):
                raise RuntimeError(f"report leakage in fold {fold}")
            fold_id[va] = fold
            state, score = train_fold(
                features, token_mask, y, weights, tr, va, fold, device
            )
            if state is None:
                raise RuntimeError(f"fold {fold} produced no checkpoint")
            head = FoundationQueryHead().to(device)
            head.load_state_dict(state, strict=True)
            oof[va] = predict_head(head, features, token_mask, va, device)
            folds.append({"fold": fold, "weak_auc": float(score), "state_dict": state})
            del head
            torch.cuda.empty_cache()
        if (fold_id < 0).any() or not np.isfinite(oof).all():
            raise RuntimeError("incomplete E11 OOF")

        weak_auc = macro_auc(y, oof)
        gold_auc = macro_auc(y[gold], oof[gold])
        audit["weak_oof_auc"] = float(weak_auc)
        audit["gold_oof_auc"] = float(gold_auc)
        log(f"E11 OOF weak macro AUC {weak_auc:.5f}")
        log(f"E11 OOF gold macro AUC {gold_auc:.5f} on 58 studies")

        # The question E11 exists to answer is not whether this arm is strong on its own but
        # whether it says something the portfolio does not already know. Both halves are
        # measured against the same 58 rows and the same rank basis the deployed blend uses.
        base_npz = find_input_file("oof.npz")
        with np.load(base_npz, allow_pickle=False) as bundle:
            if bundle["targets"].astype(str).tolist() != TARGETS:
                raise RuntimeError("E11 E2 OOF target order drift")
            if not np.array_equal(
                bundle["ids"].astype(str), train.StudyInstanceUID.astype(str).to_numpy()
            ):
                raise RuntimeError("E11 E2 OOF study order drift")
            base_prediction = bundle["pred"].astype(np.float64)
        base = _v52_rank_columns(base_prediction[gold])
        new = _v52_rank_columns(oof[gold].astype(np.float64))
        gold_y = train.loc[gold, TARGETS].to_numpy(np.float64)
        reference = _v52_target_auc(gold_y, base)
        audit["e2_base_gold_macro"] = float(np.mean([reference[t] for t in TARGETS]))
        ladder = {}
        for alpha in (0.20, 0.35, 0.50):
            scores = _v52_target_auc(gold_y, (1.0 - alpha) * base + alpha * new)
            ladder[f"{alpha:.2f}"] = {
                "macro": float(np.mean([scores[t] for t in TARGETS])),
                "per_target_delta": {t: float(scores[t] - reference[t]) for t in TARGETS},
            }
            log(f"E11 blend alpha={alpha:.2f} gold macro {ladder[f'{alpha:.2f}']['macro']:.5f}")
        audit["blend_vs_e2"] = ladder

        try:
            parent_oof = pd.read_csv(
                find_input_file("v52_oof.csv"), dtype={"StudyInstanceUID": str}
            )
            aligned = train[["StudyInstanceUID"]].merge(
                parent_oof, on="StudyInstanceUID", how="left", validate="one_to_one"
            )
            parent = _v52_rank_columns(aligned[TARGETS].to_numpy(np.float64)[gold])
            audit["correlation_with_parent_arm"] = {
                t: float(np.corrcoef(parent[:, i], new[:, i])[0, 1])
                for i, t in enumerate(TARGETS)
            }
            log(
                "E11 mean rank correlation with the parent arm: "
                f"{np.mean(list(audit['correlation_with_parent_arm'].values())):.3f}"
            )
        except FileNotFoundError:
            audit["correlation_with_parent_arm"] = None

        torch.save(
            {
                "version": "e11-radimagenet-resnet50-diverse-1",
                "targets": TARGETS,
                "encoder_sha256": audit["encoder_sha256"],
                "slots": [list(slot) for slot in E11_SLOTS],
                "crop_mm": E11_CROP_MM,
                "img": E11_IMG,
                "slices_per_plane": E11_CACHE_SLICES,
                "feature": "global_average_pool",
                "folds": folds,
                "weak_oof_auc": float(weak_auc),
                "gold_oof_auc": float(gold_auc),
            },
            output / "v52_e11_heads.pt",
        )
        oof_frame = pd.DataFrame(oof, columns=TARGETS)
        oof_frame.insert(0, "StudyInstanceUID", train.StudyInstanceUID)
        oof_frame["fold"] = fold_id
        oof_frame["is_gold"] = gold.astype(np.uint8)
        oof_frame.to_csv(output / "v52_e11_oof.csv", index=False)
        audit["status"] = "E11_TRAINED_E2_PRESERVED"
        audit["heads_sha256"] = _v52_sha256(output / "v52_e11_heads.pt")
        audit["oof_sha256"] = _v52_sha256(output / "v52_e11_oof.csv")
    except Exception as error:
        audit["status"] = "ERROR_E2_PRESERVED"
        audit["error"] = f"{type(error).__name__}: {error}"
        audit["traceback"] = traceback.format_exc()
        log(f"E11 preserves E2: {audit['error']}")
    finally:
        if preserved.is_file():
            shutil.copy2(preserved, primary)
        audit["primary_sha256"] = _v52_sha256(primary) if primary.is_file() else None
        audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


if ARM_MODE == "e11":
    main_v52_e11()
else:
    try:
        find_input_file("v52_radimagenet_heads.pt")
    except FileNotFoundError:
        main_v52()
    else:
        try:
            find_input_file("e10_contract.json")
        except FileNotFoundError:
            main_v52_pinned_e9b()
        else:
            main_v52_e10()



# ======== CELL 32 [code] ========
if ARM_MODE == 'e11':
    print('e11 training mode: V40 submission receipt skipped; artifacts are rad_e11_audit.json + rsna_rad_e11/*')
else:
    # Fail-closed V40 runtime receipt.
    import hashlib as _v40_hashlib
    import json as _v40_json
    from pathlib import Path as _V40Path
    import numpy as _v40_np
    import pandas as _v40_pd

    _v40_work = _V40Path('/kaggle/working')
    _v40_primary = _v40_work / 'submission.csv'
    _v40_parent = _v40_work / 'submission_e2_preserved.csv'
    _v40_e10_audit = _v40_work / 'rad_e10_audit.json'
    _v40_test = _v40_pd.read_csv(COMP / 'test.csv', dtype={'StudyInstanceUID': str})
    _v40_sub = _v40_pd.read_csv(_v40_primary, dtype={'StudyInstanceUID': str})
    _v40_expected = ['StudyInstanceUID', *TARGETS]
    if _v40_sub.columns.tolist() != _v40_expected:
        raise RuntimeError('V40 submission schema drift')
    if _v40_sub.StudyInstanceUID.tolist() != _v40_test.StudyInstanceUID.astype(str).tolist():
        raise RuntimeError('V40 dynamic test identity/order drift')
    _v40_values = _v40_sub[TARGETS].to_numpy(float)
    if not _v40_np.isfinite(_v40_values).all() or _v40_values.min() < 0 or _v40_values.max() > 1:
        raise RuntimeError('V40 invalid submission values')
    if not _v40_parent.is_file() or not _v40_e10_audit.is_file():
        raise RuntimeError('V40 missing parent or E10 receipt')
    _v40_audit = _v40_json.loads(_v40_e10_audit.read_text())
    if _v40_audit.get('status') != 'CANDIDATE_SELECTED' or _v40_audit.get('configuration') != 'uniform_060':
        raise RuntimeError('V40 E10 promotion contract failed')
    _v40_sha = lambda p: _v40_hashlib.sha256(p.read_bytes()).hexdigest()
    if _v40_sha(_v40_primary) != _v40_audit.get('selected_sha256'):
        raise RuntimeError('V40 E10 output hash mismatch')
    _v40_receipt = {
        'status': 'VALID_DINOV3_E10_HYBRID_ALPHA060',
        'test_studies': len(_v40_sub),
        'dynamic_test_ids_exact': True,
        'schema_exact': True,
        'finite_in_range': True,
        'parent': 'DINOv2 rank ensemble plus cross-series DINOv3',
        'correction': 'E10 RadImageNet uniform 0.60; Baker and Fracture preserved',
        'parent_sha256': _v40_sha(_v40_parent),
        'submission_sha256': _v40_sha(_v40_primary),
        'e10_contract_sha256': _v40_audit.get('contract_sha256'),
        'e10_heads_sha256': _v40_audit.get('heads_sha256'),
    }
    (_v40_work / 'v40_runtime_audit.json').write_text(
        _v40_json.dumps(_v40_receipt, indent=2, sort_keys=True) + '\n'
    )
    print(_v40_json.dumps(_v40_receipt, indent=2, sort_keys=True))


# ======== CELL 33 [markdown] ========
## License

This notebook is a derivative work of **prvsiyan**'s *RSNA Knee: read the report,
then the knee*, released under the **Apache License 2.0**, and is published under
the same licence.

Changes from the original: most of its later staging was removed, a DINOv3
ViT-S/16 was added, and the RadImageNet heads were retrained on our own fold
split.

If you fork this notebook, Apache 2.0 requires you to keep these notices and the
attributions above, and to state your own changes.

# ======== CELL 34 [markdown] ========
## V40 change notice

V40 preserves the Apache-2.0 notices and author attributions in the parent notebook. Its E10 stage is adapted from fishface's public competition notebook and the public prvsiyan pipeline, with the hash-pinned Antoine Gagnon's v15 heads. The V40 change is limited to replacing the older Rad15 correction with the precommitted E10 uniform-0.35 correction after the DINOv3 parent is produced.

