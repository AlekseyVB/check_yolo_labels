# %%
"""Read-only YOLO detection audit for Windows and VS Code.
Without --samples, asks how many evenly spaced images to show.
Run in VS Code's Python Interactive Window for inline images, or open the
generated preview.ipynb after a terminal run. Source files stay intact.
"""
import argparse
import base64
import collections
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import zipfile
from PIL import Image, ImageDraw, ImageFont

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
COLORS = ['#00ff88', '#ffbf00', '#ff5ce0', '#00dfff', '#ff654f', '#af8aff']
DEFAULT_DATASET = Path(r'G:\Мой диск\BusCam-for-DataSet\CH01-20260926-dataset\people_india_bus-1.yolo26')
SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()

def in_notebook():
    try:
        from IPython import get_ipython
        return getattr(get_ipython(), 'kernel', None) is not None
    except ImportError:
        return False

def evenly_spaced_indices(total, count):
    """Zero-based indices, including both ends when count >= 2.

    Uses integer arithmetic with rounding to the nearest available image.
    For count == 1 only the first image can be selected.
    """
    if total < 1 or count < 1:
        raise ValueError('total and count must be positive')
    count = min(total, count)
    if count == 1:
        return [0]
    denominator = count - 1
    return [(2*i*(total-1)+denominator)//(2*denominator) for i in range(count)]

def ask_sample_count(total):
    while True:
        try:
            value = int(input(f'Найдено {total} изображений. Сколько файлов показать (1–{total})? '))
        except ValueError:
            print('Введите целое число, например 10.'); continue
        except EOFError:
            raise SystemExit('Ввод недоступен. Запустите интерактивно или укажите --samples 10.')
        if 1 <= value <= total:
            return value
        print(f'Введите число от 1 до {total}.')

def show_inline(entries, out):
    from IPython.display import Image as DisplayImage, Markdown, display
    for i, r in enumerate(entries, 1):
        display(Markdown(f'### {i}. Файл {r["position"]} из {r["total"]} — {r["boxes"]} боксов\n\n`{r["source"]}`'))
        display(Markdown('**Исходный кадр**'))
        display(DisplayImage(filename=str(out/r['original']), width=1200))
        display(Markdown('**Разметка из .txt**'))
        display(DisplayImage(filename=str(out/r['overlay']), width=1200))

def write_preview_notebook(entries, out):
    """Embed images so opening the notebook shows results without running code."""
    cells = [{'cell_type':'markdown', 'metadata':{}, 'source':[
        '# Проверка YOLO-разметки\n',
        'Результаты уже встроены: исходный кадр, затем его разметка. '
        'Это не предсказания модели. Для новой выборки снова запустите check_yolo_labels.py.\n']}]
    for i, r in enumerate(entries, 1):
        cells.append({'cell_type':'markdown','metadata':{},'source':[
            f'## {i}. Файл {r["position"]} из {r["total"]} — {r["boxes"]} боксов\n',
            f'`{r["source"]}`\n\nИсходный кадр, затем кадр с боксами.\n']})
        outputs = []
        for name in [r['original'], r['overlay']]:
            # Originals can be PNG/TIFF etc.; embed a JPEG conversion for a portable notebook.
            from io import BytesIO
            buffer = BytesIO()
            with Image.open(out/name) as im:
                im.convert('RGB').save(buffer, format='JPEG', quality=93)
            outputs.append({'output_type':'display_data','metadata':{'image/jpeg':{'width':1200}},
                'data':{'image/jpeg':base64.b64encode(buffer.getvalue()).decode('ascii'),
                        'text/plain':[name]}})
        code = ('from pathlib import Path\n'
                'from IPython.display import Image, display\n'
                f'folder = Path({str(out)!r})\n'
                f'display(Image(filename=str(folder / {r["original"]!r}), width=1200))\n'
                f'display(Image(filename=str(folder / {r["overlay"]!r}), width=1200))\n')
        cells.append({'cell_type':'code','metadata':{},'source':code.splitlines(keepends=True),
                      'execution_count':i,'outputs':outputs})
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3',
        'language':'python','name':'python3'},'language_info':{'name':'python'}},
        'nbformat':4,'nbformat_minor':4}
    (out/'preview.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')

def parse(text):
    rows, issues = [], []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip(): continue
        parts = line.split()
        if len(parts)!=5:
            issues.append(f'line {n}: expected 5 fields, got {len(parts)}'); continue
        try: vals = [float(v) for v in parts]
        except ValueError:
            issues.append(f'line {n}: invalid number'); continue
        if not all(math.isfinite(v) for v in vals):
            issues.append(f'line {n}: non-finite number'); continue
        c,x,y,w,h = vals
        if c != int(c) or c<0:
            issues.append(f'line {n}: invalid class'); continue
        if not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):
            issues.append(f'line {n}: coordinates outside normalized range')
        if min(x-w/2,y-h/2)<-1e-6 or max(x+w/2,y+h/2)>1+1e-6:
            issues.append(f'line {n}: box outside image')
        rows.append((int(c),x,y,w,h))
    return rows, issues

def key(name): return re.split(r'\.rf\.',Path(name).stem)[0]

def font(size):
    for p in ['C:/Windows/Fonts/arial.ttf','DejaVuSans.ttf']:
        try: return ImageFont.truetype(p,size)
        except OSError: pass
    return ImageFont.load_default()

def overlay(path,rows):
    # Draw on actual stored pixels. Report EXIF orientation instead of auto-rotating.
    with Image.open(path) as im: im = im.convert('RGB')
    W,H=im.size; draw=ImageDraw.Draw(im); f=font(max(18,round(W/70)))
    for i,(c,x,y,w,h) in enumerate(rows,1):
        box=((x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H)
        col=COLORS[(i-1)%len(COLORS)]
        draw.rectangle(box,outline=col,width=max(3,round(W/480)))
        text=f'{i}: class {c}'; pos=(max(0,box[0]),max(0,box[1]-32))
        draw.rectangle(draw.textbbox(pos,text,font=f),fill='#10202a')
        draw.text(pos,text,font=f,fill=col)
    return im

def compare(archive_path,records):
    current=collections.defaultdict(list)
    for r in records: current[key(r['file'])].append(r)
    with zipfile.ZipFile(archive_path) as z:
        labels,images=collections.defaultdict(list),collections.defaultdict(list)
        for n in z.namelist():
            if '/labels/' in '/'+n and n.endswith('.txt'): labels[key(n)].append(n)
            if '/images/' in '/'+n and Path(n).suffix.lower() in EXTENSIONS: images[key(n)].append(n)
        result={'archive':str(archive_path),'label_files':sum(map(len,labels.values())),
                'image_files':sum(map(len,images.values())),'matching_labels':0,
                'differing_labels':[],'missing_labels':[],'ambiguous_keys':[],
                'extra_label_keys':sorted(set(labels)-set(current)),'image_hashes':[]}
        for k,candidates in current.items():
            if len(candidates)!=1 or len(labels.get(k,[]))>1:
                result['ambiguous_keys'].append(k); continue
            if k not in labels:
                result['missing_labels'].append(k); continue
            other,issues=parse(z.read(labels[k][0]).decode('utf-8-sig'))
            left,right=sorted(candidates[0]['rows']),sorted(other)
            # Geometry warnings do not imply a difference between identical exports.
            equal=len(left)==len(right)
            if equal: equal=all(all(abs(a-b)<1e-7 for a,b in zip(u,v)) for u,v in zip(left,right))
            if equal: result['matching_labels']+=1
            else: result['differing_labels'].append({'frame':k,'current':left,'archive':right,'issues':issues})
        for r in records:
            names=images.get(key(r['file']),[])
            item={'file':r['file']}
            if len(names)!=1: item['result']='missing_or_ambiguous'
            else:
                item['same_bytes']=hashlib.sha256(Path(r['path']).read_bytes()).digest()==hashlib.sha256(z.read(names[0])).digest()
            result['image_hashes'].append(item)
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description='Check exported YOLO ground-truth boxes, not predictions')
    p.add_argument('--dataset',default=DEFAULT_DATASET,type=Path)
    p.add_argument('--out',default=SCRIPT_DIR/'manual_preview',type=Path)
    p.add_argument('--samples',default=None,type=int); p.add_argument('--name',default='')
    p.add_argument('--display',choices=['auto','inline','none'],default='auto')
    p.add_argument('--compare-zip',action='append',default=[],type=Path)
    a=p.parse_args(argv); root,out=a.dataset.resolve(),a.out.resolve()
    if not root.is_dir(): raise SystemExit(f'Папка датасета не найдена: {root}')
    if out==root or root in out.parents: raise SystemExit('Output must be outside dataset')
    if a.samples is not None and a.samples<1: raise SystemExit('--samples must be >=1')
    out.mkdir(parents=True,exist_ok=True)
    records=[]; orphan=[]; dimensions=collections.Counter(); orientations=collections.Counter()
    for split in ['train','valid','val','test']:
        idir=root/split/'images'; ldir=root/split/'labels'
        if not idir.exists(): continue
        files=sorted(f for f in idir.iterdir() if f.is_file() and f.suffix.lower() in EXTENSIONS)
        stems={f.stem for f in files}
        orphan.extend(str(f) for f in ldir.glob('*.txt') if f.stem not in stems)
        for i,path in enumerate(files):
            label=ldir/(path.stem+'.txt')
            r={'file':path.name,'path':str(path),'split':split,'label':str(label),
               'label_exists':label.exists(),'rows':[],'issues':[]}
            try:
                with Image.open(path) as im:
                    r['width'],r['height']=im.size; r['orientation']=im.getexif().get(274,1)
                dimensions[f"{r['width']}x{r['height']}"]+=1; orientations[str(r['orientation'])]+=1
                if label.exists(): r['rows'],r['issues']=parse(label.read_text(encoding='utf-8-sig'))
            except Exception as exc: r['issues'].append(f'{type(exc).__name__}: {exc}')
            records.append(r)
            if (i+1)%100==0: print(f'Inspected {split}: {i+1}/{len(files)}',flush=True)
    available=[r for r in records if 'width' in r and (not a.name or a.name in r['file'])]
    if not available: raise SystemExit('No matching images')
    count=a.samples if a.samples is not None else ask_sample_count(len(available))
    idx=evenly_spaced_indices(len(available),count)
    if count>len(available): print(f'Доступно только {len(available)} файлов; показываю все.')
    print('Позиции выбранных файлов: '+', '.join(str(i+1) for i in idx),flush=True)
    chosen=[available[i] for i in idx]; entries=[]; tiles=[]
    for i,r in enumerate(chosen,1):
        name=f"{i:02d}_{Path(r['file']).stem}_boxes.jpg"; im=overlay(r['path'],r['rows'])
        im.save(out/name,quality=95)
        original=f"{i:02d}_{Path(r['file']).stem}_original{Path(r['file']).suffix}"
        shutil.copyfile(r['path'],out/original)
        entries.append({'source':r['file'],'overlay':name,'original':original,'boxes':len(r['rows']),
                        'position':idx[i-1]+1,'total':len(available),
                        'size':[r['width'],r['height']],'issues':r['issues']})
        thumb=im.copy(); thumb.thumbnail((720,405)); tile=Image.new('RGB',(720,465),'#10202a')
        tile.paste(thumb,((720-thumb.width)//2,0)); d=ImageDraw.Draw(tile)
        m=re.search(r'frame_\d+',r['file']); title=m.group(0) if m else r['file'][:45]
        d.text((12,414),f'{i:02d} {title} | {len(r["rows"])} boxes | {r["width"]}x{r["height"]}',font=font(20),fill='white')
        tiles.append(tile)
    sheet=Image.new('RGB',(1440,465*math.ceil(len(tiles)/2)),'#10202a')
    for i,t in enumerate(tiles): sheet.paste(t,((i%2)*720,(i//2)*465))
    sheet.save(out/'contact_sheet.jpg',quality=93)
    with (out/'boxes_pixels.csv').open('w',newline='',encoding='utf-8-sig') as fp:
        wri=csv.writer(fp); wri.writerow(['image','width','height','box','class','xc_norm','yc_norm','w_norm','h_norm','x1_px','y1_px','x2_px','y2_px'])
        for r in records:
            if 'width' not in r: continue
            W,H=r['width'],r['height']
            for i,(c,x,y,w,h) in enumerate(r['rows'],1):
                wri.writerow([r['file'],W,H,i,c,x,y,w,h,(x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H])
    report={'dataset':str(root),'images':len(records),'boxes':sum(len(r['rows']) for r in records),
            'dimensions':dict(dimensions),'exif_orientation':dict(orientations),
            'classes':dict(collections.Counter(str(row[0]) for r in records for row in r['rows'])),
            'missing_labels':[r['file'] for r in records if not r['label_exists']],
            'empty_labels':[r['file'] for r in records if r['label_exists'] and not r['rows']],
            'issues':[{'image':r['file'],'issues':r['issues']} for r in records if r['issues']],
            'orphan_labels':orphan,'selection':{'strategy':'evenly_spaced',
            'requested':count,'positions_1based':[i+1 for i in idx]},'samples':entries,'comparisons':[]}
    for archive in a.compare_zip:
        print(f'Comparing {archive.name}',flush=True); report['comparisons'].append(compare(archive,records))
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    cards=[]
    for r in entries:
        cards.append(f'<figure><div class="pair"><div>Исходный кадр<a href="{html.escape(r["original"])}"><img src="{html.escape(r["original"])}"></a></div><div>Боксы из .txt<a href="{html.escape(r["overlay"])}"><img src="{html.escape(r["overlay"])}"></a></div></div><figcaption>{html.escape(r["source"])}<br>{r["boxes"]} boxes; {r["size"]}</figcaption></figure>')
    page='<html lang="ru"><meta charset="utf-8"><title>Проверка YOLO-разметки</title><style>body{background:#10202a;color:#eef;font:17px Arial;margin:24px}img{width:100%}figure{margin:0 0 30px}figcaption{overflow-wrap:anywhere}main{max-width:1800px;margin:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style><main><h1>Боксы из файлов разметки YOLO</h1><p>Это исходная разметка, не предсказания модели. Слева оригинал, справа боксы. Нажмите на кадр для полного размера. Номера обозначают боксы внутри кадра, class 0 — human.</p>'+''.join(cards)+'</main></html>'
    (out/'gallery.html').write_text(page,encoding='utf-8')
    write_preview_notebook(entries,out)
    print(json.dumps({'images':report['images'],'boxes':report['boxes'],
        'dimensions':report['dimensions'],'exif_orientation':report['exif_orientation'],
        'missing_labels':len(report['missing_labels']),'empty_labels':len(report['empty_labels']),
        'images_with_geometry_warnings':len(report['issues']),'orphan_labels':len(orphan)}),flush=True)
    for c in report['comparisons']:
        print(json.dumps({'archive':Path(c['archive']).name,'image_files':c['image_files'],
            'label_files':c['label_files'],'matching_labels':c['matching_labels'],
            'differing_labels':len(c['differing_labels']),'missing_labels':len(c['missing_labels']),
            'ambiguous_keys':len(c['ambiguous_keys']),
            'images_identical':sum(v.get('same_bytes',False) for v in c['image_hashes'])}),flush=True)
    print(f'Gallery: {out/"gallery.html"}',flush=True)
    print(f'Откройте в VS Code: {out/"preview.ipynb"}',flush=True)
    if a.display=='inline' or (a.display=='auto' and in_notebook()):
        if in_notebook(): show_inline(entries,out)
        else: print('Для картинок прямо в VS Code откройте preview.ipynb или запустите файл в Python Interactive Window.')
    return report

if __name__=='__main__':
    main([] if in_notebook() else None)
