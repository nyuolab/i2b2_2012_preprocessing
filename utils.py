import xmltodict
import os
from pathlib import Path
import pathlib
import shutil

def unzip_if_not_exists(root, folder, zfile):
    assert(check_exists(root, zfile))
    if not check_exists(root, folder):
        os.system(f"tar -xf {zfile} {folder}")

def check_exists(root, file):
    path = os.path.join(root, file)
    return os.path.exists(path)

def get_paths(d, keys):
    ptr = {}
    for key in keys:
        ptr[key] = Path(d[key])
    return ptr

def make_dir(d):
    print("making dir...")
    for key in d.keys():
        path = d[key]
        if not type(path) == pathlib.PosixPath:
            path = Path(path)
        path.mkdir(parents=True, exist_ok=True)            

def copy_text(d_src, d_dest):
    for key in d_src:
        path_src = d_src[key]
        path_dest = d_dest[key]
        print(f"copying text for split {key}. source path {path_src}, dest path {path_dest}")
        gen = path_src.glob("*.txt")
        if len(list(gen)) == 0:
            raise RuntimeError(f"empty directory! please make sure you have .txt file in {path_src}")
        cnt = 0
        for each in path_src.glob("*.txt"):
            fid = each.stem.split(".")[0]
            new_file = path_dest / f"{fid}.txt"
            shutil.copyfile(each, new_file)
            if cnt <= 5:
                print(f"[top 5 log] copied doc {fid} to {new_file}")
                cnt += 1

def get_ann_files(d_path, splits):
    res = {}
    for key in splits:
        in_path = d_path[key]
        inputfiles = set()
        for f in os.listdir(in_path):
            if f.endswith('.ann'):
                inputfiles.add(f.split('.')[0].split('_')[0])
        res[key] = inputfiles
    return res

# convert Brat format into BIO format
# function for getting entity annotations from the annotation file
def get_annotation_entities(ann_file, select_types=None):
    entities = []
    with open(ann_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith('T'):
                term = line.strip().split('\t')[1].split()
                if (select_types != None) and (term[0] not in select_types): continue
                if int(term[-1]) <= int(term[1]): continue
                entities.append((int(term[1]), int(term[-1]), term[0]))
    return sorted(entities, key=lambda x: (x[0], x[1]))

# function for handling overlap by keeping the entity with largest text span
def remove_overlap_entities(sorted_entities):
    keep_entities = []
    for idx, entity in enumerate(sorted_entities):
        if idx == 0:
            keep_entities.append(entity)
            last_keep = entity
            continue
        if entity[0] < last_keep[1]:
            if entity[1]-entity[0] > last_keep[1]-last_keep[0]:
                last_keep = entity
                keep_entities[-1] = last_keep
        elif entity[0] == last_keep[1]:
            last_keep = (last_keep[0], entity[1], last_keep[-1])
            keep_entities[-1] = last_keep
        else:
            last_keep = entity
            keep_entities.append(entity)
    return keep_entities

# inverse index of entity annotations
def entity_dictionary(keep_entities, txt_file, nlp):
    #print(f"txt_file is {txt_file}")
    file_name = os.path.basename(txt_file)
    #print(f"file name is {file_name}")
    f_ann = {}
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.readlines()
        text = ''.join([i for i in text])
    for entity in keep_entities:
        entity_text = text[entity[0]:entity[1]]
        doc = nlp(entity_text)
        token_starts = [(i, doc[i:].start_char) for i in range(len(doc))]
        term_type = entity[-1]
        term_offset = entity[0]
        for i, token in enumerate(doc):
            ann_offset = token_starts[i][1]+term_offset
            if ann_offset not in f_ann:
                f_ann[ann_offset] = [i, token.text, term_type]
    return f_ann

def brat2bio(inputfiles, inputpath, outputpath, nlp, select_type, verbose=False):
    # Brat -> BIO format conversion
    print(f"converting brat2bio {inputfiles}")
    for infile in inputfiles:
        file = f"{infile}"
        ann_file = f"{inputpath}/{file}.ann"
        txt_file = f"{inputpath}/{file}.txt"
        out_file = f"{outputpath}/{file}.bio.txt"
        if verbose:
            print(f'infile is {infile}')
            print(f'outfile is {out_file}')
        sorted_entities = get_annotation_entities(ann_file, select_type)
        keep_entities = remove_overlap_entities(sorted_entities)
        f_ann = entity_dictionary(keep_entities, txt_file, nlp)
        
        with open(out_file, "w", encoding="utf-8") as f_out:
            with open(txt_file, "r", encoding="utf-8") as f:
                sent_offset = 0
                prev_label = "O"
                for line in f:
                    if '⁄' in line:
                        line = line.replace('⁄', '/') # replace non unicode characters
                    doc = nlp(line.strip())
                    # list of tuples, first value is token index, second value is char idx
                    token_starts = [(i, doc[i:].start_char) for i in range(len(doc))]
                    for token in doc:
                        token_sent_offset = token_starts[token.i][1] # sentence level local index
                        token_doc_offset = token_starts[token.i][1] + sent_offset # document level global index
                        if token_doc_offset in f_ann:
                            if prev_label == "O" or not (prev_label.split("-")[1] == f_ann[token_doc_offset][2]): # or prev_label == f"I-{f_ann[token_doc_offset][2]}" or prev_label == f"B-{f_ann[token_doc_offset][2]}":#f_ann[token_doc_offset][0] == 0: # changed edge case to I-tag according to wikipedia
                                label = f"B-{f_ann[token_doc_offset][2]}"
                            else:
                                label = f"I-{f_ann[token_doc_offset][2]}"
                            if not (f_ann[token_doc_offset][1] == token.text_with_ws.rstrip()):
                                print('{} does not match {}'.format(f_ann[token_doc_offset][1], token.text_with_ws.rstrip()))
                                assert(False)
                        else:
                            label = f"O"
                        prev_label = label # update prev_label
                        f_out.write(f"{token.text} {token_sent_offset} {token_sent_offset+len(token.text)} {token_doc_offset} {token_doc_offset+len(token.text)} {label}\n")
                    f_out.write('\n')
                    sent_offset += (len(line))  

def brat2bio_dict(ann_files_d, infiles_d, bio_out_d, nlp, select_type=None):
    for key in ann_files_d:
        ann_file = ann_files_d[key]
        infiles = infiles_d[key]
        bio_out = bio_out_d[key]
        brat2bio(ann_file, infiles, bio_out, nlp, select_type)

def load_file(file):
    with open(file, "r") as f:
        cont = f.read()
    return cont

def write_to_file(data, file):
    with open(file, "w") as f:
        f.write(data)

def xml2brat(p1, p2, BRAT_TEMP, EVENTS, verbose=False):
    # offset need to -1 on the number
    cnt = 0
    for each in p1.glob("*.xml"):
        brat_anns = []
        idx = 1
        if cnt < 5:
            print(f'[top 5 log] converting xml {each} to brat...')
            cnt += 1
        ofn = p2 / (each.stem.split(".")[0] + ".ann")
        xml = load_file(each)
        if verbose:
            print(xml)
        xml = xml.replace('&', 'AAMMPP')
        tags = xmltodict.parse(xml)['ClinicalNarrativeTemporalAnnotation']['TAGS']
        try:
            for k, v in tags.items():
                # only keep event tags
                if k == 'EVENT':
                    for d in v:
                        typ = d['@type']
                        if typ in EVENTS:
                            s = int(d['@start']) - 1 # convert from 1-index to 0-index
                            e = int(d['@end']) - 1
                            txt = d['@text']
                            brat_anns.append(BRAT_TEMP.format(idx, typ, s, e, txt))
                            idx += 1
        except:
            print(xml)
            print(tags)
            assert(False)
        ot = "\n".join(brat_anns)
        ot = ot.replace('AAMMPP', '&')
        # break # added for debug
        write_to_file(ot, ofn)

def dataset_xml2brat(d_in, d_out, BRAT_TEMP, EVENTS, verbose=False):
    for key in d_in:
        print(f"in path is {d_in[key]}, out path is {d_out[key]}")
        xml2brat(d_in[key], d_out[key], verbose=verbose, BRAT_TEMP=BRAT_TEMP, EVENTS=EVENTS)

def make_if_nonexist(dir_s):
    if not os.path.exists(dir_s):
        os.makedirs(dir_s)