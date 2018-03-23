import re
import os


def write_files(write_dir):
    global paragraphs
    for para_dict in paragraphs:
        with open(os.path.join(write_dir,"_".join([para_dict["fid"],para_dict["para_num"]])), "w") as outf:
            outf.write(para_dict["text"])
    paragraphs = []


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("transcript_dir")
    parser.add_argument("write_dir")
    args = parser.parse_args()

ts = re.compile("^\d:\d+:\d+\.\d S\d+: (.*)")
meta = re.compile("\[.*?\]")


num_files, num_paras = 0, 0
paragraphs = []
for file in tqdm(os.listdir(args.transcript_dir)):
    if os.path.splitext(file)[-1] == ".txt":
        num_files += 1
        with open(os.path.join(args.transcript_dir,file)) as f:
            paragraph = None
            start_new_para = True
            num_para = 0
            n_lines = 0
            for line in f:
                text = re.search(ts, line)
                if text:
                    if n_lines < 5 and not re.search("_", text.group(1)):
                        out_line = re.sub(meta, "", text.group(1))
                        
                        if start_new_para:
                            paragraph = out_line
                            start_new_para = False
                        else:
                            paragraph = "\n".join([paragraph, out_line])
                        n_lines += 1
                    else:
                        if paragraph:
                            paragraphs.append({"fid": os.path.splitext(file)[0], "para_num": str(num_para), "text": paragraph})
                            paragraph = None
                        num_para += 1
                        start_new_para = True
                        n_lines = 0
            if paragraph:
                paragraphs.append({"fid": os.path.splitext(file)[0], "para_num": str(num_para), "text": paragraph})
                paragraph = None
                num_para += 1
    
    if num_files % 100 == 0:
        print("Writing paragraphs...")
        write_files(args.write_dir)

if paragraphs:
    print("Writing final paragraphs...")
    write_files(args.write_dir)    